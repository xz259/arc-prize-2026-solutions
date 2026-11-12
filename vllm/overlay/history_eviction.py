"""Pure chronological observation-block registration and eviction planning.

All coordinates are original logical token positions. A registered observation
closes its predecessor only after the input containing it commits. Planning a
pending append never makes that predecessor eligible early. This module neither
changes ResidentLedger nor performs engine/worker/cache operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from decimal import Decimal,ROUND_CEILING
import math

from residency import ResidentLedger,ResidencyError,normalize_ranges


class HistoryValidationError(ValueError):
    status_code=400


class HistoryCapacityError(RuntimeError):
    """Explicit safe-boundary fallback; the caller must not submit the append."""
    status_code=413

    def __init__(self,message,*,receipt,reason="insufficient_completed_history"):
        super().__init__(message)
        self.receipt={**receipt,"capacity_fallback":reason,"eviction_committed":False}
        self.reason=reason


def _nonnegative_integer(value):
    return type(value) is int and value>=0


@dataclass(frozen=True)
class HistoryPolicy:
    policy: str="off"
    fraction: float=0.10
    resident_token_budget: int=40000
    protect_recent_fraction: float=0.0
    unit: str="blocks"                 # stage 12: "turns" enables the turn-level planner below
    protect_recent_tokens: int=0        # absolute newest-backward protection, whole segments
    target_basis_tokens: int=0          # fraction applies to this many tokens (0: resident count)

    def __post_init__(self):
        if self.unit not in ("blocks","turns","split"):
            raise HistoryValidationError("History eviction unit must be blocks, turns or split")
        if type(self.protect_recent_tokens) is not int or self.protect_recent_tokens<0:
            raise HistoryValidationError("protect_recent_tokens must be a nonnegative integer")
        if type(self.target_basis_tokens) is not int or self.target_basis_tokens<0:
            raise HistoryValidationError("target_basis_tokens must be a nonnegative integer")
        if self.policy not in ("off","oldest_interactions","qsa_frequency","oldest_recompute"):   # oldest_recompute: same plan as oldest_interactions, executed by re-prefill
            raise HistoryValidationError("Unknown history eviction policy")
        if (type(self.fraction) not in (int,float) or not 0<self.fraction<1
                or not math.isfinite(self.fraction)):
            raise HistoryValidationError("History fraction must be finite and strictly between zero and one")
        if type(self.resident_token_budget) is not int or self.resident_token_budget<1:
            raise HistoryValidationError("History resident-token budget must be a positive integer")
        if (type(self.protect_recent_fraction) not in (int,float) or not 0<=self.protect_recent_fraction<1
                or not math.isfinite(self.protect_recent_fraction)):
            raise HistoryValidationError("Recent-protection fraction must be finite and in [0, 1)")

    @classmethod
    def from_payload(cls,config,max_context):
        if type(max_context) is not int or max_context<1:
            raise HistoryValidationError("Server context limit must be a positive integer")
        if config is None:config={}
        if not isinstance(config,dict) or set(config)-{"policy","fraction","resident_token_budget","protect_recent_fraction",
                                                       "unit","protect_recent_tokens","target_basis_tokens"}:
            raise HistoryValidationError("Invalid history policy fields")
        result=cls(policy=config.get("policy","off"),fraction=config.get("fraction",0.10),
                   resident_token_budget=config.get("resident_token_budget",max_context),
                   protect_recent_fraction=config.get("protect_recent_fraction",0.0),
                   unit=config.get("unit","blocks"),protect_recent_tokens=config.get("protect_recent_tokens",0),
                   target_basis_tokens=config.get("target_basis_tokens",0))
        if result.resident_token_budget>max_context:
            raise HistoryValidationError("History resident-token budget exceeds the server context limit")
        return result

    def target_tokens(self,resident_tokens):
        # Interpret the configured decimal value, not the binary product (e.g.
        # ceil(0.07*100) can incorrectly become eight instead of seven).
        basis=self.target_basis_tokens or resident_tokens
        return int((Decimal(str(self.fraction))*basis).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class ObservationStart:
    observation_id: str
    start: int


@dataclass(frozen=True)
class HistoryRegistration:
    revision: int
    previous_observations: tuple[ObservationStart,...]
    previous_registered_through: int
    logical_offset: int
    delta_tokens: int
    registrations: tuple[ObservationStart,...]


@dataclass(frozen=True)
class HistoryDecision:
    ranges: tuple[tuple[int,int],...]
    history_ranges: tuple[tuple[int,int],...]
    suffix_ranges: tuple[tuple[int,int],...]
    eviction_plan: object
    receipt: dict


def _intersections(start,end,intervals):
    return tuple((max(start,left),min(end,right)) for left,right in intervals
                 if start<right and left<end)


def _count(ranges):
    return sum(end-start for start,end in ranges)


IMAGE_KINDS=("observation","recap","tool","animation","other")


def _valid_image_rank(mode):
    return mode in ("min","mean") or (mode.startswith("q") and mode[1:].replace(".","",1).isdigit() and 0<float(mode[1:])<=100)


def _image_rank_by_kind():
    """ARC3_QSA_IMAGE_RANK_BY_KIND (2026-09-27): {kind: rank mode}; empty when unset (every image uses ARC3_QSA_IMAGE_RANK)."""
    raw=os.environ.get("ARC3_QSA_IMAGE_RANK_BY_KIND","").strip()
    out={}
    for item in filter(None,(x.strip() for x in raw.split(","))):
        kind,_,mode=item.partition("=");kind=kind.strip();mode=mode.strip()
        if kind not in IMAGE_KINDS or not _valid_image_rank(mode):
            raise ValueError("ARC3_QSA_IMAGE_RANK_BY_KIND entries must be <kind>=<min|mean|q<percent>> with kind in "+", ".join(IMAGE_KINDS))
        out[kind]=mode
    return out


def _pin_observation_enabled():
    """ARC3_PIN_OBSERVATION (default 0): keep the latest observation out of evictable history (2026-09-26)."""
    raw=os.environ.get("ARC3_PIN_OBSERVATION","0").strip().lower()
    if raw not in ("0","1","false","true"):
        raise ValueError("ARC3_PIN_OBSERVATION must be 0/1 or false/true")
    return raw in ("1","true")


def _pinned_observation(observations,segments,logical):
    """[start,end) of the latest registered observation's segment, or None when it is not recorded."""
    if not observations:return None
    start=observations[-1].start
    for seg in reversed(segments):
        a=int(seg[1])
        if a<start:break
        if seg[0]=="observation" and a==start:
            end=min(int(seg[2]),logical)
            return (start,end) if end>start else None
    return None


def _union(ranges,extra):
    """Sorted union of disjoint ranges and one more range (overlap and adjacency merged)."""
    rows=sorted([*ranges,*([extra] if extra else [])]);out=[]
    for a,b in rows:
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out


class HistoryRegistry:
    def __init__(self):
        self.revision=0
        self.observations=()
        self.registered_through=0

    def plan_registration(self,observation_starts,*,logical_offset,delta_tokens):
        """Validate new input starts without closing any current block yet."""
        if (not _nonnegative_integer(logical_offset) or not _nonnegative_integer(delta_tokens)
                or logical_offset<self.registered_through):
            raise HistoryValidationError("History registration coordinates rewrite committed input")
        if not isinstance(observation_starts,list):
            raise HistoryValidationError("Observation starts must be a list")
        seen={record.observation_id for record in self.observations}
        registrations=[]
        previous=-1
        for record in observation_starts:
            if (not isinstance(record,dict) or not {"observation_id","local_start_token"}<=set(record)
                    or set(record)-{"observation_id","local_start_token","proof"}
                    or ("proof" in record and not isinstance(record["proof"],dict))):
                raise HistoryValidationError("Observation starts require identity, local token start and optional proof object")
            oid=record["observation_id"];local=record["local_start_token"]
            if type(oid) is not str or not oid or len(oid)>160 or oid in seen:
                raise HistoryValidationError("Observation identity is missing, too long, or reused")
            if not _nonnegative_integer(local) or not previous<local<delta_tokens:
                raise HistoryValidationError("Observation starts must increase within the appended input")
            absolute=logical_offset+local
            if self.observations and absolute<=self.observations[-1].start:
                raise HistoryValidationError("Observation start precedes committed history")
            registrations.append(ObservationStart(oid,absolute))
            previous=local;seen.add(oid)
        return HistoryRegistration(self.revision,self.observations,self.registered_through,
                                   logical_offset,delta_tokens,tuple(registrations))

    def commit(self,transition):
        """Publish only after the caller has committed the corresponding input."""
        if (not isinstance(transition,HistoryRegistration) or transition.revision!=self.revision
                or transition.previous_observations!=self.observations
                or transition.previous_registered_through!=self.registered_through):
            raise HistoryValidationError("Stale history registration")
        records=[{"observation_id":entry.observation_id,
                  "local_start_token":entry.start-transition.logical_offset}
                 for entry in transition.registrations]
        fresh=self.plan_registration(records,logical_offset=transition.logical_offset,
                                     delta_tokens=transition.delta_tokens)
        if fresh!=transition:
            raise HistoryValidationError("History registration was altered")
        self.observations+=transition.registrations
        self.registered_through=transition.logical_offset+transition.delta_tokens
        self.revision+=1

    def snapshot(self):
        """Logical spans; the active block has end=None until its successor commits."""
        rows=self.observations
        return {"revision":self.revision,"registered_through":self.registered_through,
                "protected_prefix_tokens":rows[0].start if rows else 0,
                "active_observation_id":rows[-1].observation_id if rows else None,
                "blocks":[{"observation_id":row.observation_id,"start":row.start,
                           "end":rows[i+1].start if i+1<len(rows) else None}
                          for i,row in enumerate(rows)]}

    def plan_eviction(self,policy,ledger,*,pending_suffix_ranges=(),incoming_tokens=0,frequency=None,segments=(),structural=(),images=(),incoming_observation=False,image_kinds=()):
        return plan_history_eviction(policy,self,ledger,pending_suffix_ranges=pending_suffix_ranges,
                                     incoming_tokens=incoming_tokens,frequency=frequency,segments=segments,structural=structural,images=images,
                                     incoming_observation=incoming_observation,image_kinds=image_kinds)


def _rank_by_frequency(policy,eligible,frequency,observations,logical,survivors,resident,required,receipt):
    """Reorder eligible whole interactions by selection frequency, lowest first.

    Newest-backward recent protection is applied first (whole blocks only,
    active observation counted toward the allocation). Any inability to rank
    is a recorded chronological fallback, never a silent one.
    """
    from qsa_frequency import score_blocks
    receipt.update(requested_policy="qsa_frequency",effective_policy="qsa_frequency",fallback_reason=None,
                   protect_recent_fraction=policy.protect_recent_fraction,recent_protected_block_ids=[],
                   block_scores=[],block_coverage=[])
    protect_target=int((Decimal(str(policy.protect_recent_fraction))*resident).to_integral_value(rounding=ROUND_CEILING))
    active_tokens=_count(_intersections(observations[-1].start,logical,survivors)) if observations else 0
    protected_tokens=active_tokens;kept=[];shielded=[]
    for entry in reversed(eligible):
        if protected_tokens>=protect_target:kept.append(entry)
        else:protected_tokens+=entry[3];shielded.append(entry)
    kept.reverse();shielded.reverse()
    receipt["recent_protected_block_ids"]=[e[0].observation_id for e in shielded]
    receipt["recent_protected_tokens"]=protected_tokens
    if sum(e[3] for e in kept)<required and shielded:
        # Optional protection must not by itself create a capacity failure.
        while shielded and sum(e[3] for e in kept)<required:
            kept.append(shielded.pop())
        kept.sort(key=lambda e:e[0].start)
        receipt["fallback_reason"]="recent_protection_relaxed"
    eligible=kept
    if frequency is None:
        receipt["effective_policy"]="oldest_interactions"
        receipt["fallback_reason"]=receipt["fallback_reason"] or "no_frequency_snapshot"
        return eligible
    scores,coverage,usable=score_blocks(eligible,frequency)
    receipt["block_scores"]=[{"observation_id":e[0].observation_id,"score":s,"coverage":c}
                             for e,s,c in zip(eligible,scores,coverage)]
    receipt["epoch_query_positions"]=frequency.get("epoch_query_positions")
    receipt["epoch_informative_rows"]=frequency.get("informative")
    if not usable:
        receipt["effective_policy"]="oldest_interactions"
        receipt["fallback_reason"]=receipt["fallback_reason"] or "insufficient_exposure_coverage"
        return eligible
    order=sorted(range(len(eligible)),key=lambda i:(scores[i],eligible[i][0].start))
    return [eligible[i] for i in order]



def _segment_pieces(seg,survivors):
    return _intersections(seg[1],seg[2],survivors)


def _plan_turns(policy,observations,logical,survivors,prefix,tail_start,required,frequency,segments,receipt):
    """Turn-level selection (stage 12). Segments: (kind, start, end) over logical positions, kinds observation |
    tool | assistant, in order, covering the committed history after the protected prefix. Returns the removal pieces."""
    segs=[tuple(x) for x in segments if x[2]>x[1]]
    segs=[x for x in segs if x[1]>=prefix and x[2]<=logical]
    segs.sort(key=lambda x:x[1])
    receipt.update(segments=len(segs),tail_protected_tokens=0,tail_protected_segments=0,selected_segments=[],selected_groups=0)
    if not segs:
        receipt["candidate_block_ids"]=[]
        raise HistoryCapacityError("No turn segments recorded for turn-level eviction",receipt=receipt)
    resident_of=lambda seg:_count(_segment_pieces(seg,survivors))
    # (b) newest-backward tail protection by whole segments (the last four tokens are protected separately)
    protected=set();acc=0
    for i in range(len(segs)-1,-1,-1):
        if acc>=policy.protect_recent_tokens:break
        protected.add(i);acc+=resident_of(segs[i])
    receipt["tail_protected_tokens"]=acc;receipt["tail_protected_segments"]=len(protected)
    eligible_idx=[i for i in range(len(segs)) if i not in protected and resident_of(segs[i])>0
                  and not any(start<logical and tail_start<end for start,end in _segment_pieces(segs[i],survivors))]
    receipt["eligible_history_tokens"]=sum(resident_of(segs[i]) for i in eligible_idx)
    # group segments by observation block (an observation segment opens a block)
    blocks=[];current=None
    for i,seg in enumerate(segs):
        if seg[0]=="observation" or current is None:
            current=[];blocks.append(current)
        current.append(i)
    if policy.policy=="qsa_frequency":
        pieces=_plan_turns_qsa(policy,segs,eligible_idx,survivors,required,frequency,receipt)
        if pieces is not None:
            return pieces
    receipt["effective_policy"]=receipt.get("effective_policy") or "oldest_turns"
    removed=0;history=[]
    for block in blocks:
        if removed>=required:break
        elig=[i for i in block if i in eligible_idx]
        if not elig:continue
        whole_ok=all(i in eligible_idx for i in block)   # a block with protected segments is never taken whole
        tokens=sum(resident_of(segs[i]) for i in elig)
        if whole_ok and removed+tokens<=required:
            for i in elig:
                history.extend(_segment_pieces(segs[i],survivors))
                receipt["selected_segments"].append([segs[i][0],segs[i][1],segs[i][2]])
            removed+=tokens
            receipt["selected_block_ids"].append("block@%d"%segs[block[0]][1])
            continue
        # crossing (or active, or partially protected) block: keep its observation, take assistant+tool pairs oldest-first
        pairs=[];pair=None
        for i in block:
            kind=segs[i][0]
            if kind=="observation":continue
            if kind=="assistant" or pair is None:
                pair=[];pairs.append(pair)
            pair.append(i)
        for pair in pairs:
            if removed>=required:break
            if any(i not in eligible_idx for i in pair):continue   # protected or already evicted: keep the pair intact
            for i in pair:
                history.extend(_segment_pieces(segs[i],survivors))
                receipt["selected_segments"].append([segs[i][0],segs[i][1],segs[i][2]])
                removed+=resident_of(segs[i])
    receipt["history_removed_tokens"]=removed
    if removed<required:
        receipt["candidate_block_ids"]=[]
        raise HistoryCapacityError("Too few evictable turns to meet the eviction target",receipt=receipt)
    return history


def _plan_turns_qsa(policy,segs,eligible_idx,survivors,required,frequency,receipt):
    """Descriptor-group selection inside the eligible region by ascending selection frequency; None -> fallback."""
    from qsa_frequency import RATIO
    receipt.update(requested_policy="qsa_frequency",effective_policy="qsa_frequency",fallback_reason=None)
    if frequency is None:
        receipt["effective_policy"]="oldest_turns";receipt["fallback_reason"]="no_frequency_snapshot";return None
    layers=frequency["frequency"]
    counts={}   # group -> resident positions inside eligible segments
    for i in eligible_idx:
        for start,stop in _segment_pieces(segs[i],survivors):
            for position in range(start,stop):
                counts.setdefault(position//RATIO,[]).append(position)
    if not counts:
        receipt["effective_policy"]="oldest_turns";receipt["fallback_reason"]="no_eligible_groups";return None
    scored=[];covered=0
    for g,positions in counts.items():
        values=[table.get(g) for table in layers]
        known=[v for v in values if v is not None]
        if known:covered+=len(positions)
        scored.append((sum(known)/len(layers) if known else float("inf"),g,positions))
    total=sum(len(v) for v in counts.values())
    receipt["group_coverage"]=covered/total
    receipt["epoch_query_positions"]=frequency.get("epoch_query_positions");receipt["epoch_informative_rows"]=frequency.get("informative")
    if covered/total<0.5:
        receipt["effective_policy"]="oldest_turns";receipt["fallback_reason"]="insufficient_exposure_coverage";return None
    scored.sort(key=lambda t:(t[0],t[1]))
    removed=0;history=[];groups=0
    for score,g,positions in scored:
        if removed>=required:break
        if score==float("inf"):break   # unexposed groups are never chosen on frequency
        positions.sort();run_start=positions[0];previous=positions[0]
        for pos in positions[1:]+[None]:
            if pos is not None and pos==previous+1:previous=pos;continue
            history.append((run_start,previous+1))
            if pos is not None:run_start=previous=pos
        removed+=len(positions);groups+=1
    receipt["history_removed_tokens"]=removed;receipt["selected_groups"]=groups
    if removed<required:
        receipt["effective_policy"]="oldest_turns";receipt["fallback_reason"]="insufficient_exposed_groups";return None
    return history


def _clip_pieces(seg,survivors,lo,hi):
    """Resident pieces of a (kind,start,end) segment restricted to [lo,hi)."""
    return [(max(a,lo),min(b,hi)) for a,b in _segment_pieces(seg,survivors) if min(b,hi)>max(a,lo)]


def _take_tokens(pieces,count):
    """Oldest-first prefix of `pieces` holding `count` tokens."""
    taken=[];got=0
    for a,b in pieces:
        if got>=count:break
        take=min(b-a,count-got);taken.append((a,a+take));got+=take
    return taken,got


def _subtract(pieces,removed):
    """pieces minus removed (both lists of half-open ranges); linear in the two lengths after sorting."""
    out=[];removed=sorted(removed)
    for a,b in sorted(pieces):
        cur=a
        for c,d in removed:
            if d<=cur or c>=b:continue
            if c>cur:out.append((cur,c))
            cur=max(cur,d)
            if cur>=b:break
        if cur<b:out.append((cur,b))
    return out


def _snap_cut_to_images(pieces,count,images):
    """Take `count` oldest tokens of `pieces`; if the cut would end inside an image span, extend it to the image end."""
    spans=sorted((int(s),int(e)) for s,e in images if int(e)>int(s))
    for _ in range(len(spans)+1):
        cut,rest=_take_tokens(pieces,count)
        end=cut[-1][1] if cut else None
        hit=next(((s,e) for s,e in spans if end is not None and s<end<e),None)
        if hit is None:return cut,rest
        count+=hit[1]-end
    return _take_tokens(pieces,count)


def _plan_split(policy,survivors,prefix,hend,logical,required,frequency,segments,structural,receipt,images=(),image_kinds=()):
    """Stage 12 'split' unit: strict token regions. Protected: [0,prefix) and [hend,logical); H=[prefix,hend).
    2026-09-23 guards: H ends on a pair boundary (an assistant turn and the tool/observation segments answering it
    are protected together; the trailing turn is still engine output until the next append folds it into the
    prompt), and never inside an image (a straddling image stays whole)."""
    receipt.update(pair_guard=None,image_boundary_guard=None,requested_history_end=hend,short_quantum=None)
    owner=None;prev_assistant=None
    for seg in segments:
        k,a,b=seg[0],int(seg[1]),int(seg[2])
        if k=="assistant":prev_assistant=a
        if a<hend<b or (a==hend and k!="assistant"):
            owner=prev_assistant;break
        if a>hend:break
    if owner is not None and owner<hend:
        receipt["pair_guard"]=[owner,hend];hend=max(prefix,owner)
    for s_,e_ in sorted((int(s),int(e)) for s,e in images):
        if s_<hend<e_:
            hend=max(prefix,s_);receipt["image_boundary_guard"]=[s_,e_];break
    receipt.update(split_history_range=[prefix,hend],protected_tail_range=[hend,logical],
                   tail_protected_tokens=max(0,logical-hend),selected_segments=[],selected_groups=0,
                   partial_assistant_cut=None,tool_overshoot_tokens=0,fallback_stage=None,straddling_interaction_start=None)
    hpieces=_intersections(prefix,hend,survivors) if hend>prefix else []
    receipt["eligible_history_tokens"]=_count(hpieces)
    if not hpieces:
        raise HistoryCapacityError("No resident history between the protected prefix and tail",receipt=receipt,
                                   reason="empty_history_region")
    # 2026-09-24 rule 4: fixed quantum; a guarded history region smaller than it is taken whole if that covers the real
    # capacity need. The protected tail is never cut.
    available=receipt["eligible_history_tokens"];need=int(receipt.get("capacity_needed_tokens") or 0)
    if available<required:
        if available<need:
            raise HistoryCapacityError("Protected tail leaves less resident history than the capacity need",receipt=receipt,
                                       reason="tail_exhausted")
        receipt["short_quantum"]={"target_tokens":required,"available_tokens":available,"capacity_needed_tokens":need}
        required=available
    if policy.policy=="qsa_frequency":
        pieces=_plan_split_qsa(policy,hpieces,required,frequency,structural,receipt,images,segments,(prefix,hend),image_kinds)
        if pieces is not None:
            return pieces
    receipt["effective_policy"]=receipt.get("effective_policy") or "oldest_split"
    return _plan_split_oldest(segments,survivors,hpieces,prefix,hend,required,receipt,images)


def _plan_split_oldest(segments,survivors,hpieces,prefix,hend,required,receipt,images=()):
    segs=sorted((tuple(x) for x in segments if x[2]>x[1] and x[2]>prefix and x[1]<hend),key=lambda x:x[1])
    interactions=[];current=None
    for seg in segs:                      # an observation opens an interaction; leading tool/assistant join the first
        if seg[0]=="observation" or current is None:
            current=[];interactions.append(current)
        current.append(seg)
    res=lambda seg:_clip_pieces(seg,survivors,prefix,hend)
    removed=0;history=[]
    def take(pieces):
        nonlocal removed
        history.extend(pieces);removed+=_count(pieces)
    straddled=False
    for block in interactions:
        if removed>=required:break
        whole=[piece for seg in block for piece in res(seg)]
        size=_count(whole)
        if not size:continue
        if not straddled and removed+size<=required:
            take(whole);receipt["selected_segments"].append(["interaction",block[0][1],block[-1][2]]);continue
        if not straddled:
            straddled=True;receipt["straddling_interaction_start"]=block[0][1]
        pairs=[];pair=None                # keep the observation; assistant opens a pair, following tool turns join it
        for seg in block:
            if seg[0]=="observation":continue
            if seg[0]=="assistant" or pair is None:
                pair=[];pairs.append(pair)
            pair.append(seg)
        for pair in pairs:
            if removed>=required:break
            pieces=[piece for seg in pair for piece in res(seg)];tokens=_count(pieces)
            if not tokens:continue
            if removed+tokens<=required:
                take(pieces);receipt["selected_segments"].append([pair[0][0],pair[0][1],pair[-1][2]]);continue
            need=required-removed
            assistant=[piece for seg in pair if seg[0]=="assistant" for piece in res(seg)]
            if need<=_count(assistant):
                cut,_=_take_tokens(assistant,need);take(cut)
                receipt["partial_assistant_cut"]=[cut[0][0],cut[-1][1]]
            else:                          # the count reaches the tool turn: the pair goes whole (edge case)
                take(pieces);receipt["tool_overshoot_tokens"]=removed-required
                receipt["selected_segments"].append([pair[0][0],pair[0][1],pair[-1][2]])
            break
    if removed<required:                   # last resort: oldest remaining tokens of H (kept observations included)
        receipt["fallback_stage"]="token_count_over_history"
        cut,_=_snap_cut_to_images(_subtract(hpieces,history),required-removed,images);take(cut)   # never split an image
    receipt["history_removed_tokens"]=removed
    if removed<required:
        raise HistoryCapacityError("Resident history region smaller than the eviction requirement",receipt=receipt,
                                   reason="history_region_exhausted")
    return history


# ---------------------------------------------------------------- structure-preserving QSA (2026-09-24)
HOLLOW_FLOOR=8

def _hollow_fraction():
    return float(os.environ.get("ARC3_HOLLOW_FRACTION","0.8"))

class _Cover:
    """Sorted disjoint [a,b) intervals with O(log n) covered-count and membership queries (prefix sums, built once)."""
    __slots__=("s","e","c")
    def __init__(self,iv):
        from itertools import accumulate
        self.s=[x[0] for x in iv];self.e=[x[1] for x in iv];self.c=[0,*accumulate(y-x for x,y in iv)]
    def count(self,a,b):
        return self._F(b)-self._F(a)
    def _F(self,x):
        from bisect import bisect_right
        i=bisect_right(self.s,x)-1
        return 0 if i<0 else self.c[i]+min(max(x-self.s[i],0),self.e[i]-self.s[i])
    def has(self,q):
        from bisect import bisect_right
        i=bisect_right(self.s,q)-1
        return i>=0 and q<self.e[i]
    def pieces(self,a,b):
        from bisect import bisect_right
        out=[];i=bisect_right(self.e,a)
        while i<len(self.s) and self.s[i]<b:
            x,y=max(self.s[i],a),min(self.e[i],b)
            if y>x:out.append((x,y))
            i+=1
        return out

def _hollow_blocks(segments,hpieces,history,structural,lo,hi,receipt):
    """Remaining pieces of every turn segment wholly inside [lo,hi) that is hollow after this selection.
    Only segments touched by this selection can newly cross the threshold (the rule runs at every eviction); for each,
    resident and structural counts come from O(log n) prefix-sum and binary-search queries, and pieces are built only
    for blocks that cross. Runs in the API server during planning, while the engine keeps decoding every other lane."""
    from bisect import bisect_left,bisect_right
    receipt.update(hollow_fraction=_hollow_fraction(),hollow_floor=HOLLOW_FLOOR,hollow_blocks_evicted=0,hollow_tokens=0,hollow_candidates=0)
    if not history or not hpieces:return []
    frac=_hollow_fraction();seg_starts=[int(x[1]) for x in segments]
    cand=[];seen=set()
    for a,b in history:                       # segments overlapping each selected piece
        i=max(0,bisect_right(seg_starts,a)-1)
        while i<len(segments) and seg_starts[i]<b:
            s0,s1=seg_starts[i],int(segments[i][2])
            if s1>a and s0>=lo and s1<=hi and s1>s0 and s0 not in seen:seen.add(s0);cand.append((s0,s1))
            i+=1
    receipt["hollow_candidates"]=len(cand)
    if not cand:return []
    H=_Cover(hpieces);Y=_Cover(history);extra=[];n=0
    for s0,s1 in cand:
        res_after=H.count(s0,s1)-Y.count(s0,s1)
        if res_after==0 or res_after>=s1-s0:continue
        j0,j1=bisect_left(structural,s0),bisect_left(structural,s1);n_struct=j1-j0
        content_total=(s1-s0)-n_struct;limit=max(HOLLOW_FLOOR,int((1.0-frac)*content_total))
        if content_total<=0 or res_after-n_struct>limit:continue      # cannot be hollow: skip the per-token work
        struct_after=sum(1 for q in structural[j0:j1] if H.has(q) and not Y.has(q))
        if res_after-struct_after<=limit:
            extra.extend(_subtract(H.pieces(s0,s1),Y.pieces(s0,s1)));n+=1
    receipt["hollow_blocks_evicted"]=n;receipt["hollow_tokens"]=_count(extra)
    return extra

def _plan_split_qsa(policy,hpieces,required,frequency,structural,receipt,images=(),segments=(),hbounds=None,image_kinds=()):
    """Descriptor groups over all of H by ascending selection frequency; None -> earliest fallback.
    2026-09-23: vectorized (numpy) when the snapshot carries the frequency matrix; identical semantics otherwise."""
    import numpy as np
    from qsa_frequency import RATIO
    receipt.update(requested_policy="qsa_frequency",effective_policy="qsa_frequency",fallback_reason=None)
    if frequency is None:
        receipt["effective_policy"]="oldest_split";receipt["fallback_reason"]="no_frequency_snapshot";return None
    receipt["structural_exclusion"]="on";receipt["structural_positions"]=len(structural);receipt["excluded_structural_groups"]=0
    groups={}
    for a,b in hpieces:
        for g in range(a//RATIO,(b-1)//RATIO+1):
            lo,hi=max(a,g*RATIO),min(b,(g+1)*RATIO)
            groups.setdefault(g,[]).append((lo,hi))
    receipt["excluded_structural_tokens"]=0
    if structural and groups:   # formatting never selected (2026-09-24), at token granularity (review fix 2026-09-25)
        from bisect import bisect_left,bisect_right
        lo_h,hi_h=hpieces[0][0],hpieces[-1][1];j0,j1=bisect_left(structural,lo_h),bisect_left(structural,hi_h)
        inside=structural[j0:j1];n_ex=0;sg=set()
        for s_,e_ in images:    # an image holding a structural position (not rendered in practice) stays whole
            if bisect_left(inside,int(e_))>bisect_left(inside,int(s_)):
                sg.update(g for g in range(int(s_)//RATIO,(int(e_)-1)//RATIO+1) if g in groups)
        for q in inside:        # subtract each formatting token from its group's resident pieces
            g=q//RATIO;pcs=groups.get(g)
            if pcs is None or g in sg:continue
            out=[]
            for a,b in pcs:
                if a<=q<b:
                    n_ex+=1
                    if a<q:out.append((a,q))
                    if q+1<b:out.append((q+1,b))
                else:out.append((a,b))
            if out:groups[g]=out
            else:del groups[g]
        for g in sg:del groups[g]
        receipt["excluded_structural_groups"]=len(sg);receipt["excluded_structural_tokens"]=n_ex
    if not groups:
        receipt["effective_policy"]="oldest_split";receipt["fallback_reason"]="no_eligible_groups";return None
    gids=np.fromiter(sorted(groups),dtype=np.int64,count=len(groups))
    counts=np.fromiter((_count(groups[int(g)]) for g in gids),dtype=np.int64,count=len(gids))
    inf=float("inf")
    if frequency.get("matrix") is not None:
        M=frequency["matrix"];lay=np.asarray(frequency["group_ids_array"],dtype=np.int64)
        sc=np.full(len(gids),inf)
        if len(lay):
            idx=np.searchsorted(lay,gids);ok=idx<len(lay);ok[ok]=lay[idx[ok]]==gids[ok]
            with np.errstate(invalid="ignore"):
                m=np.nanmean(M[:,idx[ok]],axis=0) if ok.any() else np.zeros(0)
            sc[ok]=np.where(np.isnan(m),inf,m)
    else:                                   # legacy per-layer dict tables
        layers=list(frequency["frequency"]);sc=np.full(len(gids),inf)
        for k,g in enumerate(gids.tolist()):
            known=[t.get(g) for t in layers];known=[v for v in known if v is not None]
            if known:sc[k]=sum(known)/len(layers)
    finite=sc<inf;total=int(counts.sum());covered=int(counts[finite].sum())
    receipt["group_coverage"]=covered/total
    receipt["epoch_query_positions"]=frequency.get("epoch_query_positions");receipt["epoch_informative_rows"]=frequency.get("informative")
    image_of={}                           # group -> image index: an image is evicted whole or not at all
    for i,(s_,e_) in enumerate(images):
        for g in range(int(s_)//RATIO,(int(e_)-1)//RATIO+1):
            if g in groups:image_of[g]=i
    receipt["image_groups"]=len(image_of)
    if covered/total<0.5:
        receipt["effective_policy"]="oldest_split";receipt["fallback_reason"]="insufficient_exposure_coverage";return None
    # units: singleton groups and whole-image bundles; image rank = min (current) or mean of member frequencies
    rank_mode=os.environ.get("ARC3_QSA_IMAGE_RANK","min");receipt["image_rank"]=rank_mode
    if not (rank_mode in ("min","mean") or (rank_mode.startswith("q") and rank_mode[1:].replace(".","",1).isdigit() and 0<float(rank_mode[1:])<=100)):
        raise ValueError("ARC3_QSA_IMAGE_RANK must be min, mean or q<percent>")
    by_kind=_image_rank_by_kind();kinds=list(image_kinds)   # 2026-09-27: per-kind image rank
    receipt["image_rank_by_kind"]=by_kind or None
    pos={int(g):k for k,g in enumerate(gids.tolist())}
    members={}
    for g,i in image_of.items():members.setdefault(i,[]).append(pos[g])
    unit_score=[];unit_members=[];unit_key=[];unit_image=[]
    in_image=np.zeros(len(gids),dtype=bool)
    for i,ks in members.items():
        ks=sorted(ks);in_image[ks]=True;vals=sc[ks];fin=vals[vals<inf]
        mode=by_kind.get(kinds[i] if i<len(kinds) else "other",rank_mode) if by_kind else rank_mode
        if mode.startswith("q"):   # 2026-09-25: percentile rank over all members (unexposed = never cheap)
            import math
            pct=float(mode[1:])/100.0;srt=np.sort(vals);score=float(srt[max(0,min(len(srt)-1,math.ceil(pct*len(srt))-1))])
        else:
            score=(inf if not len(fin) else (float(fin.min()) if mode!="mean" else float(fin.mean())))
        unit_score.append(score);unit_members.append(ks);unit_key.append(int(gids[ks[0]]));unit_image.append(i)
    receipt["image_member_scores"]=[[round(float(v),5) if v<inf else None for v in np.quantile(np.where(sc[ks]<inf,sc[ks],np.nan),(0,0.25,0.5,0.75)).tolist()] if np.isfinite(sc[ks]).any() else None
                                    for ks in members.values()]   # 2026-09-25: per resident image, member-score min/q25/q50/q75 (calibration)
    for k in np.nonzero(~in_image)[0].tolist():
        unit_score.append(float(sc[k]));unit_members.append([k]);unit_key.append(int(gids[k]));unit_image.append(None)
    order=sorted(range(len(unit_score)),key=lambda u:(unit_score[u],unit_key[u]))
    removed=0;chosen_k=[];images_evicted=0;evicted_by_kind={}
    for u in order:
        if removed>=required:break
        if unit_score[u]==inf:break        # unexposed groups are never chosen on frequency
        ks=unit_members[u];chosen_k.extend(ks);removed+=int(counts[ks].sum())
        if len(ks)>1 or in_image[ks[0]]:
            images_evicted+=1;ik=unit_image[u];kk=kinds[ik] if ik is not None and ik<len(kinds) else "other"
            evicted_by_kind[kk]=evicted_by_kind.get(kk,0)+1
    receipt["images_evicted"]=images_evicted;receipt["images_evicted_by_kind"]=evicted_by_kind
    # frequency distribution of the exposed groups and of the chosen set (receipt only)
    exposed=np.sort(sc[finite]);chosen_scores=np.sort(sc[chosen_k][sc[chosen_k]<inf]) if chosen_k else np.zeros(0)
    kept_mask=finite.copy();kept_mask[chosen_k]=False;kept=sc[kept_mask]
    def _q(v,q):return (float(v[min(len(v)-1,int(q*len(v)))]) if len(v) else None)
    receipt.update(exposed_groups=int(finite.sum()),unexposed_groups=int((~finite).sum()),zero_frequency_groups=int((exposed==0).sum()),
                   frequency_quantiles={str(q):_q(exposed,q) for q in (0,0.1,0.25,0.5,0.75,0.9,1.0)},
                   chosen_zero_frequency=int((chosen_scores==0).sum()),chosen_exposed=int(len(chosen_scores)),
                   chosen_score_quantiles={str(q):_q(chosen_scores,q) for q in (0,0.5,0.9,1.0)},
                   cut_score=(float(chosen_scores[-1]) if len(chosen_scores) else None),kept_min_score=(float(kept.min()) if len(kept) else None))
    if removed<required:
        receipt["effective_policy"]="oldest_split";receipt["fallback_reason"]="insufficient_exposed_groups";return None
    history=[]
    for k in sorted(chosen_k):
        for a,b in groups[int(gids[k])]:
            if history and history[-1][1]==a:history[-1]=(history[-1][0],b)
            else:history.append((a,b))
    if hbounds is not None:     # 2026-09-24: hollow blocks go whole (overshoot recorded)
        extra=_hollow_blocks(segments,hpieces,history,structural,hbounds[0],hbounds[1],receipt)
        if extra:
            merged=[]
            for a,b in sorted(history+extra):
                if merged and merged[-1][1]>=a:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
                else:merged.append((a,b))
            removed+=_count(extra);history=merged
    receipt["history_removed_tokens"]=removed;receipt["selected_groups"]=len(chosen_k)
    return history


def plan_history_eviction(policy,registry,ledger,*,pending_suffix_ranges=(),incoming_tokens=0,frequency=None,segments=(),structural=(),images=(),incoming_observation=False,image_kinds=()):
    """Select the shortest oldest prefix of complete resident observation blocks.

The denominator is resident tokens after pending managed-suffix retirement.
Suffix spans may lie in the current active observation: they are a separate
authorized phase-two operation. History eviction never selects that active
block. Both operations preserve the initial system/tool prefix and last four
original accepted tokens. No mutation occurs, including on capacity failure.
    """
    if not isinstance(policy,HistoryPolicy) or not isinstance(registry,HistoryRegistry):
        raise HistoryValidationError("Expected a validated history policy and registry")
    if not isinstance(ledger,ResidentLedger) or not _nonnegative_integer(incoming_tokens):
        raise HistoryValidationError("Expected a resident ledger and nonnegative incoming token count")
    logical=ledger.logical_tokens
    if registry.registered_through>logical:
        raise HistoryValidationError("History registry is ahead of the committed logical ledger")
    observations=registry.observations
    if any(record.start>=logical for record in observations):
        raise HistoryValidationError("Registered observation lies beyond committed tokens")
    prefix=observations[0].start if observations else 0
    tail_start=max(0,logical-4)
    # Prefix and tail can overlap for tiny streams; form a union, not overlapping
    # protected ranges (ResidentLedger intentionally rejects overlapping inputs).
    protected=[]
    if prefix:protected.append((0,prefix))
    if logical:
        if protected and tail_start<=prefix:protected=[(0,logical)]
        else:protected.append((tail_start,logical))
    for start,end in protected:
        if any(start<right and left<end for left,right in ledger.evicted):
            raise HistoryValidationError("Existing residency already removed protected prefix/tail tokens")
    try:
        suffix_plan=ledger.plan(pending_suffix_ranges,protected=protected)
    except ResidencyError as exc:
        raise HistoryValidationError(str(exc)) from exc
    suffix_ranges=suffix_plan.ranges
    suffix_tokens=suffix_plan.removed_tokens
    resident=suffix_plan.resident_after
    budget=policy.resident_token_budget
    triggered=resident+incoming_tokens>=budget
    target=policy.target_tokens(resident) if triggered and policy.policy in ("oldest_interactions","qsa_frequency","oldest_recompute") else 0
    capacity_needed=max(0,resident+incoming_tokens+1-budget)
    required=max(target,capacity_needed) if triggered else 0
    active_id=observations[-1].observation_id if observations else None
    receipt={"policy":policy.policy,"fraction":policy.fraction,"resident_token_budget":budget,
             "unit":policy.unit,"protect_recent_tokens":policy.protect_recent_tokens,"target_basis_tokens":policy.target_basis_tokens,
             "execution":("recompute" if policy.policy=="oldest_recompute" else "compaction"),
             "triggered":triggered,"fraction_basis":("target_basis_tokens" if policy.target_basis_tokens else "resident_tokens_after_pending_suffix"),
             "resident_tokens_before":ledger.resident_tokens,"resident_tokens_after_suffix":resident,
             "incoming_tokens":incoming_tokens,"target_tokens":target,"required_tokens":required,
             "capacity_needed_tokens":capacity_needed,"history_removed_tokens":0,
             "suffix_removed_tokens":suffix_tokens,"selected_block_ids":[],"selected_blocks":[],
             "protected_prefix_tokens":prefix,"protected_prefix_range":[0,prefix],
             "active_observation_id":active_id,
             "protected_active_range":[observations[-1].start,logical] if observations else None,
             "protected_tail_range":[tail_start,logical],"eligible_history_tokens":0,
             "skipped_zero_resident_block_ids":[],"tail_protected_block_ids":[]}
    history=[];pin=None
    if triggered and policy.policy!="off" and policy.unit=="split":
        hend=max(prefix,logical-policy.protect_recent_tokens)
        survivors=suffix_plan.survivor_intervals
        if _pin_observation_enabled():   # 2026-09-26: the latest observation stays until the next one arrives
            pin=None if incoming_observation else _pinned_observation(observations,segments,logical)
            receipt["observation_pin"]={"range":list(pin) if pin else None,"tokens":(pin[1]-pin[0]) if pin else 0,
                                        "released_for_incoming":bool(incoming_observation),"relaxed":False}
        if pin:
            try:
                history=_plan_split(policy,_subtract(survivors,[pin]),prefix,hend,logical,required,frequency,segments,structural,receipt,images,image_kinds)
            except HistoryCapacityError:   # optional protection must not by itself create a capacity failure
                pin=None;receipt["observation_pin"]["relaxed"]=True
        if pin is None:
            history=_plan_split(policy,survivors,prefix,hend,logical,required,frequency,segments,structural,receipt,images,image_kinds)
    elif triggered and policy.policy!="off" and policy.unit=="turns":
        history=_plan_turns(policy,observations,logical,suffix_plan.survivor_intervals,prefix,tail_start,
                            required,frequency,segments,receipt)
    elif triggered:
        if policy.policy=="off":
            raise HistoryCapacityError("Resident capacity reached with history eviction disabled",
                                       receipt=receipt,reason="history_eviction_disabled")
        eligible=[]
        for current,next_observation in zip(observations,observations[1:]):
            pieces=_intersections(current.start,next_observation.start,suffix_plan.survivor_intervals)
            tokens=_count(pieces)
            if not tokens:
                receipt["skipped_zero_resident_block_ids"].append(current.observation_id)
                continue
            if any(start<logical and tail_start<end for start,end in pieces):
                receipt["tail_protected_block_ids"].append(current.observation_id)
                continue
            eligible.append((current,next_observation.start,pieces,tokens))
        if policy.policy=="qsa_frequency":
            eligible=_rank_by_frequency(policy,eligible,frequency,observations,logical,
                                        suffix_plan.survivor_intervals,resident,required,receipt)
        available=sum(entry[3] for entry in eligible)
        receipt["eligible_history_tokens"]=available
        if available<required:
            receipt["candidate_block_ids"]=[entry[0].observation_id for entry in eligible]
            raise HistoryCapacityError("Too few complete resident observation blocks to meet the eviction target",
                                       receipt=receipt)
        removed=0
        for current,end,pieces,tokens in eligible:
            history.extend(pieces);removed+=tokens
            receipt["selected_block_ids"].append(current.observation_id)
            receipt["selected_blocks"].append({"observation_id":current.observation_id,
                "logical_range":[current.start,end],"resident_tokens":tokens})
            if removed>=required:break
        receipt["history_removed_tokens"]=removed
    try:
        history_ranges=normalize_ranges(history,limit=logical)
        combined=normalize_ranges((*suffix_ranges,*history_ranges),limit=logical)
        eviction_plan=ledger.plan(combined,protected=_union(protected,pin))   # a pinned observation is protected too
    except ResidencyError as exc:
        raise HistoryValidationError(str(exc)) from exc
    if eviction_plan.resident_after+incoming_tokens>=budget:
        # Defensive guard: no caller may forward an input with no output slot.
        raise HistoryCapacityError("Whole-block plan still leaves no output-token capacity",receipt=receipt)
    receipt.update(total_removed_tokens=eviction_plan.removed_tokens,
                   resident_tokens_after=eviction_plan.resident_after,
                   resident_tokens_after_input=eviction_plan.resident_after+incoming_tokens,
                   minimum_output_capacity=budget-eviction_plan.resident_after-incoming_tokens,
                   combined_ranges=[list(pair) for pair in combined],
                   history_ranges=[list(pair) for pair in history_ranges],
                   suffix_ranges=[list(pair) for pair in suffix_ranges])
    return HistoryDecision(combined,history_ranges,suffix_ranges,eviction_plan,receipt)
