"""Read saved phase-2 gameplay evidence; never contact a server or execute code.

The authoritative input is raw-harness persistent-request, not the readable
history/transcript. Engine/service journals establish resident state and work.
This auditor uses no model/runtime imports and reports unobserved gates clearly.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct
from decimal import Decimal, ROUND_CEILING
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                rows.append(row)
    return rows


class Checks:
    def __init__(self):
        self.counts, self.failed, self.failures = Counter(), Counter(), []

    def check(self, ok, key, detail=None):
        self.counts[key] += 1
        if not ok:
            self.failed[key] += 1
            if len(self.failures) < 100:
                self.failures.append(dict(check=key, detail=str(detail)[:1200]))
        return bool(ok)

    def equal(self, actual, expected, key, detail=None):
        return self.check(actual == expected, key,
                          dict(location=detail, expected=expected, actual=actual))


def png_identity(part):
    """Validate the wire PNG envelope/dimensions; not a pixel-fidelity claim."""
    url = part["image_url"]["url"]
    if not url.startswith("data:image/png;base64,"):
        raise ValueError("Expected inline PNG; remote image fetching is forbidden")
    data = base64.b64decode(url.split(",", 1)[1], validate=True)
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Malformed PNG header")
    width, height = struct.unpack(">II", data[16:24])
    if not 0 < width <= 4096 or not 0 < height <= 4096:
        raise ValueError("Unsupported PNG dimensions")
    return dict(sha256=hashlib.sha256(data).hexdigest(), width=width, height=height)


def wire_index(rows, checks):
    requests, responses, closed, contexts = {}, {}, set(), {}
    for row in rows:
        kind, payload = row.get("kind"), row.get("payload", {})
        if kind == "persistent-request":
            body = payload["body"]
            op = body["request_id"]
            checks.check(op not in requests or requests[op] == payload,
                         "wire_retry_body_identical", op)
            requests[op] = payload
            contexts[op] = {k: row.get(k) for k in ("game_id", "request", "epoch", "_file")}
        elif kind == "persistent-response":
            op = payload["id"]
            checks.check(op not in responses or responses[op] == payload,
                         "wire_retry_response_identical", op)
            responses[op] = payload
        elif kind == "persistent-close":
            if payload.get("state") == "closed":
                closed.add(payload["session_id"])
        elif kind == "persistent-close-error":
            checks.check(False, "wire_close_without_error", payload)
    checks.check(set(responses) <= set(requests), "no_unmatched_wire_response")
    return requests, responses, closed, contexts


def only(items, check, label, detail):
    check.equal(len(items), 1, label, detail)
    return items[0] if len(items) == 1 else {}


def ranges_from_positions(values):
    result=[]
    for position in sorted(set(values)):
        if result and result[-1][1]==position:result[-1][1]+=1
        else:result.append([position,position+1])
    return result


def history_decision_checks(c, *, policy, observations, logical_count, positions,
                            suffix_ranges, incoming, receipt, operation):
    """Independently select resident positions; do not import the implementation."""
    suffix={p for a,b in suffix_ranges for p in range(a,b)}
    present=set(positions)
    c.check(suffix<=present,"suffix_never_redeletes_existing_holes",operation)
    remaining=present-suffix
    budget=policy['resident_token_budget'];fraction=policy['fraction']
    triggered=len(remaining)+incoming>=budget
    target=int((Decimal(str(fraction))*len(remaining)).to_integral_value(rounding=ROUND_CEILING)) if triggered and policy['policy']!='off' else 0
    needed=max(0,len(remaining)+incoming+1-budget)
    required=max(target,needed) if triggered else 0
    prefix=observations[0][1] if observations else 0
    active=observations[-1] if observations else None
    tail=max(0,logical_count-4)
    chosen=[];history=set();skipped=[];tail_blocks=[];eligible=[]
    if triggered:
        c.equal(policy['policy'],'oldest_interactions','submitted_history_trigger_requires_enabled_policy',operation)
        for current,next_obs in zip(observations,observations[1:]):
            cells={p for p in remaining if current[1]<=p<next_obs[1]}
            if not cells:skipped.append(current[0]);continue
            if any(p>=tail for p in cells):tail_blocks.append(current[0]);continue
            eligible.append((current,next_obs[1],cells))
        c.check(sum(len(row[2]) for row in eligible)>=required,'submitted_history_has_sufficient_completed_blocks',operation)
        for current,end,cells in eligible:
            chosen.append({'observation_id':current[0],'logical_range':[current[1],end],'resident_tokens':len(cells)})
            history.update(cells)
            if len(history)>=required:break
    combined=suffix|history
    expected={'policy':policy['policy'],'fraction':fraction,'resident_token_budget':budget,
        'triggered':triggered,'fraction_basis':'resident_tokens_after_pending_suffix',
        'resident_tokens_before':len(positions),'resident_tokens_after_suffix':len(remaining),
        'incoming_tokens':incoming,'target_tokens':target,'required_tokens':required,
        'capacity_needed_tokens':needed,'history_removed_tokens':len(history),
        'suffix_removed_tokens':len(suffix),'selected_block_ids':[b['observation_id'] for b in chosen],
        'selected_blocks':chosen,'protected_prefix_tokens':prefix,'protected_prefix_range':[0,prefix],
        'active_observation_id':active[0] if active else None,
        'protected_active_range':[active[1],logical_count] if active else None,
        'protected_tail_range':[tail,logical_count],
        'eligible_history_tokens':sum(len(row[2]) for row in eligible),
        'skipped_zero_resident_block_ids':skipped,'tail_protected_block_ids':tail_blocks,
        'total_removed_tokens':len(combined),'resident_tokens_after':len(positions)-len(combined),
        'resident_tokens_after_input':len(positions)-len(combined)+incoming,
        'minimum_output_capacity':budget-len(positions)+len(combined)-incoming,
        'combined_ranges':ranges_from_positions(combined),'history_ranges':ranges_from_positions(history),
        'suffix_ranges':ranges_from_positions(suffix)}
    for key,value in expected.items():
        c.equal(receipt.get(key),value,'history_decision_'+key,operation)
        if type(value) in (int,bool):
            c.check(type(receipt.get(key)) is type(value),'history_decision_type_'+key,operation)
    c.check(not any(p<prefix or p>=tail for p in combined),'history_preserves_fixed_prefix_and_tail',operation)
    c.check(not active or all(p<active[1] for p in history),'history_preserves_current_active_block',operation)
    c.check(len(history)>=required and (not chosen or len(history)-chosen[-1]['resident_tokens']<required),
            'history_shortest_oldest_prefix_rounds_up',operation)
    c.check(expected['minimum_output_capacity']>=1,'history_plan_leaves_output_capacity',operation)
    return expected


def register_history_checks(c, *, sub, descriptors, observations, operation):
    starts=sub.get('observation_starts',[]);delta=sub['appended_token_ids'];old=sub['old_tokens']
    c.equal([r.get('observation_id') for r in starts],[d['observation_id'] for d in descriptors],
            'history_registered_start_ids_match_observations',operation)
    updated=list(observations)
    for row in starts:
        local=row.get('local_start_token');oid=row.get('observation_id');proof=row.get('proof',{})
        if not c.check(type(local)is int and 0<=local<len(delta),'history_start_inside_new_input',operation):continue
        c.check(oid not in {r[0] for r in updated} and (not updated or old+local>updated[-1][1]),
                'history_starts_unique_and_chronological',operation)
        c.equal(proof.get('preceding_token_count'),local,'history_paired_prefix_length',operation)
        c.equal(proof.get('prefix_sha256'),digest(delta[:local]),'history_paired_prefix_exact_ids',operation)
        opening=proof.get('opening_token_ids',[])
        c.check(bool(opening) and delta[local:local+len(opening)]==opening,'history_user_opening_exact_ids',operation)
        c.equal(proof.get('includes_generation_prompt'),False,'history_boundary_excludes_generation_prefix',operation)
        c.equal(proof.get('method'),'paired_no_generation_empty_user_probe_v1' if sub['initial']
                else 'paired_no_generation_preceding_messages_v1','history_paired_render_method',operation)
        updated.append((oid,old+local))
    expected={'revision':sub['revision']+1,'registered_through':old+len(delta),
        'protected_prefix_tokens':updated[0][1] if updated else 0,
        'active_observation_id':updated[-1][0] if updated else None,
        'blocks':[{'observation_id':r[0],'start':r[1],'end':updated[i+1][1] if i+1<len(updated) else None}
                  for i,r in enumerate(updated)]}
    c.equal(sub.get('history_state'),expected,'history_registry_exact_committed_blocks',operation)
    return updated,expected


def audit(engine_rows, service_rows, wire_rows, *, session_ids=None,
          deadline_epoch=None, deadline_slack=20, require_eviction=True, context_limit=40000):
    c = Checks()
    requests, responses, wire_closed, contexts = wire_index(wire_rows, c)
    wire_errors={row.get('payload',{}).get('request_id'):row.get('payload',{})
                 for row in wire_rows if row.get('kind')=='persistent-error'}
    selected = set(session_ids or [p["body"]["session_id"] for p in requests.values()
                                  if p["route"].endswith("/start")])
    c.check(bool(selected), "selected_sessions_present")
    reports, totals, observed = [], Counter(), Counter()
    used_ops = set()
    for sid in sorted(selected):
        sr = [r for r in service_rows if r.get("session_id") == sid]
        c.check(bool(sr), "service_session_present", sid)
        ids = {r.get("engine_request_id") for r in sr}
        c.check(len(ids) == 1, "stable_service_engine_identity", sid)
        external = next(iter(ids), "")
        er = [r for r in engine_rows if re.fullmatch(re.escape(external)+r"(?:-[0-9a-f]{8})?", r.get("request_id", ""))]
        c.check(bool(er) and len({r["request_id"] for r in er}) == 1, "one_engine_owner", sid)
        for r in er:
            c.equal(r.get("preemptions"), 0, "zero_preemptions", (sid, r.get("event")))
        submits = [r for r in sr if r.get("event") == "submit"]
        commits = [r for r in sr if r.get("event") == "evict_commit"]
        eevicts = [r for r in er if r.get("event") == "evict"]
        c.equal(len(eevicts), len(commits), "engine_service_eviction_count", sid)
        failures = [r for r in sr if r.get("event") in ("failed", "rejected")]
        cancelled = [r for r in failures if "CancelledError" in r.get("error", "")
                     and deadline_epoch is not None
                     and abs(r.get("time", 0)-deadline_epoch) <= deadline_slack]
        capacity_refusals=[]
        for failure in failures:
            receipt=failure.get('history_eviction');op=failure.get('operation_id')
            wire_error=wire_errors.get(op,{})
            if (failure['event']=='rejected' and isinstance(receipt,dict)
                    and receipt.get('eviction_committed') is False and receipt.get('triggered') is True
                    and receipt.get('capacity_fallback') in ('insufficient_completed_history','history_eviction_disabled')
                    and op in requests and not any(row.get('operation_id')==op for row in submits+commits)
                    and wire_error.get('status_code')==413 and wire_error.get('uncommitted_capacity_refusal') is True
                    and wire_error.get('body',{}).get('error',{}).get('history_eviction')==receipt):
                capacity_refusals.append(failure)
                c.check(op not in responses,'uncommitted_capacity_refusal_has_no_completion',op)
        c.check(all(r in cancelled or r in capacity_refusals for r in failures),
                "only_explicit_deadline_or_capacity_stop", (sid, failures))
        logical, positions, owners, live, all_owners = [], [], {}, [], set()
        history_observations=[];history_policy={'policy':'off','fraction':.10,'resident_token_budget':context_limit}
        previous_history_state=None
        revision = eviction_revision = 0
        previous_pause, assistant_ids, pending_tools = None, [], []
        report = dict(session_id=sid, completed_turns=0, completed_resume_checks=0,
                      new_observations=0, tool_only_appends=0, continuations=0,
                      committed_evictions=0, removed_tokens=0, freed_pages=0,
                      received_output_tokens=0, scheduled_prefill_tokens=0,
                      wire_images=[], unreturned_final_turns=0, history_evictions=0,
                      history_removed_tokens=0, safe_capacity_refusals=len(capacity_refusals))
        used_commits = set()
        for i, sub in enumerate(submits):
            op = sub["operation_id"]
            c.check(op not in used_ops, "single_submission_per_request_id", op)
            used_ops.add(op)
            request = requests.get(op)
            if not c.check(request is not None, "submit_has_authoritative_wire", op):
                continue
            body, initial = request["body"], sub["initial"]
            messages = body["chat"]["messages"] if initial else body["external_messages"]
            allowed = {"system", "user"} if initial else {"tool", "user"}
            c.check(all(m.get("role") in allowed for m in messages), "no_replayed_assistant_or_system", op)
            c.check(all(not any(k.startswith("_") for k in m) for m in messages), "no_private_wire_message_fields", op)
            c.equal(initial, i == 0, "initial_order", op)
            c.equal(request["route"].rsplit("/", 1)[-1], "start" if initial else "append", "wire_route", op)
            c.equal(sub["revision"], revision, "service_submit_revision", op)
            if not initial:
                c.equal(body.get("expected_revision"), revision, "wire_expected_revision", op)
                c.equal(body.get("expected_prefix_sha256"), digest(logical), "wire_expected_raw_digest", op)
            else:
                c.equal(sum(m["role"] == "system" for m in messages), 1, "system_once_at_start", op)
                requested=body.get('history_eviction',{})
                c.check(isinstance(requested,dict),'history_start_policy_object',op)
                if isinstance(requested,dict):history_policy.update(requested)
                valid_policy=(history_policy.get('policy') in ('off','oldest_interactions')
                    and type(history_policy.get('fraction')) in (int,float)
                    and math.isfinite(history_policy['fraction']) and 0<history_policy['fraction']<1
                    and type(history_policy.get('resident_token_budget')) is int
                    and 0<history_policy['resident_token_budget']<=context_limit)
                if not c.check(valid_policy,'history_pinned_start_policy_valid',op):
                    history_policy={'policy':'off','fraction':.10,'resident_token_budget':context_limit}
                c.check('history_eviction' not in body['chat'],'history_policy_outside_chat',op)
                report['history_policy']=dict(history_policy)
            if not initial:c.check('history_eviction' not in body,'history_policy_only_at_start',op)
            sidecar = body.get("managed_suffixes", {})
            c.equal(sidecar.get("version"), 1, "sidecar_version", op)
            desc = sidecar.get("observations", [])
            retire = sidecar.get("evict_observation_ids", [])
            records = sub.get("suffix_records", [])
            user_indices = [j for j,m in enumerate(messages) if m["role"] == "user"]
            c.equal(len(desc), len(user_indices), "every_new_observation_has_descriptor", op)
            c.check(len(desc) <= 1, "one_observation_per_append", op)
            c.equal(len(records), len(desc), "one_rendered_record_per_descriptor", op)
            if desc:
                observed["new_observations"] += 1
                report["new_observations"] += 1
                c.equal(retire, live, "new_observation_retires_previous_live_suffix", op)
            else:
                c.equal(retire, [], "same_observation_no_eviction", op)
                c.equal(records, [], "same_observation_no_suffix_refresh", op)
                observed["frozen_tool_or_continuation"] += int(not initial)
                report["tool_only_appends"] += int(bool(messages) and not initial)
            continuation = body.get("continuation", False)
            c.equal(continuation, not initial and not messages, "empty_input_only_continuation", op)
            report["continuations"] += int(continuation)
            ranges = [owners[oid] for oid in retire if oid in owners]
            c.equal(len(ranges), len(retire), "retired_owners_registered", op)
            history_receipt=sub.get('history_eviction')
            history_tracked=history_policy['policy']!='off' or history_receipt is not None or 'observation_starts' in sub
            expected_history=None
            if history_tracked:
                if not c.check(isinstance(history_receipt,dict),'history_decision_receipt_present',op):history_receipt={}
                expected_history=history_decision_checks(c,policy=history_policy,observations=history_observations,
                    logical_count=len(logical),positions=positions,suffix_ranges=ranges,
                    incoming=len(sub['appended_token_ids']),receipt=history_receipt,operation=op)
                ranges=expected_history['combined_ranges']
                report['history_removed_tokens']+=expected_history['history_removed_tokens']
                report['history_evictions']+=int(expected_history['history_removed_tokens']>0)
            drop = {p for a,b in ranges for p in range(a,b)}
            kept = [p for p in positions if p not in drop]
            removed = len(positions)-len(kept)
            anchor = previous_pause
            if removed:
                sc = only([r for r in commits if r.get("operation_id") == op], c, "one_commit_per_removal", op)
                used_commits.add(op)
                plan, ack = sc.get("plan", {}), sc.get("acknowledgement", {})
                c.equal(plan.get("ranges"), [list(x) for x in ranges], "eviction_only_owned_ranges", op)
                if history_tracked:
                    c.equal(sc.get('history_eviction'),history_receipt,'history_commit_exact_decision_receipt',op)
                    c.equal(plan.get('prior_evicted'),ranges_from_positions(set(range(len(logical)))-set(positions)),
                            'history_plan_prior_holes_exact',op)
                    c.equal(plan.get('survivor_intervals'),ranges_from_positions(kept),'history_plan_survivors_exact',op)
                    c.equal(plan.get('logical_tokens'),len(logical),'history_plan_keeps_full_logical_frontier',op)
                    attempt=only([r for r in sr if r.get('event')=='evict_submit' and r.get('operation_id')==op],
                                 c,'history_one_eviction_preflight_record',op)
                    c.equal(attempt.get('history_state'),previous_history_state,'history_selection_uses_preappend_closed_blocks',op)
                    c.equal(attempt.get('history_eviction'),history_receipt,'history_preflight_exact_decision',op)
                c.equal(plan.get("resident_before"), len(positions), "eviction_before_count", op)
                c.equal(plan.get("resident_after"), len(kept), "eviction_shrink_count", op)
                c.equal(plan.get("removed_tokens"), removed, "eviction_removed_count", op)
                c.equal(ack.get("plan_digest"), plan.get("digest"), "ack_plan_digest", op)
                eviction_revision += 1
                for key, value in dict(evicted_tokens=removed, resident_tokens=len(kept),
                                       logical_tokens=len(logical), eviction_revision=eviction_revision).items():
                    c.equal(ack.get(key), value, "ack_"+key, op)
                ee = only([r for r in eevicts if r.get("eviction", {}).get("plan_digest") == plan.get("digest")],
                          c, "one_engine_eviction_ack", op)
                c.equal(ee.get("eviction"), ack, "service_commits_exact_engine_ack", op)
                c.equal(ee.get("tokens_sha256"), digest([logical[p] for p in kept]), "eviction_exact_resident_ids", op)
                c.equal(ee.get("computed_tokens"), len(kept)-1, "eviction_preserves_pending_final", op)
                c.equal(ee.get("in_flight_tokens"), 0, "eviction_settled", op)
                receipts = ack.get("worker_receipts", [])
                c.check(len(receipts) == 1 and receipts[0].get("worker_synchronized") is True,
                        "synchronized_worker_receipt", op)
                copies = [p for r in receipts for p in r.get("copies", [])]
                c.check(bool(copies), "actual_cache_copy_receipts", op)
                observed["byte_hash_enabled_evictions"] += int(bool(receipts) and all(r.get("byte_verification_enabled") is True for r in receipts))
                for copy in copies:
                    proof = copy.get("byte_verification")
                    if receipts and all(r.get('byte_verification_enabled') is True for r in receipts):
                        c.check(isinstance(proof,dict),'enabled_byte_verification_has_actual_proof',op)
                    if proof is not None:
                        c.check(proof.get("equal") is True and proof.get("before") == proof.get("after")
                                and bool(re.fullmatch(r"[a-f0-9]{64}", proof.get("before", {}).get("sha256", ""))),
                                "retained_cache_bytes_equal", op)
                before = anchor.get("block_ids", []) if anchor else []
                after, freed = ee.get("block_ids", []), ack.get("freed_blocks_by_group", [])
                c.check(len(before) == len(after) == len(freed) and all(set(a)-set(b) == set(f)
                        and not set(b)-set(a) for a,b,f in zip(before,after,freed)), "freed_pages_match_owner_change", op)
                report["freed_pages"] += sum(map(len, freed))
                report["committed_evictions"] += 1
                report["removed_tokens"] += removed
                anchor = ee
            else:
                c.check(not any(r.get("operation_id") == op for r in commits), "no_unrequested_compaction", op)
            positions = kept
            if history_tracked:
                history_observations,previous_history_state=register_history_checks(
                    c,sub=sub,descriptors=desc,observations=history_observations,operation=op)
            for oid in retire:
                owners.pop(oid, None)
            if desc:
                live = []
            for d, record in zip(desc, records):
                j, oid = d["message_index"], d["observation_id"]
                c.check(j in user_indices and j == len(messages)-1, "descriptor_targets_new_final_user", op)
                message = messages[j]
                parts = message["content"]
                c.equal(d["content_sha256"], digest(parts), "wire_content_hash", op)
                for key,value in d.items():
                    c.equal(record.get(key), value, "rendered_descriptor_"+key, op)
                c.check(oid not in all_owners, "unique_observation_owner", op)
                all_owners.add(oid)
                base = d["base_parts_end"]
                suffix_parts = [x for x in (d.get("program_parts"), d.get("snapshot_parts")) if x]
                cursor = base
                for a,b in suffix_parts:
                    c.check(a == cursor and a < b <= len(parts), "contiguous_terminal_suffix_parts", op)
                    cursor = b
                c.equal(cursor, len(parts), "suffix_covers_all_extra_parts", op)
                interval = record.get("local_token_range")
                c.equal(bool(interval), bool(suffix_parts), "token_span_presence", op)
                if interval:
                    a,b = interval
                    c.check(0 < a < b < len(sub["appended_token_ids"]), "interior_expanded_span", op)
                    owners[oid] = (len(logical)+a, len(logical)+b)
                    live.append(oid)
                for n,part in enumerate(parts):
                    if part.get("type") == "image_url":
                        identity = png_identity(part)
                        report["wire_images"].append(dict(operation_id=op, part=n, managed=n >= base, **identity))
                for image in record.get("image_spans", []):
                    a,b = image["envelope_range"]
                    c.check(interval is not None and interval[0] <= a < b <= interval[1],
                            "evict_image_envelope_wholly_owned", op)
                    f,g = image["feature_range"]
                    token_ids = sub["appended_token_ids"]
                    c.check(a == f-1 and b == g+1 and token_ids[a] == 248053
                            and token_ids[b-1] == 248054 and g > f
                            and all(t == 248056 for t in token_ids[f:g]),
                            "expanded_image_tokens_match_owned_record", op)
                expected_images = sum(p.get("type") == "image_url" for p in parts[base:])
                c.equal(len(record.get("image_spans", [])), expected_images,
                        "all_managed_wire_images_have_expanded_spans", op)
            c.equal(sub.get("managed_suffixes"), dict(registered_observation_ids=[d["observation_id"] for d in desc],
                                                       evicted_observation_ids=retire), "service_ownership_receipt", op)
            for message in messages:
                if message["role"] == "tool":
                    c.check(bool(pending_tools) and message.get("tool_call_id") == pending_tools.pop(0), "tool_feedback_identity", op)
            c.check(not pending_tools, "prior_tool_calls_answered_before_generation", op)
            delta = sub["appended_token_ids"]
            c.equal(sub["old_tokens"], len(logical), "service_logical_before", op)
            c.equal(sub["prefix_sha256"], digest(logical), "service_logical_prefix", op)
            c.equal(sub["resident_before_append"], len(positions), "service_resident_before_append", op)
            computed = len(positions)-1 if i else 0
            positions.extend(range(len(logical), len(logical)+len(delta)))
            logical.extend(delta)
            c.equal(sub["logical_input_tokens"], len(logical), "service_logical_input", op)
            c.equal(sub["input_tokens"], len(positions), "service_resident_input", op)
            entries = [r for r in er if r.get("turn") == i and r.get("event") == ("start" if initial else "resume")]
            c.check(len(entries) <= 1, "at_most_one_engine_entry", op)
            entry = entries[0] if len(entries) == 1 else {}
            if entry:
                c.equal(entry.get("computed_tokens"), computed, "engine_retained_frontier", op)
                c.equal(entry.get("tokens_sha256"), digest([logical[p] for p in positions]), "engine_resident_input_ids", op)
            if not initial and entry:
                expected_blocks = (anchor or {}).get("block_ids")
                share = (entry.get("append") or {}).get("prefix_share") or {}
                if share.get("action") == "shared" and expected_blocks:   # 2026-09-27: private leading pages -> shared
                    expected_blocks = [list(group) for group in expected_blocks]; g = share["group_id"]; k = len(share["old_block_ids"])
                    if expected_blocks[g][:k] == share["old_block_ids"]:
                        expected_blocks[g][:k] = share["shared_block_ids"]
                c.equal(entry.get("block_ids"), expected_blocks, "resume_same_owned_pages", op)
                c.equal(entry.get("append", {}).get("pending_prefill_tokens"), len(delta)+1, "only_delta_plus_pending_final_prefill", op)
            pauses = [r for r in sr if r.get("event") == "pause" and r.get("operation_id") == op]
            if not pauses:
                report["unreturned_final_turns"] += 1
                c.check(i == len(submits)-1 and bool(cancelled), "unreturned_final_is_deadline_cancel", op)
                c.check(op not in responses, "unreturned_final_has_no_wire_completion", op)
                continue
            sp = only(pauses, c, "one_service_pause", op)
            c.check(bool(entry), "completed_turn_has_engine_entry", op)
            output = sp["output_token_ids"]
            response = responses.get(op, {})
            c.check(bool(response), "completed_pause_has_wire_response", op)
            choice = (response.get("choices") or [{}])[0]
            persistent = response.get("persistent", {})
            if history_tracked:
                c.equal(persistent.get('history_eviction'),history_receipt,'history_wire_receipt_matches_committed_decision',op)
                c.equal(persistent.get('history_state'),previous_history_state,'history_wire_registered_blocks_exact',op)
            c.equal(choice.get("token_ids"), output, "wire_choice_exact_emitted_ids", op)
            c.equal(persistent.get("token_ids"), output, "wire_receipt_exact_emitted_ids", op)
            parse = choice.get("persistent_parse", {})
            assistant_ids = assistant_ids + output if continuation else output
            if continuation or parse:
                c.equal(parse.get("assistant_token_ids"), assistant_ids, "cumulative_assistant_ids", op)
            if response:
                pending_tools = ([t["id"] for t in choice.get("message", {}).get("tool_calls", [])]
                                 if choice.get("finish_reason") == "tool_calls" else [])
            input_count, logical_input = len(positions), len(logical)
            positions.extend(range(len(logical), len(logical)+len(output)))
            logical.extend(output)
            revision += 1
            for key,value in dict(session_id=sid, revision=revision, resident_tokens=len(positions),
                                   logical_tokens=len(logical), logical_input_tokens=logical_input,
                                   appended_prompt_tokens=len(delta), prefix_sha256=digest(logical),
                                   prefix_caching=False, eviction=True, eviction_revision=eviction_revision,
                                   evicted_tokens=removed, managed_suffixes=sub.get("managed_suffixes")).items():
                c.equal(persistent.get(key), value, "response_"+key, op)
            c.equal(response.get("usage"), dict(prompt_tokens=input_count, completion_tokens=len(output),
                                                total_tokens=input_count+len(output)), "wire_usage", op)
            for key,value in dict(revision=revision, logical_tokens=len(logical), resident_tokens=len(positions),
                                   prefix_sha256=digest(logical)).items():
                c.equal(sp.get(key), value, "service_pause_"+key, op)
            ep = only([r for r in er if r.get("turn") == i and r.get("event") == "pause"], c, "one_engine_pause", op)
            for key,value in dict(resident_tokens=len(positions), computed_tokens=len(positions)-1,
                                   tokens_sha256=digest([logical[p] for p in positions]),
                                   computed_prefix_sha256=digest([logical[p] for p in positions[:-1]]),
                                   in_flight_tokens=0, output_placeholders=0).items():
                c.equal(ep.get(key), value, "engine_pause_"+key, op)
            ls = ep.get("logical_eviction_state", {})
            c.equal(ls.get("logical_sha256"), digest(logical), "engine_original_logical_identity", op)
            c.equal(ls.get("deleted_tokens"), len(logical)-len(positions), "engine_deleted_count", op)
            work = ep.get("turn_work", {})
            prefill = len(delta)+(not initial)
            for key,value in dict(prefill_tokens=prefill, decode_tokens=len(output)-1,
                                   scheduled_tokens=prefill+len(output)-1,
                                   resume_computed_tokens=computed, minimum_scheduled_position=computed).items():
                c.equal(work.get(key), value, "no_historical_refill_"+key, op)
            report["completed_turns"] += 1
            report["completed_resume_checks"] += int(not initial)
            report["received_output_tokens"] += len(output)
            report["scheduled_prefill_tokens"] += prefill
            previous_pause = ep
        c.equal(used_commits, {r["operation_id"] for r in commits}, "no_unmatched_committed_eviction", sid)
        for refused in capacity_refusals:
            receipt=refused['history_eviction'];op=refused['operation_id']
            for key,value in history_policy.items():c.equal(receipt.get(key),value,'capacity_refusal_pinned_'+key,op)
            c.equal(refused.get('revision'),revision,'capacity_refusal_did_not_advance_revision',op)
            c.equal(receipt.get('resident_tokens_before'),len(positions),'capacity_refusal_resident_ledger_unchanged',op)
            c.equal(receipt.get('history_removed_tokens'),0,'capacity_refusal_no_history_selected_or_removed',op)
            c.check(not any(row.get('operation_id')==op for row in sr if row.get('event')=='evict_submit'),
                    'capacity_refusal_before_engine_mutation',op)
            c.equal(wire_errors[op].get('body',{}).get('error',{}).get('type'),'persistent_session_error',
                    'capacity_refusal_is_typed_session_error',op)
            if receipt.get('capacity_fallback')=='history_eviction_disabled':
                c.equal(history_policy['policy'],'off','capacity_disabled_matches_policy',op)
            else:
                c.equal(history_policy['policy'],'oldest_interactions','capacity_insufficient_matches_policy',op)
                c.check(type(receipt.get('eligible_history_tokens')) is int and type(receipt.get('required_tokens')) is int
                        and 0<=receipt['eligible_history_tokens']<receipt['required_tokens'],
                        'capacity_refusal_reports_insufficient_completed_history',op)
        c.check(all(type(r.get("turn")) is int and 0 <= r["turn"] < len(submits)
                    for r in er if r.get("event") in ("start", "resume", "pause")),
                "no_unmatched_engine_generation_turn", sid)
        close = only([r for r in sr if r.get("event") == "close"], c, "one_service_close", sid)
        c.equal(close.get("revision"), revision, "closed_revision", sid)
        c.equal(close.get("resident_tokens"), len(positions), "closed_known_resident_ledger", sid)
        release = only([r for r in er if r.get("event") == "close_released"], c, "one_engine_close_release", sid)
        c.check(bool(release.get("block_ids")) and not any(release["block_ids"]), "close_frees_all_blocks", sid)
        c.equal(release.get("request_still_registered"), False, "close_unregisters_request", sid)
        c.check(sid in wire_closed, "wire_close_receipt", sid)
        report["cleanup_verified"] = bool(close and release and not any(release.get("block_ids", [1]))
                                           and release.get("request_still_registered") is False)
        report["deadline_cancellations"] = len(cancelled)
        for key,value in report.items():
            if type(value) is int:
                totals[key] += value
        reports.append(report)
    c.check(set(responses) <= used_ops, "all_wire_completions_audited")
    if require_eviction:
        c.check(totals["committed_evictions"] > 0, "actual_eviction_observed")
    return dict(status="fail" if c.failed else "pass", checks=dict(c.counts), failed_checks=dict(c.failed),
                failures=c.failures, totals=dict(totals), sessions=reports,
                observed=dict(observed), gates={
                    "resident_deletion": "observed" if totals["committed_evictions"] else "not_observed",
                    "whole_interaction_eviction": "observed" if totals['history_evictions'] else "not_observed",
                    "explicit_capacity_refusal": "observed" if totals['safe_capacity_refusals'] else "not_observed",
                    "physical_page_release_after_eviction": "observed" if totals["freed_pages"] else "not_observed",
                    "retained_byte_hashes": "observed" if observed["byte_hash_enabled_evictions"] else "not_observed",
                    "different_request_page_reuse": "not_checked",
                    "image_pixel_fidelity_against_source_frames": "not_checked",
                    "image_semantic_understanding": "not_checked",
                    "readable_transcript_fidelity": "not_checked"},
                limits=["Input claims use persistent HTTP wire; readable history is not replayed input.",
                        "PNG headers, ownership and exact byte identities do not establish correct game pixels or model understanding.",
                        "Work/frontier evidence detects historical prefill; retained auxiliary-state bytes are not independently checked.",
                        "Unreturned deadline-cancelled generation has no authoritative completion token ledger.",
                        "Capacity refusals are pre-submit stops, not successful eviction; refused input token counts are service-reported.",
                        "A pass is a bounded gameplay protocol/lifecycle audit, not score, throughput or numerical-equivalence validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameplay", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--allow-no-eviction", action="store_true")
    args = parser.parse_args()
    paths = sorted((args.gameplay/"raw-harness").glob("*.jsonl"))
    wire = []
    for path in paths:
        wire.extend(dict(row, _file=str(path)) for row in load(path))
    try:
        result = audit(load(args.engine), load(args.service), wire,
                       deadline_epoch=args.deadline_epoch, require_eviction=not args.allow_no_eviction)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        result = dict(status="fail", fatal_evidence_error=repr(exc))
    evidence = paths+[args.engine, args.service]
    result["provenance"] = dict(auditor=str(Path(__file__).resolve()), auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evidence=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in evidence])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k in ("status", "totals", "failed_checks", "fatal_evidence_error", "gates")}, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
