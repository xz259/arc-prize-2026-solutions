"""stage2_v1: MTP > 0 on the production composition (rollback plan, stage 2; M1 = plain requests).

Parts (ARC3_STAGE2, comma list):
  guards    The eviction runner's request staging (identity context, MRoPE rows, QSA virtual lookup) admits a
            speculative config; worker eviction itself still refuses it (eviction under speculation is M2). Persistent
            requests keep their own MTP0 refusal in engine_patch, so M1 serves plain requests only.
  rollback  GDN one-slot rollback by lazy replay, pending area in each state page (rollback.py). Needs BF16
            recurrent states and VLLM_GDN_DECODE_KERNEL=triton. Exact kernel (bit-identical to stock) by default.
  wy        With rollback: the matrix-form (WY) replay + verify kernel instead of the exact one.
  pagepad   Without rollback: only the page-size padding (reference runs with the rollback's block size).
  ctxfree   No CUDA context outside the GPU worker: glue-fusion preallocation deferred to Worker.init_device (memopt.py).
  topkshare One QSA top-k index buffer for all QSA owners instead of one per layer (memopt.py).
  pleconv   PLE short-conv prefill in padded groups bounded by max_num_batched_tokens, and the speculative short-conv
            metadata build without host synchronisation (ARC3_S2_PLECONV_SYNCFREE=0 restores the fork's) (pleconv.py).
  persist   MTP k in the persistent engine without one-step lookahead: stop fixup, kept count, resume commit, history
            eviction of paused MTP lanes (persist.py).
  schedfast The engine core's KV-allocation demand wrapper and eviction counters at O(1) per call (schedfast.py).
  mmstrip   Persistent resumes send data=None for images wholly before the resume frontier (mmstrip.py).
  overlap   Under the single-batch policy the engine core serves admissions, eviction begins and frequency snapshots
            while the dispatched step runs (overlap.py).
  draftgraph (measurement) Runtime switch forcing the drafter's decode passes above 16 requests eager (draftgraph.py).
  pwgraph   (measurement) Runtime switch running the target's PIECEWISE mixed steps above a token limit eager (pwgraph.py).
  attnprof  NVFP4 sparse QSA reader split counts retuned for MTP3 mixed-step shapes (attnprof.py).
  resumefast The worker's resume staging without redundant list round trips; needs guards (resumefast.py).
  deadlinefix An append reaching the engine after its session's deadline abort is answered as aborted, not fatal;
            since 2026-10-06 (381) an append failing the turn-capacity check aborts only its session, not the engine.
  topkfix   vLLM's top-k / top-p masking through an exact implementation (not the heavy-tie Triton kernel) (topkfix.py).
  replayfx  (export only) Exact replay of recorded sessions: recorded outputs forced as prefill + one forced token,
            recorded eviction plans re-planned and digest-checked; ARC3_S2_REPLAY=1 required (replayfx.py).
  dflashtaps DFlash target taps (attention block_input of ARC3_S2_TAPS layers) in the compiled forward; with
            ARC3_S2_TAPS_DIR, per-request feature files for the drafter's training (dflashtaps.py).
  precmode  launch-time numerics modes for the precision audit (ARC3_S2_PRECMODE: router32; precmode.py; 2026-10-15).
  kvwin     KV recent-window emulation on a BF16-KV launch (ARC3_S2_KVWIN_*; kvwin.py; 2026-10-15; exclusive with kvfq).
  kvring    BF16 recent window over the lutc main KV: the newest ARC3_S2_KVRING_N (256) rows of every target QSA owner read in
            BF16 from a per-request ring (kvring.py / kvring_kernels.py; 2026-10-15, item 4a; needs kvrot).
  prestage  a resumed persistent session is staged on the worker while the previous step runs: the engine core sends the
            resume's ids, context and images at admission (no reply), the step message carries references, ctxtail's
            staging result is taken at add_requests (prestage.py; 2026-10-15, item 5; needs ctxtail).
  fqfast    the API server's QSA frequency step (qsa_frequency.group_frequencies) from numpy arrays: same values, no
            Python-object layout (its cost grew with the process's age); ARC3_S2_FQFAST_VERIFY=N (fqfast.py; 2026-10-15).
  stepev    (diagnostics) profiler-free GPU timeline of each step from CUDA events, written to ARC3_S2_STEPEV_DIR
            (stepev.py; 2026-10-15, item 6 baseline).
  embedhost The target's input embedding table (shared with the drafter) in pinned host memory behind a UVA view:
            exact, frees 1.18 GiB of GPU memory (embedhost.py; milestone 2 A1).
  splitfix  profile_run under the allocator split limit (vLLM #58430), allocator state recorded around it (splitfix.py;
            milestone 2 A4).
  b1inc     Intel AutoRound W4A16 experts (INC config) with production's FP8 PLE and FP8 lm_head tier (b1inc.py;
            milestone 2 B1b). Inert unless the target's quant config is INCConfig.
  drafthead The MTP drafter scores a fixed vocabulary subset (ARC3_S2_DRAFTHEAD_VOCAB), -inf elsewhere; exact output
            distribution through vsampler; runtime switch to the full head (drafthead.py; milestone 2 A3).
  loadskip  Startup readers read only checkpoint files with tensors they use: the PLE worker the PLE files, the target
            (with a drafter-only folder) not the MTP+PLE file (loadskip.py; milestone 2 A10). Exact.
  ckptfetch Startup: bounded parallel prefetch of the safetensors shards into the page cache, one shard ahead of the loader,
            only the tensors it will read (ckptfetch.py; milestone 2 B9, 2026-10-02). Exact (the loader is unchanged).
            ARC3_S2_CKPTFETCH_THREADS (default 8).
  draftwin  Diagnostic (probe, 2026-10-03): the MTP drafter's QSA selection limited to its last W tokens
            (ARC3_S2_DRAFTWIN, in-run switch ARC3_S2_DRAFTWIN_SWITCH); draft numerics only (draftwin.py).
  rope2m    RoPE table extended to ARC3_S2_ROPE_ROWS (2,097,152) positions without frequency changes (rows below 262,144
            bit-identical, the rest from FP64 angles) and the three logical-position limits raised to match, so the
            rebase only fires past 2M; split-shift test hook behind ARC3_S2_ROPE_SHIFT_SWITCH (rope2m.py; 2026-10-03).
  rebaseobs the position rebase keeps the prefix plus the protected tail extended back to its interaction's observation
            (a much shorter prefill); sessions keep processed images only for that span (rebaseobs.py; 2026-10-03).
  agestat   diagnostic: the selection mass and resident groups by age at every triggered eviction plan, from the
            planner's frequency snapshot, in the receipt (agestat.py; 2026-10-03).
  agecap    age gap ARC3_S2_AGECAP: at every triggered eviction, whole interactions starting before end - A are removed
            too (latest interaction: whole assistant units only, never its observation or last unit) (agecap.py; 10-04).
  hollowgroup the hollow rule judges an assistant turn and the tool turn holding its tool's output as one block
            (hollowgroup.py; 2026-10-04).
  padbudget the session service counts the session's QSA group padding in its resident budget, and re-clamps
            max_tokens after the turn's eviction (padbudget.py; 2026-10-06, after 381's engine death).
  sessmem   API-server host memory: a closed session drops its images, stored responses and token list; a session keeps
            only its in-flight and latest completed responses (sessmem.py; 2026-10-04).
  ctxtail   per-turn history work bounded by resident + new tokens: the engine's eviction contexts carry only the ids and
            images the worker may lack (+ a digest); the worker keeps an append-only store and validates only what
            changed (ctxtail.py; 2026-10-04). Engine core and worker halves together.
  promptmove option 3: the system prompt's stored keys (K and QSA compressed keys of every owner) are rotated forward from
            per-session originals so the prompt sits just behind the oldest resident row; worker staging after each
            eviction (promptmove.py; 2026-10-04). Needs ARC3_PREFIX_SHARE=0 and codebook K formats.
  foldc     KV layout: production's fold B with the PLE state moved from compressed tensor 0 to the main-KV tensor no GDN
            layer shares, so compressed-key caches and rings are no longer padded to the PLE's size (+2.4% blocks at the
            same KV memory) (foldc.py; 2026-10-04). Needs ARC3_FIXED_STATE_FOLD=B.
  draftring KV layout (item 4b, 2026-10-15): fold D drops the MTP drafter's owners and the PLE state from the shared pool
            (24,434,688-B blocks); the worker keeps them in private extra groups: a per-session ring of P pages
            (ARC3_S2_DRAFTRING_P, 8) for the drafter's main KV and compressed keys, one page for its compressor ring and one for
            the PLE state (draftring.py). Needs ARC3_FIXED_STATE_FOLD=B; excludes foldc.
  agefix    Engine-core audit path (2026-10-16; age1 / IE): with ARC3_PERSISTENT_ENGINE_AUDIT unset, audit events return before
            the eviction wrapper's logical digest and copies; with it set, audit-path logical digests check the hashed tail and the
            full prefix every 32nd call (validation digests stay full). Exact (agefix.py).
  marlindet Marlin MoE in a fixed token order: moe_align_block_size's within-expert order canonicalised (sorted by
            expert, token), so Intel's INT4 experts and the NVFP4-W4A16 drafter are run-to-run repeatable (marlindet.py;
            milestone 2 W1).
  kvfq      Diagnostic: main-QSA K/V fake-quantized (quantize + dequantize) before a BF16 cache write by a runtime
            switch (ARC3_S2_KVFQ_SWITCH), for the 4-bit KV format study (kvfq.py; milestone 2, 2026-10-01). Off by default.
  kvevict   QSA eviction admits FP8 main KV rows (520 bytes) besides NVFP4's 296: one gate condition of the async
            eviction, source-transformed in memory; the copy is layout-generic (kvevict.py; milestone 2, 2026-10-01).
  kvq4      g4 (2026-10-26, Claude fork bdfc5402): the lutc main KV with one scale set per quad of four token rows (266-byte
            rows; kvq4.py / kvq4_kernels.py; G4_DESIGN_2026-10-26.md). Needs kvrot (lut formats) and kvring (ring ids key the
            side buffer); excludes kvevict and kvfq. The planner snaps every removal to whole QSA groups.
  eprune    expert pruning (2026-10-26, Claude fork bdfc5402): the routed experts of ARC3_S2_EPRUNE_FILE (default eprune_k16.json:
            the bottom 16 per layer by usage) removed from the target model at load time: their weights are never allocated
            (1.83 GiB), the gate's rows permuted to [kept, pruned], the pruned logits -inf before the top-k (eprune.py).
  kvrot     Main-QSA NVFP4 KV stored Hadamard-rotated with block-MSE E4M3 scales (h_nvfp4m, the KV study's choice):
            same 148-byte layout and readers; the query rotated before attention and the output rotated back
            (kvrot.py; milestone 2, 2026-10-02). Needs ARC3_STAGE_KV_FORMAT=nvfp4.
  seloverlap Diagnostic (B4): the overlap of one request's verify rows' QSA top-2,048 selections, sampled in eager
            forward_qsa calls (seloverlap.py; milestone 2, 2026-10-02). ARC3_S2_SELOVERLAP_EVERY (default 5).
  idxfq     Diagnostic (B5, record only): the QSA indexer's compressed keys fake-quantized to FP8 E4M3 per row right after
            the fused pre-indexer writes them (idxfq.py; milestone 2, 2026-10-02). ARC3_S2_IDXFQ=e4m3.
  headonly  The dense-FP8 adapter's lm_head tier without a dense tier (ARC3_DENSE_FP8_TIER=off, ARC3_DENSE_FP8_LM_HEAD=1):
            the adapter's import hook installed (its bootstrap skips it under 'off') behind ours, and install_lm_head
            redefined with one gate condition changed, in memory (headonly.py; milestone 2, 2026-10-01).
  draftqsa  The MTP drafter under the eviction runner: draft decode steps staged like target batches (QSA seq lens,
            eviction shifts, selection remap); drafter RoPE at the original index after an eviction (draftqsa.py).
  vsampler / qsaprofile   (later) candidate verify sampler, QSA 8-split verify profile.
ARC3_S2_MEMDIAG=<max trace entries> (not a part): memory attribution RPC and CUDA-init tracer (memdiag.py).
ARC3_S2_AUTOTUNE_SEED=<autotune_configs.json> (not a part): fixed FlashInfer MoE tactics for any config (memopt.py).
ARC3_S2_HOSTPROF_DIR=<dir> (not a part): trigger-file sampling profiler of every process's Python threads (hostprof.py).
ARC3_S2_STEPTIMER_DIR=<dir> (not a part): per-step host phase timings of the engine core and the worker (steptimer.py).
ARC3_S2_MEMSTATS_DIR=<dir> (not a part): trigger-file host-memory accounting of every process (memstats.py, 2026-10-03).
ARC3_S2_GC_GEN2=<n> (not a part; 2026-10-08, the eviction slowdown): CPython full collections at most every n
  generation-1 collections (CPython's default 10) in every process. Profiling run v62a167p: the API server ran a full
  collection every ~5 s and the engine core every ~25 s; each traversed the ~20M entries of the live sessions' token
  lists (0.2-0.45 s, growing with the sessions; 392b's end: 67M blocks) and freed nothing (0-122 objects a minute).
  Younger generations unchanged; cyclic garbage is still collected, only later.

Every patched module is pinned to its source hash (2026-09-29). Receipts: $ARC3_S2_RECEIPTS/proc-<pid>.json."""
import hashlib
import importlib.abc
import json
import os
import sys
import threading
import time
from pathlib import Path

PARTS = frozenset(p for p in os.environ.get('ARC3_STAGE2', '0').split(',') if p and p != '0')
REPORT = dict(schema='arc3.stage2.v1', pid=os.getpid(), parts=sorted(PARTS), patched=[], errors=[], updated=None)
_last = [0.0]


def save(force=True):
    root = os.environ.get('ARC3_S2_RECEIPTS')
    if not root or (not force and time.time() - _last[0] < 60):
        return
    _last[0] = time.time()
    if 'rollback' in PARTS:
        from . import rollback
        REPORT['rollback'] = dict(rollback.STATS, wy=rollback.WY)
        try:
            REPORT['rollback']['prefill_resumes_with_pending'] = rollback.bad_resumes()
            REPORT['rollback']['transition_pending_rows'] = rollback.transition_pending_rows()
        except Exception as exc:                    # no CUDA context in this process
            REPORT['rollback']['prefill_resumes_with_pending'] = repr(exc)[:80]
    if 'vsampler' in PARTS:
        from . import vsampler
        REPORT['vsampler'] = dict(vsampler.STATS)
        try:
            REPORT['vsampler']['overflow_rows'] = vsampler.overflow_rows()
            REPORT['vsampler']['placeholder_slots'] = vsampler.placeholder_slots()
            REPORT['vsampler']['accept_by_steps_since_mixed'] = vsampler.mix_report()
        except Exception as exc:
            REPORT['vsampler']['overflow_rows'] = repr(exc)[:80]
    if PARTS & {'ctxfree', 'topkshare'} or os.environ.get('ARC3_S2_AUTOTUNE_SEED'):
        from . import memopt
        REPORT['memopt'] = dict(memopt.STATS)
    if 'pleconv' in PARTS:
        from . import pleconv
        REPORT['pleconv'] = dict(pleconv.STATS)
    if 'persist' in PARTS:
        from . import persist
        REPORT['persist'] = dict(persist.STATS)
    if 'draftqsa' in PARTS:
        from . import draftqsa
        REPORT['draftqsa'] = dict(draftqsa.STATS)
    if 'schedfast' in PARTS:
        from . import schedfast
        REPORT['schedfast'] = dict(schedfast.STATS, verify=schedfast.VERIFY)
    if 'mmstrip' in PARTS:
        from . import mmstrip
        REPORT['mmstrip'] = dict(mmstrip.STATS)
    if 'overlap' in PARTS:
        from . import overlap
        REPORT['overlap'] = dict(overlap.STATS)
    if 'draftgraph' in PARTS:
        from . import draftgraph
        REPORT['draftgraph'] = dict(draftgraph.STATS)
    if 'pwgraph' in PARTS:
        from . import pwgraph
        REPORT['pwgraph'] = dict(pwgraph.STATS, limit=pwgraph.LIMIT)
    if 'attnprof' in PARTS:
        from . import attnprof
        REPORT['attnprof'] = dict(attnprof.STATS, splits=[list(x) for x in attnprof.SPLITS])
    if 'resumefast' in PARTS:
        from . import resumefast
        REPORT['resumefast'] = dict(resumefast.STATS, verify=resumefast.VERIFY)
    if 'deadlinefix' in PARTS:
        from . import deadlinefix
        REPORT['deadlinefix'] = dict(deadlinefix.STATS)
    if 'topkfix' in PARTS:
        from . import topkfix
        REPORT['topkfix'] = dict(topkfix.STATS)
    if 'embedhost' in PARTS:
        from . import embedhost
        REPORT['embedhost'] = dict(embedhost.STATS)
    if 'splitfix' in PARTS:
        from . import splitfix
        REPORT['splitfix'] = dict(splitfix.STATS)
    if 'b1inc' in PARTS:
        from . import b1inc
        REPORT['b1inc'] = dict(b1inc.STATS)
    if 'marlindet' in PARTS:
        from . import marlindet
        REPORT['marlindet'] = dict(marlindet.STATS)
    if 'kvfq' in PARTS:
        from . import kvfq
        REPORT['kvfq'] = dict(kvfq.STATS)
    if 'kvevict' in PARTS:
        from . import kvevict
        REPORT['kvevict'] = dict(kvevict.STATS)
    if 'kvrot' in PARTS:
        from . import kvrot
        REPORT['kvrot'] = dict(kvrot.STATS)
    if 'idxfq' in PARTS:
        from . import idxfq
        REPORT['idxfq'] = dict(idxfq.STATS)
    if 'seloverlap' in PARTS:
        from . import seloverlap
        REPORT['seloverlap'] = dict(seloverlap.STATS)
    if 'drafthead' in PARTS:
        from . import drafthead
        REPORT['drafthead'] = dict(drafthead.STATS)
    if 'loadskip' in PARTS:
        from . import loadskip
        REPORT['loadskip'] = dict(loadskip.STATS)
    if 'ckptfetch' in PARTS:
        from . import ckptfetch
        REPORT['ckptfetch'] = dict(ckptfetch.STATS)
    if 'draftwin' in PARTS:
        from . import draftwin
        REPORT['draftwin'] = dict(draftwin.STATS)
    if 'rope2m' in PARTS:
        from . import rope2m
        REPORT['rope2m'] = dict(rope2m.STATS, rows=rope2m.ROWS, trained=rope2m.TRAINED, shift_switch=rope2m.SHIFT_SWITCH)
    if 'rebaseobs' in PARTS:
        from . import rebaseobs
        REPORT['rebaseobs'] = dict(rebaseobs.STATS)
    if 'agestat' in PARTS:
        from . import agestat
        REPORT['agestat'] = dict(agestat.STATS)
    if 'agecap' in PARTS:
        from . import agecap
        REPORT['agecap'] = dict(agecap.STATS, cap=agecap.AGECAP)
    if 'hollowgroup' in PARTS:
        from . import hollowgroup
        REPORT['hollowgroup'] = dict(hollowgroup.STATS)
    if 'padbudget' in PARTS:
        from . import padbudget
        REPORT['padbudget'] = dict(padbudget.STATS)
    if 'poolpress' in PARTS:
        from . import poolpress
        REPORT['poolpress'] = dict(poolpress.STATS)
    if 'stallguard' in PARTS:
        from . import stallguard
        REPORT['stallguard'] = dict(stallguard.STATS)
    if 'sessmem' in PARTS:
        from . import sessmem
        REPORT['sessmem'] = dict(sessmem.STATS)
    if 'ctxtail' in PARTS:
        from . import ctxtail
        REPORT['ctxtail'] = dict(ctxtail.STATS)
    if 'promptmove' in PARTS:
        from . import promptmove
        REPORT['promptmove'] = promptmove.STATS
    if 'foldc' in PARTS:
        from . import foldc
        REPORT['foldc'] = foldc.STATS
    if 'draftring' in PARTS:
        from . import draftring
        REPORT['draftring'] = dict(draftring.STATS)
    if 'agefix' in PARTS:
        from . import agefix
        REPORT['agefix'] = dict(agefix.STATS)
    if 'sumpin' in PARTS:
        from . import sumpin
        REPORT['sumpin'] = dict(sumpin.STATS)
    if 'rebase74' in PARTS:
        from . import rebase74
        REPORT['rebase74'] = dict(rebase74.STATS)
    if 'mmonly' in PARTS:
        from . import mmonly
        REPORT['mmonly'] = dict(mmonly.STATS, results=dict(mmonly.STATS['results']))
    if 'dflashtaps' in PARTS:
        from . import dflashtaps
        REPORT['dflashtaps'] = dflashtaps.STATS
    if 'precmode' in PARTS:
        from . import precmode
        REPORT['precmode'] = dict(precmode.STATS)
    if 'kvwin' in PARTS:
        from . import kvwin
        REPORT['kvwin'] = dict(kvwin.STATS)
    if 'kvring' in PARTS:
        from . import kvring
        REPORT['kvring'] = dict(kvring.STATS)
    if 'kvq4' in PARTS:
        from . import kvq4
        kvq4.export_stats()
        REPORT['kvq4'] = dict(kvq4.STATS)
    if 'eprune' in PARTS:
        from . import eprune
        REPORT['eprune'] = dict(eprune.STATS)
    if 'prestage' in PARTS:
        from . import prestage
        REPORT['prestage'] = dict(prestage.STATS)
    if 'fqfast' in PARTS:
        from . import fqfast
        REPORT['fqfast'] = dict(fqfast.STATS)
    if 'stepev' in PARTS:
        from . import stepev
        REPORT['stepev'] = dict(stepev.STATS)
    if 'replayfx' in PARTS:
        from . import replayfx
        REPORT['replayfx'] = replayfx.STATS
    p = Path(root); p.mkdir(parents=True, exist_ok=True)
    REPORT['updated'] = time.time()
    dst = p / f'proc-{os.getpid()}.json'; tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(REPORT, indent=1, default=str) + '\n'); tmp.replace(dst)


# ---------------------------------------------------------------- guards
def patch_eviction_runner(mod):
    def _limits(runner):
        """eviction_runner._limits without its MTP0 refusal (identical otherwise)."""
        config = runner.vllm_config
        mod.validate_execution_mode(config)
        if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.pipeline_parallel_size != 1:
            raise mod.EvictionRunnerError('Eviction runner supports TP1/PP1 only')
        state = runner.model_state
        if type(state).__name__ != 'Qwen3_8FlashNextModelState' or state.rope_state is None:
            raise mod.EvictionRunnerError('Expected pinned Qwen3.8 conditional model state with MRoPE')
        if state._align_mode or state.mm_pruner is not None:
            raise mod.EvictionRunnerError('Eviction control requires fixed recurrent state and no image pruning')
        logical = int(config.model_config.hf_text_config.max_position_embeddings)
        return dict(max_resident_tokens=int(state.max_model_len), max_logical_tokens=logical,
                    ngram_context_len=int(state.ngram_context_len))

    stock_prepare = mod.prepare_worker_eviction

    def prepare_worker_eviction(runner, *args, **kwargs):
        config = runner.vllm_config
        if (config.num_speculative_tokens or config.speculative_config is not None) and 'persist' not in PARTS:
            raise mod.EvictionRunnerError('stage2 M1: eviction under speculative decoding needs the persist part')
        return stock_prepare(runner, *args, **kwargs)

    mod._limits = _limits
    mod.prepare_worker_eviction = prepare_worker_eviction
    if 'resumefast' in PARTS:                 # one patcher per module: resumefast chains from here
        from . import resumefast
        resumefast.patch_eviction_runner(mod)


def patch_embedding(module):
    """The MTP drafter is built with its own ParallelLMHead (and embedding), loaded from the target's checkpoint tensors;
    vLLM's load_eagle_model then shares the target's head and embedding with the drafter and deletes the drafter's
    copies (the checkpoint has no drafter head). The dense-FP8 adapter (lm_head tier B1) asserts there is one head:
    the first head stays the adapter's 'lm_head'; a second one takes the stock (BF16) path, so nothing keeps it alive
    once shared. Unshared, it would serve the drafter in BF16 (recorded in the receipt)."""
    import dense_fp8_adapter_v1.runtime as rt
    method_cls = module.UnquantizedEmbeddingMethod
    adapter_post = method_cls.process_weights_after_loading
    if 'embedding' not in rt.ORIGINAL:          # lm_head tier not installed: nothing to wrap
        return
    inner_post = rt.ORIGINAL['embedding'][0]

    def post(self, layer):
        if type(layer).__name__ == 'ParallelLMHead' and 'lm_head' in rt.LAYERS:
            REPORT['drafter_lm_head_bf16'] = REPORT.get('drafter_lm_head_bf16', 0) + 1
            save()
            return inner_post(self, layer)
        return adapter_post(self, layer)

    method_cls.process_weights_after_loading = post


def patch_qsa_frequency(mod):
    """The MTP drafter's QSA layer (13th owner) runs through the eviction selector wrapper, which counts selections
    per target QSA layer for eviction ranking; the drafter's compressed-key cache is not in that 12-layer map. Its
    selections are not counted (layer None); the target's counting is unchanged."""
    orig = mod.layer_for_cache

    def layer_for_cache(counters, k_cache):
        try:
            return orig(counters, k_cache)
        except mod.QSAFrequencyError as exc:
            if 'unregistered compressed key cache' in str(exc) and counters.layer_index:
                REPORT['qsa_frequency_uncounted_calls'] = REPORT.get('qsa_frequency_uncounted_calls', 0) + 1
                return None
            raise

    mod.layer_for_cache = layer_for_cache


def patch_eviction_qsa(mod):
    orig = mod.remap_selected_indices_

    def remap_selected_indices_(indices, token_to_req, mapping, counters=None, layer=-1, token_topk=None):
        if counters is not None and layer is None:          # drafter QSA layer: remap only, no counting
            return orig(indices, token_to_req, mapping)
        return orig(indices, token_to_req, mapping, counters, layer, token_topk)

    mod.remap_selected_indices_ = remap_selected_indices_


def transform_lookahead_bootstrap(src):
    """Production's canary bootstrap writes one startup receipt per process to runs/dense_fp8_v1/lookahead-gameplay-
    startup/process-<pid>.json with exclusive create: a recycled PID collides with an old receipt (M2f startup failure,
    479 receipts). Stage-2 processes write it under $ARC3_S2_RECEIPTS with a time suffix instead (content unchanged)."""
    a = ("    out = HERE/'lookahead-gameplay-startup'\n    out.mkdir(exist_ok=True)\n"
         "    with (out/f'process-{os.getpid()}.json').open('x') as handle:\n")
    b = ("    out = Path(os.environ.get('ARC3_S2_RECEIPTS') or HERE)/'lookahead-gameplay-startup'   # stage 2\n"
         "    out.mkdir(parents=True, exist_ok=True)\n"
         "    with (out/f'process-{os.getpid()}-{__import__(\"time\").time_ns()}.json').open('x') as handle:\n")
    assert src.count(a) == 1, 'stage2 guards: lookahead_bootstrap anchor'
    return src.replace(a, b)


def patch_persistent_endpoint(mod):
    """The API server's persistent endpoint validates the config at startup with engine_patch.validate_config, which
    refuses any speculative config. During init_state only, a relaxed copy (the same checks, minus MTP0) is swapped in
    and records the draft count; the engine core keeps the stock check in add_request, so persistent requests are
    still refused under speculation (M1 serves plain requests; M2 adds persistence)."""
    cls = mod.PersistentEndpoint
    orig = cls.init_state
    if 'persist' in PARTS:
        # M2 option C: persistent MTP lanes run without one-step lookahead. The server keeps the production canary
        # configuration (lookahead capability on); the endpoint drops a client's per-turn lookahead request.
        backend = mod.VllmBackend                   # builds the SamplingParams of every persistent turn
        orig_sampling = backend.sampling_params

        def sampling_params(self, values):
            values = dict(values)
            if values.pop('arc3_one_lookahead', False):
                REPORT['lookahead_requests_dropped'] = REPORT.get('lookahead_requests_dropped', 0) + 1
            return orig_sampling(self, values)

        backend.sampling_params = sampling_params

    async def init_state(self, engine_client, state, args):
        import engine_patch
        stock = engine_patch.validate_config

        def relaxed(config):
            if config.speculative_config is None:
                return stock(config)
            if config.cache_config.enable_prefix_caching is not False:
                raise engine_patch.PersistentStateError("Persistent phase 1 requires prefix caching disabled")
            parallel = config.parallel_config
            if parallel.pipeline_parallel_size != 1 or parallel.tensor_parallel_size != 1:
                raise engine_patch.PersistentStateError("Persistent phase 1 requires PP1 and TP1")
            if getattr(parallel, "data_parallel_size", 1) != 1:
                raise engine_patch.PersistentStateError("Persistent phase 1 requires DP1")
            if getattr(config, "kv_transfer_config", None) is not None or getattr(config, "ec_transfer_config", None) is not None:
                raise engine_patch.PersistentStateError("Persistent phase 1 does not support transfer connectors")
            if config.model_config.runner_type != "generate":
                raise engine_patch.PersistentStateError("Persistent phase 1 supports generation only")
            return {"model_runner": "v2" if config.use_v2_model_runner else "v1",
                    "async_scheduling": bool(config.scheduler_config.async_scheduling),
                    "max_model_len": int(config.model_config.max_model_len), "mtp": int(config.num_speculative_tokens),
                    "prefix_caching": False, "persistent_requests": "refused by the engine core (stage 2 M1)"}

        engine_patch.validate_config = relaxed
        try:
            return await orig(self, engine_client, state, args)
        finally:
            engine_patch.validate_config = stock

    cls.init_state = init_state


GUARD_PINS = {'eviction_runner': '03ecae34a987b0b3203df42cd609f41c051226371c517ae85d69591851253ed8',
              'persistent_endpoint': 'ceead0163e99ffcdf3f376affee74a597a343ef06781a737e0b09bf9a56b0001',
              'qsa_frequency': 'c9ccb07eb8c4ef75a7f58d62f26cca6da4d7de6b3730e09f96f17766655afb5c',
              'eviction_qsa': '568bf16b97130ec9f0ee2d033949b9aeefce4e0d7bcfe509aac6d43dcfedfbbc',
              'lookahead_bootstrap': '5ed9e4d18b906cc910f760a29c7d5d8d27f8fa1d9366c0c9827941ce6ecb4a76',
              'vllm.model_executor.layers.vocab_parallel_embedding':
                  'fc2ea26e3baee7cb34f5acb9881c22a594aa0270c41f39813260e120add76563'}
HOOKS = {}          # module name -> (pin, patcher)


def _add_hook(name, pin, patcher):
    """Register a module patcher. A second part patching the same module is chained after the first (same pin;
    module patchers only: a source transform replaces the file and cannot be chained)."""
    if name not in HOOKS:
        HOOKS[name] = (pin, patcher)
        return
    old_pin, old = HOOKS[name]
    assert old_pin == pin, f'stage2_v1: conflicting pins for {name}'
    assert not isinstance(old, tuple) and not isinstance(patcher, tuple), f'stage2_v1: cannot chain a source transform ({name})'

    def chained(module, _first=old, _second=patcher):
        _first(module)
        _second(module)
    HOOKS[name] = (pin, chained)


class _Loader(importlib.abc.Loader):
    def __init__(self, original, name):
        self.original, self.name = original, name

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        pin, patcher = HOOKS[self.name]
        data = Path(module.__spec__.origin).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert digest == pin, f'stage2_v1: {self.name} changed ({digest}); re-pin'
        if isinstance(patcher, tuple) and patcher[0] == 'source':   # source transform, executed instead of the file
            import importlib.machinery
            assert isinstance(self.original, importlib.machinery.SourceFileLoader), \
                f'stage2_v1: {self.name} has another loader ({type(self.original).__name__})'
            exec(compile(patcher[1](data.decode()), module.__spec__.origin, 'exec'), module.__dict__)
            REPORT['patched'].append(self.name + ' (source)')
            save()
            return
        self.original.exec_module(module)
        try:
            patcher(module)
            REPORT['patched'].append(self.name)
        except Exception as exc:
            REPORT['errors'].append(f'{self.name}: {exc!r}'); save()
            raise
        save()

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in HOOKS:
            return None
        active = getattr(self._local, 'active', None)
        if active is None:
            active = self._local.active = set()
        if fullname in active:
            return None
        active.add(fullname)
        try:
            started = False
            for finder in list(sys.meta_path):
                if finder is self:
                    started = True
                    continue
                if not started or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    spec.loader = _Loader(spec.loader, fullname)
                    return spec
            return None
        finally:
            active.discard(fullname)


def _thp_core_only():
    """2026-10-16 (aging arm B: GLIBC_TUNABLES=glibc.malloc.hugetlb=1 made the PLE worker's 95 GiB host load take > 25 min instead
    of 45 s): with ARC3_S2_THP_CORE_ONLY=1, a process started through multiprocessing (the engine core, spawned by the API server)
    drops GLIBC_TUNABLES from its environment after its own start, so the tunable applies to the API server and the engine core
    only; the GPU worker and the PLE worker, spawned later, start without it. glibc reads tunables at exec: popping it changes
    only the children."""
    if os.environ.get('ARC3_S2_THP_CORE_ONLY') == '1' and 'multiprocessing' in ' '.join(sys.argv[:3]):
        if os.environ.pop('GLIBC_TUNABLES', None) is not None:
            REPORT['thp_core_only'] = 'GLIBC_TUNABLES dropped for children'


def _save_atexit():
    """v7.5 (IE's review; the user 10-23): the process's last forced receipt at a normal interpreter exit, so an error counted in a run's
    last minute (non-forced saves are throttled to one a minute) reaches errors / last_error. Receipts are diagnostics: it never raises."""
    try:
        save(force=True)
    except Exception:                                   # noqa: BLE001
        pass


def install():
    _thp_core_only()
    assert PARTS <= {'guards', 'rollback', 'wy', 'vsampler', 'qsaprofile', 'pagepad', 'ctxfree', 'topkshare', 'pleconv', 'persist',
                     'draftqsa', 'schedfast', 'mmstrip', 'overlap', 'draftgraph', 'attnprof', 'resumefast', 'deadlinefix', 'topkfix',
                     'pwgraph', 'embedhost', 'splitfix', 'b1inc', 'drafthead', 'loadskip', 'marlindet', 'kvfq', 'kvevict', 'headonly',
                     'kvrot', 'idxfq', 'seloverlap', 'ckptfetch', 'draftwin', 'rope2m', 'rebaseobs', 'agestat', 'agecap',
                     'hollowgroup', 'sessmem', 'promptmove', 'ctxtail', 'foldc', 'padbudget', 'replayfx', 'dflashtaps', 'precmode', 'kvwin', 'poolpress', 'stallguard', 'kvring', 'kvq4', 'eprune', 'prestage', 'fqfast', 'stepev', 'draftring', 'agefix', 'sumpin', 'rebase74', 'mmonly'}, PARTS
    unimplemented = PARTS - {'guards', 'rollback', 'wy', 'vsampler', 'pagepad', 'ctxfree', 'topkshare', 'pleconv', 'persist',
                             'draftqsa', 'schedfast', 'mmstrip', 'overlap', 'draftgraph', 'attnprof', 'resumefast', 'deadlinefix', 'topkfix',
                     'pwgraph', 'embedhost', 'splitfix', 'b1inc', 'drafthead', 'loadskip', 'marlindet', 'kvfq', 'kvevict', 'headonly',
                     'kvrot', 'idxfq', 'seloverlap', 'ckptfetch', 'draftwin', 'rope2m', 'rebaseobs', 'agestat', 'agecap',
                     'hollowgroup', 'sessmem', 'promptmove', 'ctxtail', 'foldc', 'padbudget', 'replayfx', 'dflashtaps', 'precmode', 'kvwin', 'poolpress', 'stallguard', 'kvring', 'kvq4', 'eprune', 'prestage', 'fqfast', 'stepev', 'draftring', 'agefix', 'sumpin', 'rebase74', 'mmonly'}
    assert 'foldc' not in PARTS or os.environ.get('ARC3_FIXED_STATE_FOLD') == 'B', 'stage2_v1: foldc re-lays fold B (ARC3_FIXED_STATE_FOLD=B)'
    assert 'draftring' not in PARTS or (os.environ.get('ARC3_FIXED_STATE_FOLD') == 'B' and 'foldc' not in PARTS), \
        'stage2_v1: draftring (fold D) wraps fold B (ARC3_FIXED_STATE_FOLD=B) and replaces foldc'
    assert 'ctxtail' not in PARTS or {'guards', 'resumefast'} <= PARTS, 'stage2_v1: ctxtail stages through resumefast (guards chain)'
    assert 'prestage' not in PARTS or 'ctxtail' in PARTS, 'stage2_v1: prestage runs ctxtail\'s staging computation'
    assert 'promptmove' not in PARTS or os.environ.get('ARC3_PREFIX_SHARE', '1') == '0', \
        'stage2_v1: promptmove needs ARC3_PREFIX_SHARE=0 (a shared prefix page would move every lane)'
    assert 'persist' not in PARTS or 'rollback' in PARTS, 'stage2_v1: persist needs rollback'
    assert 'rebase74' not in PARTS or 'rebaseobs' in PARTS, 'stage2_v1: rebase74 sets rebaseobs\'s window (KEEP)'
    assert not {'kvfq', 'kvrot'} <= PARTS, 'stage2_v1: kvfq (BF16 cache) and kvrot (NVFP4 cache) exclude each other'
    assert 'kvring' not in PARTS or ('kvrot' in PARTS and not PARTS & {'kvwin', 'kvfq'}), \
        'stage2_v1: kvring needs kvrot (lut owners) and excludes the kvwin / kvfq emulations'
    assert 'kvq4' not in PARTS or ({'kvrot', 'kvring'} <= PARTS and not PARTS & {'kvevict', 'kvfq'}), \
        'stage2_v1: kvq4 needs kvrot and kvring and excludes kvevict / kvfq (one engine_eviction_async transform)'
    assert not unimplemented, f'stage2_v1: parts not implemented yet: {sorted(unimplemented)}'
    assert 'wy' not in PARTS or 'rollback' in PARTS, 'stage2_v1: wy needs rollback'
    assert 'resumefast' not in PARTS or 'guards' in PARTS, 'stage2_v1: resumefast chains from the guards eviction_runner patch'
    if 'guards' in PARTS:
        _add_hook('eviction_runner', GUARD_PINS['eviction_runner'], patch_eviction_runner)
        name = 'vllm.model_executor.layers.vocab_parallel_embedding'
        _add_hook(name, GUARD_PINS[name], patch_embedding)
        _add_hook('qsa_frequency', GUARD_PINS['qsa_frequency'], patch_qsa_frequency)
        _add_hook('persistent_endpoint', GUARD_PINS['persistent_endpoint'], patch_persistent_endpoint)
        _add_hook('eviction_qsa', GUARD_PINS['eviction_qsa'], patch_eviction_qsa)
        _add_hook('lookahead_bootstrap', GUARD_PINS['lookahead_bootstrap'], ('source', transform_lookahead_bootstrap))
    if 'rollback' in PARTS:
        assert os.environ.get('VLLM_GDN_DECODE_KERNEL') == 'triton', 'stage2_v1 rollback: set VLLM_GDN_DECODE_KERNEL=triton'
        # 2026-10-24 (Claude main session 099768dc; the solo-resume deviation, IE's review, the user): a step whose GDN rows are all
        # 1-token 0-draft decodes takes vLLM's packed decode fast path (_forward_core_decode_non_spec), which reads and writes the raw
        # ssm cache outside rollback's patched update; under scaled16 that drops the block scales (M read as the state, stored unscaled).
        if os.environ.get('ARC3_S2_GDN_STATE', '') in ('scaled16', 'scaled16sr'):
            assert os.environ.get('VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE') == '0', \
                'stage2_v1 scaled16: set VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 (the packed decode fast path bypasses the scaled16 state)'
        REPORT['packed_decode_off'] = os.environ.get('VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE') == '0'
        from . import rollback
        rollback.WY = 'wy' in PARTS
        for name, patcher in rollback.PATCHERS.items():
            _add_hook(name, rollback.PINS[name], patcher)
    if 'pagepad' in PARTS and 'rollback' not in PARTS:     # reference runs: the rollback page size, stock state path
        from . import rollback
        name = 'vllm.models.qwen3_8_flash_next.nvidia.model'
        _add_hook(name, rollback.PINS[name], rollback.install_model)
    if 'vsampler' in PARTS:
        from . import vsampler
        for name, patcher in vsampler.PATCHERS.items():
            _add_hook(name, vsampler.PINS[name], patcher)
    if PARTS & {'ctxfree', 'topkshare'}:                             # memory parts without numeric effect
        from . import memopt
        for part in sorted(PARTS & {'ctxfree', 'topkshare'}):
            for name, patcher in memopt.PATCHERS[part].items():
                _add_hook(name, memopt.PINS[name], patcher)
    if 'persist' in PARTS:                                            # MTP k in the persistent engine (M2, option C)
        from . import persist
        for name, patcher in persist.PATCHERS.items():
            _add_hook(name, persist.PINS[name], patcher)
    if 'draftqsa' in PARTS:                                           # drafter QSA staging and positions (draft numerics)
        from . import draftqsa
        for name, patcher in draftqsa.PATCHERS.items():
            _add_hook(name, draftqsa.PINS[name], patcher)
    if 'schedfast' in PARTS:                                          # O(1) KV-allocation demand wrapper (engine core)
        from . import schedfast
        for name, patcher in schedfast.PATCHERS.items():
            _add_hook(name, schedfast.PINS[name], patcher)
    if 'mmstrip' in PARTS:                                            # no re-sent pixels for computed images (engine core)
        from . import mmstrip
        for name, patcher in mmstrip.PATCHERS.items():
            _add_hook(name, mmstrip.PINS[name], patcher)
    if 'overlap' in PARTS:                                            # input served during in-flight steps (engine core)
        assert 'persist' in PARTS, 'stage2_v1: overlap needs persist (the single-batch policy with MTP k)'
        from . import overlap
        for name, patcher in overlap.PATCHERS.items():
            _add_hook(name, overlap.PINS[name], patcher)
    if 'deadlinefix' in PARTS:                                        # deadline-race append answered as aborted
        from . import deadlinefix
        for name, patcher in deadlinefix.PATCHERS.items():
            _add_hook(name, deadlinefix.PINS[name], patcher)
    if 'topkfix' in PARTS:                                            # exact top-k / top-p masking (worker)
        from . import topkfix
        for name, patcher in topkfix.PATCHERS.items():
            _add_hook(name, topkfix.PINS[name], patcher)
    if 'draftgraph' in PARTS:                                         # drafter decode graphs switchable (measurement)
        from . import draftgraph
        for name, patcher in draftgraph.PATCHERS.items():
            _add_hook(name, draftgraph.PINS[name], patcher)
    if 'pwgraph' in PARTS:                                            # target PIECEWISE graphs switchable (measurement)
        from . import pwgraph
        for name, patcher in pwgraph.PATCHERS.items():
            _add_hook(name, pwgraph.PINS[name], patcher)
    if 'attnprof' in PARTS:                                           # NVFP4 sparse reader split counts (worker)
        from . import attnprof
        for name, patcher in attnprof.PATCHERS.items():
            _add_hook(name, attnprof.PINS[name], patcher)
    if 'pleconv' in PARTS:                                            # memory-bounded PLE short-conv prefill
        from . import pleconv
        for name, patcher in pleconv.PATCHERS.items():
            _add_hook(name, pleconv.PINS[name], patcher)
    if 'embedhost' in PARTS:                                          # embedding table in pinned host memory (worker)
        from . import embedhost
        for name, patcher in embedhost.PATCHERS.items():
            _add_hook(name, embedhost.PINS[name], patcher)
    if 'splitfix' in PARTS:                                           # profile_run under the split limit (worker)
        from . import splitfix
        for name, patcher in splitfix.PATCHERS.items():
            _add_hook(name, splitfix.PINS[name], patcher)
    if 'b1inc' in PARTS:                                              # INC (Intel W4A16) compatibility (B1b)
        from . import b1inc
        for name, patcher in b1inc.PATCHERS.items():
            _add_hook(name, b1inc.PINS[name], patcher)
    if 'drafthead' in PARTS:                                          # drafter vocabulary subset head (A3)
        assert os.path.isfile(os.environ.get('ARC3_S2_DRAFTHEAD_VOCAB', '')), 'stage2_v1 drafthead: set ARC3_S2_DRAFTHEAD_VOCAB'
        from . import drafthead
        for name, patcher in drafthead.PATCHERS.items():
            _add_hook(name, drafthead.PINS[name], patcher)
    if 'marlindet' in PARTS:                                          # deterministic Marlin MoE order (W1)
        from . import marlindet
        for name, patcher in marlindet.PATCHERS.items():
            _add_hook(name, marlindet.PINS[name], patcher)
    if 'kvfq' in PARTS:                                               # diagnostic K/V fake quantization (KV study)
        from . import kvfq
        for name, patcher in kvfq.PATCHERS.items():
            _add_hook(name, kvfq.PINS[name], patcher)
    if 'kvevict' in PARTS:                                            # FP8 main-KV rows through QSA eviction
        from . import kvevict
        for name, patcher in kvevict.PATCHERS.items():
            _add_hook(name, kvevict.PINS[name], patcher)
    if 'kvrot' in PARTS:                                              # h_nvfp4m main KV (rotated NVFP4, MSE scales)
        from . import kvrot
        for name, patcher in kvrot.PATCHERS.items():
            _add_hook(name, kvrot.PINS[name], patcher)
    if 'kvring' in PARTS:                                             # BF16 recent window (item 4a, 10-15): right after
        from . import kvring                                          # kvrot, it wraps kvrot's forward_qsa
        for name, patcher in kvring.PATCHERS.items():
            _add_hook(name, kvring.PINS[name], patcher)
    if 'kvq4' in PARTS:                                               # g4 quad KV (2026-10-26): after kvrot and kvring
        from . import kvq4
        for name, patcher in kvq4.PATCHERS.items():
            _add_hook(name, kvq4.PINS[name], patcher)
    if 'eprune' in PARTS:                                             # 2026-10-26: routed experts removed at load (the flag)
        from . import eprune
        for name, patcher in eprune.PATCHERS.items():
            _add_hook(name, eprune.PINS[name], patcher)
    if 'seloverlap' in PARTS:                                         # B4 diagnostic: verify rows' selection overlap
        from . import seloverlap                                      # (after kvrot: chained on the same module)
        for name, patcher in seloverlap.PATCHERS.items():
            _add_hook(name, seloverlap.PINS[name], patcher)
    if 'idxfq' in PARTS:                                              # B5 diagnostic: FP8 compressed indexer keys
        from . import idxfq
        for name, patcher in idxfq.PATCHERS.items():
            _add_hook(name, idxfq.PINS[name], patcher)
    if 'loadskip' in PARTS:                                           # startup readers skip unused files (A10)
        from . import loadskip
        for name, patcher in loadskip.PATCHERS.items():
            _add_hook(name, loadskip.PINS[name], patcher)
    if 'ckptfetch' in PARTS:                                          # bounded parallel shard prefetch (B9)
        from . import ckptfetch
        for name, patcher in ckptfetch.PATCHERS.items():
            _add_hook(name, ckptfetch.PINS[name], patcher)
    if 'draftwin' in PARTS:                                           # drafter selection window (probe, 10-03)
        from . import draftwin
        for name, patcher in draftwin.PATCHERS.items():
            _add_hook(name, draftwin.PINS[name], patcher)
    if 'rope2m' in PARTS:                                             # RoPE table and position limits to 2M (10-03)
        from . import rope2m                                          # (last: chained after guards and schedfast)
        for name, patcher in rope2m.PATCHERS.items():
            _add_hook(name, rope2m.PINS[name], patcher)
    if 'rebaseobs' in PARTS:                                          # rebase to the tail's full interactions (10-03)
        from . import rebaseobs
        for name, patcher in rebaseobs.PATCHERS.items():
            _add_hook(name, rebaseobs.PINS[name], patcher)
    if 'agestat' in PARTS:                                            # selection mass by age per eviction (10-03)
        from . import agestat
        for name, patcher in agestat.PATCHERS.items():
            _add_hook(name, agestat.PINS[name], patcher)
    if 'agecap' in PARTS:                                             # age gap: whole old interactions (10-04)
        from . import agecap                                          # (after agestat: its stats see production's plan)
        for name, patcher in agecap.PATCHERS.items():
            _add_hook(name, agecap.PINS[name], patcher)
    if 'hollowgroup' in PARTS:                                        # hollow rule on assistant + tool units (10-04)
        from . import hollowgroup
        for name, patcher in hollowgroup.PATCHERS.items():
            _add_hook(name, hollowgroup.PINS[name], patcher)
    if 'padbudget' in PARTS:                                          # padding in the session budget (10-06)
        from . import padbudget
        for name, patcher in padbudget.PATCHERS.items():
            _add_hook(name, padbudget.PINS[name], patcher)
    if 'poolpress' in PARTS:                                          # KV overcommit: pool-pressure cuts at turn starts (API server)
        assert 'padbudget' in PARTS, 'stage2_v1: poolpress goes through padbudget\'s base'
        from . import poolpress
        for name, patcher in poolpress.PATCHERS.items():
            _add_hook(name, poolpress.PINS[name], patcher)
    if 'stallguard' in PARTS:                                         # no persistent preemption: stall, then a length pause
        from . import stallguard
        for name, patcher in stallguard.PATCHERS.items():
            _add_hook(name, stallguard.PINS[name], patcher)
    if 'sessmem' in PARTS:                                            # session host memory (10-04)
        from . import sessmem
        for name, patcher in sessmem.PATCHERS.items():
            _add_hook(name, sessmem.PINS[name], patcher)
    if 'ctxtail' in PARTS:                                            # per-turn history work O(resident + new) (10-04)
        from . import ctxtail                                         # (before promptmove: promptmove wraps its staging)
        for name, patcher in ctxtail.PATCHERS.items():
            _add_hook(name, ctxtail.PINS[name], patcher)
    if 'prestage' in PARTS:                                           # resume staged during the previous step (10-15)
        from . import prestage
        for name, patcher in prestage.PATCHERS.items():
            _add_hook(name, prestage.PINS[name], patcher)
    if 'fqfast' in PARTS:                                             # frequency step from numpy (10-15)
        from . import fqfast
        for name, patcher in fqfast.PATCHERS.items():
            _add_hook(name, fqfast.PINS[name], patcher)
    if 'stepev' in PARTS:                                             # step GPU timeline from CUDA events (10-15)
        from . import stepev
        for name, patcher in stepev.PATCHERS.items():
            _add_hook(name, stepev.PINS[name], patcher)
    if 'promptmove' in PARTS:                                         # option 3: prompt keys follow the context (10-04)
        from . import promptmove                                      # (after guards / resumefast / rope2m staging)
        for name, patcher in promptmove.PATCHERS.items():
            _add_hook(name, promptmove.PINS[name], patcher)
    if 'foldc' in PARTS:                                              # PLE state out of the padded compressed pages (10-04)
        from . import foldc                                           # (after production's fold B import hook)
        for name, patcher in foldc.PATCHERS.items():
            _add_hook(name, foldc.PINS[name], patcher)
    if 'draftring' in PARTS:                                          # drafter + PLE out of the pool (item 4b, 10-15)
        from . import draftring                                       # (after production's fold B import hook, as foldc)
        for name, patcher in draftring.PATCHERS.items():
            _add_hook(name, draftring.PINS[name], patcher)
    if 'agefix' in PARTS:                                             # engine-core audit path (process-age slowdown, 10-16)
        from . import agefix
        for name, patcher in agefix.PATCHERS.items():
            _add_hook(name, agefix.PINS[name], patcher)
    if 'dflashtaps' in PARTS:                                         # DFlash target taps / feature export (10-10)
        from . import dflashtaps
        for name, patcher in dflashtaps.PATCHERS.items():
            _add_hook(name, dflashtaps.PINS[name], patcher)
    if 'precmode' in PARTS:                                           # precision audit: numerics modes (10-15)
        from . import precmode
        for name, patcher in precmode.PATCHERS.items():
            _add_hook(name, precmode.PINS[name], patcher)
    if 'kvwin' in PARTS:                                              # KV recent-window emulation (10-15)
        assert 'kvfq' not in PARTS, 'stage2_v1: kvwin and kvfq both wrap the QSA cache update'
        from . import kvwin
        for name, patcher in kvwin.PATCHERS.items():
            _add_hook(name, kvwin.PINS[name], patcher)
    if 'sumpin' in PARTS:                                             # v6.5 C5: the restart opener's pinned section (10-17);
        from . import sumpin                                          # after agestat / agecap / hollowgroup: the outermost
        for name, patcher in sumpin.PATCHERS.items():                 # planner wrapper (replayfx stays last when on)
            _add_hook(name, sumpin.PINS[name], patcher)
    if 'rebase74' in PARTS:                                           # the rebase arm (10-23): forced rebase, suspend /
        from . import rebase74                                        # resume, the P-aware cap, the pin's keep-first;
        for name, patcher in rebase74.PATCHERS.items():               # after rebaseobs / poolpress / sumpin (outermost)
            _add_hook(name, rebase74.PINS[name], patcher)
    if 'mmonly' in PARTS:                                             # 10-23: vLLM's multimodal processor cache in its
        from . import mmonly                                          # processor_only mode in every process (no P0 / P1
        for name, patcher in mmonly.PATCHERS.items():                 # drift, numerics unchanged; V67_DESIGN 287)
            _add_hook(name, mmonly.PINS[name], patcher)
    if 'kvq4' in PARTS:                                               # g4: the planner's whole-group snap, outermost
        from . import kvq4                                            # (after agecap / hollowgroup / sumpin)
        for name, patcher in kvq4.PLANNER_PATCHERS.items():
            _add_hook(name, kvq4.PINS[name], patcher)
    if 'replayfx' in PARTS:                                           # exact replay (10-10): last, so its planner
        from . import replayfx                                        # and backend wrappers are outermost
        assert replayfx.ENABLED, 'stage2_v1: replayfx needs ARC3_S2_REPLAY=1 (never a production launch)'
        for name, patcher in replayfx.PATCHERS.items():
            _add_hook(name, replayfx.PINS[name], patcher)
    if os.environ.get('ARC3_S2_AUTOTUNE_SEED'):                      # fixed FlashInfer MoE tactics (memopt.py)
        from . import memopt
        assert os.path.isfile(os.environ['ARC3_S2_AUTOTUNE_SEED']), os.environ['ARC3_S2_AUTOTUNE_SEED']
        name = 'vllm.model_executor.warmup.flashinfer_autotune_cache'
        _add_hook(name, memopt.PINS[name], memopt.patch_autotune_cache)
        REPORT['autotune_seed'] = os.environ['ARC3_S2_AUTOTUNE_SEED']
    if os.environ.get('ARC3_S2_GC_GEN2'):                            # full-collection pacing (10-08; not a part)
        import gc
        before = list(gc.get_threshold())
        gc.set_threshold(before[0], before[1], int(os.environ['ARC3_S2_GC_GEN2']))
        REPORT['gc_threshold'] = dict(before=before, after=list(gc.get_threshold()))
    if os.environ.get('ARC3_S2_HOSTPROF_DIR'):                       # host sampling profiler (hostprof.py), diagnostics
        from . import hostprof
        hostprof.start()
        REPORT['hostprof'] = os.environ['ARC3_S2_HOSTPROF_DIR']
    if os.environ.get('ARC3_S2_MEMSTATS_DIR'):                       # host-memory accounting on trigger (memstats.py)
        from . import memstats
        memstats.start()
        REPORT['memstats'] = os.environ['ARC3_S2_MEMSTATS_DIR']
    if os.environ.get('ARC3_S2_STEPTIMER_DIR'):                      # per-step host phase timings (steptimer.py)
        from . import steptimer
        steptimer.start()
        REPORT['steptimer'] = dict(dir=os.environ['ARC3_S2_STEPTIMER_DIR'], stats=steptimer.STATS)
    if os.environ.get('ARC3_S2_CANARY_DIR'):                         # process-age canary (canary.py; 10-15; not a part)
        from . import canary
        canary.start()
        REPORT['canary'] = dict(dir=os.environ['ARC3_S2_CANARY_DIR'])
    if os.environ.get('ARC3_S2_MEMDIAG', '0') not in ('', '0'):     # diagnostics only (memdiag.py)
        from . import memdiag
        _add_hook('arc3_host_diagnostics', memdiag.PINS['arc3_host_diagnostics'], memdiag.patch_diagnostics)
        REPORT['memdiag'] = os.environ['ARC3_S2_MEMDIAG']
    loaded = sorted(n for n in HOOKS if n in sys.modules)
    assert not loaded, f'stage2_v1: modules imported before the hook: {loaded}'
    if 'headonly' in PARTS:     # FP8 lm_head with BF16 dense: the adapter's hook and gate (headonly.py); before our finder
        from . import headonly
        REPORT['headonly'] = dict(headonly.apply_now(_Finder))
    if HOOKS and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if os.environ.get('ARC3_S2_SAVE_ATEXIT', '0') == '1':        # v7.5 (IE's review): a last forced receipt at a normal exit
        import atexit
        atexit.register(_save_atexit)
        REPORT['save_atexit'] = True
    save()
