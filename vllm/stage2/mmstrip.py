"""stage2_v1 'mmstrip' part: resumed persistent sessions stop re-sending the pixel data of images they have already
computed (GAP_PLAN item 1, 2026-09-30).

A persistent session's resume reaches the worker as a new request (NewRequestData.from_request) carrying the session's
whole history, including every image feature with its processed data: 22 images and 17.5 MB of tensors per resume in
g4d (steptimer payload fields), which overflow the 16 MB shared-memory chunk and go through the zmq socket (send 10.8 ms
per resume, plus the worker's receive). The worker reads MultiModalFeatureSpec.data only in
EncoderRunner.prepare_mm_inputs, for the step's scheduled encoder inputs (images inside the new prefill window), and
skips data=None there by design (processor-cache hits already arrive that way). Positions (MRoPE) come from the eviction
context; gather_mm_embeddings and the prefix ranges read mm_position and modality only.

So the transmitted copy replaces data with None for images lying wholly before the resume frontier
(offset + length <= num_computed_tokens): their tokens already have KV and are never prefilled again. Images that
straddle or follow the frontier keep their data. The engine's own request keeps everything (a new list and
dataclasses.replace copies; the request's list is never mutated), so any restart from frontier 0 or preemption re-add
sends full data. Only persistent requests (engine_patch.is_persistent) are touched. A stripped image that were ever
scheduled for encoding would fail loudly (encoder output missing), not silently."""
import dataclasses

PINS = {'vllm.v1.core.sched.output': '51643386fab3a56b7747aa784ad657833966ebe823acee1f37222e33c1e5696a'}
STATS = dict(calls=0, persistent_resumes=0, stripped_features=0, kept_features=0)


def _persistent(request):
    try:
        import engine_patch
        return engine_patch.is_persistent(request)
    except Exception:
        return False


def strip_features(features, frontier):
    """New list with data=None for features wholly before frontier; None when nothing changes."""
    out = None
    for i, f in enumerate(features):
        pos = f.mm_position
        if f.data is not None and pos.offset + pos.length <= frontier:
            if out is None:
                out = list(features[:i])
            out.append(dataclasses.replace(f, data=None))
            STATS['stripped_features'] += 1
        else:
            if out is not None:
                out.append(f)
            STATS['kept_features'] += 1
    return out


def patch_output(mod):
    cls = mod.NewRequestData
    original = cls.from_request.__func__

    def from_request(klass, request, block_ids, *args, **kwargs):
        nrd = original(klass, request, block_ids, *args, **kwargs)
        STATS['calls'] += 1
        frontier = nrd.num_computed_tokens
        feats = nrd.mm_features
        if frontier and feats and _persistent(request):
            STATS['persistent_resumes'] += 1
            out = strip_features(feats, frontier)
            if out is not None:
                nrd.mm_features = out
        return nrd

    cls.from_request = classmethod(from_request)


PATCHERS = {'vllm.v1.core.sched.output': patch_output}
