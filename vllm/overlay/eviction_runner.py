"""MRV2 original-position preservation with opt-in stable graph metadata.

Imports do not initialize torch or install hooks. The worker receives only
JSON-safe original token/image-layout information; old image tensors are never
encoded here. Runtime hooks are restricted to the pinned Qwen3.8 MTP0 model.
"""
from __future__ import annotations

import contextvars
import copy
from dataclasses import dataclass, replace
import functools
import hashlib
import os
import time
import bisect
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from scalar_publication import allocate_zero_fields, publish_scalars


class EvictionRunnerError(RuntimeError):
    pass


def _efficiency_enabled(name):
    value=os.environ.get(name,'0')
    if value not in ('0','1'):
        raise EvictionRunnerError(f'{name} must be 0 or 1')
    return value=='1'


CONTEXT_KEY = 'arc3_eviction_context'
_ACTIVE_ROPE = contextvars.ContextVar('arc3_eviction_original_rope', default=None)
_ACTIVE_QSA = contextvars.ContextVar('arc3_eviction_qsa_batch', default=None)
_READDING = contextvars.ContextVar('arc3_eviction_streaming_readd', default=frozenset())
_INSTALLED = False
SOURCE_HASHES = {
    'models/qwen3_8_flash_next/nvidia/model.py': 'd900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17',
    'models/qwen3_8_flash_next/nvidia/model_state.py': 'f127380fcd884c1fb7b010a2c32065d55c319f65a1268c18ef21376e39ba8425',
    'models/qwen3_8_flash_next/nvidia/indexer_qsa.py': 'e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066',
    'models/qwen3_8_flash_next/common/qsa_cache.py': '5ef7ddee8934e27e05d43b905487da06619a45d32bcec41d49b71ea5e9d25732',
    'model_executor/models/qwen3_vl.py': 'e041cc630f3a68ad2e058291799da81cb9355ff31a542abad326d77494f1f4e4',
    'model_executor/models/qwen3_5.py': 'd5fa8f7c99c5cac89bfb0c0288bab58c846ee27b9733178bf957ca44b9d1392d',
    'v1/worker/gpu/mm/rope.py': '8d1de6397e64e657a70a6ed4fc9c0fb64e7ca71c470d559bc7b7647281581d94',
    'v1/worker/gpu/model_states/default.py': 'ef4866fde8b3e0513a5d56f2fb80aabd809ac9e64d551175d9d9ca7c9cb7eec1',
    'v1/worker/gpu/model_states/mamba_hybrid.py': '2e65f1da0a440f11a784bd36afe9206a7a2f674d522c047b838ad604e4c6acc0',
    'v1/worker/gpu/model_runner.py': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8',
}
GRAPH_SOURCE_HASHES = {
    'v1/worker/gpu/cudagraph_utils.py': 'ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c',
    'v1/worker/gpu/input_batch.py': '3929c92e42ae90e4410bb4537dcbdcd171a662e44afc1189a9a3e19037a84410',
    'v1/attention/backend.py': '666c1928ea840d3dc65a45c82c2d5a1ceebee88939af66b9ba3de9fa5e0fc6d2',
    'v1/worker/gpu/attn_utils.py': '663743e98ad85c0dcb6c53080dd88d58a0fa9c87fbb1ed18c546c1f825f53f7a',
}


def verify_sources(runtime_root):
    actual = {}
    hashes = dict(SOURCE_HASHES)
    if graph_mapping_enabled():
        hashes.update(GRAPH_SOURCE_HASHES)
    for name,expected in hashes.items():
        value = hashlib.sha256((Path(runtime_root)/name).read_bytes()).hexdigest()
        if value != expected:
            raise EvictionRunnerError(f'Unreviewed worker source {name}: {value}')
        actual[name] = value
    return actual


def _ints(values, name, *, ordered=False):
    # 2026-09-24: C-speed type scan + numpy range/order checks; same accept/reject set as the element loop
    if not isinstance(values, (list,tuple)):
        raise EvictionRunnerError(f'{name} must contain nonnegative integers')
    n = len(values)
    if n:
        if set(map(type, values)) != {int}:
            raise EvictionRunnerError(f'{name} must contain nonnegative integers')
        try:
            arr = np.fromiter(values, dtype=np.int64, count=n)
        except OverflowError:
            arr = None
        if arr is None:
            if any(x < 0 for x in values):
                raise EvictionRunnerError(f'{name} must contain nonnegative integers')
            if ordered and any(a >= b for a,b in zip(values, values[1:])):
                raise EvictionRunnerError(f'{name} must be strictly increasing')
        else:
            if int(arr.min()) < 0:
                raise EvictionRunnerError(f'{name} must contain nonnegative integers')
            if ordered and n > 1 and not bool((arr[1:] > arr[:-1]).all()):
                raise EvictionRunnerError(f'{name} must be strictly increasing')
    return list(values)


def _ol_flag(name, default):
    value = os.environ.get(name, default)
    if value not in ('0', '1'):
        raise EvictionRunnerError(f'{name} must be 0 or 1')
    return value == '1'


def _gather(ids, positions):
    import operator
    if not positions:
        return []
    if len(positions) == 1:
        return [ids[positions[0]]]
    return list(operator.itemgetter(*positions)(ids))


def trusted_ids(existing):
    """The worker's own validated logical ids for a request (2026-09-25 O(L) fast path), or None."""
    if existing is None or not _ol_flag('ARC3_OL_FASTPATH', '1'):
        return None
    ids = existing.get('logical_token_ids')
    return ids if type(ids) is list and ids else None


def _ids_with_trusted_prefix(values, trusted):
    """_ints for the logical ids, except that a prefix equal to the worker's already validated copy is not rescanned:
    only the tail is checked, and the result reuses the validated prefix elements."""
    if (trusted is None or type(values) is not list or len(values) < len(trusted)
            or values[:len(trusted)] != trusted):
        return _ints(values, 'Logical token IDs')
    tail = values[len(trusted):]
    return trusted + (_ints(tail, 'Logical token IDs') if tail else [])


def validate_context(context, *, dense_ids=None, dense_computed=None,
                     max_resident_tokens, max_logical_tokens, ngram_context_len=2,
                     allow_closed_frontier=False, trusted_ids=None):
    """Prove dense storage is a retained subsequence with a contiguous new tail."""
    if trusted_ids is not None and _ol_flag('ARC3_OL_FASTPATH_VERIFY', '0'):
        kw = dict(dense_ids=dense_ids, dense_computed=dense_computed, max_resident_tokens=max_resident_tokens,
                  max_logical_tokens=max_logical_tokens, ngram_context_len=ngram_context_len,
                  allow_closed_frontier=allow_closed_frontier)
        fast = _validate_context(context, trusted_ids=trusted_ids, **kw); full = _validate_context(context, **kw)
        if fast != full:
            raise EvictionRunnerError('Trusted-prefix context validation differs from the full validation')
        return fast
    return _validate_context(context, dense_ids=dense_ids, dense_computed=dense_computed,
                             max_resident_tokens=max_resident_tokens, max_logical_tokens=max_logical_tokens,
                             ngram_context_len=ngram_context_len, allow_closed_frontier=allow_closed_frontier,
                             trusted_ids=trusted_ids)


def _validate_context(context, *, dense_ids=None, dense_computed=None,
                      max_resident_tokens, max_logical_tokens, ngram_context_len=2,
                      allow_closed_frontier=False, trusted_ids=None):
    required = {'logical_token_ids','resident_positions','logical_computed',
                'eviction_revision','deleted_tokens','deleted_groups','logical_image_features'}
    if not isinstance(context,dict) or not required <= set(context):
        raise EvictionRunnerError('Missing original-position worker context')
    ids = _ids_with_trusted_prefix(context['logical_token_ids'], trusted_ids)
    positions = _ints(context['resident_positions'],'Resident positions',ordered=True)
    if not ids or not positions or positions[-1] != len(ids)-1:
        raise EvictionRunnerError('Original/resident stream must preserve the accepted tail')
    if len(ids) > max_logical_tokens or len(positions) > max_resident_tokens:
        raise EvictionRunnerError('Original/resident position capacity exceeded')
    computed = context['logical_computed']
    if type(allow_closed_frontier) is not bool:
        raise EvictionRunnerError('Closed-frontier allowance must be an explicit boolean')
    last_computed = len(ids) if allow_closed_frontier else len(ids)-1
    if type(computed) is not int or not 0 <= computed <= last_computed:
        raise EvictionRunnerError('Original computed frontier is outside the pending stream')
    for name in ('eviction_revision','deleted_tokens','deleted_groups'):
        if type(context[name]) is not int or context[name] < 0:
            raise EvictionRunnerError(f'Invalid {name}')
    if context['deleted_tokens'] != len(ids)-len(positions):
        raise EvictionRunnerError('Deleted token count disagrees with retained positions')
    efficient_copy=_efficiency_enabled('ARC3_EFFICIENCY_VALIDATED_CONTEXT_COPY')
    if dense_ids is not None:
        # 2026-09-24 stage E: numpy comparison with the retained subsequence; list fallback for odd inputs
        try:
            if set(map(type, dense_ids)) != {int}:
                raise TypeError('non-int dense ids')
            if _ol_flag('ARC3_OL_FASTPATH', '1'):   # 2026-09-25: C-level gather, no full-length array of the logical ids
                _ok = len(dense_ids) == len(positions) and list(dense_ids) == _gather(ids, positions)
            else:
                _dense = np.fromiter(dense_ids, dtype=np.int64, count=len(dense_ids))
                _ok = (_dense.shape[0] == len(positions) and bool(np.array_equal(
                    _dense, np.fromiter(ids, dtype=np.int64, count=len(ids))[np.fromiter(positions, dtype=np.int64, count=len(positions))])))
        except (TypeError, ValueError, OverflowError):
            _ok = list(dense_ids) == [ids[p] for p in positions]
        if not _ok:
            raise EvictionRunnerError('Dense prompt IDs differ from the original retained subsequence')
    computed_dense = bisect.bisect_left(positions, computed)   # positions are strictly increasing (validated above)
    if dense_computed is not None and dense_computed != computed_dense:
        raise EvictionRunnerError('Dense/original computed frontiers disagree')
    if positions[computed_dense:] != list(range(computed,len(ids))):
        raise EvictionRunnerError('A deletion intersects the newly forwarded contiguous tail')
    # This is enough to preserve the existing PLE gather exactly. After this
    # boundary every newly processed token has ordinary contiguous predecessors.
    predecessors = list(range(max(0,computed-ngram_context_len),computed))
    if predecessors and positions[max(0,computed_dense-len(predecessors)):computed_dense] != predecessors:
        raise EvictionRunnerError('Eviction changes PLE immediate predecessor tokens')
    _pos = np.fromiter(positions, dtype=np.int64, count=len(positions))
    deleted_groups = computed//4-int(np.unique(_pos[:bisect.bisect_left(positions, 4*(computed//4))]//4).size)
    if context['deleted_groups'] != deleted_groups:
        raise EvictionRunnerError('Deleted QSA group count disagrees with original groups')
    images = context['logical_image_features']
    if not isinstance(images,list):
        raise EvictionRunnerError('Original image layout must be a list')
    previous_end = 0
    for image in images:
        if not isinstance(image,dict) or set(image) != {'offset','length','grid_thw'}:
            raise EvictionRunnerError('Original image layout supports offset/length/grid_thw only')
        offset,length = image['offset'],image['length']
        grid = _ints(image['grid_thw'],'Image grid dimensions')
        if (type(offset) is not int or type(length) is not int or length <= 0
                or offset < previous_end or offset+length > len(ids)
                or len(grid) != 3 or grid[0] != 1 or min(grid) <= 0):
            raise EvictionRunnerError('Invalid or overlapping original image layout')
        previous_end = offset+length
    if not efficient_copy:
        return copy.deepcopy(context)
    # Both integer arrays have already been validated and copied above. Avoid
    # recursively copying each immutable int again, but retain deepcopy's
    # aliasing and extra-field semantics. Subclasses still use ordinary deepcopy.
    memo = {}
    for name, validated in (('logical_token_ids',ids),('resident_positions',positions)):
        original = context[name]
        if id(original) not in memo:
            if type(original) is list:
                memo[id(original)] = validated
            elif type(original) is tuple:
                memo[id(original)] = original
    return copy.deepcopy(context,memo)


class _Grid:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def original_image_proxies(context):
    # Exact fields read by the pinned Qwen3VL image MRoPE implementation. No
    # tensor or pixel payload is needed for this CPU positional calculation.
    return [SimpleNamespace(modality='image',
        mm_position=SimpleNamespace(offset=image['offset'],length=image['length']),
        data={'image_grid_thw':SimpleNamespace(data=_Grid(image['grid_thw']))})
        for image in context['logical_image_features']]


def _mrope_full(model, context):
    """The model's original MRoPE positions for context's logical ids and images: (3, L) integer array, raw delta."""
    merge = int(model.config.vision_config.spatial_merge_size)
    for image in context['logical_image_features']:
        t,h,w = image['grid_thw']
        if t != 1 or h%merge or w%merge or (h//merge)*(w//merge) != image['length']:
            raise EvictionRunnerError('Original image grid and visual token length disagree')
    full,delta = model.get_mrope_input_positions(context['logical_token_ids'],
                                                 original_image_proxies(context))
    try:
        arr = np.asarray(full.numpy() if hasattr(full, 'numpy') else full)
    except Exception as exc:
        raise EvictionRunnerError('Conditional model returned unexpected original MRoPE dimensions') from exc
    if arr.ndim != 2 or arr.shape[0] != 3 or arr.shape[1] != len(context['logical_token_ids']):
        raise EvictionRunnerError('Conditional model returned unexpected original MRoPE dimensions')
    if not np.issubdtype(arr.dtype, np.integer) or (arr.size and int(arr.min()) < 0):
        raise EvictionRunnerError('Original MRoPE positions must contain nonnegative integers')
    if type(delta) is not int:
        raise EvictionRunnerError('Conditional model returned a noninteger MRoPE delta')
    return arr, delta


def _mrope_rows(runner, req_id, context):
    """original_mrope_rows with a per-request prefix cache (2026-09-25 O(L) fast path). Qwen3-VL positions are built
    segment by segment, each continuing from the previous segment's maximum, so when the logical ids extend the cached
    ones (private copy, compared at C speed), the cached images are unchanged and no image straddles the boundary,
    the model's function runs on the appended suffix only and its positions continue from the cached maximum."""
    model = runner.model_state.model
    if not _ol_flag('ARC3_OL_FASTPATH', '1'):
        return original_mrope_rows(model, context)
    caches = runner.__dict__.setdefault('_arc3_mrope_cache', {})
    ids = context['logical_token_ids']; images = context['logical_image_features']; L = len(ids)
    entry = caches.get(req_id)
    if entry is not None:
        n, k = len(entry['ids']), len(entry['images'])
        if not (n <= L and len(images) >= k and ids[:n] == entry['ids'] and images[:k] == entry['images']
                and all(image['offset'] >= n for image in images[k:])):
            entry = None
    if entry is None:
        arr, _ = _mrope_full(model, context)
        buf = np.empty((3, max(L + 8192, 2 * L)), dtype=np.int32); buf[:, :L] = arr
        entry = caches[req_id] = dict(ids=list(ids), images=copy.deepcopy(images), buf=buf,
                                      max=int(arr.max()) if L else -1)
    elif L > n:
        suffix = dict(logical_token_ids=ids[n:],
                      logical_image_features=[dict(image, offset=image['offset']-n) for image in images[k:]])
        arr, _ = _mrope_full(model, suffix)
        base = entry['max'] + 1
        buf = entry['buf']
        if L > buf.shape[1]:
            grown = np.empty((3, max(L + 8192, 2 * buf.shape[1])), dtype=np.int32); grown[:, :n] = buf[:, :n]
            buf = entry['buf'] = grown
        buf[:, n:L] = arr + base
        entry['max'] = base + int(arr.max())
        entry['ids'].extend(ids[n:]); entry['images'].extend(copy.deepcopy(images[k:]))
    _pos = np.fromiter(context['resident_positions'], dtype=np.int64, count=len(context['resident_positions']))
    rows = entry['buf'][:, _pos].tolist(); delta = entry['max'] + 1 - L + context['deleted_tokens']
    if _ol_flag('ARC3_OL_FASTPATH_VERIFY', '0') and (rows, delta) != tuple(original_mrope_rows(model, context)):
        raise EvictionRunnerError('Cached MRoPE rows differ from the full construction')
    return rows, delta


def original_mrope_rows(model, context):
    """Invoke the real outer conditional model, then gather resident columns."""
    merge = int(model.config.vision_config.spatial_merge_size)
    for image in context['logical_image_features']:
        t,h,w = image['grid_thw']
        if t != 1 or h%merge or w%merge or (h//merge)*(w//merge) != image['length']:
            raise EvictionRunnerError('Original image grid and visual token length disagree')
    full,delta = model.get_mrope_input_positions(context['logical_token_ids'],
                                                 original_image_proxies(context))
    # 2026-09-24 stage E: numpy gather of the resident columns (same lists as tolist + comprehensions)
    try:
        arr = np.asarray(full.numpy() if hasattr(full, 'numpy') else full)
    except Exception as exc:
        raise EvictionRunnerError('Conditional model returned unexpected original MRoPE dimensions') from exc
    if arr.ndim != 2 or arr.shape[0] != 3 or arr.shape[1] != len(context['logical_token_ids']):
        raise EvictionRunnerError('Conditional model returned unexpected original MRoPE dimensions')
    if not np.issubdtype(arr.dtype, np.integer) or (arr.size and int(arr.min()) < 0):
        raise EvictionRunnerError('Original MRoPE positions must contain nonnegative integers')
    if type(delta) is not int:
        raise EvictionRunnerError('Conditional model returned a noninteger MRoPE delta')
    _pos = np.fromiter(context['resident_positions'], dtype=np.int64, count=len(context['resident_positions']))
    selected = arr[:, _pos].tolist()
    return selected,delta+context['deleted_tokens']


def historical_virtual_lookup(context):
    """Retained original groups in row order, with -1 for removed members."""
    # Callers validate the complete context first. With no deletion, the
    # existing analytic tail maps every nonnegative virtual index to itself.
    # A zero cutoff also makes stale table contents unreachable on slot reuse.
    if (_efficiency_enabled('ARC3_EFFICIENCY_IDENTITY_LOOKUP')
            and context['deleted_tokens'] == 0):
        if context['deleted_groups'] != 0:
            raise EvictionRunnerError('Identity mapping cannot delete QSA groups')
        return [],0,0
    positions = context['resident_positions']
    _pos = np.fromiter(positions, dtype=np.int64, count=len(positions))   # 2026-09-24: numpy table build
    _groups = np.unique(_pos//4)
    _full = np.full((len(_groups), 4), -1, dtype=np.int64)
    _full[np.searchsorted(_groups, _pos//4), _pos%4] = np.arange(len(_pos))
    table = _full.ravel().tolist()
    padding = context['deleted_tokens']-4*context['deleted_groups']
    # Cut off before unprocessed future members of the last partial group.
    cutoff = len(positions)+padding
    if cutoff < 0 or cutoff > len(table):
        raise EvictionRunnerError('Invalid QSA virtual frontier')
    return table[:cutoff],cutoff,padding


def _limits(runner):
    config = runner.vllm_config
    validate_execution_mode(config)
    if config.num_speculative_tokens or config.speculative_config is not None:
        raise EvictionRunnerError('Eviction runner supports MTP0 only')
    if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.pipeline_parallel_size != 1:
        raise EvictionRunnerError('Eviction runner supports TP1/PP1 only')
    state = runner.model_state
    if type(state).__name__ != 'Qwen3_8FlashNextModelState' or state.rope_state is None:
        raise EvictionRunnerError('Expected pinned Qwen3.8 conditional model state with MRoPE')
    if state._align_mode or state.mm_pruner is not None:
        raise EvictionRunnerError('Eviction control requires fixed recurrent state and no image pruning')
    logical = int(config.model_config.hf_text_config.max_position_embeddings)
    return dict(max_resident_tokens=int(state.max_model_len),max_logical_tokens=logical,
                ngram_context_len=int(state.ngram_context_len))


def graph_mapping_enabled():
    """Keep the proven eager path as the default until live graph validation."""
    value = os.environ.get('ARC3_EVICTION_GRAPH_MAPPING', '0')
    if value not in ('0', '1'):
        raise EvictionRunnerError('ARC3_EVICTION_GRAPH_MAPPING must be 0 or 1')
    return value == '1'


def validate_execution_mode(config):
    mode = config.compilation_config.cudagraph_mode
    name = getattr(mode, 'name', str(mode))
    eager = bool(config.model_config.enforce_eager)
    enabled = graph_mapping_enabled()
    if not enabled:
        if name != 'NONE' or not eager:
            raise EvictionRunnerError('Initial eviction correctness candidate requires --enforce-eager')
    elif not ((eager and name == 'NONE')
              or (not eager and name == 'FULL_AND_PIECEWISE')):
        raise EvictionRunnerError('Graph mapping supports eager NONE or FULL_AND_PIECEWISE only')
    return dict(eager_only=not enabled, graph_mapping=enabled, cudagraph_mode=name)


@dataclass(frozen=True)
class PreparedWorkerEviction:
    request_id: str
    context: dict
    mrope_rows: list[list[int]]
    mrope_delta: int
    virtual_lookup: list[int]
    virtual_cutoff: int
    group_padding: int


def prepare_worker_eviction(runner, request_id, context, block_ids, *, allow_closed_frontier=False):
    """Failure-prone CPU checks before any in-place cache copy or page free."""
    limits = _limits(runner)
    checked = validate_context(context,allow_closed_frontier=allow_closed_frontier,**limits)
    if request_id not in runner.req_states.req_id_to_index:
        raise EvictionRunnerError('Eviction target has no resident worker request')
    if not block_ids or any(not group for group in block_ids):
        raise EvictionRunnerError('Eviction requires explicit retained block ownership')
    rows,delta = original_mrope_rows(runner.model_state.model,checked)
    table,cutoff,padding = historical_virtual_lookup(checked)
    if cutoff > limits['max_logical_tokens']:
        raise EvictionRunnerError('QSA virtual mapping exceeds allocated worker capacity')
    return PreparedWorkerEviction(request_id,checked,rows,delta,table,cutoff,padding)


def apply_worker_eviction(runner, request_id, prepared):
    """Publish only CPU state; old dense wrappers remain unusable until re-add."""
    if not isinstance(prepared,PreparedWorkerEviction) or prepared.request_id != request_id:
        raise EvictionRunnerError('Prepared worker eviction belongs to a different request')
    runner._arc3_eviction_states[request_id] = dict(prepared.context,awaiting_readd=True)
    return {'awaiting_readd':True,'eviction_revision':prepared.context['eviction_revision']}


def _allocate_buffers(runner):
    import torch
    limits = _limits(runner)
    state = runner.model_state
    rows = int(state.max_num_reqs)
    width = limits['max_logical_tokens']
    runner._arc3_eviction_states = {}
    state._arc3_eviction_runner = runner
    state._arc3_eviction_graph_mapping = graph_mapping_enabled()
    trace = os.environ.get('ARC3_EVICTION_GRAPH_TRACE', '0')
    if trace not in ('0', '1'):
        raise EvictionRunnerError('ARC3_EVICTION_GRAPH_TRACE must be 0 or 1')
    state._arc3_eviction_graph_trace = trace == '1'
    state._arc3_eviction_batch_counts = {}
    state._arc3_eviction_buffers = SimpleNamespace(
        **allocate_zero_fields(torch,rows,device=state.device),
        lookup=torch.full((rows,width),-1,dtype=torch.int32,device=state.device))
    import qsa_frequency
    state._arc3_qsa_frequency = None
    if qsa_frequency.enabled():
        # Off by default; when on, the selector wrapper counts into these static
        # buffers and the device flag stays 0 until the runner enables it after warmup.
        state._arc3_qsa_frequency = qsa_frequency.allocate(
            torch, slots=rows, capacity=qsa_frequency.descriptor_capacity(limits['max_resident_tokens']),
            device=state.device, bound_layers=lambda: runner.compilation_config.static_forward_context)
        global _QSA_FREQUENCY
        _QSA_FREQUENCY = state._arc3_qsa_frequency
    if state._arc3_eviction_graph_mapping:
        # Full CUDA graphs retain these addresses. Each normal prepare_attn
        # updates their contents before replay; graph capture does not freeze
        # a Python-selected request order or a temporary seq_lens allocation.
        buffers = state._arc3_eviction_buffers
        buffers.batch_slots = torch.full((rows,), -1, dtype=torch.int64, device=state.device)
        buffers.raw_seq_lens = torch.zeros(rows, dtype=torch.int32, device=state.device)
        buffers.virtual_seq_lens = torch.zeros(rows, dtype=torch.int32, device=state.device)


def _stage_graph_batch(buffers, input_batch, *, for_capture, padded):
    """Refresh graph-owned vectors outside capture, masking unused batch rows.

    All three returned tensor views always share the same lifetime-long base
    allocations. Request slots are not inferred from row numbers: eviction,
    streaming re-add, scheduler reordering and slot reuse can change them.
    """
    real = int(input_batch.num_reqs)
    count = int(input_batch.num_reqs_after_padding) if padded else real
    if not 0 <= real <= count <= buffers.batch_slots.shape[0]:
        raise EvictionRunnerError('Graph QSA batch exceeds stable request capacity')
    if (input_batch.idx_mapping.shape[0] < real
            or input_batch.seq_lens.shape[0] < count):
        raise EvictionRunnerError('Graph QSA input vectors are smaller than the active batch')
    cpu_slots = input_batch.idx_mapping_np[:real]
    if len(cpu_slots) != real or any(slot < 0 or slot >= buffers.batch_slots.shape[0]
                                     for slot in cpu_slots):
        raise EvictionRunnerError('Graph QSA request slot is outside worker capacity')
    slots = buffers.batch_slots[:count]
    raw_lens = buffers.raw_seq_lens[:count]
    virtual_lens = buffers.virtual_seq_lens[:count]
    slots.fill_(-1)
    raw_lens.zero_()
    virtual_lens.zero_()
    if real:
        slots[:real].copy_(input_batch.idx_mapping[:real])
        base = input_batch.seq_lens[:real]
        if for_capture:
            # Capture can run dummy forwards with a live worker populated.
            # Dummy rows are identity mappings, independent of resident slots.
            slots[:real].fill_(-1)
            raw_lens[:real].copy_(base)
            virtual_lens[:real].copy_(base)
        else:
            # Indexing temporaries are outside the captured model graph; only
            # the stable output vectors are referenced from captured kernels.
            selected = slots[:real]
            raw_lens[:real].copy_(base + buffers.deleted[selected].to(base.dtype))
            virtual_lens[:real].copy_(base + buffers.padding[selected].to(base.dtype))
    return SimpleNamespace(lookup=buffers.lookup, cutoffs=buffers.cutoffs,
        state_padding=buffers.padding, slots=slots, mapping=None,
        raw_seq_lens=raw_lens, virtual_seq_lens=virtual_lens, graph_mapping=True)


def _stage_request(runner, req_index, new_req_data, context):
    import torch
    profiling=os.environ.get('ARC3_EVICTION_PROFILE','0')=='1'
    started=phase_start=time.perf_counter() if profiling else 0
    phases={}
    limits = _limits(runner)
    existing = runner._arc3_eviction_states.get(new_req_data.req_id)
    checked = validate_context(context,dense_ids=new_req_data.prefill_token_ids,
        dense_computed=new_req_data.num_computed_tokens,trusted_ids=trusted_ids(existing),**limits)
    if existing is not None:
        old = existing['logical_token_ids']
        if checked['logical_token_ids'][:len(old)] != old:
            raise EvictionRunnerError('Worker original token prefix was rewritten')
        if checked['eviction_revision'] != existing['eviction_revision']:
            raise EvictionRunnerError('Worker eviction revision changed without a committed transaction')
    if profiling:
        now=time.perf_counter();phases['validate_context_seconds']=now-phase_start;phase_start=now
    rows,delta = _mrope_rows(runner,new_req_data.req_id,checked)
    if profiling:
        now=time.perf_counter();phases['original_mrope_seconds']=now-phase_start;phase_start=now
    table,cutoff,padding = historical_virtual_lookup(checked)
    if profiling:
        now=time.perf_counter();phases['lookup_build_seconds']=now-phase_start;phase_start=now
    buffers = runner.model_state._arc3_eviction_buffers
    if cutoff > buffers.lookup.shape[1]:
        raise EvictionRunnerError('QSA virtual mapping exceeds worker buffer width')
    # Per-request staging runs outside CUDA graphs; full graph replay reads the
    # stable buffers through the current batch-to-state mapping.
    with publish_scalars(buffers,req_index,checked['deleted_tokens'],padding,cutoff):
        if cutoff:
            buffers.lookup[req_index,:cutoff].copy_(torch.tensor(table,dtype=torch.int32,device=buffers.lookup.device))
    runner._arc3_eviction_states[new_req_data.req_id] = dict(checked,awaiting_readd=False,req_index=req_index)
    if profiling:
        now=time.perf_counter();phases['buffer_staging_seconds']=now-phase_start
        phases['total_seconds']=now-started
        totals=getattr(runner.model_state,'_arc3_eviction_staging_profile',None)
        if totals is None:
            totals={'completed_calls':0,'identity_calls':0,'lookup_entries_uploaded':0,'seconds':{}}
            runner.model_state._arc3_eviction_staging_profile=totals
        totals['completed_calls']+=1
        totals['identity_calls']+=int(checked['deleted_tokens']==0)
        totals['lookup_entries_uploaded']+=cutoff
        for key,value in phases.items():
            totals['seconds'][key]=totals['seconds'].get(key,0)+value
    return rows,delta


def _assert_active(runner, input_batch):
    for index,req_id in enumerate(input_batch.req_ids):
        state = runner._arc3_eviction_states.get(req_id)
        if state is not None and state.get('awaiting_readd'):
            raise EvictionRunnerError('Compacted request reached forward before streaming re-add')
        computed = getattr(input_batch,'num_computed_tokens_np',None)
        if (state is not None and computed is not None
                and int(computed[index])+state['deleted_tokens'] < state['logical_computed']):
            raise EvictionRunnerError('Worker attempted to re-forward the historical compacted prefix')


def _trace_batch(state, input_batch, mode):
    if not state._arc3_eviction_graph_trace:
        return
    name = getattr(mode, 'name', str(mode))
    real = int(input_batch.num_reqs)
    padded = int(input_batch.num_reqs_after_padding) if name == 'FULL' else real
    # Request cardinality/mode counters only: no per-token JSON IO or GPU reads.
    key = (name, real, padded)
    counts = state._arc3_eviction_batch_counts
    counts[key] = counts.get(key, 0) + 1


def arc3_eviction_graph_status(worker):
    """Read-only configuration, prepared batch counts, and resident slot IDs."""
    runner = worker.model_runner
    state = runner.model_state
    from scalar_publication import status as scalar_publication_status
    manager = runner.cudagraph_manager
    mode = runner.vllm_config.compilation_config.cudagraph_mode
    return dict(graph_mapping=bool(getattr(state, '_arc3_eviction_graph_mapping', False)),
        trace_enabled=bool(getattr(state, '_arc3_eviction_graph_trace', False)),
        enforce_eager=bool(runner.vllm_config.model_config.enforce_eager),
        cudagraph_mode=getattr(mode, 'name', str(mode)),
        captured_token_counts=[] if manager is None else manager.captured_token_counts(),
        prepared_batches=[dict(mode=key[0],real_requests=key[1],padded_requests=key[2],count=value)
            for key,value in sorted(getattr(state,'_arc3_eviction_batch_counts',{}).items())],
        request_slots={key:int(value) for key,value in runner.req_states.req_id_to_index.items()},
        staging_profile=copy.deepcopy(getattr(state,'_arc3_eviction_staging_profile',None)),
        scalar_publication=scalar_publication_status(state._arc3_eviction_buffers),
        staging_profile_scope='Successful _stage_request host elapsed time only; no per-token IO or added CUDA synchronization; excludes the subsequent original add_request.',
        evidence_scope='Prepared metadata batches; successful responses and engine audit establish completed work')


_QSA_FREQUENCY = None


def _qsa_frequency_counters():
    # One runner process, one model state: the wrapper reaches counters through this global.
    return _QSA_FREQUENCY


def install_runner_hooks():
    """Install isolated worker hooks after phase-one/core installation."""
    global _INSTALLED
    if _INSTALLED:
        return {'installed':True,'already_installed':True,'eager_only':not graph_mapping_enabled(),
                'graph_mapping':graph_mapping_enabled()}
    import vllm
    verify_sources(Path(vllm.__file__).parent)
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.worker_base import WorkerBase
    from vllm.models.qwen3_8_flash_next.nvidia.model_state import Qwen3_8FlashNextModelState
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.models.qwen3_8_flash_next.common.qsa_cache import QSAMetadataBuilder
    from eviction_qsa import QSASelectionMapping, install_qsa_selector_wrapper, mapping_from_forward_context

    original_load = GPUModelRunner.load_model
    original_add_many = GPUModelRunner.add_requests
    original_remove = GPUModelRunner._remove_request
    original_add = Qwen3_8FlashNextModelState.add_request
    original_rope = RopeState.init_prefill_positions
    original_attn = Qwen3_8FlashNextModelState.prepare_attn
    original_inputs = Qwen3_8FlashNextModelState.prepare_inputs
    original_qsa = QSAMetadataBuilder.build

    @functools.wraps(original_load)
    def load_model(self,*args,**kwargs):
        result = original_load(self,*args,**kwargs)
        _allocate_buffers(self)
        return result

    @functools.wraps(original_add_many)
    def add_requests(self,scheduler_output):
        token = _READDING.set(frozenset(req.req_id for req in scheduler_output.scheduled_new_reqs))
        try:
            return original_add_many(self,scheduler_output)
        finally:
            _READDING.reset(token)

    @functools.wraps(original_remove)
    def remove_request(self,req_id):
        result = original_remove(self,req_id)
        if req_id not in _READDING.get():
            getattr(self,'_arc3_eviction_states',{}).pop(req_id,None)
            getattr(self,'_arc3_mrope_cache',{}).pop(req_id,None)
        return result

    @functools.wraps(original_add)
    def add_request(self,req_index,new_req_data):
        runner = getattr(self,'_arc3_eviction_runner',None)
        extra = getattr(new_req_data.sampling_params,'extra_args',None) or {}
        context = extra.get(CONTEXT_KEY)
        if runner is None or context is None:
            if runner is not None:
                if new_req_data.req_id in runner._arc3_eviction_states:
                    raise EvictionRunnerError('Existing eviction session lost its original-position context')
                buffers = self._arc3_eviction_buffers
                with publish_scalars(buffers,req_index,0,0,0):
                    pass
            return original_add(self,req_index,new_req_data)
        fresh_session = new_req_data.req_id not in runner._arc3_eviction_states
        rows,delta = _stage_request(runner,req_index,new_req_data,context)
        if fresh_session and _QSA_FREQUENCY is not None:
            # Re-adds after eviction keep their statistics; only a new owner clears the slot.
            import qsa_frequency
            qsa_frequency.on_fresh_admission(_QSA_FREQUENCY, int(req_index))
        token = _ACTIVE_ROPE.set((self.rope_state,req_index,rows,delta))
        try:
            return original_add(self,req_index,new_req_data)
        finally:
            _ACTIVE_ROPE.reset(token)

    @functools.wraps(original_rope)
    def init_prefill_positions(self,req_idx,model,prefill_token_ids,mm_features):
        active = _ACTIVE_ROPE.get()
        if active is None or active[0] is not self or active[1] != req_idx:
            return original_rope(self,req_idx,model,prefill_token_ids,mm_features)
        _,_,rows,delta = active
        # Replace the original staging call, not an additional overlapping
        # staged write: the fused writer does not order duplicate row writes.
        for axis,row in enumerate(rows):
            self.prefill_positions.stage_write(self.num_dims*req_idx+axis,0,row)
        self.prefill_delta.np[req_idx] = delta

    @functools.wraps(original_inputs)
    def prepare_inputs(self,input_batch,req_states):
        runner = getattr(self,'_arc3_eviction_runner',None)
        if runner is not None:
            _assert_active(runner,input_batch)
        return original_inputs(self,input_batch,req_states)

    @functools.wraps(original_attn)
    def prepare_attn(self,input_batch,*args,**kwargs):
        runner = getattr(self,'_arc3_eviction_runner',None)
        if runner is None:
            return original_attn(self,input_batch,*args,**kwargs)
        capture = bool(kwargs.get('for_capture',False))
        mode = args[0] if args else kwargs.get('cudagraph_mode')
        if not capture and any(req_id in runner._arc3_eviction_states for req_id in input_batch.req_ids):
            _trace_batch(self,input_batch,mode)
        if self._arc3_eviction_graph_mapping:
            if not capture:
                _assert_active(runner,input_batch)
            # Capture uses NONE but real full replay uses padded FULL sizes.
            padded = capture or getattr(mode,'name',str(mode)) == 'FULL'
            active = _stage_graph_batch(self._arc3_eviction_buffers, input_batch,
                                       for_capture=capture, padded=padded)
            token = _ACTIVE_QSA.set(active)
            try:
                return original_attn(self,input_batch,*args,**kwargs)
            finally:
                _ACTIVE_QSA.reset(token)
        if capture:
            return original_attn(self,input_batch,*args,**kwargs)
        _assert_active(runner,input_batch)
        if not any(runner._arc3_eviction_states.get(req_id,{}).get('deleted_tokens',0)
                   for req_id in input_batch.req_ids):
            return original_attn(self,input_batch,*args,**kwargs)
        buffers = self._arc3_eviction_buffers
        slots = input_batch.idx_mapping.long()
        active = SimpleNamespace(deleted=buffers.deleted[slots],padding=buffers.padding[slots],
            lookup=buffers.lookup,cutoffs=buffers.cutoffs,state_padding=buffers.padding,
            slots=slots,mapping=None)
        token = _ACTIVE_QSA.set(active)
        try:
            return original_attn(self,input_batch,*args,**kwargs)
        finally:
            _ACTIVE_QSA.reset(token)

    @functools.wraps(original_qsa)
    def qsa_build(self,common_prefix_len,common_attn_metadata,fast_build=False):
        active = _ACTIVE_QSA.get()
        if active is None:
            return original_qsa(self,common_prefix_len,common_attn_metadata,fast_build)
        if self.is_circular_buffer:
            kind = 'raw'
        elif self.compress_ratio == 4:
            kind = 'virtual'
        else:
            raise EvictionRunnerError('Unexpected QSA metadata owner in eviction control')
        original_lens = common_attn_metadata.seq_lens
        if getattr(active, 'graph_mapping', False):
            shifted_lens = (active.raw_seq_lens if kind == 'raw' else active.virtual_seq_lens)
        else:
            shift = active.deleted if kind == 'raw' else active.padding
            shifted_lens = original_lens + shift.to(original_lens.dtype)
        if original_lens.shape != shifted_lens.shape:
            raise EvictionRunnerError('QSA batch/state-slot dimensions disagree')
        shifted = replace(common_attn_metadata,seq_lens=shifted_lens)
        result = original_qsa(self,common_prefix_len,shifted,fast_build)
        if not self.is_circular_buffer:
            if active.mapping is None:
                active.mapping = QSASelectionMapping(
                    virtual_query_positions=result.logical_positions,
                    virtual_seq_lens=result.seq_lens,virtual_to_resident=active.lookup,
                    virtual_cutoffs=active.cutoffs,group_padding=active.state_padding,
                    request_to_state_slot=active.slots)
            result._arc3_qsa_selection_mapping = active.mapping
        return result

    GPUModelRunner.load_model = load_model
    GPUModelRunner.add_requests = add_requests
    GPUModelRunner._remove_request = remove_request
    Qwen3_8FlashNextModelState.add_request = add_request
    RopeState.init_prefill_positions = init_prefill_positions
    Qwen3_8FlashNextModelState.prepare_inputs = prepare_inputs
    Qwen3_8FlashNextModelState.prepare_attn = prepare_attn
    QSAMetadataBuilder.build = qsa_build
    WorkerBase.arc3_eviction_graph_status = arc3_eviction_graph_status
    install_qsa_selector_wrapper(mapping_from_forward_context, counters_provider=_qsa_frequency_counters)
    _INSTALLED = True
    return {'installed':True,'eager_only':not graph_mapping_enabled(),
            'graph_mapping':graph_mapping_enabled(), 'original_mrope':True,
            'stock_ple_protected_tail':True}
