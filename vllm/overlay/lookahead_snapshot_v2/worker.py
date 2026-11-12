"""Bounded read-only values from real MRV2 caches, never allocator padding."""
import hashlib
import math
import struct
from . import receipt_schema as contract
from .guards import validate_config, validate_wire, no_staged_writes
from .layout import (validate_specs, spec_record, row_chunks, hash_chunks,
                     allocation_records, require_disjoint_ranges)
from .provenance import verify_sources

require = contract.require


def tensor_identity(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
        device=str(tensor.device), data_ptr=tensor.data_ptr(), storage_offset=tensor.storage_offset(),
        storage_bytes=tensor.untyped_storage().nbytes())


def check_tensor(tensor, torch, *, dtype, tail, device, main_nvfp4=False):
    require(type(tensor) is torch.Tensor and tensor.dtype == dtype and tensor.device == device,
            'Unsupported tensor type/dtype/device')
    require(tensor.is_cuda and list(tensor.shape[1:]) == list(tail) and tensor.shape[0] > 1,
            'Unexpected cache dimensions')
    stride, running = tensor.stride(), 1
    if main_nvfp4:
        # Pinned QSA backend inherits FlashAttention NHD: logical B,H,N,D
        # is stored B,N,H,D, then permuted back without copying.
        require(dtype == torch.uint8 and tuple(tail) == (2, 2768, 296),
                'Unknown main NVFP4 geometry')
        require(tuple(stride[1:]) == (296, 592, 1),
                'Unknown main NVFP4 NHD inner stride')
        running = 2*2768*296
    else:
        for dim in range(tensor.ndim-1, 0, -1):
            require(stride[dim] == running, 'Unknown inner cache stride')
            running *= tensor.shape[dim]
    require(stride[0] >= running, 'Overlapping cache pages')
    end = tensor.storage_offset() + (tensor.shape[0]-1)*stride[0] + running
    require(end*tensor.element_size() <= tensor.untyped_storage().nbytes(), 'View exceeds allocation')


def byte_copy(tensor, torch):
    require(0 < tensor.numel()*tensor.element_size() <= contract.MAX_CHUNK_BYTES,
            'Byte copy exceeds budget')
    # Reinterpret, never convert numeric values (including BF16 NaN/zero bits).
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes(order='C')


def cache_bindings(runner, wire, torch):
    specs = validate_specs(runner.kv_cache_config.kv_cache_groups)
    require(contract.same({n:spec_record(s) for n,s in sorted(specs.items())}, wire['core_specs']),
            'Worker physical specs disagree with preserved core specs')
    declared = runner.vllm_config.compilation_config.static_forward_context
    bound = runner.compilation_config.static_forward_context
    names = {n for n, layer in declared.items() if hasattr(layer, 'kv_cache')}
    require(names == set(specs), 'Bound cache inventory is incomplete or unknown')
    require({n for n, layer in bound.items() if hasattr(layer, 'kv_cache')} == names,
            'Runtime bound cache inventory differs')
    allocations = allocation_records(runner.kv_cache_config, specs)
    result, owned_ranges = {}, []
    for owner in contract.expected_owners():
        name, group_id = owner['owner'], owner['group_id']
        layer = declared[name]
        require(bound.get(name) is layer, 'Different runtime layer binding')
        cache = layer.kv_cache
        if group_id >= 2:
            require(type(cache) is tuple and len(cache) == len(specs[name].shapes), 'Unbound recurrent state')
            tensors = cache
            for value, shape in zip(cache, specs[name].shapes):
                check_tensor(value, torch, dtype=torch.bfloat16, tail=shape, device=runner.device)
            # MambaBase packs typed states at consecutive offsets in each page.
            offset = 0
            for value in cache:
                require(value.data_ptr() == cache[0].data_ptr()+offset, 'Recurrent components not packed as pinned')
                offset += math.prod(value.shape[1:])*value.element_size()
        else:
            tail = {0: (2, 2768, 296) if owner['categories'] == ['main_nvfp4'] else (692, 1, 128),
                    1: (4, 1, 140)}[group_id]
            dtype = torch.uint8 if owner['categories'] == ['main_nvfp4'] else torch.bfloat16
            check_tensor(cache, torch, dtype=dtype, tail=tail, device=runner.device,
                         main_nvfp4=owner['categories'] == ['main_nvfp4'])
            tensors = (cache,)
            if group_id == 1:
                require(layer.key_head_size == 128 and layer.cache_rope_positions is True
                        and layer.rope_position_offset == 128, 'Unknown QSA raw key/position split')
                require(tensor_identity(layer.key_cache) == tensor_identity(cache[..., :128]),
                        'Raw key view not bound to selected cache')
                require(tensor_identity(layer.rope_position_cache)
                        == tensor_identity(cache[..., 128:].view(torch.int64)), 'Raw position view differs')
        page = wire['block_ids'][group_id][0]
        allocation = allocations[name]
        offset = allocation['offset']
        for value in tensors:
            require(page < value.shape[0], 'Owned page outside allocation')
            require(value.shape[0] == allocation['num_blocks']
                    and value.untyped_storage().nbytes() == allocation['size']
                    and value.storage_offset()*value.element_size() == offset
                    and value.stride(0)*value.element_size() == allocation['stride'],
                    'Bound typed view differs from actual allocation descriptor')
            offset += math.prod(value.shape[1:])*value.element_size()
        start = tensors[0].data_ptr()+page*allocation['stride']
        owned_ranges.append((start, start+allocation['page'], name))
        result[name] = cache
    require_disjoint_ranges(owned_ranges)
    return result


def metadata(runner, wire, torch):
    require(runner.execute_model_state is None, 'Partially executed model output remains')
    states, model = runner.req_states, runner.model_state
    require(type(model).__name__ == 'Qwen3_8FlashNextModelState', 'Unknown model state')
    require(model._align_mode is False and model.mm_pruner is None, 'Alignment/MM pruning unsupported')
    require(model.uses_ngram_embedding is True and model.ngram_context_len == 2, 'Unknown PLE context')
    rope = model.rope_state
    require(rope is not None and rope.num_dims == 3 and rope.has_delta is True, 'Expected three-axis MRoPE')
    for buffer in (states.all_token_ids, states.total_len, states.num_computed_tokens,
                   rope.prefill_positions, *runner.block_tables.block_tables):
        no_staged_writes(buffer)
    slot = states.req_id_to_index[wire['request_id']]
    require(states.index_to_req_id.get(slot) == wire['request_id'], 'Nonreciprocal request slot')
    computed = int(states.num_computed_tokens.gpu[slot].item())
    total = int(states.total_len.gpu[slot].item())
    require(computed == wire['computed_tokens'] and total == len(wire['accepted_token_ids']),
            'Worker actual computed/accepted counters differ from core')
    ids = states.all_token_ids.gpu[slot, :total].cpu().tolist()
    require(contract.same(ids, wire['accepted_token_ids']), 'Worker accepted IDs differ from core')
    prompt = int(states.prompt_len.gpu[slot].item())
    prefill = int(states.prefill_len.gpu[slot].item())
    require(prompt == wire['original_prompt_tokens'] and 0 < prompt <= prefill <= computed,
            'Worker query boundary differs from core')
    require(int(states.prompt_len.np[slot]) == prompt and int(states.prefill_len.np[slot]) == prefill,
            'UVA prompt/prefill view is stale')
    delta = int(rope.prefill_delta.gpu[slot].item())
    require(delta == 0 and int(rope.prefill_delta.np[slot]) == 0, 'Text MRoPE delta changed')
    pos = rope.read_prefill_positions(slot, prefill)
    require(pos.dtype == torch.int32 and list(pos.shape) == [3, prefill], 'Unknown stored MRoPE layout')
    pos_bytes = byte_copy(pos, torch)
    expected_pos = b''.join(struct.pack('<i', p) for p in range(prefill))*3
    require(pos_bytes == expected_pos, 'Stored text MRoPE positions differ from logical prefix')
    logical = dict(original_prompt_tokens=prompt, prefill_len=prefill, mrope_axes=3, mrope_delta=delta,
        mrope_prefill_positions_sha256=hashlib.sha256(pos_bytes).hexdigest(), ple_context_len=2,
        ple_predecessor_ids_sha256=contract.token_digest(ids[computed-2:computed]))
    require(contract.same(logical, contract.expected_logical_metadata(ids[:computed], prompt, prefill)),
            'Unexpected logical metadata')
    tables = runner.block_tables
    require(tables.blocks_per_kv_block == [1]*6 and tables.cp_size == 1, 'Unsupported expanded block IDs')
    require(len(tables.block_tables) == 6 and tables.num_kv_cache_groups == 6, 'Unknown worker cache groups')
    actual_blocks = []
    for group, table in enumerate(tables.block_tables):
        n = int(tables.num_blocks.gpu[group, slot].item())
        require(n == len(wire['block_ids'][group]) and int(tables.num_blocks.np[group, slot]) == n,
                'Worker block count disagrees with core')
        ids_on_device = table.gpu[slot, :n].cpu().tolist()
        require(contract.same(ids_on_device, wire['block_ids'][group]), 'Worker persistent block table differs')
        actual_blocks.append(ids_on_device)
    return dict(slot=slot, computed=computed, total=total, ids=ids, logical=logical,
        blocks=actual_blocks, req_id_to_index=dict(states.req_id_to_index),
        index_to_req_id=dict(states.index_to_req_id),
        buffer_views=[tensor_identity(x.gpu) for x in
            (states.all_token_ids, states.total_len, states.num_computed_tokens,
             states.prompt_len, states.prefill_len, rope.prefill_positions, rope.prefill_delta,
             tables.num_blocks, *tables.block_tables)])


def component_chunks(cache, descriptor, blocks, computed, torch):
    category = descriptor['category']
    if category in ('gdn_conv', 'gdn_ssm', 'ple_short_conv'):
        value = cache[descriptor['coordinates']['index']][blocks[0]]
        require(value.is_contiguous(), 'Selected recurrent state not contiguous')
        flat = value.view(torch.uint8).view(-1)
        for start in range(0, flat.numel(), contract.MAX_CHUNK_BYTES):
            yield byte_copy(flat[start:start+contract.MAX_CHUNK_BYTES], torch)
        return
    if category == 'main_nvfp4':
        rows, size, storage = range(computed), 592, 2768
    elif category == 'qsa_compressed':
        rows, size, storage = range(computed//4), 256, 692
    else:
        rows, size, storage = range(computed-4, computed), (256 if category == 'qsa_raw_key' else 24), 4
    for page, offset, count in row_chunks(rows, size, storage, blocks, ring=storage == 4):
        if category == 'main_nvfp4':
            part = cache[page, :, offset:offset+count, :].movedim(1, 0)
        elif category == 'qsa_compressed':
            part = cache[page, offset:offset+count, 0, :]
        elif category == 'qsa_raw_key':
            part = cache[page, offset:offset+count, 0, :128]
        else:
            part = cache[page, offset:offset+count, 0, 128:].view(torch.int64)
        raw = byte_copy(part, torch)
        if category == 'qsa_raw_rope_position':
            # Validate the ring's interpretation independently of its byte digest.
            positions = [p for p in rows if p % 4 in range(offset, offset+count)]
            expected = b''.join(struct.pack('<qqq', p, p, p) for p in positions)
            require(raw == expected, 'Raw ring positions are not the computed text suffix')
        yield raw


def worker_snapshot(worker, wire):
    """Explicit fences and byte copies are intentionally outside all timing."""
    manifest, source_sha = verify_sources()
    computed = validate_wire(wire)
    require(wire['source_manifest_sha256'] == source_sha, 'Core/worker source manifests differ')
    runner = worker.model_runner
    validate_config(runner.vllm_config)
    import torch
    torch.cuda.synchronize(runner.device)
    before = metadata(runner, wire, torch)
    bindings = cache_bindings(runner, wire, torch)
    view_ids = {name: dict(layer_id=id(runner.compilation_config.static_forward_context[name]),
                    tensors=[tensor_identity(t) for t in (cache if type(cache) is tuple else (cache,))])
                for name,cache in bindings.items()}
    owners = {x['owner']: x for x in contract.expected_owners()}
    components = []
    for descriptor in contract.expected_components(computed):
        name = descriptor['owner']
        blocks = wire['block_ids'][owners[name]['group_id']]
        digest = hash_chunks(component_chunks(bindings[name], descriptor, blocks, computed, torch),
                             descriptor['bytes'])
        components.append(dict(descriptor, sha256=digest))
    torch.cuda.synchronize(runner.device)
    after = metadata(runner, wire, torch)
    after_bindings = cache_bindings(runner, wire, torch)
    after_views = {name: dict(layer_id=id(runner.compilation_config.static_forward_context[name]),
                      tensors=[tensor_identity(t) for t in (cache if type(cache) is tuple else (cache,))])
                   for name,cache in after_bindings.items()}
    require(contract.same(before, after) and contract.same(view_ids, after_views),
            'Frontier, ownership or view changed during readout')
    require(verify_sources()[1] == source_sha, 'Sources changed during readout')
    canonical = contract.canonical_state(before['ids'][:computed], components, before['logical'])
    result = dict(schema=contract.SCHEMA, scope=contract.SCOPE,
        **{flag: True for flag in contract.FLAGS}, source_manifest=manifest,
        source_manifest_sha256=source_sha, owner_inventory_sha256=contract.json_digest(contract.expected_owners()),
        coverage_manifest_sha256=contract.coverage_digest(computed), coverage=contract.expected_coverage(computed),
        components=components, canonical=canonical, comparison_digest=contract.json_digest(canonical),
        computed_tokens=computed, computed_token_ids_sha256=canonical['computed_token_ids_sha256'],
        logical_positions_sha256=canonical['logical_positions_sha256'],
        ownership=dict(request_id=wire['request_id'], slot=before['slot'], core_block_ids=wire['block_ids'],
            worker_block_ids=before['blocks'], block_tables_match=True, exclusive_pages=True,
            frontier_verified=True, core_proof=wire['core_proof'], views=view_ids,
            accepted_tokens=before['total'], sampled_only_tail=before['ids'][computed:]))
    contract.validate_receipt({'worker': result}, expected_ids=before['ids'][:computed],
        expected_source_manifest_sha256=source_sha, expected_prompt_tokens=before['logical']['original_prompt_tokens'],
        expected_prefill_len=before['logical']['prefill_len'])
    return result
