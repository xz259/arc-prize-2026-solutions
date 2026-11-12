"""L4 port lever 4 (2026-10-31, ARC2 - main; the user's lever order; design: l4port/INDEXER_DEDUPE_DESIGN_2026-10-31.md): the QSA
indexer's compressed keys split by block parity over the TP ranks of a stage. Env ARC3_L4PORT_IDX_SHARD=1 (unset: unchanged).

Measured first (4090 chain n7): the compressed-key spec at half its row bytes gives x1.0977 KV tokens at equal KV GiB (~+2.9 lanes on
the L4). Post-import hooks, no source edits:
  - common.qsa_cache.QSACompressedKeyCache.get_kv_cache_spec: the target model's indexers (language_model.model.layers.*.self_attn.
    indexer; not the drafter's) at TP = w > 1 report head_size / w, so vLLM allocates [pages, S, 1, 128 / w] per rank, viewed here as
    [pages, S / w, 1, 128]: rank r keeps the global blocks g with g % w == r at local row (g % S) // w of the same physical page;
  - nvidia.indexer_qsa.QSAIndexer.forward: the original code with (a) the compressed store into that view at local slots
    (idx_shard/merge.local_slots: the fused kernel takes its page size from the tensor and skips slots < 0), (b) the selection split:
    the unchanged scoring kernel on the view (visibility comes from positions only; each block's score is the same per-column dot
    product), the local visible prefix, the native local top-k, one all-gather of (score, global block) per layer, the exact merge
    (idx_shard/merge.merge, CPU-tested against exact_topk), the unchanged expansion;
  - engine_eviction.worker_compact and engine_eviction_async.worker_compact_async (production's mode is async): refuse (loudly)
    to compact a sharded compressed owner until that path is adapted (they would move half rows); promptmove's own shape check
    refuses a sharded cache.
  - VLLM_BUILD_URL gains '-idxshard' (vLLM's AOT cache key misses class-attribute patches; the PLE handoff's install v6 salts it too).
Verify mode ARC3_L4PORT_IDX_SHARD_VERIFY=N (N > 0; off by default): every N-th eager selection call also all-gathers the local logits,
rebuilds the full logits and runs exact_topk on them; mismatching rows are counted, and split into tie-only (the same selected score
multiset: the native op's tie order) and real. Receipts: one line per process, counts every 256 checks."""
import importlib.abc
import importlib.util
import os
import sys

ON = os.environ.get("ARC3_L4PORT_IDX_SHARD") == "1"
VERIFY_EVERY = int(os.environ.get("ARC3_L4PORT_IDX_SHARD_VERIFY", "0") or 0)
CACHE_MOD = "vllm.models.qwen3_8_flash_next.common.qsa_cache"
IDX_MOD = "vllm.models.qwen3_8_flash_next.nvidia.indexer_qsa"
EVICT_MOD = "engine_eviction"
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = dict(spec=0, forward_printed=False, calls=0, checks=0, rows=0, differ=0, tie_only=0, real=0)
_MERGE = None


def _merge():
    global _MERGE
    if _MERGE is None:
        spec = importlib.util.spec_from_file_location("l4port_idx_merge", os.path.join(HERE, "..", "idx_shard", "merge.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MERGE = mod
    return _MERGE


def _is_target(prefix):
    return prefix.startswith("language_model.model.layers.") and ".self_attn.indexer." in prefix and "mtp" not in prefix


def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def _say(msg):
    print("[l4port] idxshard: " + msg, flush=True)


def _patch_cache(module):
    cls = module.QSACompressedKeyCache
    orig = cls.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig(self, vllm_config)
        rank, world = _tp()
        if world == 1 or not _is_target(self.prefix):
            return spec
        rows = spec.block_size // spec.compress_ratio
        if spec.head_size % world or rows % world or spec.block_size % spec.compress_ratio:
            raise RuntimeError("idxshard: compressed cache rows %d / head %d not divisible by TP %d" % (rows, spec.head_size, world))
        STATE["spec"] += 1
        if STATE["spec"] == 1:
            _say("compressed-key spec head %d -> %d per rank (rank %d of %d; %d rows per page, %d per rank)" % (
                spec.head_size, spec.head_size // world, rank, world, rows, rows // world))
        return module.MLAAttentionSpec(block_size=spec.block_size, num_kv_heads=spec.num_kv_heads, head_size=spec.head_size // world,
                                       dtype=spec.dtype, compress_ratio=spec.compress_ratio)

    cls.get_kv_cache_spec = get_kv_cache_spec


def _half_view(cache, world):
    n, s, h, d = cache.shape
    if h != 1 or s % world:
        raise RuntimeError("idxshard: unexpected compressed cache shape %s" % (tuple(cache.shape),))
    return cache.view(n, s // world, 1, d * world)


def _sharded_select(q, half, page_table, token_to_req, query_positions, sequence_lengths, token_topk, compress_ratio, out, rank, world):
    """qsa_select_paged_tokens over this rank's half of the compressed blocks, merged over the TP group (see the module text)."""
    import torch
    import arc3_determinism
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.platforms import current_platform
    from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as ops
    mg = _merge()
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out
    columns = page_table.shape[1] * half.shape[1]          # local columns
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, ops._LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = torch.empty((chunk_rows, block_topk), dtype=torch.int32, device=q.device)
    topk_workspace = torch.empty((ops._TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device)
    eager = not torch.cuda.is_current_stream_capturing()
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        rs = slice(row_start, row_end)
        logits, visible = ops.qsa_mqa_paged(q[rs], half, page_table, token_to_req[rs], query_positions[rs], sequence_lengths,
                                            compress_ratio)
        local_len = mg.local_length(visible, rank, world).to(visible.dtype)
        blocks = blocks_buffer[: row_end - row_start]
        use_cooperative_topk = (blocks.shape[0] <= 32 and logits.stride(0) % 4 == 0 and current_platform.has_device_capability(90)
                                and not current_platform.is_device_capability_family(120))
        topk_op = torch.ops._C.cooperative_topk if use_cooperative_topk else torch.ops._C.persistent_topk
        arc3_determinism.select_topk(topk_op, logits, local_len, blocks, topk_workspace, block_topk, columns)
        scores, gidx = mg.local_candidates(logits, blocks, rank, world)
        packed = torch.cat([scores, gidx.to(torch.float32)], dim=1)          # block indices < 2^24 and INVALID = 2^30: exact
        gathered = tensor_model_parallel_all_gather(packed, dim=1).reshape(packed.shape[0], world, 2, block_topk)
        all_scores = gathered[:, :, 0, :].reshape(packed.shape[0], world * block_topk)
        all_gidx = gathered[:, :, 1, :].reshape(packed.shape[0], world * block_topk).to(torch.long)
        merged = mg.merge(all_scores, all_gidx, block_topk)
        if eager and VERIFY_EVERY > 0:
            STATE["calls"] += 1
            if STATE["calls"] % VERIFY_EVERY == 0:
                _verify(logits, visible, merged, block_topk, columns, rank, world)
        ops.expand_qsa_block_indices_cuda(merged, query_positions[rs], sequence_lengths, token_to_req[rs], compress_ratio, token_topk,
                                          out[rs])
    return out


def _verify(logits, visible, merged, k, columns, rank, world):
    import torch
    import arc3_determinism
    from vllm.distributed import tensor_model_parallel_all_gather
    full_local = tensor_model_parallel_all_gather(logits[:, :columns].contiguous(), dim=1).reshape(logits.shape[0], world, columns)
    full = full_local.permute(0, 2, 1).reshape(logits.shape[0], columns * world)   # global column g = local * world + rank
    ref = torch.empty_like(merged)
    arc3_determinism.exact_topk(full, visible, ref, k, columns * world)
    differ = (ref != merged).any(dim=1)
    n = int(differ.sum())
    STATE["checks"] += 1
    STATE["rows"] += int(logits.shape[0])
    if n:
        def picked(sel):
            v = torch.where(sel >= 0, full.gather(1, sel.clamp(min=0).long()), torch.full_like(sel, -float("inf"), dtype=full.dtype))
            return torch.sort(v, dim=1).values
        same = (picked(ref) == picked(merged)) | (torch.isinf(picked(ref)) & torch.isinf(picked(merged)))
        tie = (same.all(dim=1) & differ)
        STATE["differ"] += n
        STATE["tie_only"] += int(tie.sum())
        STATE["real"] += n - int(tie.sum())
    if STATE["checks"] % 256 == 1 or (n and STATE["real"]):
        _say("verify rank %d: checks %d rows %d differ %d (tie-only %d, real %d)" % (rank, STATE["checks"], STATE["rows"], STATE["differ"],
                                                                                     STATE["tie_only"], STATE["real"]))


def _patch_indexer(module):
    cls = module.QSAIndexer
    orig_forward = cls.forward

    def forward(self, hidden_states, positions, out=None):
        rank, world = _tp()
        if world == 1 or not _is_target(self.compressed_key_cache.prefix):
            return orig_forward(self, hidden_states, positions, out)
        metadata = self._metadata()
        if metadata is None:                        # dummy / profile forwards: the original's early return (no cache access)
            return orig_forward(self, hidden_states, positions, out)
        import torch
        from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_compress_groups_with_ratio, qsa_store_cache_rows
        mg = _merge()
        if not STATE["forward_printed"]:
            STATE["forward_printed"] = True
            _say("sharded indexer forward on rank %d of %d (%s)" % (rank, world, self.prefix))
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        hidden_states = hidden_states[:num_tokens]
        positions = positions[..., :num_tokens]
        projected_qk, _ = self.index_qk_proj(hidden_states)
        projected_q, raw_keys = projected_qk.split(
            (self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim), dim=-1)
        raw_key_state_cache = self.raw_key_cache
        half = _half_view(self.compressed_key_cache.kv_cache, world)
        local = mg.local_slots(compressed_metadata.slot_mapping, rank, world)
        if self.use_fused_pre_indexer:
            q = projected_q.new_empty(num_tokens, self.index_n_heads, self.index_head_dim)
            module.qsa_pre_indexer(
                projected_q, raw_keys, positions, self.rotary_emb.cos_sin_cache, self.q_layernorm.weight, self.k_layernorm.weight,
                self.q_layernorm.variance_epsilon, q, raw_key_state_cache.kv_cache, raw_metadata.slot_mapping,
                raw_metadata.block_table, raw_metadata.query_start_loc, raw_metadata.logical_positions,
                half, local, compressed_metadata.k_work_metadata,
                compress_ratio=self.compress_ratio, mrope_section=getattr(self.rotary_emb, "mrope_section", None),
                rope_pos_offset=(raw_key_state_cache.rope_position_offset
                                 if raw_key_state_cache.rope_position_cache is not None else None))
        else:
            from flashinfer.norm import gemma_rmsnorm
            q = projected_q.reshape(-1, self.index_n_heads, self.index_head_dim)
            q = gemma_rmsnorm(q.reshape(-1, self.index_head_dim), self.q_layernorm.weight,
                              self.q_layernorm.variance_epsilon).reshape_as(q)
            q = module.apply_qsa_rope(self.rotary_emb, positions, q)
            raw_key_cache = raw_key_state_cache.key_cache
            rope_position_cache = raw_key_state_cache.rope_position_cache
            if rope_position_cache is None:
                position_rows = raw_metadata.logical_positions.view(-1, 1, 1).expand(-1, 1, 3)
            else:
                position_rows = module.canonical_qsa_rope_positions(positions).to(device=raw_key_cache.device)
            pooled, first_positions = qsa_compress_groups_with_ratio(
                raw_keys.reshape(-1, 1, self.index_head_dim), position_rows, raw_key_cache, raw_metadata.block_table,
                raw_metadata.token_to_req, raw_metadata.query_start_loc, raw_metadata.logical_positions,
                compressed_metadata.slot_mapping, self.compress_ratio, rope_position_cache)
            compressed_keys = gemma_rmsnorm(pooled.reshape(-1, self.index_head_dim), self.k_layernorm.weight,
                                            self.k_layernorm.variance_epsilon).reshape(-1, 1, self.index_head_dim)
            if getattr(self.rotary_emb, "mrope_section", None):
                first_positions = first_positions.transpose(0, 1)
            else:
                first_positions = first_positions[:, 0]
            compressed_keys = module.apply_qsa_rope(self.rotary_emb, first_positions, compressed_keys)
            qsa_store_cache_rows(half, local, compressed_keys)
            qsa_store_cache_rows(raw_key_cache, raw_metadata.slot_mapping, raw_keys)
            if rope_position_cache is not None:
                qsa_store_cache_rows(rope_position_cache, raw_metadata.slot_mapping, position_rows)
        if self.skip_topk:
            if out is None:
                raise RuntimeError("QSA top-k reuse requires an output buffer")
            return out
        return _sharded_select(q, half, compressed_metadata.block_table, compressed_metadata.token_to_req,
                               compressed_metadata.logical_positions, compressed_metadata.seq_lens, self.token_topk,
                               self.compress_ratio, out, rank, world)

    cls.forward = forward


def _sharded_owner_in(wire):
    """The first target compressed owner named anywhere in an eviction wire (any nesting), or None."""
    stack = [wire]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            name = o.get("layer_name")
            if o.get("kind") == "compressed" and isinstance(name, str) and _is_target(name):
                return name
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return None


def _guard(module, fn_name, err_name):
    orig = getattr(module, fn_name)

    def guarded(worker, request_id, wire, *args, **kwargs):
        name = _sharded_owner_in(wire)
        if name is not None:
            err = getattr(module, err_name, RuntimeError)
            raise err("idxshard: compaction of a sharded compressed owner is not adapted yet: " + name)
        return orig(worker, request_id, wire, *args, **kwargs)

    setattr(module, fn_name, guarded)


def _patch_eviction(module):
    _guard(module, "worker_compact", "EngineEvictionError")


def _patch_eviction_async(module):
    for fn_name in ("worker_compact_async",):
        if hasattr(module, fn_name):
            _guard(module, fn_name, "EngineEvictionError")


_PATCHES = {CACHE_MOD: _patch_cache, IDX_MOD: _patch_indexer, EVICT_MOD: _patch_eviction,
            "engine_eviction_async": _patch_eviction_async}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHES:
            return None
        for finder in list(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec, _name=fullname):
                    _orig(module)
                    _PATCHES[_name](module)

                spec.loader.exec_module = exec_module
                return spec
        return None


def install():
    if not ON:
        return
    os.environ["VLLM_BUILD_URL"] = (os.environ.get("VLLM_BUILD_URL") or "arc3-l4port") + "-idxshard"
    pending = []
    for name, fn in _PATCHES.items():
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            pending.append(name)
    if pending:
        sys.meta_path.insert(0, _Finder())
