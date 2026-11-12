"""L4 port (2026-11-01, ARC2 - main; the Kaggle gap): the PP sampled-token receives posted when they are consumed, env
ARC3_L4PORT_PP_DEFER=1 (unset: unchanged).
Why (the Kaggle diagnostic ka2t11010339 against the matched pod profile pbp1): MRv2's PPHandler.receive posts the non-last stage's
receives of step T's sampled tokens and counts (and the port's drafts, l4port_ppdraftbc) on its side stream in sample_tokens(T); they
are consumed pp_size steps later. The receive kernel (NCCL LL broadcast) then spins on the GPU until the last stage samples step T,
i.e. through stage 0's next forward. On Kaggle (driver 580: the full decode graph's branches on many internal streams, 136 stream ids
in the trace against the pods' 6) one branch of that forward, layer 2's shared experts, waits behind the spinning receive's stream
every step (45-81 ms per replay, released 0.03 ms after the receive completes): stage 0 waits for stage 1's previous microbatch, the
two microbatches never overlap and the decode period is stage 0 + stage 1 (110 ms) instead of the slower stage (69 ms on the pods).
What: on non-last ranks, receive() keeps stock's bookkeeping (the need mask, the generation snapshot, the slot) and allocates the
tensors, but posts nothing; get_prev_sampled_outputs() (start of execute_model pp_size steps later) posts the slot's broadcasts, in
the original order (tokens, counts, then the drafts when l4port_ppdraftbc is on), on the current (main) stream, then filters and
applies them exactly as stock and l4port_ppdraftbc do. v2: the first receive that posts anything is posted at once through the
original path (stock + l4port_ppdraftbc, side stream, events): the broadcast group's NCCL communicator is created lazily at its first
collective, a host-side rendezvous of both stages, and v1 (6e6d3424, kept as l4port_ppdefer_6e6d3424.py) deferred it, so the last
stage blocked in its first broadcast and the engine hung (pbx3: DistBackendError on PP 1 at startup). In fact the first
IMMEDIATE_POSTS (256) posting receives go the original way: warmup runs PP broadcasts too, and while the receiver defers, the last
stage's small broadcasts queue in the connection's FIFO (NCCL_STEPS = 8 messages), so deferral starts only in steady serving, where
at most pp_size steps x 3 messages are outstanding. Slots carry their kind; get_prev handles both. The NCCL call order per rank is unchanged (the last rank's broadcasts are not
touched; the receives are posted in the same order, later): the last rank's small broadcasts complete into the connection's FIFO
(at most pp_size steps x 3 messages outstanding). The receive now runs after the previous forward on the main stream and right
before the forward that needs it, so no receive is pending while a forward runs. Same values, same rows. Post-import hook on
vllm.v1.worker.gpu.model_runner (PPHandler is patched once model_runner has executed, after l4port_ppdraftbc's patch); one receipt
line per process."""
import importlib
import importlib.abc
import os
import sys

PP = "vllm.v1.worker.gpu.pp_utils"
MR = "vllm.v1.worker.gpu.model_runner"
STATE = dict(installed=0, immediate=0, deferred=0, posted=0, printed=False)
IMMEDIATE_POSTS = int(os.environ.get("ARC3_L4PORT_PP_DEFER_AFTER", "256") or 256)
_BUSY = set()


def _patch_pp(module):
    import numpy as np
    import torch
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
    cls = module.PPHandler
    if getattr(cls, "_l4_ppdefer", False):
        return
    need_fn = module.compute_need_sampled_mask
    drafts_on = getattr(cls, "l4_broadcast_drafts", None) is not None       # l4port_ppdraftbc patched this class first
    if os.environ.get("ARC3_L4PORT_PP_DRAFTBC") == "1" and not drafts_on:
        raise RuntimeError("l4port_ppdefer: ARC3_L4PORT_PP_DRAFTBC=1 but PPHandler carries no draft broadcast (install order)")

    orig_receive = cls.receive                 # l4port_ppdraftbc's wrapper of stock's receive (or stock's)
    orig_get = cls.get_prev_sampled_outputs

    def receive(self, input_batch):
        assert not self.is_last_rank
        if getattr(self, "_l4_immediate_posts", 0) < IMMEDIATE_POSTS:
            # the original path: the first one pairs with the last stage's first broadcast, which creates the group's
            # communicator (a rendezvous); warmup's broadcasts stay paired at once; later receives are deferred
            out = orig_receive(self, input_batch)
            if self.queue and self.queue[-1] is not None:
                self._l4_immediate_posts = getattr(self, "_l4_immediate_posts", 0) + 1
                STATE["immediate"] = STATE.get("immediate", 0) + 1
            return out
        need = need_fn(input_batch)
        if need is None:
            return False                                                   # stock: this step's reserved slot stays None
        gen = self.req_idx_gen_np[input_batch.idx_mapping_np]
        n = input_batch.num_reqs
        sampled = torch.empty(n, self.max_sample_len, dtype=torch.int64, device=self.device)
        combined = torch.empty(2, n, dtype=torch.int32, device=self.device)
        k = self.max_sample_len - 1
        drafts = torch.empty(n, k, dtype=torch.int64, device=self.device) if (drafts_on and k > 0) else None
        num_sampled, num_rejected = combined.unbind(dim=0)
        slot = module.PendingRecv(None, sampled, num_sampled, num_rejected, input_batch.idx_mapping, input_batch.idx_mapping_np,
                                  need, gen)
        slot.l4_defer = (sampled, combined, drafts)
        self.queue[-1] = slot
        STATE["deferred"] += 1
        return bool(need.all())

    def get_prev_sampled_outputs(self):
        if not self.queue:
            return None
        head = self.queue[0]
        if head is None or getattr(head, "l4_defer", None) is None:
            return orig_get(self)              # a None slot or an immediate (stock) one: the original path
        slot = self.queue.popleft()
        self.queue.append(None)
        sampled, combined, drafts = slot.l4_defer
        # the receives, in stock's order, now: on the current stream, right before the forward that consumes them
        torch.distributed.broadcast(sampled, src=self.last_rank, group=self.broadcast_group)
        torch.distributed.broadcast(combined, src=self.last_rank, group=self.broadcast_group)
        if drafts is not None:
            torch.distributed.broadcast(drafts, src=self.last_rank, group=self.broadcast_group)
        STATE["posted"] += 1
        if not STATE["printed"]:
            STATE["printed"] = True
            print("[l4port] ppdefer: PP receives posted at the consuming step from now on (after %d immediate; first: %d rows, drafts %s, "
                  "pid %d)" % (STATE["immediate"], sampled.shape[0], "none" if drafts is None else drafts.shape[1], os.getpid()), flush=True)
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude = freed | ~slot.need_sampled_mask
        rs = getattr(self, "_l4_req_states", None)
        if drafts is not None and rs is not None and (~exclude).any():  # l4port_ppdraftbc's write of the kept rows' drafts
            rows = np.nonzero(~exclude)[0]
            r = torch.from_numpy(rows.astype(np.int64)).pin_memory().to(self.device, non_blocking=True)
            dst = torch.from_numpy(slot.idx_mapping_np[rows].astype(np.int64)).pin_memory().to(self.device, non_blocking=True)
            rs.draft_tokens.index_copy_(0, dst, drafts.index_select(0, r))
        idx_mapping = slot.idx_mapping
        if exclude.any():                                                  # stock's filter
            if exclude.all():
                return None
            idx_mapping = async_copy_to_gpu(np.where(exclude, -1, slot.idx_mapping_np), device=self.device)
        return dict(sampled_tokens=slot.sampled_tokens, num_sampled=slot.num_sampled, num_rejected=slot.num_rejected,
                    idx_mapping=idx_mapping)

    receive.__wrapped__ = orig_receive
    get_prev_sampled_outputs.__wrapped__ = orig_get
    cls.receive = receive
    cls.get_prev_sampled_outputs = get_prev_sampled_outputs
    cls._l4_ppdefer = True
    STATE["installed"] += 1
    print("[l4port] ppdefer: installed (drafts %s, pid %d)" % ("on" if drafts_on else "off", os.getpid()), flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MR or fullname in _BUSY:
            return None
        _BUSY.add(fullname)
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def exec_module(module, _orig=orig_exec):
                        _orig(module)
                        _patch_pp(importlib.import_module(PP))

                    spec.loader.exec_module = exec_module
                    return spec
            return None
        finally:
            _BUSY.discard(fullname)


def install():
    if os.environ.get("ARC3_L4PORT_PP_DEFER") != "1":
        return
    if MR in sys.modules:
        _patch_pp(importlib.import_module(PP))
    else:
        sys.meta_path.insert(0, _Finder())
