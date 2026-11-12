"""L4 port v2 (2026-10-30, ARC2 - main; v1 recursed with l4port_ppbcast's finder): MTP drafts carried from the last PP stage to the others, env ARC3_L4PORT_PP_DRAFTBC=1.
MRv2 builds a verify step's input ids from req_states.draft_tokens on every rank, but only the last rank's speculator writes it, and
in the engine's batch-queue (PP) path the scheduler output carries placeholder spec tokens only, so stage 0 embedded zeros at every
draft position (4090 c22a tau 1.787, c23a 1.70 with the scheduler-copy attempt; TP=4 3.2-3.5). This adds a third broadcast to
PPHandler's dedicated group, after the sampled tokens and counts:
  - last rank: GPUModelRunner.sample_tokens is wrapped; after the stock call (sampling, the stock broadcasts, the speculator's
    propose writing req_states.draft_tokens) the batch's draft rows [num_reqs, K] are broadcast on the handler's side stream, under
    the same skip condition as the stock broadcast (compute_need_sampled_mask(input_batch) is None -> nothing sent);
  - other ranks: PPHandler.receive posts the matching receive right after the stock two (same stream, same group, same order) and
    keeps the tensor and its event in the step's slot; PPHandler.get_prev_sampled_outputs (called by update_pp_decode_requests at the
    start of execute_model, before prepare_inputs, pp_size steps later) writes the drafts into req_states.draft_tokens for the rows
    the stock code keeps (freed / no-sample rows excluded with its host-side masks; no device sync).
Draft numerics only: the target verifies every draft. PP 1 and K 0 are unchanged. Post-import hooks on vllm.v1.worker.gpu.pp_utils
and vllm.v1.worker.gpu.model_runner; one receipt line per process per side."""
import importlib
import importlib.abc
import os
import sys

PP = "vllm.v1.worker.gpu.pp_utils"
MR = "vllm.v1.worker.gpu.model_runner"
STATE = dict(sent=0, recv=0, applied=0, printed_s=False, printed_r=False, printed_a=False)


def _say(key, msg):
    if not STATE[key]:
        STATE[key] = True
        print("[l4port] PP drafts: " + msg, flush=True)


def _patch_pp(module):
    import numpy as np
    import torch
    cls = module.PPHandler
    if getattr(cls, 'l4_broadcast_drafts', None) is not None:
        return
    need = module.compute_need_sampled_mask
    orig_receive = cls.receive
    orig_get = cls.get_prev_sampled_outputs

    def receive(self, input_batch):
        out = orig_receive(self, input_batch)
        k = self.max_sample_len - 1
        slot = self.queue[-1] if self.queue else None
        if k > 0 and slot is not None:
            with torch.cuda.stream(self.broadcast_stream):
                drafts = torch.empty(input_batch.num_reqs, k, dtype=torch.int64, device=self.device)
                torch.distributed.broadcast(drafts, src=self.last_rank, group=self.broadcast_group)
                ev = self.broadcast_stream.record_event()
                drafts.record_stream(self.main_stream)
            slot.l4_drafts = drafts
            slot.l4_event = ev
            STATE["recv"] += 1
            _say("printed_r", "receiving [%d, %d] drafts after the sampled tokens" % (input_batch.num_reqs, k))
        return out

    def get_prev_sampled_outputs(self):
        slot = self.queue[0] if self.queue else None
        out = orig_get(self)
        drafts = getattr(slot, "l4_drafts", None) if slot is not None else None
        rs = getattr(self, "_l4_req_states", None)
        if drafts is not None and rs is not None:
            freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
            keep = ~(freed | ~slot.need_sampled_mask)
            if keep.any():
                rows = np.nonzero(keep)[0]
                self.main_stream.wait_event(slot.l4_event)
                r = torch.from_numpy(rows.astype(np.int64)).pin_memory().to(self.device, non_blocking=True)
                dst = torch.from_numpy(slot.idx_mapping_np[rows].astype(np.int64)).pin_memory().to(self.device, non_blocking=True)
                rs.draft_tokens.index_copy_(0, dst, drafts.index_select(0, r))
                STATE["applied"] += 1
                _say("printed_a", "drafts written into req_states.draft_tokens (first: %d rows)" % len(rows))
        return out

    def l4_broadcast_drafts(self, drafts, input_batch):
        if need(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            d = drafts.contiguous()
            torch.distributed.broadcast(d, src=self.last_rank, group=self.broadcast_group)
            d.record_stream(self.broadcast_stream)
        STATE["sent"] += 1
        _say("printed_s", "sending [%d, %d] drafts after propose" % tuple(d.shape))

    receive.__wrapped__ = orig_receive
    get_prev_sampled_outputs.__wrapped__ = orig_get
    cls.receive = receive
    cls.get_prev_sampled_outputs = get_prev_sampled_outputs
    cls.l4_broadcast_drafts = l4_broadcast_drafts


def _patch_mr(module):
    cls = module.GPUModelRunner
    if getattr(cls, '_l4_draftbc', False):
        return
    cls._l4_draftbc = True
    orig_sample = cls.sample_tokens
    orig_upd = cls.update_pp_decode_requests

    def update_pp_decode_requests(self):
        if self.pp_handler is not None and getattr(self.pp_handler, "_l4_req_states", None) is None:
            self.pp_handler._l4_req_states = self.req_states
        return orig_upd(self)

    def sample_tokens(self, grammar_output):
        st = self.execute_model_state
        ib = None
        if (st is not None and self.pp_handler is not None and self.is_last_pp_rank and self.speculator is not None
                and getattr(self, "num_speculative_steps", 0) > 0):
            ib = st.input_batch
        out = orig_sample(self, grammar_output)
        if ib is not None:
            self.pp_handler.l4_broadcast_drafts(self.req_states.draft_tokens[ib.idx_mapping], ib)
        return out

    update_pp_decode_requests.__wrapped__ = orig_upd
    sample_tokens.__wrapped__ = orig_sample
    cls.update_pp_decode_requests = update_pp_decode_requests
    cls.sample_tokens = sample_tokens


_BUSY = set()


class _Finder(importlib.abc.MetaPathFinder):
    # Only model_runner is hooked: it imports pp_utils at module level, so PPHandler is patched from here once model_runner has
    # executed. (v1 also hooked pp_utils, next to l4port_ppbcast's finder for the same module; each delegated to "every other
    # finder", so they recursed into each other: RecursionError on the 4090's c24a.) A re-entrancy guard covers other finders.
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
                        _patch_mr(module)
                        _patch_pp(importlib.import_module(PP))

                    spec.loader.exec_module = exec_module
                    return spec
            return None
        finally:
            _BUSY.discard(fullname)


def install():
    if os.environ.get("ARC3_L4PORT_PP_DRAFTBC") != "1":
        return
    if MR in sys.modules:
        _patch_mr(sys.modules[MR])
        _patch_pp(importlib.import_module(PP))
    else:
        sys.meta_path.insert(0, _Finder())
