"""Offload-process side of the direct handoff: accept the flag registration and poll the flag instead of ZMQ.

Applied to the executed `vllm.v1.ple_offload.worker` module by the import hook. The flag layout is
int32[4] = [num_tokens, num_reqs, reserved, seq]; seq is written last by the D2H stream (a single 16-byte DMA
after the staging copies), so a new seq implies the staged inputs are complete. The reader uses a seqlock-style
double read so a torn 16-byte view is never acted on."""
import json
import os
import pickle
import time

RECEIPT = dict(version='arc3-ple-direct-handoff-v1', pid=os.getpid(), role='offload', handled=0, batches=0,
               skipped_seq=0, torn_reads=0, idle_polls=0, poll_sleep_us=None, written=None)


def write_receipt():
    d = os.environ.get('ARC3_PLE_DIRECT_RECEIPTS')
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        RECEIPT['written'] = time.time()
        tmp = os.path.join(d, f'offload-{os.getpid()}.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(RECEIPT, f, indent=1)
        os.replace(tmp, os.path.join(d, f'offload-{os.getpid()}.json'))
    except Exception:
        pass


def install(module, flag_registration_cls_name='PleDirectFlagRegistration'):
    """Patch PleOffloadRunner: __init__ (flag map), accept_registrations (second message), busy_loop (poll)."""
    Runner = module.PleOffloadRunner
    PleOffloadRequest = module.PleOffloadRequest
    original_init = Runner.__init__
    original_accept = Runner.accept_registrations

    def __init__(self, vllm_config):
        original_init(self, vllm_config)
        self._direct_flag_bufs = {}

    def accept_registrations(self, pull_socket, num_workers):
        original_accept(self, pull_socket, num_workers)
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        for _ in range(dp_size):
            item = pickle.loads(pull_socket.recv())
            if type(item).__name__ != flag_registration_cls_name:
                raise RuntimeError(f'Expected {flag_registration_cls_name}, got {type(item).__name__}')
            buf = item.request_flag_buf
            if buf.dtype != module.torch.int32 or buf.numel() != 4 or not buf.is_shared():
                raise RuntimeError('direct-handoff flag buffer must be a shared int32[4] CPU tensor')
            if item.dp_rank in self._direct_flag_bufs:
                raise RuntimeError(f'duplicate flag registration for dp_rank={item.dp_rank}')
            self._direct_flag_bufs[item.dp_rank] = buf
        if set(self._direct_flag_bufs) != set(range(dp_size)):
            raise RuntimeError(f'flag registrations {sorted(self._direct_flag_bufs)} != DP ranks {list(range(dp_size))}')
        module.logger.info('PleOffload(direct): flag buffers registered for dp_ranks=%s', sorted(self._direct_flag_bufs))

    def busy_loop(self, pull_socket, shutdown_event):
        """Poll the per-DP request flags; batch whatever is new; sleep briefly when idle."""
        poll_sleep = float(os.environ.get('ARC3_PLE_DIRECT_POLL_SLEEP_US', '50')) * 1e-6
        RECEIPT['poll_sleep_us'] = poll_sleep * 1e6
        views = {dp: buf.numpy() for dp, buf in self._direct_flag_bufs.items()}
        last = {dp: int(view[3]) for dp, view in views.items()}   # a flag written before the loop starts is stale
        module.logger.info('PleOffload(direct): busy-loop polling %d flag(s), sleep %.0f us', len(views), poll_sleep * 1e6)
        write_receipt()
        while not shutdown_event.is_set():
            requests = []
            for dp, view in views.items():
                seq = int(view[3])
                if seq == last[dp]:
                    continue
                num_tokens = int(view[0]); num_reqs = int(view[1]); seq2 = int(view[3])
                if seq2 != seq:                      # torn view: the next poll sees the settled record
                    RECEIPT['torn_reads'] += 1
                    continue
                if seq != last[dp] + 1:
                    RECEIPT['skipped_seq'] += 1      # only possible if this process fell a whole step behind
                last[dp] = seq
                requests.append(PleOffloadRequest(dp_rank=dp, num_tokens=num_tokens, num_reqs=num_reqs))
            if requests:
                self._handle_requests(requests)
                RECEIPT['handled'] += len(requests); RECEIPT['batches'] += 1
                if RECEIPT['batches'] % 2000 == 0:
                    write_receipt()
            else:
                RECEIPT['idle_polls'] += 1
                time.sleep(poll_sleep)
        write_receipt()

    Runner.__init__ = __init__
    Runner.accept_registrations = accept_registrations
    Runner.busy_loop = busy_loop
    Runner._arc3_ple_direct_handoff = True
    return Runner
