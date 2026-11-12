"""Derive ple_hostpoll_tp/connector.py from production's ple_hostpoll_v1/connector.py (md5-pinned) by exact replacements.

L4 port (2026-10-31, ARC2 - main; the user: "Let's do it", the PLE handoff without the offload process's CUDA contexts, for
the 30th lane). Production's host-poll handoff (TP = 1 on the RTX) has two gaps on TP2 x PP2:
  (a) every worker's first registration (PleOffloadRegistration) still carries CUDA tensors (the GPU output buffers, dummy
      semaphores), and the offload side's base accept opens them over CUDA IPC with a copy stream per GPU: a CUDA context
      on each GPU of the PLE stage (238 / 206 MiB on the L4's GPUs 0-1);
  (b) only TP rank 0 registers shared result buffers, pins them and advances the per-step seq, so TP rank 1's placeholder
      would poll an unpinned flag that never advances.
Changes here (worker side; the offload side is worker_patch_tp.py):
  1. the first registration carries no CUDA tensors;
  2. TP rank 0 broadcasts its host-poll registration payload (the shared result buffers and flags) over the TP group; every
     TP rank polls the same flag and copies the same rows, so the offload process writes each step once. Safe to reuse the
     rows: the next request is published by TP rank 0 only after its forward completes on the GPU, which needs every TP
     rank's collectives of that forward, each after that rank's H2D copy on its model stream;
  3. TP ranks != 0 page-lock the shared rows / map the shared flags in their own process and advance their own seq and
     expected_dev in lockstep (every TP rank runs prepare_forward for the same forwards);
  4. opt-in ARC3_L4PORT_PLE_HP_VERIFY=1 (a test mode, off by default): a shared int64[2] aux per layer = [seq, checksum of
     the rows' int16 view], filled by the offload before the flag; after each real forward every rank compares the
     checksum of its GPU copy (a skipped count when the aux already holds a later seq), raises on a mismatch, and writes
     counts to ARC3_PLE_HOSTPOLL_RECEIPTS/verify-PID.json.
TP = 1 is unchanged in behaviour (the broadcast returns at once; with the verify mode off nothing else differs but the
empty CUDA fields of the first registration, which the TP-aware offload side requires).
Usage: python3 derive_tp_connector.py PRODUCTION_CONNECTOR OUT_CONNECTOR"""
import hashlib
import os
import sys

SRC_MD5 = "de0a97a577822799bdccb2800407d18a"   # /workspace/arc3-inference/runs/dense_fp8_v1/ple_hostpoll_v1/connector.py

src, out = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
assert hashlib.md5(raw).hexdigest() == SRC_MD5, ("production connector changed", hashlib.md5(raw).hexdigest())
t = raw.decode()


def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:100], t.count(old))
    t = t.replace(old, new)


rep('arc3 ple_hostpoll_v1 (2026-09-20): direct staging + host-poll results.',
    'arc3 ple_hostpoll_tp (2026-10-31, L4 port, ARC2 - main): ple_hostpoll_v1 for TP > 1, derived by\n'
    'l4port/ple_hostpoll_tp/derive_tp_connector.py (see there): no CUDA tensors in the first registration; every TP rank\n'
    'polls TP rank 0\'s shared result buffers; an opt-in verify mode (ARC3_L4PORT_PLE_HP_VERIFY=1).\n\n'
    'arc3 ple_hostpoll_v1 (2026-09-20): direct staging + host-poll results.')

rep('''    result_flags: dict[str, torch.Tensor]          # layer -> shared pinned int32[1] = seq of the rows present
''', '''    result_flags: dict[str, torch.Tensor]          # layer -> shared pinned int32[1] = seq of the rows present
    aux_bufs: dict[str, torch.Tensor] | None = None   # L4 port verify mode: layer -> shared int64[2] = [seq, checksum]
''')

rep('''        self._timed_out_host: torch.Tensor | None = None
        self._layers = self._setup_layers(vllm_config, model)
''', '''        self._timed_out_host: torch.Tensor | None = None
        # L4 port (TP > 1): the verify mode's shared aux buffers and device counters [checked, mismatched, skipped]
        self._hp_verify = os.environ.get("ARC3_L4PORT_PLE_HP_VERIFY", "0") == "1"
        self._aux_bufs: dict[str, torch.Tensor] = {}
        self._aux_dev: dict[str, torch.Tensor] = {}
        self._v_counts: torch.Tensor | None = None
        self._v_host: torch.Tensor | None = None
        self._v_last: tuple[int, int] | None = None   # (num_tokens, seq) of the last real forward
        self._v_steps = 0
        self._layers = self._setup_layers(vllm_config, model)
        if self._hp_verify:
            for name in self._layers:
                if self.tp_rank == 0:
                    self._aux_bufs[name] = torch.zeros(2, dtype=torch.int64, device="cpu").share_memory_()
                self._aux_dev[name] = torch.zeros(2, dtype=torch.int64, device=self.device)
            self._v_counts = torch.zeros(3, dtype=torch.int64, device=self.device)
            self._v_host = torch.zeros(3, dtype=torch.int64, device="cpu", pin_memory=True)
''')

rep('''                self._register_flag_with_offload_worker()
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
''', '''                payload = self._register_flag_with_offload_worker()
                # L4 port (TP > 1): the other TP ranks of this group poll the same flags and copy the same rows.
                # bytes: ForkingPickler.dumps returns a memoryview (ZMQ takes it; the group's object broadcast pickles it)
                get_tp_group().broadcast_object(bytes(payload), src=0)
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
''')

rep('''                        self._flag_src = torch.zeros(2, 4, dtype=torch.int32, device="cpu", pin_memory=True)
        except Exception:
''', '''                        self._flag_src = torch.zeros(2, 4, dtype=torch.int32, device="cpu", pin_memory=True)
            else:
                self._adopt_shared_results(get_tp_group().broadcast_object(None, src=0))
        except Exception:
''')

rep('''        buffers.extend(self._result_bufs.values())
        mapped = list(self._result_flags.values())   # polled by a kernel through their device pointers
''', '''        buffers.extend(self._result_bufs.values())
        buffers.extend(self._aux_bufs.values())   # L4 port verify mode (async H2D after each real forward)
        mapped = list(self._result_flags.values())   # polled by a kernel through their device pointers
''')

rep('''            gpu_output_buffers={
                name: layer._gpu_output_buffer for name, layer in self._layers.items()
            },
            sem_flag_tensors=dict(self._dummy_sem_flags),
''', '''            # L4 port: no CUDA tensors (the host-poll offload never writes GPU memory; CUDA IPC handles would give the
            # offload process a context on every GPU of the PLE stage)
            gpu_output_buffers={},
            sem_flag_tensors={},
''')

rep('''                                               result_bufs=dict(self._result_bufs), result_flags=dict(self._result_flags))
''', '''                                               result_bufs=dict(self._result_bufs), result_flags=dict(self._result_flags),
                                               aux_bufs=dict(self._aux_bufs) if self._hp_verify else None)
''')

rep('''        logger.info("PleOffload(hostpoll): registered request flag + %d result buffer(s) for dp_rank=%d", len(self._result_bufs), self.dp_rank)
''', '''        logger.info("PleOffload(hostpoll): registered request flag + %d result buffer(s) for dp_rank=%d", len(self._result_bufs), self.dp_rank)
        return payload

    def _adopt_shared_results(self, payload: bytes) -> None:
        """L4 port (TP rank != 0): poll TP rank 0's shared result buffers / flags (and aux) instead of this rank's own."""
        registration = ForkingPickler.loads(payload)
        if type(registration).__name__ != "PleHostpollRegistration" or registration.dp_rank != self.dp_rank:
            raise RuntimeError("PLE host-poll TP: unexpected payload from TP rank 0")
        if set(registration.result_bufs) != set(self._layers) or set(registration.result_flags) != set(self._layers):
            raise RuntimeError("PLE host-poll TP: TP rank 0's layers differ from this rank's")
        for name, layer in self._layers.items():
            result_buf, result_flag = registration.result_bufs[name], registration.result_flags[name]
            mine = self._result_bufs[name]
            if result_buf.shape != mine.shape or result_buf.dtype != mine.dtype or not result_buf.is_shared():
                raise RuntimeError(f"PLE host-poll TP: result buffer {name} differs from TP rank 0's")
            self._result_bufs[name] = result_buf
            self._result_flags[name] = result_flag
            layer.setup_hostpoll_offload(layer._gpu_output_buffer, result_buf, result_flag, self._expected_dev[name],
                                         self._timed_out_dev[name])
        if self._hp_verify:
            if not registration.aux_bufs or set(registration.aux_bufs) != set(self._layers):
                raise RuntimeError("PLE host-poll TP: verify mode on, but TP rank 0 sent no aux buffers")
            self._aux_bufs = dict(registration.aux_bufs)
        # Page-lock the shared rows (async H2D) and map the shared flags (polled through their device pointers) in this
        # process too; close() unregisters them with the rest.
        mapped = list(self._result_flags.values())
        with torch.accelerator.device_index(self.device.index):
            for buffer in list(self._result_bufs.values()) + list(self._aux_bufs.values()) + mapped:
                flags = cuda_driver.CU_MEMHOSTREGISTER_PORTABLE
                if any(buffer is m for m in mapped):
                    flags = flags | cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP
                _cuda_check(
                    cuda_driver.cuMemHostRegister(buffer.data_ptr(), buffer.numel() * buffer.element_size(), flags),
                    "cuMemHostRegister(PLE shared result)",
                )
                self._pinned_input_buffers.append(buffer)
                if not buffer.is_pinned():
                    raise RuntimeError("CUDA did not page-lock a shared PLE result buffer")
        logger.info("PleOffload(hostpoll-tp): TP rank %d polls TP rank 0's %d shared result buffer(s) (dp_rank=%d)",
                    self.tp_rank, len(self._result_bufs), self.dp_rank)
''')

rep('''        if self.tp_rank != 0:
            return

        if self._uses_cuda_inputs:
''', '''        if self.tp_rank != 0:
            # L4 port (TP > 1): keep this rank's seq in lockstep with TP rank 0's (every TP rank runs prepare_forward for
            # the same forwards), so its placeholders poll the shared flag for this step's rows; rank 0 alone stages.
            self._request_seq += 1
            for expected in self._expected_dev.values():
                expected.fill_(self._request_seq)
            self._v_last = (num_tokens, self._request_seq)
            return

        if self._uses_cuda_inputs:
''')

rep('''        for expected in self._expected_dev.values():
            expected.fill_(request.seq)

    def prepare_forward(
''', '''        for expected in self._expected_dev.values():
            expected.fill_(request.seq)
        self._v_last = (num_tokens, request.seq)

    def prepare_forward(
''')

rep('''        for name, layer in self._layers.items():
            layer._gpu_output_buffer[:num_tokens].zero_()
''', '''        self._v_last = None   # L4 port verify mode: nothing to check after a dummy forward
        for name, layer in self._layers.items():
            layer._gpu_output_buffer[:num_tokens].zero_()
''')

rep('''        for index, timed_out in enumerate(self._timed_out_dev.values()):
            self._timed_out_host[index : index + 1].copy_(timed_out, non_blocking=True)
''', '''        for index, timed_out in enumerate(self._timed_out_dev.values()):
            self._timed_out_host[index : index + 1].copy_(timed_out, non_blocking=True)
        if self._hp_verify:
            self._verify_last()

    def _verify_last(self) -> None:
        """L4 port verify mode: compare each layer's GPU rows of the last real forward with the offload's checksum."""
        assert self._v_counts is not None and self._v_host is not None
        if self._v_last is not None:
            num_tokens, seq = self._v_last
            for name, layer in self._layers.items():
                aux = self._aux_dev[name]
                aux.copy_(self._aux_bufs[name], non_blocking=True)
                checksum = layer._gpu_output_buffer[:num_tokens].view(torch.int16).sum(dtype=torch.int64)
                same_seq = aux[0] == seq
                self._v_counts[0] += same_seq.to(torch.int64)
                self._v_counts[1] += (same_seq & (aux[1] != checksum)).to(torch.int64)
                self._v_counts[2] += (~same_seq).to(torch.int64)
            self._v_last = None
        self._v_steps += 1
        if self._v_steps % 256 == 0:
            # The mirror copied 256 steps ago has landed (the model thread never runs that far ahead of the GPU).
            if int(self._v_host[1]) != 0:
                self._write_verify_receipt()
                raise RuntimeError("PLE host-poll TP verify: %d mismatched forward(s); see ple-hostpoll receipts" % int(self._v_host[1]))
            if self._v_steps % 4096 == 0:
                self._write_verify_receipt()
            self._v_host.copy_(self._v_counts, non_blocking=True)

    def _write_verify_receipt(self) -> None:
        """L4 port verify mode: counts so far into ARC3_PLE_HOSTPOLL_RECEIPTS/verify-PID.json."""
        root = os.environ.get("ARC3_PLE_HOSTPOLL_RECEIPTS")
        if not root or self._v_host is None:
            return
        try:
            import json
            import time
            os.makedirs(root, exist_ok=True)
            record = dict(version="arc3-ple-hostpoll-tp", role="gpu-verify", pid=os.getpid(), tp_rank=self.tp_rank,
                          dp_rank=self.dp_rank, forwards=self._v_steps, checked=int(self._v_host[0]),
                          mismatched=int(self._v_host[1]), skipped=int(self._v_host[2]), epoch=time.time())
            tmp = os.path.join(root, "verify-%d.json.tmp" % os.getpid())
            with open(tmp, "w") as f:
                json.dump(record, f)
            os.replace(tmp, os.path.join(root, "verify-%d.json" % os.getpid()))
        except Exception:
            logger.exception("PLE host-poll TP: verify receipt failed")
''')

rep('''    def close(self) -> None:
        """Stop request transport and release host registrations."""
        if self._d2h_stream is not None:
''', '''    def close(self) -> None:
        """Stop request transport and release host registrations."""
        if self._hp_verify and self._v_host is not None:
            self._write_verify_receipt()
        if self._d2h_stream is not None:
''')

tmp = out + ".tmp"
with open(tmp, "w") as f:
    f.write(t)
os.replace(tmp, out)
print("wrote %s md5 %s" % (out, hashlib.md5(t.encode()).hexdigest()))
