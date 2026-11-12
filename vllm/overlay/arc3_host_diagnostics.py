"""Optional worker RPCs; no instrumentation is installed by importing this file."""
from worker_diagnostics import WorkerDiagnostics as ExistingDiagnostics


class WorkerDiagnostics(ExistingDiagnostics):
    def arc3_dense_source_identity(self):
        import hashlib
        import os
        from pathlib import Path
        import sys

        linear = sys.modules['vllm.model_executor.layers.linear']
        source = Path(linear.__file__)
        runtime = sys.modules.get('dense_projection_adapter_v1.runtime')
        return dict(pid=os.getpid(), source=str(source),
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    apply_file=linear.UnquantizedLinearMethod.apply.__code__.co_filename,
                    post_file=linear.UnquantizedLinearMethod.process_weights_after_loading.__code__.co_filename,
                    tuning_report=None if runtime is None else dict(runtime.REPORT))

    def arc3_register_failure_trace(self):
        """Register a passive stack-dump signal; never schedule periodic signals."""
        import faulthandler
        import json
        import os
        from pathlib import Path
        import signal

        root = Path(__file__).resolve().parent
        assert root.parent == Path('/workspace/arc3-inference/runs')
        directory = root/'startup-trace'
        directory.mkdir(exist_ok=True)
        if hasattr(self, '_arc3_failure_trace_handle'):
            return self._arc3_failure_trace_receipt
        assert signal.getsignal(signal.SIGUSR2) == signal.SIG_DFL
        handle = (directory/f'worker-{os.getpid()}.stacks.txt').open('x')
        try:
            faulthandler.register(signal.SIGUSR2, file=handle, all_threads=True, chain=False)
        except BaseException:
            handle.close()
            raise
        self._arc3_failure_trace_handle = handle
        fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
        receipt = dict(pid=os.getpid(), start_ticks=int(fields[19]),
                       signal='SIGUSR2', registered=True, output=handle.name)
        self._arc3_failure_trace_receipt = receipt
        (directory/f'worker-{os.getpid()}.signal-registered.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt

    def arc3_ple_source_identity(self):
        import hashlib
        import inspect
        from pathlib import Path

        connector = self.model_runner._ple_offload_connector
        if connector is None:
            return dict(present=False)
        path = Path(inspect.getsourcefile(type(connector)))
        return dict(present=True, source=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    class_name=type(connector).__qualname__,
                    has_submission_ack=hasattr(connector, '_input_copy_submitted'))

    def arc3_host_profile_start(self, seconds=20, hz=50):
        from host_profile import start_for_worker
        try:
            return start_for_worker(self, seconds, hz)
        except Exception as exc:
            return dict(error=repr(exc))

    def arc3_host_profile_result(self):
        from host_profile import result_for_worker
        return result_for_worker(self)
