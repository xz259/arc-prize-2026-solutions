"""Low-overhead wall-clock, vLLM, GPU and owned-process samples for bounded runs."""
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import Request, urlopen


class Monitor:
    def __init__(self, output, base_url, identity_path):
        self.output, self.base_url = Path(output), base_url
        self.identity_path = Path(identity_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def metrics(self):
        with urlopen(self.base_url + "/metrics", timeout=5) as r:
            return r.read().decode()

    def start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output/"metrics-before.txt").write_text(self.metrics())
        self._memory_snapshot("memory-before.json", reset_peaks=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=20)
        try:
            (self.output/"metrics-after.txt").write_text(self.metrics())
        except Exception as exc:
            (self.output/"metrics-after-error.json").write_text(json.dumps({"error":str(exc)})+"\n")
        self._memory_snapshot("memory-after.json")

    def _memory_snapshot(self, name, reset_peaks=False):
        identity = json.loads(self.identity_path.read_text())
        if '--worker-extension-cls' not in identity.get('argv', []):
            return
        try:
            def rpc(method):
                request = Request(self.base_url+'/collective_rpc',
                                  data=json.dumps({'method':method, 'timeout':15}).encode(),
                                  headers={'Content-Type':'application/json'})
                with urlopen(request, timeout=20) as response:
                    return json.load(response)
            if reset_peaks:
                rpc('arc3_reset_memory_peaks')
            data = rpc('arc3_memory_snapshot')
        except Exception as exc:
            data = {'error':str(exc)}
        (self.output/name).write_text(json.dumps(data, indent=2)+'\n')

    def _loop(self):
        with (self.output/"telemetry.jsonl").open("a") as f:
            while not self.stop_event.is_set():
                row = {"epoch": time.time()}
                try:
                    gpu = subprocess.check_output([
                        "nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu",
                        "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                    row["gpu"] = dict(zip(["util_percent", "memory_util_percent", "used_mib", "power_w", "sm_mhz", "temperature_c"],
                                          map(float, gpu.split(","))))
                    metrics = self.metrics()
                    wanted = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
                              "num_preemptions", "prompt_tokens_total", "generation_tokens_total",
                              "prefix_cache", "spec_decode", "time_to_first_token_seconds_sum",
                              "time_per_output_token_seconds_sum", "request_prefill_time_seconds_sum",
                              "request_decode_time_seconds_sum", "request_queue_time_seconds_sum")
                    row["metrics"] = [line for line in metrics.splitlines()
                                      if not line.startswith("#") and any(k in line for k in wanted)]
                    identity = json.loads(self.identity_path.read_text())
                    row["processes"] = []
                    for p in identity.get("processes", []):
                        root = Path(f'/proc/{p["pid"]}')
                        try:
                            stat = (root/"stat").read_text().rsplit(")", 1)[1].split()
                            if int(stat[19]) != p["start_ticks"]:
                                continue
                            row["processes"].append(dict(pid=p["pid"], cpu_ticks=int(stat[11])+int(stat[12]),
                                                         rss_pages=int(stat[21]), minor_faults=int(stat[7]),
                                                         major_faults=int(stat[9]),
                                                         command=(root/"comm").read_text().strip()))
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                    cgroup = Path("/sys/fs/cgroup/cpu.stat")
                    if cgroup.exists():
                        row["cgroup_cpu"] = cgroup.read_text()
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.stop_event.wait(2)
