"""CPU-only, bounded HTTP/proc sampling. No CUDA context or collective RPC."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.request import urlopen


def parse_metrics(text):
    result = {}
    lines = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        name = fields[0].split('{', 1)[0]
        if not name.startswith('vllm:') or name.endswith('_bucket'):
            continue
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        lines.append(line)
        result[name] = result.get(name, 0.0) + value
    for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting',
                'vllm:generation_tokens_total', 'vllm:prompt_tokens_total'):
        if key not in result:
            raise ValueError('Missing required metric: '+key)
    return result, lines


def proc_sample(expected):
    p = Path('/proc') / str(expected['pid'])
    try:
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if int(fields[19]) != expected['start_ticks'] or fields[0] == 'Z':
            return None
        return dict(pid=expected['pid'], start_ticks=expected['start_ticks'],
                    cpu_ticks=int(fields[11])+int(fields[12]),
                    rss_pages=int(fields[21]), minor_faults=int(fields[7]),
                    major_faults=int(fields[9]))
    except (OSError, ValueError, IndexError):
        return None


class HTTPObserver:
    def __init__(self, path, identity, *, interval=.5, max_seconds=1500):
        self.path, self.identity = Path(path), identity
        self.interval, self.max_seconds = interval, max_seconds
        self.stop_event = threading.Event()
        self.process_lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.summary = dict(complete=False, samples=0, errors=0, successful_samples=0, interval=interval,
                            max_gap_seconds=0., source='loopback metrics and owned proc stats',
                            cpu_ticks_per_second=os.sysconf('SC_CLK_TCK'),page_bytes=os.sysconf('SC_PAGE_SIZE'),
                            cpu_coverage='server tree at run start plus harness root; excludes later sandbox children')

    def start(self):
        if self.path.exists():
            raise FileExistsError(self.path)
        self.thread.start()
        return self

    def track_processes(self, processes):
        with self.process_lock:
            old = self.identity.get('processes', [])
            self.identity['processes'] = list({(p['pid'],p['start_ticks']):p
                                               for p in old+processes}.values())

    def _loop(self):
        self.summary['started_epoch'] = time.time()
        deadline = time.monotonic()+self.max_seconds
        previous = None
        try:
            with self.path.open('x') as f:
                while not self.stop_event.is_set() and time.monotonic()<deadline:
                    mono = time.monotonic(); wall = time.time()
                    row = dict(wall_start=wall, mono_start=mono)
                    try:
                        with urlopen('http://127.0.0.1:1234/metrics', timeout=3) as response:
                            row['metrics'], row['raw_lines'] = parse_metrics(response.read().decode())
                        with self.process_lock:
                            processes = list(self.identity.get('processes', []))
                        row['processes'] = [v for p in processes
                                            if (v := proc_sample(p)) is not None]
                    except Exception as exc:
                        row['error'] = type(exc).__name__+': '+str(exc)
                        self.summary['errors'] += 1
                    row['mono_end'] = time.monotonic(); row['wall_end'] = time.time()
                    row['epoch'] = (wall+row['wall_end'])/2
                    row['rtt_seconds'] = row['mono_end']-mono
                    if 'error' not in row:
                        self.summary['successful_samples'] += 1
                        self.summary.setdefault('first_success_epoch',row['epoch'])
                        self.summary['last_success_epoch']=row['epoch']
                    if previous is not None:
                        self.summary['max_gap_seconds'] = max(self.summary['max_gap_seconds'],row['epoch']-previous)
                    previous = row['epoch']; self.summary['samples'] += 1
                    f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
                    self.stop_event.wait(max(0., self.interval-(time.monotonic()-mono)))
            self.summary['complete'] = True
        except Exception as exc:
            self.summary['fatal_error'] = repr(exc)
        finally:
            self.summary['ended_epoch'] = time.time()

    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.summary['complete'] = False
            self.summary['fatal_error'] = 'Observer thread did not stop'
        self.path.with_suffix('.summary.json').write_text(json.dumps(self.summary,indent=2)+'\n')
        return self.summary
