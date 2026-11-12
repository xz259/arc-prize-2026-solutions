"""Read-only, bounded device-memory sampling for one owned candidate process.

Runs one persistent nvidia-smi process. No CUDA context or vLLM RPC is created.
Driver-reported free memory, rather than total minus used, defines free margin.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], pgid=int(fields[2]), sid=int(fields[3]),
                    start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected):
    actual = identity(expected['pid'])
    return actual and actual['state'] != 'Z' and all(
        actual[k] == expected[k] for k in ('pid', 'pgid', 'sid', 'start_ticks'))


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def accumulate(stats, row):
    stats['samples'] = stats.get('samples', 0) + 1
    if not stats.get('peak_used') or row['used_mib'] > stats['peak_used']['used_mib']:
        stats['peak_used'] = row
    if not stats.get('minimum_free') or row['free_mib'] < stats['minimum_free']['free_mib']:
        stats['minimum_free'] = row
    for mib in (1024, 512):
        key = f'samples_below_{mib}_mib_free'
        stats[key] = stats.get(key, 0) + int(row['free_mib'] < mib)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interval-ms', type=int, default=200)
    p.add_argument('--max-seconds', type=int, default=7200)
    args = p.parse_args()
    assert 100 <= args.interval_ms <= 5000
    assert 1 <= args.max_seconds <= 14400
    state = json.loads(args.state.read_text())
    owner = state['controller']
    assert socket.gethostname() == state['hostname']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == state['boot_id']
    assert same_process(owner) and not state.get('complete') and state['stage'] != 'failed'
    deadline = min(state['campaign_deadline_epoch'] + 120, time.time() + args.max_seconds)
    assert deadline > time.time()
    args.output.mkdir(exist_ok=False)
    command = ['nvidia-smi', '--id=0',
        '--query-gpu=timestamp,uuid,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits', f'--loop-ms={args.interval_ms}']
    summary = dict(started_epoch=time.time(), complete=False, interval_ms=args.interval_ms,
        controller=owner, hostname=state['hostname'], boot_id=state['boot_id'],
        observer=identity(os.getpid()), deadline_epoch=deadline,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        state_path=str(args.state), command=command, overall={}, stages={},
        coverage_start_delay_seconds=time.time()-state['started_epoch'],
        max_sample_gap_seconds=0, parse_errors=0,
        note='Sampled device peaks; brief spikes and time before attachment may be missed. '
             'free_mib is driver-reported; total-used-free is unavailable/driver-reserved memory. '
             'Existing per-stage PyTorch allocator peaks are recorded separately by the candidate.')
    save(args.output/'observer-identity.json', summary)
    stop = []
    def on_signal(signum, frame):
        stop.append(f'signal_{signum}')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    child = None
    try:
        with (args.output/'samples.jsonl').open('x') as samples, (args.output/'nvidia-smi.stderr').open('x') as errors:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, bufsize=0)
            summary['sampler'] = identity(child.pid)
            sel = selectors.DefaultSelector()
            sel.register(child.stdout, selectors.EVENT_READ)
            buffer = b''
            last_sample = None
            next_refresh = 0
            stage = state['stage']
            save(args.output/'summary.json', summary)
            while not stop:
                now = time.time()
                if now >= next_refresh:
                    state = json.loads(args.state.read_text())
                    stage = state['stage']
                    summary.update(updated_epoch=now, current_stage=stage)
                    save(args.output/'summary.json', summary)
                    next_refresh = now + 1
                    if state.get('complete') or stage == 'failed':
                        stop.append(f'candidate_{stage}')
                    elif not same_process(owner):
                        stop.append('controller_exited')
                    elif now >= deadline:
                        stop.append('observer_deadline')
                    if stop:
                        break
                for key, _ in sel.select(timeout=.25):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        stop.append('sampler_exited')
                        break
                    buffer += chunk
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        try:
                            ts, uuid, total, used, free, util = next(csv.reader([line.decode()]))
                            row = dict(epoch=time.time(), stage=stage, driver_timestamp=ts.strip(), uuid=uuid.strip(),
                                total_mib=float(total), used_mib=float(used), free_mib=float(free), util_percent=float(util))
                            row['unavailable_mib'] = row['total_mib']-row['used_mib']-row['free_mib']
                        except (ValueError, UnicodeDecodeError):
                            summary['parse_errors'] += 1
                            continue
                        if last_sample is not None:
                            summary['max_sample_gap_seconds'] = max(summary['max_sample_gap_seconds'], row['epoch']-last_sample)
                        last_sample = row['epoch']
                        summary['latest'] = row
                        accumulate(summary['overall'], row)
                        accumulate(summary['stages'].setdefault(stage, {}), row)
                        samples.write(json.dumps(row) + '\n')
                    samples.flush()
            sel.close()
        summary.update(complete=True, stop_reason=stop[0])
    except BaseException as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        summary.update(finished_epoch=time.time(), sampler_returncode=child.returncode if child else None)
        save(args.output/'summary.json', summary)


if __name__ == '__main__':
    main()
