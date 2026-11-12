"""Fixed scalar/runner-pin proof sequence after completed v2 measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        return
    here = args.candidate_dir.resolve()
    assert here == Path('/workspace/arc3-inference/runs/persistent-efficiency-20260913-v3-scalar-proof')
    previous = here.parent / 'persistent-efficiency-20260913-v2-performance'
    sequence = json.loads((previous / 'performance-sequence.json').read_text())
    assert sequence['complete'] and sequence['stage'] == 'passed'
    for label in ('bench-2-persistent-n25-v1', 'bench-3-persistent-api-n25-v1',
                  'bench-4-eviction-n25-v1', 'staggered-deferred-drain-v1'):
        state = json.loads((previous / (label + '-controller.json')).read_text())
        assert state['complete'] and state['stage'] == 'passed' and not state['cleanup_errors']
        assert json.loads((previous / label / 'results.json').read_text())['status'] == 'passed'
    assert json.loads((previous / 'staggered-deferred-drain-v1/results.json').read_text())['coverage_observed'] is True
    state_path = here / 'proof-sequence.json'
    assert not state_path.exists()
    state = dict(stage='startup', complete=False, started_epoch=time.time(), steps=[])

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)

    def run(command, label):
        state['stage'] = label
        save()
        started = time.time()
        with (here / (label + '-dispatch.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state['steps'].append(dict(label=label, started=started, ended=time.time(), returncode=result.returncode))
        save()
        if result.returncode:
            raise RuntimeError('Bounded controller failed: ' + label)

    save()
    try:
        run(['python3', str(here / 'startup_control.py'), '--candidate-dir', str(here), '--execute'], 'startup')
        for job in ('small-graph-slot-reuse-v1-job.json', 'graph-full-v1-job.json'):
            spec = json.loads((here / job).read_text())
            run(['python3', str(here / 'run_control.py'), '--candidate-dir', str(here),
                 '--job', job, '--execute'], spec['label'])
        state['stage'] = 'passed'
    except BaseException as exc:
        state.update(stage='failed', error=repr(exc))
    finally:
        state.update(complete=True, ended_epoch=time.time())
        save()
    if state['stage'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
