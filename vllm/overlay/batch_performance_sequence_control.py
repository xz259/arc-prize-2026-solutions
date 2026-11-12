"""Fixed performance queue, gated on successful scheduling/scalar/runner-pin GPU proofs."""
import argparse
import hashlib
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
    assert here == Path('/workspace/arc3-inference/runs/persistent-efficiency-20260913-v6-batch-performance')
    proof = here.parent / 'persistent-efficiency-20260913-v5-batch-proof'
    sequence = json.loads((proof / 'proof-sequence.json').read_text())
    assert sequence['complete'] and sequence['stage'] == 'passed'
    for label in ('small-graph-slot-reuse-v1', 'graph-full-v1'):
        state = json.loads((proof / (label + '-controller.json')).read_text())
        assert state['complete'] and state['stage'] == 'passed' and not state['cleanup_errors']
        assert json.loads((proof / label / 'results.json').read_text())['status'] == 'passed'
    for concurrency in (25, 30):
        result = json.loads((proof / 'graph-full-v1' / f'concurrency-{concurrency}' / 'results.json').read_text())
        assert result['status'] == 'passed'
        for graph in result['graph_states'].values():
            publication = graph['scalar_publication']
            assert publication['enabled'] and publication['initialized']
            counts = publication['counts']
            assert counts['failed'] == 0 and counts['published'] == counts['calls']
            assert counts['issued_writes'] + counts['skipped_writes'] == 3 * counts['calls']
        final = list(result['graph_states'].values())[-1]['scalar_publication']['counts']
        assert final['issued_writes'] > 0 and final['skipped_writes'] > 0
    requested = json.loads((proof / 'requested-config.json').read_text())
    assert requested['environment']['VLLM_USE_V2_MODEL_RUNNER'] == '1'
    assert requested['environment']['ARC3_PERSISTENT_SINGLE_BATCH'] == '1'
    assert json.loads((proof / 'plugin-import-smoke.json').read_text())['batch_policy']['installed']
    # Serving source is byte-identical to the validated proof deployment.
    proof_manifest = json.loads((proof / 'overlay-manifest.json').read_text())
    for path in here.glob('*.py'):
        if path.name == Path(__file__).name:
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == proof_manifest[path.name], path.name
    state_path = here / 'performance-sequence.json'
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
        for job in ('bench-2-persistent-n25-v1-job.json', 'bench-3-persistent-api-n25-v1-job.json',
                    'bench-4-eviction-n25-v1-job.json'):
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
