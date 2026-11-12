"""One bounded quantized serving candidate against a completed BF16 reference.

Measures serving replay, fixed-target likelihood drift, and the preselected
public-game panel. These remain separate evidence; this does not promote a
format or modify gameplay defaults. The owned server remains available after
completion so capacity or diagnostic work can follow without another reload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import re
import time

from minimal_inference import ROOT, IDENTITY, matches, owned_snapshot, port_open, process, save


def verify_storage(snapshot, fmt):
    """Check resolved worker specs/allocation, not only the launch flag."""
    report = json.loads(snapshot.read_text())
    assert len(report['results']) == 1
    inventory = report['results'][0]['cache_inventory']
    content_bytes = 296 if fmt == 'nvfp4' else 520
    page_tokens = 5536 if fmt == 'nvfp4' else 3152
    expected_blocks = 181 if fmt == 'nvfp4' else 189
    expected_mode = 'KVQuantMode.NVFP4' if fmt == 'nvfp4' else 'KVQuantMode.FP8_PER_TOKEN_HEAD'
    main = []
    side = []
    for group in inventory['groups']:
        for name, spec in group['spec'].get('kv_cache_specs', {}).items():
            if name.endswith('.self_attn.attn'):
                assert spec.startswith('FullAttentionSpec(')
                assert 'dtype=torch.uint8,' in spec and expected_mode in spec
                assert f'state_content_bytes={content_bytes},' in spec
                assert f'block_size={page_tokens},' in spec
                main.append(name)
            else:
                assert 'dtype=torch.bfloat16,' in spec
                assert 'KVQuantMode.NONE' in spec
                side.append(name)
        if group['spec_type'] == 'MambaSpec':
            expected = ['torch.bfloat16'] if group['spec']['tp_replicated'] else ['torch.bfloat16', 'torch.float32']
            assert group['spec']['dtypes'] == expected
    assert len(main) == 13 and len(side) == 26
    assert inventory['num_blocks'] == expected_blocks
    block_bytes = 13 * (page_tokens * 2 * content_bytes + (page_tokens//4)*128*2)
    assert inventory['unique_storage_bytes'] == expected_blocks * block_bytes
    assert 0 < 8*2**30-inventory['unique_storage_bytes'] < block_bytes
    return dict(passed=True, format=fmt, main_layers=len(main), side_layers=len(side),
                page_tokens=page_tokens, num_blocks=expected_blocks,
                unique_storage_bytes=inventory['unique_storage_bytes'],
                scope='Resolved worker cache specs and unique physical allocation; TP1/MTP3/KV8/prefix off')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', required=True, choices=['fp8_e4m3', 'fp8_e5m2', 'nvfp4'])
    parser.add_argument('--baseline', type=Path, default=ROOT/'runs/radix-serving-graphs-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--from-stopped', action='store_true',
                        help='Resume after the owned server was stopped for an isolated GPU probe')
    parser.add_argument('--fp8-reader', choices=['baseline','scale_placement'], default='baseline')
    parser.add_argument('--tactic-reference', type=Path,
                        help='Require the loaded/saved MoE plans to match this preserved JSON')
    args = parser.parse_args()
    if args.fp8_reader != 'baseline' and (not args.format.startswith('fp8_') or not args.tactic_reference):
        parser.error('An FP8 reader candidate requires FP8 storage and a preserved tactic reference')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state = dict(stage='preflight', format=args.format, fp8_reader=args.fp8_reader, started_epoch=time.time(),
                 controller=process(os.getpid()), runs=[])

    def update(**fields):
        state.update(**fields)
        save(args.output/'state.json', state)
        print(json.dumps(state), flush=True)

    def command(label, argv, timeout, env=None):
        update(stage=label)
        with (args.output/(label+'.log')).open('w') as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        state['runs'].append(dict(name=label, argv=argv, returncode=result.returncode))
        update(stage=label)
        if result.returncode:
            raise RuntimeError(label+' failed; inspect preserved evidence')

    try:
        update()
        baseline = json.loads((args.baseline/'state.json').read_text())
        assert baseline['stage'] == 'complete' and not matches(baseline['controller'])
        likelihood = args.baseline/'multimodal-likelihood-v1'
        assert json.loads((likelihood/'result.json').read_text())['complete']
        panel = json.loads((args.baseline/'public-panel.json').read_text())
        assert panel == json.loads((ROOT/'quantization-public-panel-v1.json').read_text())
        assert panel['context'] == 32768 and panel['request_slots'] == 8
        save(args.output/'public-panel.json', panel)
        expected_tactics = None
        if args.tactic_reference:
            raw = args.tactic_reference.read_bytes()
            expected_tactics = json.loads(raw)
            (args.output/'moe-tactics-reference.json').write_bytes(raw)
        identity = json.loads(IDENTITY.read_text())
        assert identity['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if args.from_stopped:
            assert not port_open() and not any(matches(row) for row in owned_snapshot(identity))
        else:
            assert matches(identity['root'])
        command('serving', ['python3', '-B', 'run_config_candidate.py',
            *(['--from-stopped'] if args.from_stopped else []),
            '--batch', '8192', '--mtp', '3', '--selector', 'radix',
            '--topk-threads', '512', '--topk-cache-row', '1',
            '--kv-format', args.format,
            '--fp8-reader', args.fp8_reader,
            '--reference', str(args.baseline/'serving/benchmark/correctness.json'),
            '--output', str(args.output/'serving')], 4200)
        if expected_tactics is not None:
            log = (args.output/'serving/server.log').read_text()
            paths = re.findall(r'Using FlashInfer autotune cache file: (\S+)', log)
            assert len(set(paths)) == 1, paths
            tactic_path = Path(paths[0])
            assert f'configs from {tactic_path}' in log, 'Tactics were not loaded from the saved file'
            raw = tactic_path.read_bytes()
            assert json.loads(raw) == expected_tactics, 'MoE tactics differ from the paired reference'
            (args.output/'moe-tactics-used.json').write_bytes(raw)
            save(args.output/'moe-tactics-verification.json', dict(passed=True,
                 path=str(tactic_path), sha256=hashlib.sha256(raw).hexdigest(),
                 scope='Loaded cache file and post-startup plans match the preserved reference'))
        update(stage='verify_storage')
        save(args.output/'storage-verification.json', verify_storage(
            args.output/'serving/benchmark/forced-c8-r0/memory-after.json', args.format))
        command('multimodal_likelihood', ['venv-duck/bin/python', '-B', 'probe_multimodal_likelihood.py',
            '--panel', str(likelihood/'panel.json'),
            '--reference', str(likelihood/'result.json'),
            '--output', str(args.output/'multimodal-likelihood-v1')], 1200)
        env = os.environ.copy()
        env['LOCAL_ANALYZER_SEED'] = str(panel['sampler_seed'])
        env['ARC3_CONTEXT'] = str(panel['context'])
        command('public', ['venv-duck/bin/python', '-B', 'run_scheduled.py',
            '--games', ','.join(panel['games']), '--seed', str(panel['game_shuffle_seed']),
            '--token-cap', str(panel['token_cap_per_game']),
            '--request-slots', str(panel['request_slots']),
            '--total-seconds', str(panel['total_seconds']),
            '--shutdown-margin', str(panel['shutdown_margin']),
            '--output', str(args.output/'public'), '--execute'],
            panel['total_seconds']+180, env=env)
        update(stage='complete', finished_epoch=time.time())
    except Exception as exc:
        update(stage='failed', error=f'{type(exc).__name__}: {exc}', finished_epoch=time.time())
        raise


if __name__ == '__main__':
    main()
