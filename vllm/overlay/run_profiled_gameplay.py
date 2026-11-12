"""Explicit bounded gameplay profiling contract; dry plan unless --execute.

The reviewed PRO-LONG runner owns preparation, game scheduling, transcripts,
deadline, scoring and cleanup. This wrapper adds private event-only profiling
and the persistent adapter through locally bound imports, with source pins.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace
from urllib.parse import urlsplit

import run_persistent_gameplay as inherited

VERSION = 'arc3-profiled-gameplay-v1'
HARNESS_MANIFEST_SHA256 = '95b9edb2ce3f413999f4cfa84f6fb4c9039c95f9dd552ad321215cbfe5e00f7a'
LOCAL_PINS = {
    'persistent_harness_adapter.py': 'b5ddd82192e50816fc2766303a0009958706c7c960ea11f0ff09f226e304b864',
    'append_only_history.py': '97cc3ce58debce44a7db54b64f8243dac9d869fd284ca67c635c980de52f855b',
    'managed_suffix_history.py': '961e7943958c910242d30659daea7cfe456d6ae927b550e655090e64ac288471',
    'run_persistent_gameplay.py': 'ddd28eae111f5d1d433e2d90224ed665c826aff2a348e86aa4a4fe7a2f175b89',
}
OPTION = inherited._option


def validate_contract(argv, contract):
    """A new explicit workload contract, not a bypass of old <=4/600 checks."""
    if contract not in ('public25', 'smoke'):
        raise ValueError('Select public25 or smoke profiling contract')
    values = {
        '--total-seconds': '900' if contract == 'public25' else '150',
        '--replicas': '1', '--expected-mtp': '0', '--expected-sequences': '30',
        '--context': '40000', '--context-profile': 'nvfp4',
        '--expected-gpu-memory-utilization': '0.95', '--active-games': '0',
        '--request-slots': '0', '--clock-mode': 'gameplay', '--shutdown-margin': '0',
        '--min-p': '0', '--reasoning-effort': 'xhigh', '--image-scale': '4',
        '--stdout-token-limit': '1024', '--temperature': '1',
        '--response-headroom-fraction': '0.10',
    }
    numeric = set(values)-{'--context-profile', '--clock-mode', '--reasoning-effort'}
    for key, expected in values.items():
        actual = OPTION(argv, key)
        if actual is None or (float(actual) != float(expected) if key in numeric else actual != expected):
            raise ValueError(f'{contract} requires explicit {key} {expected}')
    for name in ('--no-token-limit', '--no-response-cap'):
        if argv.count(name) != 1:
            raise ValueError(f'Require {name} exactly once')
    for name in ('--token-cap', '--total-token-budget', '--throughput-tps', '--response-cap',
                 '--expected-kv-gib', '--start-epoch', '--no-program-index', '--no-snapshot'):
        if any(arg == name or arg.startswith(name+'=') for arg in argv):
            raise ValueError('Conflicting benchmark option '+name)
    games = OPTION(argv, '--games')
    if contract == 'public25':
        if games != 'all':
            raise ValueError('public25 requires --games all; resolved official IDs are checked again')
    elif (not games or games == 'all' or len(games.split(',')) != 4
          or len(set(games.split(','))) != 4 or not all(games.split(','))):
        raise ValueError('smoke requires four explicit distinct public game IDs')
    if not OPTION(argv, '--output'):
        raise ValueError('An explicit fresh --output directory is required')
    if argv.count('--execute') > 1:
        raise ValueError('Repeated --execute')
    return list(argv)


def verify_sources(root, manifest_path):
    root, manifest_path = Path(root).resolve(), Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HARNESS_MANIFEST_SHA256:
        raise ValueError('Harness manifest is not the reviewed PRO-LONG v5.22 release')
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or len(manifest) != 111:
        raise ValueError('Expected 111 source entries in the reviewed overlay')
    for relative, expected in manifest.items():
        path = root / relative
        if (Path(relative).is_absolute() or '..' in Path(relative).parts
                or not path.resolve().is_relative_to(root)
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('Reviewed harness source mismatch: '+relative)
    source = Path(__file__).resolve().parent
    for name, expected in LOCAL_PINS.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Persistent adapter source mismatch: '+name)
    return dict(harness_root=str(root), harness_manifest=str(manifest_path),
                harness_manifest_sha256=HARNESS_MANIFEST_SHA256, harness_files=111,
                adapter_sha256=dict(LOCAL_PINS))


def load_reviewed_runner(root):
    root = Path(root).resolve()
    # Script directory retains first priority for experimental adapter modules;
    # every loaded module whose filename is in the harness manifest is checked.
    sys.path.insert(1, str(root))
    runner = importlib.import_module('run_prolong')
    if Path(runner.__file__).resolve() != root/'run_prolong.py':
        raise ValueError('A different run_prolong is already imported')
    return runner


def validate_loaded_overlay(root, manifest):
    root = Path(root).resolve()
    for name in manifest:
        if not name.endswith('.py') or '/' in name:
            continue
        module = sys.modules.get(name[:-3])
        if module is not None and Path(module.__file__).resolve() != root/name:
            raise ValueError('Wrong cached harness module: '+name)


def validate_resolved(settings, contract):
    ids = settings['original_game_ids']
    n = 25 if contract == 'public25' else 4
    if (len(ids) != n or len(set(ids)) != n or len(settings['game_instances']) != n
            or settings['request_admission'] != 'all_games'
            or settings['request_slots'] != n or settings['game_concurrency'] != n
            or settings['initial_total_generated_tokens'] is not None
            or settings['context_policy']['response_cap'] is not None):
        raise ValueError('Resolved game population or allocation differs from the profiling contract')
    if not settings['program_index'] or not settings['snapshot']:
        raise ValueError('Keep the reviewed library and snapshot behavior enabled')


def archive_profile_sources(output):
    output = Path(output)
    manifest_path = output/'code-sha256.json'
    manifest = json.loads(manifest_path.read_text())
    for name in ('gameplay_harness_profile.py', 'run_profiled_gameplay.py'):
        raw = Path(__file__).with_name(name).read_bytes()
        relative = Path('gameplay-profile')/name
        target = output/'implementation'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest[str(relative)] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


def make_bound_runner(run_prolong, *, options, history_config, source_receipt, state):
    import gameplay_harness_profile as profile
    overlay_manifest = json.loads(Path(options.harness_manifest).read_text())

    def configure(solver, module, **kwargs):
        validate_loaded_overlay(options.harness_root, overlay_manifest)
        module.configure_solver(solver, **kwargs)
        return profile.configure_solver(solver, base_url=options.persistent_base_url,
                                        history_eviction=history_config)

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = builtins.__import__(name, globals, locals, fromlist, level)
        if name == 'prolong_duck' and 'configure_solver' in fromlist:
            return SimpleNamespace(configure_solver=lambda solver, **kw: configure(solver, module, **kw))
        return module

    def enrich(settings):
        validate_resolved(settings, options.profile_contract)
        inherited.apply_persistent_settings(settings, base_url=options.persistent_base_url,
                                             history_eviction=history_config)
        settings['harness_profile'] = dict(version=VERSION, contract=options.profile_contract,
            journal='gameplay-harness-profile.jsonl', source=source_receipt,
            timing='nested inclusive spans; HTTP transport includes generation; no model-visible telemetry',
            unknown_deadline_tail='engine generation without a verified received completion is separate')

    def arm(settings, solver, **kwargs):
        enrich(settings)
        output = Path(settings['output'])
        inherited.archive_experiment_sources(output)
        archive_profile_sources(output)
        state['settings'] = settings
        state['recorder'] = profile.activate(output/'gameplay-harness-profile.jsonl')
        deadline = run_prolong.arm_run_clock(settings, solver, **kwargs)
        profile.emit('run_clock', wall_start=settings['gameplay_started_epoch'], mono_start=time.monotonic(),
                     gameplay_started_epoch=settings['gameplay_started_epoch'], deadline_epoch=deadline,
                     total_seconds=settings['total_seconds'], game_ids=settings['original_game_ids'])
        return deadline

    def print_settings(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):
            try:
                settings = json.loads(args[0])
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(settings, dict) and 'context_policy' in settings and 'gameplay' in settings:
                    enrich(settings)
                    args = (json.dumps(settings, indent=2),)
        builtins.print(*args, **kwargs)

    scope = dict(run_prolong.main.__globals__, arm_run_clock=arm, print=print_settings,
                 __builtins__=dict(vars(builtins), __import__=scoped_import))
    bound = FunctionType(run_prolong.main.__code__, scope, run_prolong.main.__name__,
                         run_prolong.main.__defaults__, run_prolong.main.__closure__)
    bound.__kwdefaults__ = run_prolong.main.__kwdefaults__
    return bound


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--persistent-base-url', required=True)
    parser.add_argument('--harness-root', required=True, type=Path)
    parser.add_argument('--harness-manifest', required=True, type=Path)
    parser.add_argument('--profile-contract', choices=('public25', 'smoke'), required=True)
    parser.add_argument('--kv-eviction-policy', default='oldest_interactions', choices=('oldest_interactions',))
    parser.add_argument('--kv-evict-fraction', default=0.10, type=float)
    try:
        for name in ('--persistent-base-url', '--harness-root', '--harness-manifest',
                     '--profile-contract', '--kv-eviction-policy', '--kv-evict-fraction'):
            OPTION(arguments, name)
        options, remaining = parser.parse_known_args(arguments)
        remaining = validate_contract(remaining, options.profile_contract)
        if options.kv_evict_fraction != 0.10:
            raise ValueError('Keep the reviewed 10% earliest-interaction eviction policy')
        url = urlsplit(options.persistent_base_url)
        if (url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost') or not url.port
                or url.username or url.password or url.path or url.query or url.fragment):
            raise ValueError('Use an explicit loopback HTTP persistent server URL without path')
        receipt = verify_sources(options.harness_root, options.harness_manifest)
    except ValueError as exc:
        parser.error(str(exc))
    history = inherited._history_settings(options.kv_eviction_policy, options.kv_evict_fraction)
    runner = load_reviewed_runner(options.harness_root)
    if f'http://{runner.HOST}:{runner.PORT}' != options.persistent_base_url:
        raise ValueError('Persistent URL must match the reviewed runner model/metrics endpoint')
    state = {}
    bound = make_bound_runner(runner, options=options, history_config=history, source_receipt=receipt, state=state)
    completed = False
    error = None
    try:
        result = bound(remaining)
        completed = True
        return result
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        if '--execute' in remaining:
            import gameplay_harness_profile as profile
            output = Path(OPTION(remaining, '--output'))
            validation = inherited.write_persistent_validation(output, runner_completed=completed)
            recorder = state.get('recorder')
            if recorder is not None:
                profile.emit('run_finished', wall_end=time.time(), mono_end=time.monotonic(),
                    runner_completed=completed, error_type=error,
                    gameplay_started_epoch=state['settings'].get('gameplay_started_epoch'),
                    deadline_epoch=state['settings'].get('deadline_epoch'))
                status = recorder.status()
                profile.deactivate()
                (output/'harness-profile-status.json').write_text(json.dumps(status, indent=2)+'\n')
                valid = bool(completed and validation and validation['valid'] and status['valid'])
                (output/'profiled-gameplay-result.json').write_text(json.dumps(dict(
                    version=VERSION, status='passed' if valid else 'failed', runner_completed=completed,
                    error_type=error, profile=status, persistent_validation=validation,
                    gameplay_started_epoch=state['settings'].get('gameplay_started_epoch'),
                    deadline_epoch=state['settings'].get('deadline_epoch'), finished_epoch=time.time()), indent=2)+'\n')
                if completed and not valid:
                    raise RuntimeError('Profile or persistent cleanup validation failed; evidence preserved')


if __name__ == '__main__':
    main()
