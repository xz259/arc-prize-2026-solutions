"""Harness v6.0 runner (BUILD_PLAN components 9-10): one arm of a gameplay run on the running persistent server.

taaf's Benchmark with dfranzen's HarnessSolver (10882e3; auto-RESET after a death, echo, trace, guards, animation
chain), the V6Agent per game (v6/agent.py) on our server session (v6/transport.py). Writes, under --output:
  raw-harness/game-NNN.jsonl     {kind, game_id, epoch, payload}: persistent-request / -response / -injection / -error,
                                 model-turn (usage), tool-output, v6-* events (the control's format: g7_common.py reads it)
  prolong/game-NNN/log.jsonl     {kind: 'action', game_id, epoch, payload: {result}} per executed action
  score.json                     taaf evaluate_runs / save_score_file (as the control runner)
  run_settings.json, v6-stats.json
Port deltas applied here (spec §9, §5.2, §5.3):
  - an action whose animation has interior transient cells is never a NO-OP: verdict 'effect (animated; see the
    animation note)', and it does not trigger the batch no-op stop;
  - sandbox processes: nice 10 and an address-space limit (ARC3_V6_SANDBOX_AS_GIB, default 4); 10-s time limit.
usage: python -m v6.run_v6 --games <ids or @file> --seconds 3600 --output <dir> [--base-url http://127.0.0.1:1234]
       [--context 35484] [--basis 30000] [--protect 5484] [--tokenizer <tokenizer.json>] [--expected-context N]"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# dfranzen's submitted configuration (notebook cell_04) with the v6 deltas; set before his modules are imported
ENV = {
    'ARC3_REASONING_HISTORY_KEY': 'reasoning_content', 'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'LOCAL_ANALYZER_TOOL_STEPS': '0', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '10', 'LOCAL_ANALYZER_YIELD_SECONDS': '0',
    'LOCAL_ANALYZER_YIELD_TOKENS': '2048', 'LOCAL_ANALYZER_TEMPERATURE': '1.0', 'LOCAL_ANALYZER_TOP_P': '0.95',
    'LOCAL_ANALYZER_TOP_K': '20', 'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '2048', 'ARC3_MIDDLE_TRUNCATION': '1',
    'ARC3_ACTION_ECHO': '1', 'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1', 'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1', 'ARC3_ACTION_INFO': '1', 'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1', 'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8',
    'ARC3_DIFF_IMAGE': '1', 'ARC3_GAMEOVER_DIFF_IMAGE': '1', 'ARC3_ANIMATION': '1', 'ARC3_GUARDS_FROM_LEVEL': '2',
    'ARC3_BATCH_NOOP_BLOCK': '1', 'ARC3_STALE_STATE_BLOCK': '1', 'ARC3_MEMORY_SECTIONS': 'off',
    'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
    'EXPOSE_UNDO': 'on', 'ONLY_RESET_LEVELS': 'true', 'MPLBACKEND': 'Agg', 'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness v6',
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--games', required=True)
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:1234')
    p.add_argument('--context', type=int, default=35484)
    p.add_argument('--basis', type=int, default=30000)
    p.add_argument('--protect', type=int, default=5484)
    p.add_argument('--tokenizer', default='/workspace/models/Qwen3.8-Flash-Next-W4A16-intel/tokenizer.json')
    p.add_argument('--environments-dir', default='/workspace/arc3-inference/public/environment_files')
    p.add_argument('--label', default='v6')
    p.add_argument('--concurrency', type=int, default=0,
                   help='games played at once (0 = all); later games in --games order take freed slots (long-context runs)')
    p.add_argument('--fake-model', action='store_true',
                   help='dry run: scripted replies (random valid action each turn) instead of the server')
    return p.parse_args()


def main():
    args = parse()
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault('ARC3_ASSISTANT_TURN_CAPS', '8000,12000')
    os.environ.setdefault('ARC3_ASSISTANT_TURN_CHECK', '8000')
    from urllib.request import urlopen
    if args.fake_model:
        served, max_len = 'fake-model', args.context
    else:
        models = json.loads(urlopen(args.base_url + '/v1/models', timeout=30).read())['data']
        served, max_len = models[0]['id'], models[0].get('max_model_len')
    if max_len != args.context:
        raise SystemExit(f'server max_model_len {max_len} != --context {args.context}')
    os.environ['LOCAL_ANALYZER_MODEL_ID'] = served
    os.environ['INFERENCE_ANALYZER_MODEL'] = served
    os.environ['LOCAL_ANALYZER_BASE_URL'] = args.base_url + '/v1'

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    im_end = tok.encode('<|im_end|>', add_special_tokens=False).ids
    count_tokens = lambda text: len(tok.encode(text, add_special_tokens=False).ids)   # noqa: E731
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)                   # noqa: E731

    import v6.patches as patches          # noqa: F401  (solver verdict/no-op exemption, sandbox limits)
    import arc_agi
    import taaf.game_api
    from taaf.benchmark import Benchmark
    from inference.tools.eval import evaluate_runs, save_score_file
    from v6.agent import V6Agent, TOOL
    from v6.solver import V6Solver
    from v6.transport import PersistentSession
    from v6 import lazy_history                # spec §5.4 lazy history store, only with ARC3_V6_LAZY_HISTORY=1
    lazy_history.install()

    games = (open(args.games[1:]).read().split() if args.games.startswith('@') else args.games.split(','))
    games = [g.strip() for g in ','.join(games).split(',') if g.strip()]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'raw-harness').mkdir()
    (out / 'prolong').mkdir()
    index_of = {g: i for i, g in enumerate(games)}
    locks = {}

    def writer(game_id):
        i = index_of.get(game_id, len(index_of))
        path = out / 'raw-harness' / f'game-{i:03d}.jsonl'
        lock = locks.setdefault(game_id, threading.Lock())

        def log(kind, payload):
            line = json.dumps(dict(kind=kind, game_id=game_id, epoch=time.time(), payload=payload), ensure_ascii=False)
            with lock, open(path, 'a') as f:
                f.write(line + '\n')
        return log

    # 2026-10-04 (user ~02:20): per-tier eviction fraction (10% / 20% / 30% at the 36,748 / 64,684 / 101,932 tiers)
    history_eviction = dict(fraction=float(os.environ.get('ARC3_V6_EVICT_FRACTION', '0.1')), policy='qsa_frequency',
                            protect_recent_tokens=args.protect,
                            target_basis_tokens=args.basis, unit='split')
    # 2026-10-04 (user ~11:32: 64K runs at T = 0.6 / 0.7): the persistent transport's temperature follows
    # LOCAL_ANALYZER_TEMPERATURE (ENV default 1.0; an exported value overrides it, as for dfranzen's own requests)
    sampling = dict(temperature=float(os.environ['LOCAL_ANALYZER_TEMPERATURE']), top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0,
                    presence_penalty=0.0, frequency_penalty=0.0, max_tokens=12000)
    template_kwargs = dict(enable_thinking=True, reasoning_effort='xhigh', preserve_thinking=True)
    agents = {}

    def factory(game, index):
        gid = game.game_run.game_id if game.game_run is not None else games[index]
        log = writer(gid)
        if args.fake_model:
            from v6.fake_session import FakeSession
            session = FakeSession(log=log)
        else:
            session = PersistentSession(base_url=args.base_url, model=served, tools=[TOOL], template_kwargs=template_kwargs,
                                        sampling=sampling, history_eviction=history_eviction, im_end_ids=im_end,
                                        decode=decode, log=log)
        agent = V6Agent(session=session, logger=log, count_tokens=count_tokens, model=served, timeout=900.0,
                        base_url=args.base_url + '/v1', provider='vllm')
        agents[gid] = agent
        return agent

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=args.environments_dir)
    solver = V6Solver(analyzer_factory=factory, max_runtime_s_per_game=args.seconds, concurrency=args.concurrency or len(games),
                      analyzer_timeout=900.0, action_log_root=out / 'prolong', game_index=index_of)
    bm = Benchmark(games=[taaf.game_api.GameAPI(env_name=g, external_game_id=g, arcade_spec=spec) for g in games],
                   solver=solver, job_dir=out, n_passes=1)
    settings = dict(label=args.label, games=games, concurrency=args.concurrency or len(games), seconds=args.seconds, context=args.context, basis=args.basis,
                    protect=args.protect, served_model=served, env={k: os.environ.get(k) for k in ENV},
                    caps=os.environ.get('ARC3_ASSISTANT_TURN_CAPS'), check=os.environ.get('ARC3_ASSISTANT_TURN_CHECK'),
                    yield_tokens=os.environ.get('ARC3_V6_YIELD_TOKENS', '4096'), start_epoch=time.time(),
                    variants={k: os.environ.get(k) for k in ('ARC3_V6_A13', 'ARC3_V6_A45', 'ARC3_V6_ANIM_PUSH',
                                                             'ARC3_V6_LAZY_HISTORY', 'ARC3_V6_EVICT_FRACTION')},
                    history_eviction=history_eviction, sampling=sampling, template_kwargs=template_kwargs)
    (out / 'run_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    deadline = datetime.fromtimestamp(time.time() + args.seconds + 60)
    try:
        asyncio.run(bm.run(soft_end_time=deadline, runtime_environment=None, minimal_diagnostics=True))
    finally:
        bm._save_json()
        for gid, agent in agents.items():
            try:
                agent._v6_session.close()
            except Exception:  # noqa: BLE001
                pass
        stats = {gid: a._v6_stats for gid, a in agents.items()}
        (out / 'v6-stats.json').write_text(json.dumps(stats, indent=1) + '\n')
        if lazy_history.ENABLED:
            (out / 'lazy-history-stats.json').write_text(json.dumps(lazy_history.STATS) + '\n')
    try:
        score = evaluate_runs([out])
        save_score_file(score, run_dirs=[out], output_path=out / 'score.json')
    except ValueError as exc:
        # 2026-10-03 step 2: a game that crashed mid-run (server connection reset) leaves no pass taaf will score,
        # and the arm then exited 1 with complete gameplay data. Fall back to the runs' own final scores.
        games = {}
        for run in getattr(bm, 'game_runs', None) or []:
            s = run.final_score if run.final_score is not None else run._compute_final_score()
            games[run.game_id] = dict(score=float(s or 0.0), state=run.state)
        mean = sum(g['score'] for g in games.values()) / max(1, len(games))
        (out / 'score.json').write_text(json.dumps(dict(score=mean, games=games, partial=True,
                                                        note='evaluate_runs failed: %s' % str(exc)[:160]), indent=1) + '\n')
        print('evaluate_runs failed (%s); fallback score.json from the runs\' final scores' % exc, flush=True)
    print('Saved:', out, 'score', json.load(open(out / 'score.json')).get('score'), flush=True)


if __name__ == '__main__':
    sys.exit(main())
