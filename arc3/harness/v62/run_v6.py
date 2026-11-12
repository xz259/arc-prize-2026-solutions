"""Harness v6.2 runner (v6.1 + the allocator, 2026-10-05; v6.1 = the 2026-10-05 revision, HARNESS_REVISION_ITEMS_2026-10-05.md):
one arm of a gameplay run on the running persistent server. The level restart is on by default (150K; user 2026-10-05:
"from now on with restart on"); ARC3_MAX_ACTIVE_STREAMS = lanes turns on the allocator (dfranzen's priority gate: a game
stalled 150K on a level gives up its lane and comes back with the restart; 0 = off, restarts in place).

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
usage: python -m v61.run_v6 --games <ids or @file> --seconds 3600 --output <dir> [--base-url http://127.0.0.1:1234]
       [--context 35484] [--basis 30000] [--protect 5484] [--tokenizer <tokenizer.json>] [--expected-context N]
SIGTERM or SIGUSR1 ends the run early through the deadline teardown; benchmark.json, v6-stats.json and score.json are
written as at the deadline."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
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
    'LOCAL_ANALYZER_TOP_K': '20', 'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072', 'ARC3_MIDDLE_TRUNCATION': '1',
    'ARC3_ACTION_ECHO': '1', 'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1', 'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1', 'ARC3_ACTION_INFO': '1', 'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1', 'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8',
    'ARC3_DIFF_IMAGE': '1', 'ARC3_GAMEOVER_DIFF_IMAGE': '1', 'ARC3_ANIMATION': '1', 'ARC3_GUARDS_FROM_LEVEL': '2',
    'ARC3_BATCH_NOOP_BLOCK': '1', 'ARC3_STALE_STATE_BLOCK': '1', 'ARC3_MEMORY_SECTIONS': 'off',   # C6R (10-17)
    'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
    'EXPOSE_UNDO': 'on', 'ONLY_RESET_LEVELS': 'true', 'MPLBACKEND': 'Agg', 'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness v6',
    # v6.1 (user decisions 2026-10-05): repeat-state guard and death ledger on (death repeat guard off), voluntary
    # RESET exposed, A3 attempt lengths on, the v6.1 texts and the level-completion summary on, a 40-action per-call
    # cap; the level restart (run B) off unless ARC3_V61_RESTART_TOKENS is set (V62_BUILD item 126, user 2026-10-09: the
    # default restart budget is 200K, as the job scripts pass since run 396)
    'ARC3_REPEAT_STATE_GUARD': '0', 'ARC3_DEATH_LEDGER': '1', 'EXPOSE_RESET': 'on', 'ARC3_V6_A3': '1',
    'ARC3_V61_TEXTS': '1', 'ARC3_V61_SUMMARY': '1', 'ARC3_V61_CALL_CAP': '40', 'ARC3_V61_RESTART_TOKENS': '200000',
    'ARC3_V61_RESTART_HISTORY': 'level',
    # v6.2: restart on by default; the allocator gate off unless ARC3_MAX_ACTIVE_STREAMS is set (queue re-scored on refills)
    'ARC3_MAX_ACTIVE_STREAMS': '0', 'ARC3_PRIORITY_REFRESH_QUEUE': '1', 'ARC3_V62_LANE_TOKS': '50',   # v6.5 RF
    'ARC3_V62_LEVEL_TOKENS': '60000', 'ARC3_V62_START_STAGGER': '3',
    # V62_BUILD item 127 (user 2026-10-09: "We should use the production harness, i.e., our latest harness used for
    # yesterday's runs"): the defaults are the production launch's values (job 397: lc_run3 LAZY=1, FRAC=0.2 at the
    # 64,684 tier, the only tier for the finals)
    'ARC3_V6_LAZY_HISTORY': '1', 'ARC3_V6_EVICT_FRACTION': '0.2',
}

# v7.9 KAGGLE (apply_v79.py; KAGGLE_PORT_PLAN_2026-10-24.md 1b; the user 10-24: no guards on Kaggle, the games start early, the
# game-order flag). K2 ARC3_KAGGLE=1: no blocking /v1/models query at start (the server may still be loading): the served name from
# ARC3_KAGGLE_SERVED_MODEL, the context from --context, and a thread logs the server's values once it answers (log only).
# K3 --game-source gateway: the competition gateway (ARC_BASE_URL; OperationMode.COMPETITION; the games as available_environments
# lists them); --games @all (offline): every environment in --environments-dir. K4 ARC3_KAGGLE_SHUFFLE=1: the order shuffled with
# random.Random(ARC3_KAGGLE_SHUFFLE_SEED). Unset: the pod's behaviour.
KAGGLE = os.environ.get('ARC3_KAGGLE', '0') == '1'
KAGGLE_SERVED = 'nvidia/Qwen3.8-Flash-Next-NVFP4'          # minimal_inference.py's --served-model-name


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--games', default='')                  # v7.9 K3: required unless --game-source gateway
    p.add_argument('--game-source', choices=('offline', 'gateway'), default='offline')   # v7.9 K3
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:1234')
    p.add_argument('--context', type=int, default=64684)       # item 127: the production tier (64,684 x 32)
    p.add_argument('--basis', type=int, default=52684)
    p.add_argument('--protect', type=int, default=12000)
    p.add_argument('--tokenizer', default='/workspace/models/Qwen3.8-Flash-Next-W4A16-intel/tokenizer.json')
    p.add_argument('--environments-dir', default='/workspace/arc3-inference/public/environment_files')
    p.add_argument('--label', default='v62')
    p.add_argument('--concurrency', type=int, default=0,
                   help='games played at once (0 = all); later games in --games order take freed slots (long-context runs)')
    p.add_argument('--fake-model', action='store_true',
                   help='dry run: scripted replies (random valid action each turn) instead of the server')
    args = p.parse_args()
    if args.game_source == 'offline' and not args.games:
        p.error('--games is required with --game-source offline')
    return args


def _v79_gateway_url():
    return os.environ.get('ARC_BASE_URL') or 'http://gateway:8001/'


def _v79_arcade_games(args):
    """v7.9 K3: the game ids as the arcade lists them (available_environments, in its order). gateway: after the gateway's api/games
    answers (any status below 500, a 4xx included: urllib raises those as HTTPError; IE's review A1), or after
    ARC3_KAGGLE_GATEWAY_WAIT_S (default 1800 s), when it lists anyway."""
    import arc_agi
    from urllib.error import HTTPError
    from urllib.request import urlopen
    if args.game_source == 'gateway':
        base, t0, last = _v79_gateway_url(), time.time(), ''
        wait = float(os.environ.get('ARC3_KAGGLE_GATEWAY_WAIT_S', '1800') or 1800)
        while True:
            try:
                with urlopen(base.rstrip('/') + '/api/games', timeout=10) as r:
                    if r.status < 500:
                        break
                    last = 'status %s' % r.status
            except HTTPError as exc:                  # a 4xx (e.g. no API key on the probe) is an answer: the gateway is up
                if exc.code < 500:
                    break
                last = 'status %s' % exc.code
            except Exception as exc:  # noqa: BLE001
                last = repr(exc)[:200]
            if time.time() - t0 > wait:
                print('v79 kaggle: the gateway %s did not answer in %.0f s (%s); listing anyway' % (base, wait, last), flush=True)
                break
            time.sleep(5)
        print('v79 kaggle: the gateway %s answered after %.0f s' % (base, time.time() - t0), flush=True)
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=base, environments_dir='')
    else:
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=args.environments_dir)
    games = [e.game_id for e in arcade.available_environments]
    print('v79 kaggle: %d games from the %s arcade: %s' % (len(games), args.game_source, ','.join(games[:6]) + (' ...' if len(games) > 6 else '')),
          flush=True)
    return games


def _v79_server_check(base_url, served, context):
    """v7.9 K2: once the server answers, log its model id and max_model_len against the client's (log only)."""
    from urllib.request import urlopen
    t0 = time.time()
    while True:
        try:
            m = json.loads(urlopen(base_url + '/v1/models', timeout=10).read())['data'][0]
            same = (m['id'], m.get('max_model_len')) == (served, context)
            print('v79 kaggle: the server answered after %.0f s: model %r, max_model_len %r (client %r, %r)%s' % (
                time.time() - t0, m['id'], m.get('max_model_len'), served, context, '' if same else '; MISMATCH (logged only)'), flush=True)
            return
        except Exception:  # noqa: BLE001
            time.sleep(10)


def main():
    args = parse()
    # v6.9 REX HIDE_RESET (decision 49; apply_rex69.py item R69HR): arm B hides RESET; EXPOSE_RESET goes off before ENV's default
    # 'on' applies (a caller's EXPOSE_RESET on is refused)
    if os.environ.get('ARC3_V69_HIDE_RESET', '0') not in ('0', '1'):
        raise SystemExit('ARC3_V69_HIDE_RESET must be 0 or 1, not %r' % os.environ.get('ARC3_V69_HIDE_RESET'))
    if os.environ.get('ARC3_V69_HIDE_RESET', '0') == '1':
        if os.environ.get('EXPOSE_RESET', 'off').strip().lower() in ('on', '1', 'true', 'yes'):
            raise SystemExit('ARC3_V69_HIDE_RESET=1 with EXPOSE_RESET on: refused (arm B does not expose RESET)')
        os.environ['EXPOSE_RESET'] = 'off'
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault('ARC3_ASSISTANT_TURN_CAPS', '12000,16000')   # user 10-07 12:40: 12K cue, 16K hard cap (item 127)
    os.environ.setdefault('ARC3_V66_TEXTS', '1')                # v6.6 (user 10-17): the scoring fact, the 16K cue text
    os.environ.setdefault('ARC3_V66_SCO', '1')                  # v6.6g (user 10-18): the scoring fact alone; the control sets 0
    os.environ.setdefault('ARC3_V67_RESTART_KIND', 'hard')      # v6.7 SOFT (IE 10-18): hard = today's restart; arm B: soft
    os.environ.setdefault('ARC3_V67_DEAD_GAMEOVERS', '3')       # v6.7 SOFT: hard under soft at 3 GAME OVERs from the board
    os.environ.setdefault('ARC3_V67_DEAD_WINDOW', '0')          # v6.7 SOFT: the no-new-state window (0 = off)
    os.environ.setdefault('ARC3_V67_LOOP', '0')                 # v6.7 LOOP (IE 10-18): the loop ladder (both arms)
    os.environ.setdefault('ARC3_V67_LOOP_RUNG2_KIND', 'soft')   # v6.7x LOOPHARD (user decision 50): hard in the next runs
    os.environ.setdefault('ARC3_V68_NO_SCORE', '0')             # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_LEDGER', '0')               # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_NEVER_TYPES', '0')          # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_NOTES_NEG', '0')            # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_LOOP_TEXTS', '0')           # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_LOOP_FRAME', '0')           # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_REFRESH_TEXTS', '0')        # v6.8 texts (IE agreement 10-18)
    os.environ.setdefault('ARC3_V68_LOOP_FALLBACK', '16' if os.environ.get('ARC3_V68_LOOP_FRAME') == '1' else '0')   # v6.8b
    os.environ.setdefault('ARC3_V68_REFRESH_PROGRAMS', '0')     # v6.8 REFRESH_PROGRAMS (the user, 18:02 UTC)
    os.environ.setdefault('ARC3_V68_REFRESH_PROG_FREQ', '0')    # v6.8 REFRESH_PROGRAMS: the most called
    os.environ.setdefault('ARC3_V68_REFRESH_PROG_RECENT', '8')  # v6.8 REFRESH_PROGRAMS: the most recent
    os.environ.setdefault('ARC3_V69_CLEAR_STEP', '0')           # v6.9 CLEAR_STEP (decisions 82, 87): the clearing-step boards
    os.environ.setdefault('ARC3_V69_INLEVEL_REFRESH', '0')      # v6.9 INLEVEL (decisions 94-99): the in-level refresh's threshold (0 = off)
    os.environ.setdefault('ARC3_V69_ACTREV_SHADOW', '0')        # v6.9 ACT-REV shadow (decisions 95, 97, 99): log only
    os.environ.setdefault('ARC3_V69_RESTART_ACROSS', '0')       # v6.9 RAB (decisions 105, 117): the budget over conversations
    os.environ.setdefault('ARC3_V69_PARK_TOKENS', '0')          # v6.9 RAB (decisions 111, 117): the allocation rung (0 = off)
    os.environ.setdefault('ARC3_V69_POOL_RESERVE', '0')         # v6.9 RAB (decisions 111, 117): the pool's reserve (0 = off)
    os.environ.setdefault('ARC3_V69_HIDE_RESET', '0')           # v6.9 REX (decision 49): RESET hidden from the model (arm B: 1)
    for _k in ('ARC3_V70_SOFT_CAP', 'ARC3_V70_INDEX', 'ARC3_V70_INDEX_SHADOW', 'ARC3_V70_INDEX_ADMIT', 'ARC3_V70_INDEX_SWAP',
               'ARC3_V70_NO_HARD_RESTART'):                     # v7.0 SWAP70 (apply_swap70.py): each off by default
        os.environ.setdefault(_k, '0')
    for _k in ('ARC3_V70H_ATTEMPT_LEDGER', 'ARC3_V70H_IL_OPENER', 'ARC3_V70H_NO_CLOCK', 'ARC3_V70H_ALIASES',
               'ARC3_V70H_ACTION_FACTS', 'ARC3_V70H_ACCESSORS', 'ARC3_V70H_BOARD_LEDGER', 'ARC3_V70H_REFUSAL_ONCE',
               'ARC3_V70H_ACTREV', 'ARC3_V70H_RESIDENT_POINTER', 'ARC3_V70H_RESET0'):   # v7.0 harness revision (apply_v70h.py)
        os.environ.setdefault(_k, '0')
    for _k in ('ARC3_V71_TEXTS', 'ARC3_V71_ALLOC'):             # v7.1 (apply_v71.py): the prompt cleanup, the allocation
        os.environ.setdefault(_k, '0')
    os.environ.setdefault('ARC3_V67_TEXTS', '0')                # v6.7 TEXTS67 (IE 10-18): four factual text items
    os.environ.setdefault('ARC3_V67_REFRESH_LOGICAL', '0')      # v6.7 REFRESH (variant B: 500000): level-up refresh
    os.environ.setdefault('ARC3_V67_NOTES', '0')                # v6.7 NOTES (IE 10-18): notes format, ledger, facts
    os.environ.setdefault('ARC3_V67_SCORE_TEXT', '0')           # v6.7 SCORE (IE 10-18): the complete scoring rule
    os.environ.setdefault('ARC3_V67_GAME_POOL', '0')            # v6.7x GAMEBUDGET (user decision 55): the run's token pool
    os.environ.setdefault('ARC3_V67_SUMLOOKUP', '0')            # v6.7x SUMLOOKUP (user decision 53): summaries()
    os.environ.setdefault('ARC3_V67_EXT200', '0')               # v6.7x EXT200 (user decision 48): the extended fraction
    os.environ.setdefault('ARC3_V67_EXT200_TOKENS', '50000')    # v6.7x EXT200: the extension of the attempt's budget
    os.environ.setdefault('ARC3_V67_EXT200_SEED', '20261018')   # v6.7x EXT200: the assignment hash's seed
    os.environ.setdefault('ARC3_V62_NOACT_WARN', '16000,32000,48000')   # v6.6 NA3 (recorded explicitly in run_settings)
    os.environ.setdefault('ARC3_V66_SEEDS', '1')                # v6.6 item C + SEED1 (user 10-17): part of v6.6; control arm sets 0
    os.environ.setdefault('ARC3_V65_BATCH', '0')                # C6R (user + IE 10-17): v6.4's first-no-change batch stop
    os.environ.setdefault('ARC3_ASSISTANT_TURN_CHECK', '8000')    # v6.5 (user 10-17 ~01:4x: back to 8K; was 12000, v6.3 item F,
                                                                  # 12K: the first pause coincides with the 12K cue)
    from urllib.request import urlopen
    if args.fake_model:
        served, max_len = 'fake-model', args.context
    elif KAGGLE:                                 # v7.9 K2: never block on the server here (it may still be loading)
        served, max_len = os.environ.get('ARC3_KAGGLE_SERVED_MODEL') or KAGGLE_SERVED, args.context
        threading.Thread(target=_v79_server_check, args=(args.base_url, served, args.context), name='v79-server-check',
                         daemon=True).start()
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

    import v62.patches as patches          # noqa: F401  (session layer, verdicts, sandbox limits and v6.1 sandbox edits)
    import v62.retention as retention
    retention.install()                    # item 8: literal UPPER_CASE constants persist
    import arc_agi
    import taaf.game_api
    from taaf.benchmark import Benchmark
    from inference.tools.eval import evaluate_runs, save_score_file
    from v62.agent import V6Agent, TOOL
    from v62.solver import V6Solver
    from v62.transport import PersistentSession
    from v62 import lazy_history                # spec §5.4 lazy history store, only with ARC3_V6_LAZY_HISTORY=1
    lazy_history.install()
    from v62 import kernel_ext                  # v8.0 (10-25): aliases, the 60-s backstop; ARC3_V80_STATEFUL=1: the kernel
    kernel_ext.install()
    from v62 import agent as v61agent
    v61agent.install_history_scope()            # run B: a restarted conversation's history omits the level's earlier attempts

    if args.game_source == 'gateway' or args.games == '@all':     # v7.9 K3: the games as the arcade lists them
        games = _v79_arcade_games(args)
    else:
        games = (open(args.games[1:]).read().split() if args.games.startswith('@') else args.games.split(','))
        games = [g.strip() for g in ','.join(games).split(',') if g.strip()]
    v79_shuffle = None
    if os.environ.get('ARC3_KAGGLE_SHUFFLE', '0') == '1':          # v7.9 K4: the seeded shuffle (the user: default off)
        _seed = int(os.environ.get('ARC3_KAGGLE_SHUFFLE_SEED', '0') or 0)
        v79_shuffle = dict(seed=_seed, served=list(games))
        random.Random(_seed).shuffle(games)
        print('v79 kaggle: the game order shuffled with seed %d: %s ...' % (_seed, ','.join(games[:6])), flush=True)
    if not games:
        raise SystemExit('no games to play (game source %s, --games %r)' % (args.game_source, args.games))
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
        def make_session():
            if args.fake_model:
                from v62.fake_session import FakeSession
                return FakeSession(log=log)
            return PersistentSession(base_url=args.base_url, model=served, tools=[TOOL], template_kwargs=template_kwargs,
                                     sampling=sampling, history_eviction=history_eviction, im_end_ids=im_end,
                                     decode=decode, log=log)
        agent = V6Agent(session=make_session(), logger=log, count_tokens=count_tokens, model=served, timeout=900.0,
                        base_url=args.base_url + '/v1', provider='vllm', session_factory=make_session)
        agents[gid] = agent
        return agent

    if args.game_source == 'gateway':                              # v7.9 K3: the competition gateway
        spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=_v79_gateway_url(),
                                        environments_dir='')
    else:
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
                                                             'ARC3_V6_LAZY_HISTORY', 'ARC3_V6_EVICT_FRACTION',
                                                             'ARC3_V61_RESTART_TOKENS', 'ARC3_V61_CALL_CAP',
                                                             'ARC3_V61_NO_ACTION_CALLS', 'ARC3_V61_RESTART_HISTORY',
                                                             'ARC3_MAX_ACTIVE_STREAMS', 'ARC3_PRIORITY_REFRESH_QUEUE',
                                                             'ARC3_V62_LANE_TOKS', 'ARC3_V62_LEVEL_TOKENS', 'ARC3_V62_START_STAGGER',
                                                             'ARC3_V62_NOACT_WARN', 'ARC3_V62_NOACT_TOKENS', 'ARC3_V62_ZERO_LEVEL_BAND',
                                                             'ARC3_V62_RESTART_MIN_LEFT', 'ARC3_V63_LEVELS_COMPLETE_END',
                                                             'ARC3_V67_RESTART_KIND', 'ARC3_V67_DEAD_GAMEOVERS', 'ARC3_V67_DEAD_WINDOW',
                                                             'ARC3_V67_LOOP', 'ARC3_V67_TEXTS',
                                                             'ARC3_V67_LOOP_RUNG2_KIND',
                                                             'ARC3_V68_NO_SCORE', 'ARC3_V68_LEDGER', 'ARC3_V68_NEVER_TYPES',
                                                             'ARC3_V68_NOTES_NEG', 'ARC3_V68_LOOP_TEXTS', 'ARC3_V68_LOOP_FRAME',
                                                             'ARC3_V68_REFRESH_TEXTS',
                                                             'ARC3_V68_LOOP_FALLBACK',
                                                             'ARC3_V68_REFRESH_PROGRAMS', 'ARC3_V68_REFRESH_PROG_FREQ',
                                                             'ARC3_V68_REFRESH_PROG_RECENT',
                                                             'ARC3_V69_CLEAR_STEP',
                                                             'ARC3_V69_INLEVEL_REFRESH',
                                                             'ARC3_V69_ACTREV_SHADOW',
                                                             'ARC3_V69_RESTART_ACROSS',
                                                             'ARC3_V69_PARK_TOKENS',
                                                             'ARC3_V69_POOL_RESERVE',
                                                             'ARC3_V69_HIDE_RESET',
                                                             'ARC3_V70_SOFT_CAP', 'ARC3_V70_INDEX', 'ARC3_V70_INDEX_FILE',
                                                             'ARC3_V70_INDEX_SHADOW', 'ARC3_V70_INDEX_ADMIT',
                                                             'ARC3_V70_INDEX_SWAP', 'ARC3_V70_NO_HARD_RESTART',
                                                             'ARC3_V70H_ATTEMPT_LEDGER', 'ARC3_V70H_IL_OPENER',
                                                             'ARC3_V70H_IL_OPENER_TOKENS', 'ARC3_V70H_IL_OPENER_KMAX',
                                                             'ARC3_V70H_NO_CLOCK', 'ARC3_V70H_ALIASES', 'ARC3_V70H_ACTION_FACTS',
                                                             'ARC3_V70H_ACCESSORS', 'ARC3_V70H_BOARD_LEDGER',
                                                             'ARC3_V70H_REFUSAL_ONCE', 'ARC3_V70H_ACTREV',
                                                             'ARC3_V70H_RESIDENT_POINTER', 'ARC3_V70H_RESET0',
                                                             'ARC3_V71_TEXTS', 'ARC3_V71_ALLOC', 'ARC3_V71_LEVEL_TOKENS',
                                                             'ARC3_V71_GUARD_S',
                                                             'ARC3_V72_IL_BOARDS', 'ARC3_V72_REFRESH_MIN_LEFT', 'ARC3_V72_PARK_RANK',
                                                             'ARC3_V72_PARK_FIRST', 'ARC3_V72_RV', 'ARC3_V72_PARK_INDEX_FILE', 'ARC3_V72_PARK_BANDS',
                                                             'ARC3_V73', 'ARC3_V73_STRICT_LOGICAL', 'ARC3_V73_ILB_BOARDS', 'ARC3_V73_ILB_ACTIONS',
                                                             'ARC3_V73_CLEAR_BOARDS', 'ARC3_V73_NO_LEVELUP_REFRESH',
                                                             'ARC3_REBASE_MODE', 'ARC3_V74_REBASE_AT', 'ARC3_V74_RESET_MAX', 'ARC3_V75_POLISH',
                                                             'ARC3_V76_HYBRID', 'ARC3_V76_REFRESH_AT', 'ARC3_V76_REBASE_MIN_LEFT',
                                                             'ARC3_V76_CUE_MARK', 'ARC3_V77',
                                                             'ARC3_V67_REFRESH_LOGICAL', 'ARC3_V67_NOTES', 'ARC3_V67_SCORE_TEXT',
                                                             'ARC3_V67_GAME_POOL',
                                                             'ARC3_V67_SUMLOOKUP',
                                                             'ARC3_V67_EXT200', 'ARC3_V67_EXT200_TOKENS', 'ARC3_V67_EXT200_SEED',
                                                             'ARC3_V66_SCO', 'ARC3_V66_TEXTS', 'ARC3_V66_SEEDS', 'ARC3_V65_BATCH')},
                    history_eviction=history_eviction, sampling=sampling, template_kwargs=template_kwargs)
    if v61agent.V72_RANK != 'fifo' or v61agent.V72_C is not None or v61agent.V72_RML or v61agent.V72_BANDS:   # v7.2b / c: as loaded
        import hashlib
        _f = os.environ.get('ARC3_V72_PARK_INDEX_FILE', '').strip()
        settings['v72b'] = dict(refresh_min_left=v61agent.V72_RML, rank=v61agent.V72_RANK, first=v61agent.V72_FIRST,
                                bands=v61agent.V72_BANDS,
                                rv=dict(v61agent.V72_RV),
                                index_file=(dict(path=_f, md5=hashlib.md5(open(_f, 'rb').read()).hexdigest(),
                                                 name=v61agent.V72_C['name'], status=v61agent.V72_C['status'],
                                                 constants=v61agent.V72_C) if v61agent.V72_C is not None else None))
    if v61agent.V70_INDEX:                       # v7.0 INDEX: the constants file (path, md5, name, status), as loaded
        import hashlib
        _f = os.environ['ARC3_V70_INDEX_FILE']
        settings['v70_index_file'] = dict(path=_f, md5=hashlib.md5(open(_f, 'rb').read()).hexdigest(),
                                          name=v61agent.V70_C['name'], status=v61agent.V70_C['status'],
                                          schema=v61agent.V70_C['schema'], constants=v61agent.V70_C)
    if KAGGLE or args.game_source != 'offline' or args.games == '@all' or v79_shuffle is not None:   # v7.9: the switches as run
        settings['v79_kaggle'] = dict(kaggle=KAGGLE, game_source=args.game_source, games_arg=args.games[:200], shuffle=v79_shuffle,
                                      served_model=served, server_wait_s=os.environ.get('ARC3_KAGGLE_SERVER_WAIT_S'),
                                      warmup_action_games=os.environ.get('ARC3_WARMUP_ACTION_GAMES'),
                                      gateway=_v79_gateway_url() if args.game_source == 'gateway' else None)
    (out / 'run_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    import v62.solver as v67x_solver             # v6.7x GAMEBUDGET: the per-game caps in the run's pool (off at 0)
    v67x_solver.gamebudget_install(games, writer)
    deadline = datetime.fromtimestamp(time.time() + args.seconds + 60)

    if v61agent.V70_ON:                          # v7.0 SWAP70: the gate monitor (free lanes, waiting games by kind) every 60 s
        def v70_gate_monitor():
            from inference.agent import tool_agent as _ta
            path = out / 'v70-gate.jsonl'
            while True:
                try:
                    gate = _ta._priority_gate()
                    if gate is not None:
                        with gate._cond:
                            free, waiting, snaps = gate._free, list(gate._waiting), dict(gate._snapshots)
                        kinds = {}
                        for _neg, tok in waiting:
                            sn = snaps.get(tok)
                            k = (('never' if sn.state.get('never_started') else sn.band) if isinstance(sn, v61agent._V70Snapshot)
                                 else 'parked' if getattr(sn, 'parked', False) else 'swapped' if sn is not None else 'never')
                            kinds[k] = kinds.get(k, 0) + 1
                        with open(path, 'a') as f:
                            f.write(json.dumps(dict(epoch=time.time(), free=free, waiting=len(waiting), kinds=kinds,
                                                    running=sum(1 for e in v61agent._s70.REGISTRY.running().values()),
                                                    claims=v61agent._s70.CLAIMS.count())) + '\n')
                except Exception as exc:  # noqa: BLE001
                    print('v70 gate monitor: %s' % exc, flush=True)
                time.sleep(60)
        threading.Thread(target=v70_gate_monitor, name='v70-gate', daemon=True).start()

    if v61agent.LOOP:                            # v6.7 LOOP (Kaggle visibility): the LOOPS line every LOOP_REPORT_S
        loops_t0 = time.time()

        def loops_reporter():
            while True:
                time.sleep(v61agent.LOOP_REPORT_S)
                try:
                    tot, _ = v61agent.loops_totals({g: a._v6_stats for g, a in list(agents.items())})
                    print(v61agent.loops_line(tot, time.time() - loops_t0), flush=True)
                except Exception as exc:  # noqa: BLE001
                    print('LOOPS line failed: %s' % exc, flush=True)
        threading.Thread(target=loops_reporter, name='v67-loops', daemon=True).start()

    async def run_with_stop():
        # 2026-10-05: graceful early stop. SIGTERM / SIGUSR1 reuse taaf's soft-deadline teardown (Benchmark.request_stop,
        # R2.33); before the solver task exists the main task is cancelled instead. The handlers close with the loop, so
        # a second SIGTERM during the saves below still kills the process.
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()

        def stop(signame):
            print('%s: stopping the run through the deadline teardown' % signame, flush=True)
            task = bm._solver_task
            if task is not None and not task.done():
                bm.request_stop()
            else:
                main_task.cancel()

        for sig in (signal.SIGTERM, signal.SIGUSR1):
            loop.add_signal_handler(sig, stop, sig.name)
        await bm.run(soft_end_time=deadline, runtime_environment=None, minimal_diagnostics=True)

    try:
        asyncio.run(run_with_stop())
    except asyncio.CancelledError:
        print('run stopped early by a signal; saving the results', flush=True)
    finally:
        bm._save_json()
        for gid, agent in agents.items():
            try:
                agent._v6_session.close()
            except Exception:  # noqa: BLE001
                pass
        if v61agent.EXT200:                      # v6.7x EXT200: attempts still open at the run's end (outcome run_end)
            for agent in agents.values():
                try:
                    agent._v67x_run_end()
                except Exception as exc:  # noqa: BLE001  (the stats below are written regardless)
                    print('EXT200 run-end close failed: %s' % exc, flush=True)
        stats = {gid: a._v6_stats for gid, a in agents.items()}
        if v61agent.LOOP:                        # v6.7 LOOP: the totals and one line per game with a rung 1-3 event
            loops_tot, loops_games = v61agent.loops_totals(stats)
            stats['LOOPS'] = dict(totals=loops_tot, games=loops_games)
            for gid in loops_games:
                print(v61agent.loops_game_line(gid, loops_games[gid]), flush=True)
            print(v61agent.loops_line(loops_tot, time.time() - settings['start_epoch']), flush=True)
        if v67x_solver.GAMEBUDGET is not None:   # v6.7x GAMEBUDGET: the run-level line (pool, caps, leftover)
            stats['GAMEBUDGET'] = v67x_solver.GAMEBUDGET.summary()
        if v61agent.V70_ON:                      # v7.0 SWAP70: the registry and the admissions at the end
            stats['SWAP70'] = dict(registry=v61agent._s70.REGISTRY.running(), claims=v61agent._s70.CLAIMS.count(),
                                   admissions=v61agent._s70.ADMITS.count(), admission_rows=v61agent._s70.ADMITS.records())
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
    if v61agent.LOOP:                            # v6.7 LOOP: the totals as the last stdout line
        print(v61agent.loops_line(v61agent.loops_totals({g: a._v6_stats for g, a in agents.items()})[0],
                                  time.time() - settings['start_epoch']), flush=True)


if __name__ == '__main__':
    sys.exit(main())
