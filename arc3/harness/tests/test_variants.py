"""A1-A5 variant checks (run with ARC3_V6_A13=1 ARC3_V6_A45=1): texts applied once each; the A4 rule capture; the rules
block in the next opener; the restart opener's attempt number and lengths (A2/A3); the A2 per-action verdict.
usage: ARC3_V6_A13=1 ARC3_V6_A3=1 ARC3_V6_A45=1 PYTHONPATH=src:. python3 tests/test_variants.py (A3 is off by default
since 2026-10-03 ~16:20; this test turns it on to check its text)"""
import os
import sys

assert os.environ.get('ARC3_V6_A13') == '1' and os.environ.get('ARC3_V6_A45') == '1' and os.environ.get('ARC3_V6_A3') == '1'
from inference.agent.runtime_state import Frame  # noqa: E402
from v6.agent import TEXTS, V6Agent  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('%-66s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


class Sess:
    rebased_away, blocks = set(), []

    def span_evicted(self, a, b):
        return 0


class GameSession:
    v6_level_attempts = {2: [14, 9]}

    def step_env(self, arguments):
        return {}


def s_text():
    return TEXTS['system']


def main():
    s = TEXTS['system']
    check('A1 merged into Objective / Reading the board / Actions (no new section)',
          'Work from mechanics toward the goal' in s and 'priors are the first hypotheses' in s
          and 'usually a piece, a cursor or a selection' in s and '## Priors and probing' not in s)
    check('A2 verdict list names LEVEL RESTARTED', 'LEVEL RESTARTED (the attempt ended here; the flag is `game_over`)' in s)
    a = V6Agent(session=Sess(), logger=lambda k, p: None, count_tokens=lambda t: max(1, len(t) // 4), model='M',
                base_url='http://x/v1', provider='vllm')
    a._v6_started = True
    a._step_env_callback = GameSession().step_env
    f = Frame(grid=[[0] * 64 for _ in range(64)], step=40, level=2)
    text, kind = a._build_user_prompt(40, valid_actions=['UP'], current_frame=f, history_entries=[],
                                      previous_step_summary=dict(executed_actions=['UP'], executed_count=1,
                                                                 level_transition=True, level=2))
    check('level-up opener carries A4 request and A5 line', kind == 'levelup' and 'Rule L1:' in text
          and 'act on the first probe or plan the evidence supports' in text, kind)
    a._v6_capture_rule('Some thinking.\nRule L1: push the yellow block onto the red pad.\nThe pad must be empty first.\n\nNext.')
    check('A4 rule captured (two lines, to the blank)', a._v6_rules.get(1) == 'push the yellow block onto the red pad. '
          'The pad must be empty first.', a._v6_rules)
    text2, kind2 = a._build_user_prompt(41, valid_actions=['UP'], current_frame=f, history_entries=[],
                                        previous_step_summary=dict(executed_actions=['UP'], executed_count=1, level=2))
    check('rules block in the next opener', 'Rules you recorded for solved levels:\n- L1: push the yellow block' in text2)
    text3, kind3 = a._build_user_prompt(42, valid_actions=['UP'], current_frame=f, history_entries=[],
                                        previous_step_summary=dict(executed_actions=['LEFT'], executed_count=1, level=2,
                                                                   game_over=True, fatal_action='LEFT',
                                                                   game_over_trace=['1: LEFT -> LEVEL RESTARTED (attempt 2 ended here)']))
    check('restart opener: attempt 3 of level 2, two restarts, lengths 14, 9', kind3 == 'restart'
          and 'LEVEL RESTARTED (attempt 3 of level 2)' in text3 and 'restarted 2 time(s); the attempts lasted 14, 9 actions'
          in text3, text3[:200])
    # HD review (07:00): A2 in the per-action trace, A3 capped at the last 5 lengths, A4 capped at the 5 latest rules
    text4, _ = a._build_user_prompt(43, valid_actions=['UP'], current_frame=f, history_entries=[],
                                    previous_step_summary=dict(executed_actions=['LEFT'], executed_count=1, level=2,
                                                               game_over=True, fatal_action='LEFT',
                                                               game_over_trace=['1: LEFT -> GAME OVER (attempt ended immediately after this action)']))
    check('A2: the trace verdict says LEVEL RESTARTED, never GAME OVER', 'GAME OVER' not in text4
          and '1: LEFT -> LEVEL RESTARTED (attempt 2 ended here)' in text4, text4[:120])
    GameSession.v6_level_attempts = {2: [3, 4, 5, 6, 7, 8]}
    text5, _ = a._build_user_prompt(44, valid_actions=['UP'], current_frame=f, history_entries=[],
                                    previous_step_summary=dict(executed_actions=['LEFT'], executed_count=1, level=2,
                                                               game_over=True, fatal_action='LEFT'))
    check('A3: six restarts -> the last 5 attempts lasted 4, 5, 6, 7, 8',
          'restarted 6 time(s); the last 5 attempts lasted 4, 5, 6, 7, 8 actions' in text5, text5[:260])
    a._v6_rules = {n: 'rule %d' % n for n in range(1, 8)}
    block = a._v6_rules_block()
    check('A4: rules block keeps the 5 most recent levels', '- L3: rule 3' in block and '- L7: rule 7' in block
          and '- L2:' not in block, block.replace(chr(10), ' | '))
    check('A1 wording: "clear before the mechanics"', 'When the goal is clear before the mechanics' in s_text())
    from v6 import patches  # noqa: F401  (needs arcengine on the pod; skipped locally)
    print('TEST_VARIANTS', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ModuleNotFoundError as exc:
        print('TEST_VARIANTS', 'PASS (patches import skipped locally: %s)' % exc if not FAIL else 'FAIL ' + ', '.join(FAIL))
        sys.exit(1 if FAIL else 0)
