"""v6.2 round 5g unit tests (user 2026-10-06 ~21:40, after reading the drift cases): the yield resumption names an error
when the last Python call returned one (item 76); a level restart in place near the server's RoPE rows instead of the
server's rebase (item 77); the context line says parts of later turns can be cut and what always remains (item 78)."""
import inspect
import json
import os
import types
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa: F401
import v62.retention as r; r.install()
import v62.agent as A
fails = []


def check(name, got, want):
    ok = got == want
    print('%-62s -> %-48r %s' % (name, got if not isinstance(got, str) else got[:48], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


# item 78: the context line
S = A.TEXTS['system']
check('system: the new context line', "- Older turns, and parts of later turns, may be removed from this conversation as it "
      "grows, so earlier messages can be missing or cut short; this system message and the latest board always remain. "
      "`history` keeps every action and board of the game, and your library functions persist." in S, True)
check('system: the old context line is gone', '- Older turns may be removed from this conversation as it grows;' in S, False)

# item 76: the texts
check('resumption unchanged', A.TEXTS['resumption'].startswith('Your last Python call executed no actions. That is fine: '
                                                                 'it computed or inspected state.'), True)
check('resumption_error first sentence', A.TEXTS['resumption_error'].startswith(
    'Your last Python call ended with an error and executed no actions. The board is unchanged'), True)
check('resumption_error drops "That is fine"', 'That is fine' in A.TEXTS['resumption_error'], False)
check('resumption_error keeps the step line', 'Step {step}, level {level}. Valid actions: {valid_actions}.' in
      A.TEXTS['resumption_error'], True)


def tool(content):
    return {'role': 'tool', 'tool_call_id': 'x', 'content': content}


agent = object.__new__(A.V6Agent)
agent._v6_stats, agent._logs = {}, []
agent._v6_log = lambda kind, payload: agent._logs.append((kind, payload))
err = json.dumps({'tool': 'python', 'error': 'NameError: name x is not defined'})
fmt = json.dumps({'tool': 'python', 'error': A._FORMAT_ERROR})
ok = json.dumps({'tool': 'python', 'stdout': '3 objects'})
check('last call errored: tool error', agent._v62_last_call_errored([tool(err)]), True)
check('last call errored: format error', agent._v62_last_call_errored([tool(fmt)]), True)
check('last call errored: inspect-only output', agent._v62_last_call_errored([tool(ok)]), False)
check('last call errored: error then a good call', agent._v62_last_call_errored([tool(err), tool(ok)]), False)
check('last call errored: list content with an error', agent._v62_last_call_errored(
    [tool([dict(type='text', text=err), dict(type='text', text='caption')])]), True)
check('last call errored: no tool messages', agent._v62_last_call_errored([{'role': 'user', 'content': 'x'}]), False)

# rendering: the first line follows the flag
agent._step_env_callback, agent._v61_noact_calls = None, 0
frame = types.SimpleNamespace(step=12, level='?')
agent._v62_resume_error = False
check('rendered (no error)', agent._v6_resumption_text(frame, ['UP', 'DOWN']).split('\n')[0],
      'Your last Python call executed no actions. That is fine: it computed or inspected state. The board is unchanged.')
agent._v62_resume_error = True
check('rendered (error)', agent._v6_resumption_text(frame, ['UP', 'DOWN']).split('\n')[0],
      'Your last Python call ended with an error and executed no actions. The board is unchanged.')

# both yield branches set the flag; it is cleared with the resume; the position check runs before the restart decision
src = inspect.getsource(A.V6Agent._v62_analyze)
check('flag set in both yield branches (+ the v6.3 no-action limit break)', src.count('self._v62_resume_error = self._v62_last_call_errored(tool_messages)'), 3)
check('flag cleared when the resumption is sent', 'self._v62_resume_error = False' in src, True)
check('position check before the restart decision',
      0 <= src.find('self._v62_position_check()') < src.find("forced = getattr(self, '_v62_force_restart', False)"), True)

# item 77: the position restart
check('default limit', A.POSITION_RESTART, 1950000)
agent = object.__new__(A.V6Agent)
agent._v6_stats, agent._logs = {}, []
agent._v6_log = lambda kind, payload: agent._logs.append((kind, payload))
agent._v6_started = True
agent._v6_session = types.SimpleNamespace(last_receipt={'logical_tokens': 1000000})
check('below the limit: no restart', (agent._v62_position_check(), getattr(agent, '_v62_force_restart', False)), (False, False))
agent._v6_session.last_receipt = {'logical_tokens': 1960000}
check('at the limit: restart in place', (agent._v62_position_check(), agent._v62_force_restart), (True, True))
check('counted and logged', (agent._v6_stats.get('position_restarts'), agent._logs[-1][0], agent._logs[-1][1]['logical_tokens']),
      (1, 'v62-position-restart', 1960000))
agent2 = object.__new__(A.V6Agent)
agent2._v6_stats, agent2._v6_log, agent2._v6_started = {}, lambda k, p: None, False
agent2._v6_session = types.SimpleNamespace(last_receipt={'logical_tokens': 1960000})
check('not before the session started', agent2._v62_position_check(), False)
agent2._v6_started, agent2._v6_session = True, None
check('no session yet', agent2._v62_position_check(), False)
A.POSITION_RESTART = 0
agent2._v6_session = types.SimpleNamespace(last_receipt={'logical_tokens': 1960000})
check('off when 0', agent2._v62_position_check(), False)

# item 79: the summary as a bullet list of up to 10 items
L = A.TEXTS['levelup']
# round 5h review: the wording now leads with the header line and enumerates items in any form (superseded check)
check('levelup asks for at most 10 items within 4,000 tokens', 'at most 10 items' in L and
      'Keep the summary within 4,000 tokens; a longer one is truncated.' in L, True)
check('levelup: "five lines" gone', 'five lines' in L, False)
check('caps', (A.RULE_LINES, A.SUMMARY_TOKENS), (10, 4000))


class D:
    pass


def cap(texts, n=3, turns=2):
    d = D(); d._v6_rule_capture = n; d._v62_rule_turns = turns; d._v6_rules = {}; d._v6_stats = {}; d._v6_rules_dirty = False
    A.V6Agent._v6_capture_rule(d, texts)
    return d._v6_rules.get(n)


items = ['- block %d slides until it hits a wall or another block' % i for i in range(1, 13)]
check('list right after the header', cap('Rule L3:\n' + '\n'.join(items[:4]) + '\n\nNow the plan.'), '\n'.join(items[:4]))
check('blank line between header and list', cap('Rule L3:\n\n' + '\n'.join(items[:3])), '\n'.join(items[:3]))
check('12 items -> the first 10', cap('Rule L3:\n' + '\n'.join(items)), '\n'.join(items[:10]))
check('list ends at the first non-item line', cap('Rule L3:\n' + '\n'.join(items[:2]) + '\nNow level 4: the board shows ...'),
      '\n'.join(items[:2]))
check('mid-sentence mention of the next level kept', cap('Rule L3:\n- keys open doors; level 4 may add more doors\n- the goal is '
      'the exit'), '- keys open doors; level 4 may add more doors\n- the goal is the exit')
check('next level named at a sentence start ends the list (ka59)', cap('Rule L3:\n- the goal is the exit door on the right. Now '
      'level 4: the board shows a key\n- more planning'), '- the goal is the exit door on the right.')
check('numbered and starred items kept as written', cap('Rule L3:\n1. the red block is pushed by the player\n* walls stop it\n2) '
      'the goal is the green pad'), '1. the red block is pushed by the player\n* walls stop it\n2) the goal is the green pad')
check('loose list (blank lines between items)', cap('Rule L3:\n- the red block is pushed\n\n- walls stop it\n\n- goal: the pad\n\nplan'),
      '- the red block is pushed\n- walls stop it\n- goal: the pad')
# round 5h (G1, Fable): a header line ending in ':' is all header, so its lead-in is no longer kept
check('header line ending in ":" dropped (5h; was: lead-in kept)', cap('Rule L3: mechanics so far:\n- the red block is pushed by '
      'the player\n- walls stop it'), '- the red block is pushed by the player\n- walls stop it')
check('prose summary as before (cut at the next level)', cap('Rule L3: push the red block onto the green pad first. Now level 4 '
      'has two pads.\nmore'), 'push the red block onto the green pad first.')
check('stub still skipped', cap('Rule L3: ...\nthinking'), None)
long_items = ['- %s' % ('mechanic %d: ' % i + ' '.join(['x'] * 1000)) for i in range(10)]     # ~20,000 characters, ~5,000 tokens (estimate)
got = cap('Rule L3:\n' + '\n'.join(long_items))
check('over 4,000 tokens: whole items, then "- ..."', ((len(got) + 3) // 4 <= A.SUMMARY_TOKENS, got.endswith('\n- ...'),
      all(l in long_items or l == '- ...' for l in got.split('\n'))), (True, True, True))
words = lambda text: len(text.split())            # an exact counter stands in for the served tokenizer
check('cap with the tokenizer: list', A._cap_rule('\n'.join('- ' + ' '.join(['w'] * 1500) for _ in range(4)), words).count('\n'), 2)
got = A._cap_rule(' '.join('w%d' % i for i in range(5000)), words)
check('cap with the tokenizer: prose cut at a word, marked', (got.endswith(' ...'), words(got) <= A.SUMMARY_TOKENS), (True, True))
check('under the cap: unchanged', A._cap_rule('- a\n- b', words), '- a\n- b')
check('display: a list under its level', A._rule_display(3, '- a moves\n- b stops'), '- L3:\n  - a moves\n  - b stops')
check('display: numbered list under its level', A._rule_display(3, '1. a moves\n2. b stops'), '- L3:\n  1. a moves\n  2. b stops')
check('display: lead-in then items', A._rule_display(3, 'so far:\n- a moves'), '- L3: so far:\n  - a moves')
check('display: prose unchanged', A._rule_display(3, 'push the block'), '- L3: push the block')
print('FAILS:', fails)
