"""v6.2 round 5b unit tests."""
import os
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa
import v62.retention as r; r.install()
import v62.agent as A
import v62.session_ext as se
from inference.framework import solver as sv
fails = []
def check(name, got, want):
    ok = got == want
    print('%-50s -> %-44r %s' % (name, got if not isinstance(got, str) else got[:44], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)
check('level-start fact', sv._action_verdict(dict(gameplay_changed=True, level_start_match='interior', edge_cells_restored=84)),
      "effect | the interior now matches the level's starting board, with no GAME OVER reported; 84 cells of the outer band also returned to their starting values")  # round 5d wording
check('repeat fact', sv._action_verdict(dict(gameplay_changed=True, repeat_in_call=2)),
      'effect | already executed from this board earlier in this call (2 times; boards compared on the interior)')  # round 5d wording
class E:
    def __init__(self, a, **res): self.action, self.result = a, res
ents = [E('RESET', action_name='RESET'), E('UP'), E('SPACE'), E('LEFT'), E('SPACE'), E('SPACE', game_over=True), E('RESET', automatic=True)]
check('fatal position', A.V6Agent._v62_fatal_position(ents), ' It was the 3rd SPACE and the 5th action of the attempt.')
check('fatal position after level-up', A.V6Agent._v62_fatal_position([E('UP', level_completed=True), E('MOUSE(row=1, col=2)', game_over=True), E('RESET', automatic=True)]),
      ' It was the 1st MOUSE and the 1st action of the attempt.')
T = A.TEXTS
S = T['system']
check('no repeat-refusal sentence (guard off)', 'RepeatedActionInStateError' in S, False)
check('edge-strip paragraph cut', 'usually a step budget that changes as you act' in S, False)
check('neutral band sentence', 'their verdict is BAND-ONLY CHANGE, not NO-OP.' in S, True)   # round 5l (item 102)
check('fields documented', '`level_start_match`' in S and '`repeat_in_call`' in S, True)
check('edge check wording (v6.3 G: merged into the default loop)', 'inspect exactly what changed and where it lies: in the interior or in the outer band' in S, True)
check('reminder bar gone', 'edge-bar' in T['reminder'], False)
check('restart has position slot', '{fatal_position}' in T['restart'] and 'the attempts lasted {lengths}' in T['restart'], True)
g = [[0] * 16 for _ in range(16)]
for rr in range(4, 9):
    for cc in range(4, 9):
        if rr in (4, 8) or cc in (4, 8): g[rr][cc] = 2
check('MOUSE neutral (no background label; round 5c adds the relative box fact)', 'background' in se.hit_text(se.mouse_hit(g, 6, 6)), False)
print(se.hit_text(se.mouse_hit(g, 6, 6)))
print('FAILS:', fails)
