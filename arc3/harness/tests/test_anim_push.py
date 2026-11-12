"""B2 animation push checks (run with ARC3_V6_ANIM_PUSH=1): frames only for interior transient cells; at most 6, evenly
spaced, settled board excluded; captions that give the server's 'animation' image kind; nothing when the call's last
animated action has no interior transient cells.
usage: ARC3_V6_ANIM_PUSH=1 PYTHONPATH=src:. python3 tests/test_anim_push.py"""
import os
import sys

assert os.environ.get('ARC3_V6_ANIM_PUSH') == '1'
from inference.agent.runtime_state import Frame  # noqa: E402
from v6 import agent as ag  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('%-70s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


def pack(g):
    return (len(g), len(g[0]), bytes(v for row in g for v in row))


class Entry:
    def __init__(self, action):
        self.action = action


class Game:
    def __init__(self):
        self.history_entries = [Entry('')]
        self.v6_anim = {}

    def add(self, action, chain):
        self.history_entries.append(Entry(action))
        if chain is not None:
            self.v6_anim[len(self.history_entries) - 1] = dict(chain=[pack(g) for g in chain], step=len(self.history_entries) - 1)


def board(cells=()):
    g = [[0] * 64 for _ in range(64)]
    for r, c, v in cells:
        g[r][c] = v
    return g


class Sess:
    rebased_away, blocks = set(), []

    def span_evicted(self, a, b):
        return 0


def main():
    a = ag.V6Agent(session=Sess(), logger=lambda k, p: None, count_tokens=lambda t: max(1, len(t) // 4), model='M',
                   base_url='http://x/v1', provider='vllm')
    g = Game()
    n0 = len(g.history_entries)
    # an interior blink: (30,30) goes 0 -> 5 -> 0 over 10 frames, (30,31) flickers in frames 2..8
    frames = [board()] + [board([(30, 30, 5)] + ([(30, 31, 3)] if 2 <= k <= 8 else [])) for k in range(1, 10)] + [board()]
    g.add('UP', frames)
    parts = a._v6_anim_push(g, n0)
    imgs = [p for p in parts if p.get('type') == 'image_url']
    caps = [p['text'] for p in parts if p.get('type') == 'text']
    check('interior transient -> frames pushed, at most 6', 1 <= len(imgs) <= 6, len(imgs))
    check('header caption first, spec wording', caps[0].startswith('Animation of UP: frames ') and
          'the settled board is the current board.' in caps[0], caps[0][:90])
    check('per-frame captions carry the animation kind marker',
          all(c.startswith('Animation, action event ') for c in caps[1:]) and len(caps) - 1 == len(imgs))
    check('settled board (frame 11 of 11) not pushed', not any(c.endswith('frame 11 of 11') for c in caps[1:]), caps[-1])
    check('frames at 4x (256 px images)', all(p['image_url']['url'].startswith('data:image/png') for p in imgs))
    # border-only transient: nothing
    g2 = Game(); m0 = len(g2.history_entries)
    g2.add('LEFT', [board(), board([(1, 1, 4)]), board()])
    check('edge-band transient only -> no push', a._v6_anim_push(g2, m0) == [])
    # last animated action without transient cells (monotone change): nothing, even if an earlier one blinked
    g3 = Game(); k0 = len(g3.history_entries)
    g3.add('UP', [board(), board([(30, 30, 5)]), board()])
    g3.add('DOWN', [board(), board([(20, 20, 1)]), board([(20, 20, 1), (21, 20, 1)])])
    check('last animated action has no transient cells -> no push', a._v6_anim_push(g3, k0) == [])
    # an action from an earlier call is not pushed
    g4 = Game()
    g4.add('UP', [board(), board([(30, 30, 5)]), board()])
    check('animation from an earlier call -> no push', a._v6_anim_push(g4, len(g4.history_entries)) == [])
    # the opener's animation line: location filled (port fix) and, after a push of the same action, the pointer
    g5 = Game(); j0 = len(g5.history_entries)
    g5.add('UP', frames)
    a._v6_started = True
    a._step_env_callback = None
    pushed = a._v6_anim_push(g5, j0)
    f = Frame(grid=board(), step=5, level=1)
    summ = dict(executed_actions=['UP'], executed_count=1, level=1, animation_action='UP',
                animation=dict(frames=10, transient_pixels=2, transient_bbox=[30, 30, 30, 31]))
    text, _ = a._build_user_prompt(5, valid_actions=['UP'], current_frame=f, history_entries=[], previous_step_summary=summ)
    check('animation line has the location', 'changed and changed back in row 30, cols 30-31' in text, text[:200])
    check('animation line points at the pushed frames', bool(pushed) and 'Its frames are shown above.' in text)
    text2, _ = a._build_user_prompt(6, valid_actions=['UP'], current_frame=f, history_entries=[], previous_step_summary=summ)
    check('pointer only in the opener right after the push', 'Its frames are shown above.' not in text2)
    print('TEST_ANIM_PUSH', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
