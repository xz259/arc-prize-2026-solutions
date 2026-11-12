"""Recap port checks (ARC3_V6_RECAP=1): boards whose opener's board image left the context are re-shown (at most 10,
oldest first, 4x) with step / level / attempt / action captions; not again while the carrying opener's board is resident
and nothing new has left; again after the carrier left and newer boards left; duplicate boards are captioned only.
usage: ARC3_V6_RECAP=1 PYTHONPATH=src:. python3 tests/test_recap.py"""
import os
import sys

assert os.environ.get('ARC3_V6_RECAP') == '1'
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
from v6.agent import V6Agent  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('%-70s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


class Sess:
    def __init__(self):
        self.rebased_away, self.blocks = set(), []

    def span_evicted(self, a, b):
        return 0


class Game:
    def __init__(self):
        self.history_entries = [HistoryEntry(action='', frame=None, result={})]
        self.v6_level_attempts = {1: [7]}

    def step_env(self, a):
        return {}


def board(i):
    g = [[0] * 64 for _ in range(64)]
    g[10][i % 64] = 3
    return g


def main():
    sess, game = Sess(), Game()
    a = V6Agent(session=sess, logger=lambda k, p: None, count_tokens=lambda t: 1, model='M', base_url='http://x/v1',
                provider='vllm')
    a._step_env_callback = game.step_env
    for obs in range(1, 13):                       # 12 openers, each followed by two actions
        a._v6_obs_count = obs
        grid = board(obs if obs != 6 else 5)        # obs 6 repeats obs 5's board
        a._v6_record_observation(Frame(grid=grid, step=obs * 2, level=1))
        for k in range(2):
            game.history_entries.append(HistoryEntry(action='UP' if k == 0 else 'LEFT', frame=None,
                                                     result=dict(board_changed=k == 0, game_over=(obs == 4 and k == 1))))
    sess.blocks = [dict(observation_id='obs-%08d' % t, board_resident=t > 8) for t in range(1, 13)]
    a._v6_obs_count = 13
    p = a._v6_recap_parts()
    imgs = [x for x in p if x.get('type') == 'image_url']
    texts = [x['text'] for x in p if x.get('type') == 'text']
    check('recap at obs 13: 8 left boards, one duplicate captioned only', len(texts) == 9 and len(imgs) == 7, (len(texts), len(imgs)))
    check('header names the steps', texts[0].startswith('Recap: the 8 most recent boards no longer in your context (steps 2-16)'), texts[0][:90])
    check('caption: step, level, attempt, actions, changes', texts[1] == 'Step 2 (level 1, attempt 2) -> UP, LEFT; board changed 1/2', texts[1])
    check('A2 wording in captions (default A1-A3 on)', 'level restarted' in texts[4], texts[4])
    check('duplicate board noted', texts[6].endswith('(same board as the previous entry)'), texts[6])
    a._v6_record_observation(Frame(grid=board(13), step=26, level=1))
    a._v6_obs_count = 14
    check('no recap while the carrier (obs 13) is resident', a._v6_recap_parts() == [])
    sess.blocks = [dict(observation_id='obs-%08d' % t, board_resident=t > 13) for t in range(1, 14)]
    a._v6_record_observation(Frame(grid=board(14), step=28, level=1))
    a._v6_obs_count = 15
    p2 = a._v6_recap_parts()
    t2 = [x['text'] for x in p2 if x.get('type') == 'text']
    check('recap again after the carrier left: the 10 most recent left (steps 8-26)', t2 and t2[0].startswith(
        'Recap: the 10 most recent boards no longer in your context (steps 8-26)'), t2[0][:90] if t2 else None)
    sess.rebased_away.add('obs-%08d' % 14)
    a._v6_record_observation(Frame(grid=board(15), step=30, level=1))
    a._v6_obs_count = 16
    check('no recap while the newer carrier (obs 15) is resident even if a rebase dropped obs 14', a._v6_recap_parts() == [])
    print('stats', a._v6_stats.get('recaps'), a._v6_stats.get('recap_boards'))
    print('TEST_RECAP', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
