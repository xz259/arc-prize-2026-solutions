"""history_state(session) (2026-09-27): per observation block, board_resident (the block's first image labelled
'observation' fully resident) and text_evicted_fraction (evicted share of the text before it); None without a board;
the registry snapshot fields are unchanged. argv: none (run from the overlay directory)."""
from pathlib import Path
import sys, unittest
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_service as ss
from history_eviction import HistoryRegistry, ObservationStart
from residency import ResidentLedger


def session(evicted=(), kinds=None):
    reg = HistoryRegistry()
    reg.observations = (ObservationStart('obs1', 100), ObservationStart('obs2', 400), ObservationStart('obs3', 700))
    reg.registered_through = 900
    images = [[150, 216], [300, 366], [450, 516], [800, 866]]      # obs1 board, a tool board in obs1's block, obs2, obs3
    kinds = kinds or {150: 'observation', 300: 'tool', 450: 'observation', 800: 'observation'}
    return NS(history_registry=reg, residency=ResidentLedger(900, list(evicted)), images=images, image_kinds=kinds)


class HistoryState(unittest.TestCase):
    def blocks(self, **kw):
        return {b['observation_id']: b for b in ss.history_state(session(**kw))['blocks']}

    def test_all_resident(self):
        b = self.blocks()
        self.assertEqual([b[k]['board_resident'] for k in ('obs1', 'obs2', 'obs3')], [True, True, True])
        self.assertEqual([b[k]['text_evicted_fraction'] for k in ('obs1', 'obs2', 'obs3')], [0.0, 0.0, 0.0])
        self.assertIsNone(b['obs3']['end'])

    def test_board_and_text_evictions(self):
        b = self.blocks(evicted=[(100, 130), (160, 170), (420, 450)])
        self.assertFalse(b['obs1']['board_resident'])                       # part of the board evicted: not resident
        self.assertEqual(b['obs1']['text_evicted_fraction'], 0.6)            # 30 of the 50 text tokens before the board
        self.assertTrue(b['obs2']['board_resident']); self.assertEqual(b['obs2']['text_evicted_fraction'], 0.6)
        self.assertTrue(b['obs3']['board_resident'])

    def test_tool_board_is_not_the_observation_board(self):
        b = self.blocks(evicted=[(300, 366)])
        self.assertTrue(b['obs1']['board_resident'])

    def test_block_without_labelled_board(self):
        b = self.blocks(kinds={150: 'recap', 300: 'tool', 450: 'observation', 800: 'observation'})
        self.assertIsNone(b['obs1']['board_resident']); self.assertIsNone(b['obs1']['text_evicted_fraction'])
        self.assertTrue(b['obs2']['board_resident'])

    def test_snapshot_fields_kept(self):
        s = ss.history_state(session()); r = session().history_registry.snapshot()
        self.assertEqual({k: s[k] for k in r if k != 'blocks'}, {k: r[k] for k in r if k != 'blocks'})
        self.assertEqual([{k: b[k] for k in ('observation_id', 'start', 'end')} for b in s['blocks']], r['blocks'])


if __name__ == '__main__':
    unittest.main()
