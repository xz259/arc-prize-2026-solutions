import json
from pathlib import Path
import tempfile
import unittest
from run_profile_control import attempted_sessions


class OwnershipTests(unittest.TestCase):
    def test_only_this_output_persisted_start_requests_define_cleanup_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'raw-harness').mkdir()
            rows=[{'kind':'persistent-request','payload':{'route':'/arc3/persistent/start','body':{'session_id':'ours-01'}}},
                  {'kind':'persistent-response','payload':{'session_id':'unrelated'}},
                  {'kind':'persistent-request','payload':{'route':'/arc3/persistent/another/append','body':{'session_id':'another'}}},
                  {'kind':'persistent-request','payload':{'route':'/arc3/persistent/start','body':{'session_id':'bad/id'}}}]
            (root/'raw-harness'/'game.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n{"partial":')
            self.assertEqual(attempted_sessions(root),{'ours-01'})

    def test_no_raw_start_means_no_authority_to_close_new_foreign_session(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(attempted_sessions(Path(directory)),set())

if __name__=='__main__':unittest.main()
