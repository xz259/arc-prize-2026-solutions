"""CPU lineage/config checks; no imports of launcher or serving runtime."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
BASE=json.loads((HERE/'BASELINE_FILES.json').read_text())
sha=lambda raw:hashlib.sha256(raw).hexdigest()


class StageTests(unittest.TestCase):
    def test_all_baseline_sources_unchanged_except_label_and_environment(self):
        allowed={'launch_scaling.py','optimization-config.json'}
        for name,expected in BASE.items():
            if name not in allowed:self.assertEqual(sha((HERE/name).read_bytes()),expected,name)
    def test_launcher_exact_inverse_to_baseline(self):
        text=(HERE/'launch_scaling.py').read_text();extra=(HERE/'launch-environment-insertion.txt').read_text()
        self.assertEqual(text.count(extra),1)
        self.assertEqual(sha(text.replace(extra,'').encode()),BASE['launch_scaling.py'])
        self.assertLess(text.index("os.environ['PYTHONPATH'] = str(HERE)"),text.index(extra))
        self.assertLess(text.index(extra),text.index('import minimal_inference as m'))
    def test_only_label_changes_config(self):
        c=json.loads((HERE/'optimization-config.json').read_text());self.assertEqual(c['label'],'nvfp4-private-moe-mode3-20260914-v1')
        c['label']='nvfp4-optimization-baseline-20260914-v1'
        # Original baseline config has same indent/newline.
        self.assertEqual(sha((json.dumps(c,indent=2)+'\n').encode()),BASE['optimization-config.json'])
    def test_normal_graph_argv_unchanged(self):
        tree=ast.parse((HERE/'launch_scaling.py').read_text());f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='server_command')
        source=ast.get_source_segment((HERE/'launch_scaling.py').read_text(),f)
        self.assertNotIn('compile-sizes',source);self.assertNotIn('worker-extension',source)
        c=json.loads((HERE/'optimization-config.json').read_text());self.assertEqual(c['eager'],0);self.assertEqual(c['capture_sizes'],[1,2,4,8,16,24,25,30,32])

if __name__=='__main__':unittest.main()
