"""Normal parent/sibling imports stay intact, only pinned modules are redirected."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import py_compile
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bootstrap'))
import arc3_private_moe_bootstrap as b


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        for rel in ['fakepkg/__init__.py','fakepkg/sub/__init__.py']:
            path=self.root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('')
        self.target=self.root/'fakepkg/sub/target.py';self.target.write_text('from .sibling import value\nanswer = value + 1\n')
        (self.root/'fakepkg/sub/sibling.py').write_text('value = 41\n')
        self.c={'targets':{'fakepkg.sub.target':dict(path=str(self.target),sha256=b.digest(self.target))}}
        self.events=[];self.state=NS(wrap_warmup=lambda m:self.events.append(m.answer),record=lambda *a,**k:None)
        self.finder=b.Finder(self.c,self.state);sys.meta_path.insert(0,self.finder);sys.path.insert(0,str(self.root))
        self.addCleanup(self.cleanup)
    def cleanup(self):
        if self.finder in sys.meta_path:sys.meta_path.remove(self.finder)
        sys.path.remove(str(self.root))
        for name in list(sys.modules):
            if name=='fakepkg' or name.startswith('fakepkg.'):sys.modules.pop(name)
    def test_exact_target_preserves_parent_and_sibling_and_removes_finder(self):
        m=importlib.import_module('fakepkg.sub.target');self.assertEqual(m.answer,42);self.assertEqual(self.events,[42])
        self.assertNotIn(self.finder,sys.meta_path)
        self.assertEqual(Path(sys.modules['fakepkg.sub.sibling'].__file__),self.root/'fakepkg/sub/sibling.py')
    def test_changed_source_fails_before_execution(self):
        self.target.write_text('raise AssertionError("must not execute")\n')
        with self.assertRaisesRegex(RuntimeError,'changed after admission'):importlib.import_module('fakepkg.sub.target')
    def test_loader_uses_verified_bytes_not_timestamp_bytecode(self):
        old='from .sibling import value\nanswer = value + 0\n';new=self.target.read_text()
        self.assertEqual(len(old),len(new));self.target.write_text(old)
        cache=self.root/'old.pyc';py_compile.compile(str(self.target),cfile=str(cache),doraise=True)
        self.target.write_text(new)
        # Direct verified get_code must compile source, independent of any cache.
        loader=b.Loader('fakepkg.sub.target',self.target,self.finder)
        self.assertIn(1,loader.get_code('fakepkg.sub.target').co_consts)
    def test_reload_or_shadowed_target_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Repeated/reloaded'):
            self.finder.find_spec('fakepkg.sub.target',[],object())
        self.c['targets']['fakepkg.sub.target']['path']=str(self.root/'different.py')
        (self.root/'different.py').write_text(self.target.read_text())
        with self.assertRaisesRegex(RuntimeError,'shadows target'):importlib.import_module('fakepkg.sub.target')


if __name__=='__main__':unittest.main()
