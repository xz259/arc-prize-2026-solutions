"""Source-pinned PLE handoff overlay; no installed-runtime files are modified."""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time

TARGET='vllm.v1.ple_offload.connector'
ORIGINAL_SHA='b911d00cf4a39556d946e86ac34ea70e6230683b22adf5ef14ac299e9ef36206'
PATCHED_SHA='5c9e6822ff8d18647f975124c38fcabd228cfb8e83a501edfe272f76d9f011ed'
SOURCE=Path(__file__).with_name('connector.py')

class Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):
        callback=getattr(self.original,'create_module',None)
        return callback(spec) if callback else None
    def exec_module(self,module):
        assert hashlib.sha256(Path(module.__spec__.origin).read_bytes()).hexdigest()==ORIGINAL_SHA
        raw=SOURCE.read_bytes();assert hashlib.sha256(raw).hexdigest()==PATCHED_SHA
        exec(compile(raw,str(SOURCE),'exec'),module.__dict__)
        root=Path(os.environ['ARC3_PLE_HANDOFF_RECEIPTS']);root.mkdir(parents=True,exist_ok=True)
        (root/f'worker-{os.getpid()}.json').write_text(json.dumps(dict(installed=True,pid=os.getpid(),epoch=time.time(),
            original_sha256=ORIGINAL_SHA,patched_sha256=PATCHED_SHA,source=str(SOURCE)),indent=2)+'\n')
    def __getattr__(self,name):return getattr(self.original,name)

class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname!=TARGET:return None
        for finder in list(sys.meta_path):
            if finder is self:continue
            spec=finder.find_spec(fullname,path,target)
            if spec is not None:
                assert spec.loader is not None
                spec.loader=Loader(spec.loader);return spec
        raise ImportError(fullname)

def install():
    assert TARGET not in sys.modules
    assert not any(isinstance(x,Finder) for x in sys.meta_path)
    assert Path(os.environ['ARC3_PLE_HANDOFF_RECEIPTS']).is_absolute()
    sys.meta_path.insert(0,Finder())
