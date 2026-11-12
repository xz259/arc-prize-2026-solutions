"""NHD admission and canonical bytes: CPU construction, never native proof."""
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lookahead_snapshot_v2 import worker, layout, receipt_schema as contract
from lookahead_snapshot_v2.strict_receipt import ContractFailure
from lookahead_snapshot_v2.test_snapshot import CpuTensor

PAGE = 2*2768*296


class GeometryTensor:
    """Metadata only; dtype/device labels do not establish native admission."""
    def __init__(self, *, shape=(3,2,2768,296), stride=(PAGE,296,592,1),
                 dtype='uint8', itemsize=1, offset=0, nbytes=None):
        self.shape, self._stride = shape, stride
        self.dtype, self.device, self.is_cuda = dtype, 'cuda:0', True
        self.ndim, self._itemsize, self._offset = len(shape), itemsize, offset
        self._nbytes = (offset+shape[0]*stride[0])*itemsize if nbytes is None else nbytes
    def stride(self): return self._stride
    def element_size(self): return self._itemsize
    def storage_offset(self): return self._offset
    def untyped_storage(self): return NS(nbytes=lambda:self._nbytes)


TORCH = NS(Tensor=GeometryTensor, uint8='uint8', bfloat16='bfloat16')


def admit(tensor, *, main=True, dtype='uint8', tail=(2,2768,296)):
    worker.check_tensor(tensor,TORCH,dtype=dtype,tail=tail,device='cuda:0',main_nvfp4=main)


class NHDTests(unittest.TestCase):
    def test_exact_nhd_and_packed_page_gap(self):
        admit(GeometryTensor())
        admit(GeometryTensor(stride=(PAGE*2,296,592,1),offset=4096))
        # Maximum touched byte exactly equals the complete per-page payload.
        self.assertEqual((2-1)*296+(2768-1)*592+(296-1)+1,PAGE)

    def test_unknown_transposes_partial_and_overlapping_layouts_rejected(self):
        for stride in ((PAGE,2768*296,296,1), (PAGE,592,296,1),
                       (PAGE,0,592,1), (PAGE,296,593,1), (PAGE,296,592,2),
                       (PAGE,296,-592,1), (PAGE-1,296,592,1)):
            with self.subTest(stride=stride),self.assertRaises(ContractFailure):
                admit(GeometryTensor(stride=stride))
        for tensor in (GeometryTensor(shape=(3,1,2768,296)),
                       GeometryTensor(nbytes=3*PAGE-1),
                       GeometryTensor(offset=4,nbytes=3*PAGE+3)):
            with self.assertRaises(ContractFailure): admit(tensor)
        with self.assertRaises(ContractFailure):
            admit(GeometryTensor(dtype='bfloat16',itemsize=2),dtype='bfloat16')

    def test_dense_categories_keep_exact_strides(self):
        cases=[((3,692,1,128),(88576,128,128,1)),
               ((3,4,1,140),(88576,140,140,1)),
               ((3,3,10240),(819328,10240,1)),
               ((3,48,128,128),(819328,16384,128,1)),
               ((3,9,10240),(819328,10240,1))]
        for shape,stride in cases:
            admit(GeometryTensor(shape=shape,stride=stride,dtype='bfloat16',itemsize=2),
                  main=False,dtype='bfloat16',tail=shape[1:])
            bad=tuple(stride[:-1])+(2,)
            with self.assertRaises(ContractFailure):
                admit(GeometryTensor(shape=shape,stride=bad,dtype='bfloat16',itemsize=2),
                      main=False,dtype='bfloat16',tail=shape[1:])
        with self.assertRaises(ContractFailure): admit(GeometryTensor(),main=False)

    def test_nhd_byte_order_padding_and_selected_bytes(self):
        import numpy as np
        torch=NS(uint8=np.uint8,int64=np.int64)
        stride=PAGE+4096;offset=256
        raw=np.full(4*stride,0xA7,dtype=np.uint8)
        # Same logical view returned by permuting physical B,N,H,D back to B,H,N,D.
        view=np.ndarray((3,2,2768,296),dtype=np.uint8,buffer=raw,offset=offset,
                        strides=(stride,296,592,1))
        token=np.arange(2768,dtype=np.uint32)[:,None,None]
        head=np.arange(2,dtype=np.uint32)[None,:,None]
        column=np.arange(296,dtype=np.uint32)[None,None,:]
        logical=((token*17+head*103+column*29+(column//128)*11)%256).astype(np.uint8)
        view[1]=logical.transpose(1,0,2);view[2]=view[1]
        cache=CpuTensor(view)
        for computed in (511,2048):
            descriptor=next(x for x in contract.expected_components(computed)
                            if x['category']=='main_nvfp4')
            def extract(page=1):
                return list(worker.component_chunks(cache,descriptor,[page],computed,torch))
            parts=extract();expected=logical[:computed].tobytes(order='C')
            self.assertEqual(b''.join(parts),expected)
            self.assertEqual(sum(map(len,parts)),computed*592)
            self.assertTrue(all(0<len(x)<=1048576 for x in parts))
            self.assertEqual(len(parts),1 if computed==511 else 2)
            self.assertEqual(layout.hash_chunks(extract(2),descriptor['bytes']),
                             hashlib.sha256(expected).hexdigest())
            # Future/uncomputed row, physical page gap and another page are excluded.
            view[1,:,computed:,:]^=np.uint8(1)
            raw[offset+stride+PAGE:offset+2*stride]^=np.uint8(1)
            view[0]^=np.uint8(1)
            self.assertEqual(b''.join(extract()),expected)
            view[1,:,computed:,:]^=np.uint8(1)
            # Every byte in both heads is covered, including outer/block scale bytes.
            for h in (0,1):
                for c in (0,127,128,131,147,148,275,276,295):
                    view[1,h,computed-1,c]^=np.uint8(1)
                    self.assertNotEqual(b''.join(extract()),expected)
                    view[1,h,computed-1,c]^=np.uint8(1)

    def test_source_delta_is_only_category_specific_guard(self):
        original=ast.parse((HERE/'fixtures'/'worker_v1.py').read_text())
        current=ast.parse((HERE/'worker.py').read_text())
        old={n.name:n for n in original.body if isinstance(n,ast.FunctionDef)}
        new={n.name:n for n in current.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(set(old),set(new))
        for name in old.keys()-{'check_tensor','cache_bindings'}:
            self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
        # Reverse the sole call-site keyword addition and require exact AST parity.
        calls=[n for n in ast.walk(new['cache_bindings'])
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='check_tensor' and any(k.arg=='main_nvfp4' for k in n.keywords)]
        self.assertEqual(len(calls),1)
        calls[0].keywords=[k for k in calls[0].keywords if k.arg!='main_nvfp4']
        self.assertEqual(ast.dump(old['cache_bindings']),ast.dump(new['cache_bindings']))


if __name__=='__main__':unittest.main()
