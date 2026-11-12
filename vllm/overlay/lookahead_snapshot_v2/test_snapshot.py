"""CPU construction tests. NumPy adapter is not a native Torch/GPU proof."""
import copy
from enum import Enum, IntEnum
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lookahead_snapshot_v2 import core, guards, layout, worker, receipt_schema as schema
from lookahead_snapshot_v2 import provenance
from lookahead_snapshot_v2.strict_receipt import ContractFailure


class KVQuantMode(IntEnum):
    NONE = 0
    NVFP4 = 5


class MambaAttentionBackendEnum(Enum):
    GDN_ATTN = 1
    SHORT_CONV = 2


def named(name, **kw):
    return type(name, (), dict(kw))()


def groups():
    result = []
    for group in range(6):
        owners = [x for x in schema.expected_owners() if x['group_id'] == group]
        specs = {}
        for owner in owners:
            common = dict(indexes_kv_by_block_stride=True, sliding_window=None,
                          attention_chunk_size=None, non_causal=False)
            if owner['categories'] == ['main_nvfp4']:
                s = named('FullAttentionSpec', **common, block_size=2768, storage_block_size=2768,
                    num_kv_heads=2, head_size=256, head_size_v=256, dtype='torch.uint8',
                    state_content_bytes=296, kv_quant_mode=KVQuantMode.NVFP4)
            elif owner['categories'] == ['qsa_compressed']:
                s = named('MLAAttentionSpec', **common, block_size=2768, storage_block_size=692,
                    num_kv_heads=1, head_size=128, head_size_v=0, dtype='torch.bfloat16',
                    compress_ratio=4, kv_quant_mode=KVQuantMode.NONE)
            elif group == 1:
                s = named('CircularBufferSpec', **common, block_size=4, storage_block_size=4,
                    num_kv_heads=1, head_size=140, head_size_v=0, dtype='torch.bfloat16',
                    page_size_padded=177152, kv_quant_mode=KVQuantMode.NONE)
            else:
                ple = group == 5
                s = named('MambaSpec', block_size=40000, storage_block_size=40000,
                    shapes=((9,10240),) if ple else ((3,10240),(48,128,128)),
                    dtypes=('torch.bfloat16',)*(1 if ple else 2), mamba_cache_mode='none',
                    num_speculative_blocks=0, tp_replicated=ple, page_size_padded=1638656,
                    mamba_type=MambaAttentionBackendEnum.SHORT_CONV if ple else MambaAttentionBackendEnum.GDN_ATTN)
            specs[owner['owner']] = s
        result.append(NS(layer_names=list(specs), kv_cache_spec=(named('UniformTypeKVCacheSpecs',
            kv_cache_specs=specs) if group < 2 else next(iter(specs.values())))))
    return result


def config():
    return NS(use_v2_model_runner=True, speculative_config=None, num_speculative_tokens=0,
        scheduler_config=NS(async_scheduling=True),
        parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1,
            enable_dbo=False,decode_context_parallel_size=1),
        cache_config=NS(enable_prefix_caching=False,mamba_cache_mode='none'),
        kv_transfer_config=None,ec_transfer_config=None,model_config=NS(runner_type='generate'))


def core_fixture():
    request = NS(request_id='r', status=NS(name='WAITING_FOR_STREAMING_REQ'),
        num_in_flight_tokens=0,num_output_placeholders=0,num_stale_output_tokens=0,spec_token_ids=[],
        streaming_queue=[],num_preemptions=0,mm_features=[],prompt_embeds=None,lora_request=None,
        num_computed_tokens=511,num_prompt_tokens=511,_all_token_ids=list(range(512)))
    blocks = [[i+1] for i in range(6)]
    coord = NS(kv_cache_config=NS(kv_cache_groups=groups()),single_type_managers=[
        NS(req_to_blocks={'r':[NS(block_id=x[0],ref_cnt=1)]}) for x in blocks])
    manager = NS(coordinator=coord,get_blocks=lambda rid:NS(get_block_ids=lambda:blocks))
    scheduler = NS(vllm_config=config(),requests={'r':request},kv_cache_manager=manager)
    owner = NS(scheduler=scheduler,batch_queue=[],model_executor=NS(futures_queue=[]))
    def settled(r):
        schema.require(all(getattr(r,k)==0 for k in ('num_in_flight_tokens',
            'num_output_placeholders','num_stale_output_tokens')) and not r.spec_token_ids, 'Not settled')
    ep = NS(is_persistent=lambda r:True,assert_settled=settled)
    ev = NS(_request_for_external=lambda s,ext:s.requests[ext],
        worker_owner_groups=lambda c:c.kv_cache_groups)
    la = NS(enabled=lambda:True,pending=lambda s:{},marked=lambda r:False,VERSION='test')
    return owner, ep, ev, la


class CpuTensor:
    """Byte-preserving NumPy construction adapter for component_chunks only."""
    def __init__(self, array): self.array = array
    def __getitem__(self, key): return CpuTensor(self.array[key])
    def movedim(self, a, b):
        import numpy as np
        return CpuTensor(np.moveaxis(self.array,a,b))
    def view(self, arg):
        if arg == -1: return CpuTensor(self.array.reshape(-1))
        return CpuTensor(self.array.view(arg))
    def contiguous(self):
        import numpy as np
        return CpuTensor(np.ascontiguousarray(self.array))
    def cpu(self): return self
    def numpy(self): return self.array
    def numel(self): return self.array.size
    def element_size(self): return self.array.dtype.itemsize
    def is_contiguous(self): return self.array.flags.c_contiguous


class SnapshotTests(unittest.TestCase):
    def test_exact_inventory_and_enum_names(self):
        self.assertEqual(len(layout.validate_specs(groups())),73)
        schema.verify_inventory()
        self.assertEqual(schema.expected_coverage(2048)['selected_bytes'],75154560)

    def test_specs_fail_closed(self):
        for change in ('duplicate','unknown','normalized','geometry'):
            g=groups()
            if change=='duplicate':g[0].layer_names.append(g[0].layer_names[0])
            elif change=='unknown':g[5].layer_names=['unrecognized']
            elif change=='normalized':g[0].kv_cache_spec=next(iter(g[0].kv_cache_spec.kv_cache_specs.values()))
            else:next(iter(g[0].kv_cache_spec.kv_cache_specs.values())).storage_block_size=100
            with self.subTest(change=change), self.assertRaises(ContractFailure):layout.validate_specs(g)

    def test_row_chunks_page_and_copy_bounds(self):
        chunks=list(layout.row_chunks(range(4096),592,2768,[7,9]))
        self.assertEqual(sum(n for _,_,n in chunks),4096)
        self.assertEqual(chunks[0][0],7);self.assertEqual(chunks[-1][0],9)
        self.assertTrue(all(n*592<=1048576 and o+n<=2768 for _,o,n in chunks))
        with self.assertRaises(ContractFailure):list(layout.row_chunks(range(2769),592,2768,[7]))

    def test_all_ring_phases_in_chronological_order(self):
        for c in range(511,515):
            slots=[None]*4
            for p in range(c-4,c):slots[p%4]=p
            actual=[p for _,o,n in layout.row_chunks(range(c-4,c),256,4,[12],ring=True)
                    for p in slots[o:o+n]]
            self.assertEqual(actual,list(range(c-4,c)))

    def test_no_unbounded_or_missing_hash_copy(self):
        for chunks,n in (([b''],1),([b'x'*1048577],1048577),([b'x'],2),([b'xx'],1)):
            with self.assertRaises(ContractFailure):layout.hash_chunks(chunks,n)
        self.assertEqual(layout.hash_chunks([b'\x00\x80',b'\xff\x7f'],4),hashlib.sha256(b'\x00\x80\xff\x7f').hexdigest())

    def test_core_exact_readonly_wire(self):
        owner,ep,ev,la=core_fixture()
        before=copy.deepcopy(owner.scheduler.requests['r'].__dict__)
        r,wire=core.prepare_wire(owner,'r',ep,ev,la)
        self.assertEqual(len(wire['core_specs']),73)
        self.assertEqual(wire['computed_tokens'],511)
        self.assertTrue(wire['core_proof']['exclusive_pages'])
        self.assertEqual(r.__dict__,before)

    def test_core_rejects_queue_shared_page_and_other_active_owner(self):
        for change in ('batch','future','shared','null','active','inflight','mm','queued_append','preempted'):
            c,ep,ev,la=core_fixture();r=c.scheduler.requests['r']
            if change=='batch':c.batch_queue.append(object())
            elif change=='future':c.model_executor.futures_queue.append(object())
            elif change in ('shared','null'):
                b=c.scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks['r'][0]
                if change=='shared':b.ref_cnt=2
                else:b.block_id=0
            elif change=='active':c.scheduler.requests['other']=NS(**dict(r.__dict__,status=NS(name='RUNNING')))
            elif change=='inflight':r.num_in_flight_tokens=1
            elif change=='mm':r.mm_features=[object()]
            elif change=='queued_append':r.streaming_queue=[object()]
            else:r.num_preemptions=1
            with self.subTest(change=change),self.assertRaises(ContractFailure):core.prepare_wire(c,'r',ep,ev,la)

    def test_core_post_rpc_changes_are_rejected(self):
        c,ep,ev,la=core_fixture()
        def rpc(*args,**kw):
            c.scheduler.requests['r'].num_computed_tokens=510
            return [{}]
        c.collective_rpc=rpc
        with patch.dict(sys.modules,{'engine_patch':ep,'engine_eviction':ev,'one_lookahead':la}),\
             patch.object(core,'verify_sources',return_value=({},'a'*64)):
            with self.assertRaises(ContractFailure):core.core_diagnostic(c,'r')

    def test_core_rpc_error_is_not_retried_or_repaired(self):
        c,ep,ev,la=core_fixture();calls=[]
        def rpc(*args,**kw):calls.append(kw);raise TimeoutError('uncertain')
        c.collective_rpc=rpc
        with patch.dict(sys.modules,{'engine_patch':ep,'engine_eviction':ev,'one_lookahead':la}),\
             patch.object(core,'verify_sources',return_value=({},'a'*64)):
            with self.assertRaises(TimeoutError):core.core_diagnostic(c,'r')
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]['timeout'],120)

    def test_staged_writes_and_scope_fail_closed(self):
        b=NS(**{k:[] for k in ('_staged_write_indices','_staged_write_starts',
                            '_staged_write_contents','_staged_write_cu_lens')})
        guards.no_staged_writes(b);b._staged_write_contents=[42]
        with self.assertRaises(ContractFailure):guards.no_staged_writes(b)
        c=config();c.parallel_config.enable_dbo=True
        with self.assertRaises(ContractFailure):guards.validate_config(c)
        for bad in ([[0]]*6,[[True]]*6,[[1,2]]*6,[[1]]*5):
            with self.assertRaises(ContractFailure):layout.validate_block_ids(bad)

    def test_component_extraction_all_categories_bits_and_padding(self):
        import numpy as np
        torch=NS(uint8=np.uint8,int64=np.int64)
        descriptors={x['category']:x for x in schema.expected_components(511)}
        for category,d in descriptors.items():
            with self.subTest(category=category):
                if category=='main_nvfp4':
                    a=np.arange(3*2*2768*296,dtype=np.uint8).reshape(3,2,2768,296)
                    cache=CpuTensor(a);expected=a[1,:,:511,:].transpose(1,0,2).tobytes()
                    mutate=lambda:a.__setitem__((1,0,0,295),a[1,0,0,295]^1)
                    pad=lambda:a.__setitem__((1,0,600,0),a[1,0,600,0]^1)
                elif category=='qsa_compressed':
                    a=np.arange(3*692*128,dtype=np.uint16).reshape(3,692,1,128)
                    cache=CpuTensor(a);expected=a[1,:127,0,:].tobytes()
                    mutate=lambda:a.__setitem__((1,0,0,0),a[1,0,0,0]^1)
                    pad=lambda:a.__setitem__((1,200,0,0),a[1,200,0,0]^1)
                elif category.startswith('qsa_raw'):
                    a=np.arange(3*4*140,dtype=np.uint16).reshape(3,4,1,140)
                    for p in range(507,511):a[1,p%4,0,128:].view(np.int64)[:]=[p]*3
                    cache=CpuTensor(a)
                    expected=b''.join((a[1,p%4,0,:128] if category=='qsa_raw_key'
                                     else a[1,p%4,0,128:]).tobytes() for p in range(507,511))
                    mutate=lambda:a.__setitem__((1,3,0,0 if category=='qsa_raw_key' else 128),
                        a[1,3,0,0 if category=='qsa_raw_key' else 128]^1)
                    pad=lambda:a.__setitem__((2,0,0,0),a[2,0,0,0]^1)
                else:
                    a=np.arange(3*math_prod(d['shape']),dtype=np.uint16).reshape(3,*d['shape'])
                    # Known signed zero / NaN payload patterns remain raw bits.
                    a.reshape(-1)[:4]=[0,0x8000,0x7fc1,0xffff]
                    cache=[None]*(d['coordinates']['index']+1);cache[-1]=CpuTensor(a)
                    expected=a[1].tobytes()
                    mutate=lambda:a[1].reshape(-1).__setitem__(0,int(a[1].reshape(-1)[0])^1)
                    pad=lambda:a[2].reshape(-1).__setitem__(0,int(a[2].reshape(-1)[0])^1)
                def extract():return b''.join(worker.component_chunks(cache,d,[1],511,torch))
                self.assertEqual(extract(),expected)
                pad();self.assertEqual(extract(),expected)
                mutate()
                if category=='qsa_raw_rope_position':
                    with self.assertRaises(ContractFailure):extract()
                else:self.assertNotEqual(extract(),expected)

    def test_page_relocation_does_not_change_logical_hash(self):
        import numpy as np
        torch=NS(uint8=np.uint8,int64=np.int64)
        d=next(x for x in schema.expected_components(511) if x['category']=='main_nvfp4')
        a=np.zeros((4,2,2768,296),dtype=np.uint8)
        a[1]=np.arange(2*2768*296,dtype=np.uint8).reshape(2,2768,296);a[3]=a[1]
        cache=CpuTensor(a)
        self.assertEqual(layout.hash_chunks(worker.component_chunks(cache,d,[1],511,torch),d['bytes']),
                         layout.hash_chunks(worker.component_chunks(cache,d,[3],511,torch),d['bytes']))

    def test_packed_cross_owner_pool_and_owned_overlap(self):
        specs=layout.validate_specs(groups());page=1638656;stride=page*80
        descriptors=[]
        for i,name in enumerate(specs):
            # Repeated offsets across groups are legal for different owned pages.
            descriptors.append(NS(size=7*stride,offset=(i%12)*page,block_stride=stride,shared_by=[name]))
        cfg=NS(kv_cache_tensors=descriptors,num_blocks=7)
        actual=layout.allocation_records(cfg,specs)
        self.assertEqual(len(actual),73)
        self.assertTrue(all(x['stride']>x['page'] for x in actual.values()))
        layout.require_disjoint_ranges([(1000,2000,'group0'),(3000,4000,'group1')])
        with self.assertRaises(ContractFailure):
            layout.require_disjoint_ranges([(1000,2000,'group0'),(1500,2500,'group1')])
        descriptors[0].offset=stride
        with self.assertRaises(ContractFailure):layout.allocation_records(cfg,specs)

    def test_absent_ple_and_unknown_or_poisoned_state_rejected(self):
        g=groups();g[5].layer_names=[]
        with self.assertRaises(ContractFailure):layout.validate_specs(g)
        c,ep,ev,la=core_fixture()
        c.scheduler.requests['r']._arc3_eviction_state=NS(deleted_tokens=0,eviction_revision=1,poisoned=False)
        with self.assertRaises(ContractFailure):core.prepare_wire(c,'r',ep,ev,la)
        c.scheduler.requests['r']._arc3_eviction_state=NS(deleted_tokens=0,eviction_revision=0,poisoned=True)
        with self.assertRaises(ContractFailure):core.prepare_wire(c,'r',ep,ev,la)

    def test_exact_local_source_inventory(self):
        # Read the actual pinned sources without importing the serving runtime.
        repo=Path(os.environ.get('ARC3_SNAPSHOT_TEST_REPO', str(HERE.parents[3])))
        experiments=repo/'inference_stack/experiments'
        runtime=experiments/'persistent_state_20260912/source-snapshot/vllm'
        campaign=experiments/'private_gameplay_20260916'
        stack=experiments.parent
        paths={'vllm':runtime/'__init__.py',
            'engine_patch':campaign/'resume_lookahead_candidate/engine_patch.py',
            'one_lookahead':campaign/'resume_lookahead_candidate/one_lookahead.py',
            'engine_eviction':campaign/'evidence/restored-live-v1/server/engine_eviction.py',
            'arc3_qsa_cache':stack/'arc3_qsa_cache.py', 'arc3_qsa_nvfp4':stack/'arc3_qsa_nvfp4.py'}
        with patch.dict(sys.modules,{n:NS(__file__=str(p)) for n,p in paths.items()}):
            manifest,digest=provenance.verify_sources()
            self.assertEqual(digest,schema.json_digest(manifest))
            self.assertEqual(len(manifest['runtime_files']),29)
            # A mapped helper with a different source cannot inherit admission.
            sys.modules['engine_patch'].__file__=str(paths['one_lookahead'])
            with self.assertRaises(ContractFailure):provenance.verify_sources()


def math_prod(values):
    value=1
    for x in values:value*=x
    return value


if __name__=='__main__': unittest.main()
