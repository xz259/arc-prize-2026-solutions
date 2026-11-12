import ast
import copy
import hashlib
import importlib
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
image=importlib.import_module(HERE.name+'.image_metadata')
encoder=importlib.import_module(HERE.name+'.encoder')
schema=importlib.import_module(HERE.name+'.receipt_schema')
worker=importlib.import_module(HERE.name+'.worker')
fixtures=importlib.import_module(HERE.name+'.test_snapshot')
closed=importlib.import_module(HERE.name+'.closed')
receipt_fixtures=importlib.import_module(HERE.name+'.test_receipts')
REPO=Path(os.environ.get('ARC3_SNAPSHOT_TEST_REPO',str(HERE.parents[4])))
SOURCE=REPO/'inference_stack/experiments/persistent_state_20260912/source-snapshot/vllm/model_executor/models/qwen3_vl.py'


def real_position_model():
    module=ast.parse(SOURCE.read_text())
    cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3VLForConditionalGeneration')
    functions=[copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
               for name in ('_iter_mm_grid_hw','_get_mrope_input_positions')]
    for fn in functions:fn.decorator_list=[]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])
    ns={'np':np,'torch':NS(from_numpy=lambda x:x)}
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),ns)
    ns['Qwen3VLForConditionalGeneration']=NS(_iter_mm_grid_hw=ns['_iter_mm_grid_hw'])
    config=NS(video_token_id=248057,vision_start_token_id=248053,vision_end_token_id=248054,
              vision_config=NS(spatial_merge_size=2))
    return NS(config=config,get_mrope_input_positions=lambda ids,features:ns['_get_mrope_input_positions'](ids,features,config))


def layout(offset=8,identifier='red'):
    return dict(identifier=identifier,offset=offset,length=4,grid_thw=[1,4,4])


def feature(row):
    return NS(modality='image',identifier=row['identifier'],mm_position=NS(offset=row['offset'],length=row['length'],is_embed=None),
              data={'image_grid_thw':NS(data=NS(tolist=lambda:row['grid_thw']))})


class EncoderTensor(fixtures.CpuTensor):
    def __init__(self,array):
        super().__init__(array);self.dtype='bfloat16';self.device='cuda:0';self.is_cuda=True
    @property
    def shape(self):return self.array.shape


def encoder_fixture():
    images=[layout()];rid='r'
    manager=NS(request_cached_ids={},cached={'red':set()},freeable={'red':4},freed=[])
    proof=encoder.core_encoder(NS(encoder_cache_manager=manager),NS(request_id=rid),images)
    wire=dict(request_id=rid,original_prompt_tokens=511,image_layout=images,encoder_proof=proof)
    value=EncoderTensor(np.arange(4*2560,dtype=np.uint16).reshape(4,2560))
    runner=NS(device='cuda:0',encoder_cache=NS(mm_features={rid:[feature(images[0])]},encoder_outputs={'red':value}),
              model_state=NS(model=NS(config=NS(vision_config=NS(out_hidden_size=2560,deepstack_visual_indexes=[])))))
    torch=NS(Tensor=EncoderTensor,uint8=np.uint8,bfloat16='bfloat16')
    return runner,wire,torch,value


class ImageTests(unittest.TestCase):
    def test_actual_qwen_method_and_analytic_rows_match(self):
        model=real_position_model()
        for images in ([],[layout()],[layout(8),layout(24,'blue')],[layout(0)]):
            actual,delta=image.actual_position_rows(model,list(range(511)),images)
            self.assertEqual((actual,delta),image.position_rows(511,images))
            if images:self.assertLess(delta,0);self.assertNotEqual(actual[1],actual[2])
        self.assertEqual(image.position_rows(511,[]),(list(map(list,[range(511)]*3)),0))

    def test_delta_zero_prepend_u_preserves_global_image_positions(self):
        accepted=600;relative=20
        reference=layout(accepted+1+relative);candidate=layout(accepted+(1+relative))
        self.assertEqual(image.position_rows(700,[reference]),image.position_rows(700,[candidate]))
        wrong=layout(accepted+relative)
        self.assertNotEqual(image.position_rows(700,[reference]),image.position_rows(700,[wrong]))

    def test_signed_delta_decode_rows_and_ring_inside_image(self):
        images=[layout(507)]
        rows,delta=image.computed_rows(511,520,images)
        self.assertEqual([r[519] for r in rows],[519+delta]*3)
        self.assertNotEqual([r[510] for r in rows],[510]*3)
        # Every ring phase carries actual three-axis image/decode coordinates.
        for computed in range(511,519):
            full,_=image.computed_rows(511,computed,images)
            ring=np.zeros((2,4,1,140),dtype=np.uint16)
            for p in range(computed-4,computed):ring[1,p%4,0,128:].view(np.int64)[:]=[r[p] for r in full]
            descriptor=next(x for x in schema.expected_components(computed) if x['category']=='qsa_raw_rope_position')
            torch=NS(uint8=np.uint8,int64=np.int64)
            raw=b''.join(worker.component_chunks(fixtures.CpuTensor(ring),descriptor,[1],computed,torch,full))
            self.assertEqual(raw,b''.join(struct.pack('<qqq',*(r[p] for r in full)) for p in range(computed-4,computed)))

    def test_layout_rejects_video_pruning_overlap_missing_and_changed_grid(self):
        for images in ([dict(layout(),grid_thw=[2,4,4])],[dict(layout(),length=3)],
                       [layout(),layout(10)],[dict(layout(),offset=510)],[layout()]*3):
            with self.assertRaises(schema.ContractFailure):image.validate_images(images,511)
        f=feature(layout());self.assertEqual(image.feature_layout([f],511),[layout()])
        f.data=None
        with self.assertRaises(schema.ContractFailure):image.feature_layout([f],511)
        f=feature(layout());f.mm_position.is_embed=[True]*4
        with self.assertRaises(schema.ContractFailure):image.feature_layout([f],511)

    def test_actual_core_encoder_refs_and_retained_cache_scope(self):
        request=NS(request_id='r');manager=NS(request_cached_ids={},cached={'red':set()},freeable={'red':4},freed=[])
        scheduler=NS(encoder_cache_manager=manager)
        self.assertTrue(encoder.core_encoder(scheduler,request,[layout()])['entries'][0]['retained'])
        for mutate in (lambda m:m.request_cached_ids.update(r={0}),lambda m:m.cached['red'].add('r'),
                       lambda m:m.cached.clear(),lambda m:m.freed.append('red'),lambda m:m.freeable.update(red=3)):
            bad=copy.deepcopy(manager);mutate(bad)
            with self.assertRaises(schema.ContractFailure):encoder.core_encoder(NS(encoder_cache_manager=bad),request,[layout()])
        manager.cached['red'].add('r')
        with self.assertRaises(schema.ContractFailure):encoder.core_encoder(scheduler,request,[])

    def test_text_default_and_image_logical_metadata_are_distinct(self):
        ids=list(range(511));text=schema.expected_logical_metadata(ids,511,511)
        expected=b''.join(struct.pack('<i',p) for p in range(511))*3
        self.assertEqual(text['mrope_delta'],0);self.assertEqual(text['mrope_prefill_positions_sha256'],hashlib.sha256(expected).hexdigest())
        mm=schema.expected_logical_metadata(ids,511,511,[layout()])
        self.assertLess(mm['mrope_delta'],0);self.assertNotEqual(mm['mrope_prefill_positions_sha256'],text['mrope_prefill_positions_sha256'])
        self.assertEqual(mm['ple_predecessor_ids_sha256'],text['ple_predecessor_ids_sha256'])
        self.assertEqual(len(schema.expected_components(511)),121)
        self.assertEqual(image.MAX_ENCODER_BYTES,5242880)

    def test_worker_encoder_bytes_views_and_same_identity_mutation(self):
        runner,wire,torch,value=encoder_fixture()
        view=lambda t:dict(shape=list(t.shape),dtype=t.dtype,device=t.device,data_ptr=123)
        def read():return encoder.worker_encoder(runner,wire,torch,worker.byte_copy,view)
        initial=read();self.assertEqual(initial['outputs'][0]['sha256'],hashlib.sha256(value.array.tobytes()).hexdigest())
        value.array[0,0]^=np.uint16(1)
        changed=read();self.assertEqual(initial['views'],changed['views'])
        self.assertNotEqual(initial['outputs'],changed['outputs'])
        self.assertEqual(initial['outputs'][0]['bytes'],4*2560*2)
        for mutate in (lambda r,v:r.encoder_cache.encoder_outputs.clear(),lambda r,v:setattr(v,'dtype','float32'),
                       lambda r,v:setattr(v,'device','cpu'),lambda r,v:setattr(v,'is_cuda',False),
                       lambda r,v:setattr(v,'array',v.array[:3]),
                       lambda r,v:setattr(v,'array',v.array[:,::-1]),
                       lambda r,v:r.encoder_cache.mm_features.clear()):
            runner,wire,torch,value=encoder_fixture();mutate(runner,value)
            with self.assertRaises(schema.ContractFailure):read()

    def test_worker_rejects_unknown_or_unreleased_core_encoder_proof(self):
        for mutate in (lambda p:p.update(images=[]),lambda p:p.update(entries=[]),
                       lambda p:p['entries'][0].update(refs=['r']),
                       lambda p:p['entries'][0].update(freeable_tokens=True)):
            runner,wire,torch,_=encoder_fixture();mutate(wire['encoder_proof'])
            with self.assertRaises(schema.ContractFailure):
                encoder.worker_encoder(runner,wire,torch,worker.byte_copy,lambda t:{'test':True})

    @patch.dict(sys.modules,{'image_admission':NS(enabled=lambda:True)})
    def test_closed_owner_proof_allows_global_retention_but_rejects_stale_owner(self):
        c,ep,ev,la=fixtures.core_fixture();c.scheduler.requests.clear()
        manager=NS(request_cached_ids={},cached={'red':set()},freeable={'red':4},freed=[])
        c.scheduler.encoder_cache_manager=manager
        state=closed.closed_core_state(c,'r',[layout()],la)
        self.assertTrue(state['entries'][0]['cached'])
        runner=NS(vllm_config=fixtures.config(),execute_model_state=None,
                  req_states=NS(req_id_to_index={},index_to_req_id={}),
                  encoder_cache=NS(mm_features={},encoder_outputs={'red':object()}))
        wire=dict(request_id='r',image_layout=[layout()],source_manifest_sha256='a'*64)
        with patch.object(closed,'verify_sources',return_value=({},'a'*64)):
            proof=closed.closed_worker_state(NS(model_runner=runner),wire)
            self.assertEqual(proof['retained_identifiers'],['red'])
            runner.encoder_cache.mm_features['r']=[]
            with self.assertRaises(schema.ContractFailure):closed.closed_worker_state(NS(model_runner=runner),wire)
        manager.cached['red'].add('r')
        with self.assertRaises(schema.ContractFailure):closed.closed_core_state(c,'r',[layout()],la)
        manager.cached['red'].clear();c.batch_queue.append(object())
        with self.assertRaises(schema.ContractFailure):closed.closed_core_state(c,'r',[layout()],la)

    def test_canonical_image_layout_mrope_and_encoder_bytes_are_required(self):
        receipt=receipt_fixtures.fixture();w=receipt['worker'];images=[layout()]
        runner,wire,torch,_=encoder_fixture()
        encoded=encoder.worker_encoder(runner,wire,torch,worker.byte_copy,lambda t:{'data_ptr':123})
        w['canonical']=schema.canonical_state(list(range(512)),w['components'],
            schema.expected_logical_metadata(list(range(512)),511,511,images),images,encoded['outputs'])
        w['comparison_digest']=schema.json_digest(w['canonical'])
        w['ownership'].update(encoder_core_proof=wire['encoder_proof'],encoder_worker=encoded)
        def check(value):return schema.validate_receipt(value,expected_ids=list(range(512)),
            expected_source_manifest_sha256=w['source_manifest_sha256'],expected_prompt_tokens=511,
            expected_prefill_len=511,expected_image_layout=images)
        check(receipt)
        for mutate in (lambda b:b['canonical'].update(image_layout=[]),
                       lambda b:b['canonical'].update(encoder_outputs=[]),
                       lambda b:b['canonical']['logical_metadata'].update(mrope_delta=0),
                       lambda b:b['ownership'].pop('encoder_core_proof'),
                       lambda b:b['ownership']['encoder_worker'].update(views=[])):
            bad=copy.deepcopy(receipt);mutate(bad['worker'])
            bad['worker']['comparison_digest']=schema.json_digest(bad['worker']['canonical'])
            with self.assertRaises(schema.ContractFailure):check(bad)
        changed=copy.deepcopy(receipt)
        changed['worker']['canonical']['encoder_outputs'][0]['sha256']='b'*64
        changed['worker']['comparison_digest']=schema.json_digest(changed['worker']['canonical'])
        check(changed)
        with self.assertRaises(schema.ContractFailure):
            receipt_fixtures.compare_diagnostics(receipt,changed,expected_computed_ids=list(range(512)))

    @patch.dict(sys.modules,{'image_admission':NS(enabled=lambda:True)})
    def test_closed_rpc_checks_before_and_after_without_mutation_or_retry(self):
        c,ep,ev,la=fixtures.core_fixture();c.scheduler.requests.clear()
        c.scheduler.encoder_cache_manager=NS(request_cached_ids={},cached={'red':set()},freeable={'red':4},freed=[])
        source={'own_files':{'closed.py':'a'*64}};sha=schema.json_digest(source);calls=[]
        def rpc(*args,**kwargs):
            calls.append(kwargs)
            return [dict(request_id='r',worker_requests_empty=True,owner_features_absent=True,retained_identifiers=['red'])]
        c.collective_rpc=rpc
        with patch.dict(sys.modules,{'one_lookahead':la}),patch.object(closed,'verify_sources',return_value=(source,sha)):
            result=closed.core_closed_encoder(c,'r',[layout()])
            closed.validate_closed_receipt(result,request_id='r',image_layout=[layout()],source_sha=sha)
            self.assertEqual(len(calls),1);self.assertEqual(calls[0]['timeout'],30)
            def changed_rpc(*args,**kwargs):
                c.scheduler.encoder_cache_manager.cached['red'].add('r')
                return rpc(*args,**kwargs)
            c.collective_rpc=changed_rpc
            with self.assertRaises(schema.ContractFailure):closed.core_closed_encoder(c,'r',[layout()])
            self.assertEqual(len(calls),2)

    def test_image_core_readout_requires_flag_even_for_unmarked_reference(self):
        c,ep,ev,la=fixtures.core_fixture();r=c.scheduler.requests['r'];r.mm_features=[feature(layout())]
        c.scheduler.encoder_cache_manager=NS(request_cached_ids={},cached={'red':set()},freeable={'red':4},freed=[])
        with patch.dict(sys.modules,{'image_admission':NS(enabled=lambda:False)}):
            with self.assertRaises(schema.ContractFailure):fixtures.core.prepare_wire(c,'r',ep,ev,la)
        with patch.dict(sys.modules,{'image_admission':NS(enabled=lambda:True)}):
            _,wire=fixtures.core.prepare_wire(c,'r',ep,ev,la)
            self.assertEqual(wire['image_layout'],[layout()])


if __name__=='__main__':unittest.main()
