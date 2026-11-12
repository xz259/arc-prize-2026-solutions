"""CPU transaction invariants; worker byte preservation is tested separately.

The fake worker acknowledges a completed transaction but never represents a
GPU execution. These tests prove core validation, publication, and page-release
ordering using the actual ResidentLedger and transaction implementation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import engine_eviction as ee
import engine_patch
from residency import ResidentLedger


@dataclass
class FullAttentionSpec:
    storage_block_size: int=16

    @property
    def block_size(self):return self.storage_block_size


@dataclass
class MLAAttentionSpec:
    storage_block_size: int=4
    compress_ratio: int=4

    @property
    def block_size(self):return self.storage_block_size*self.compress_ratio


class MambaSpec:
    block_size=80


class CircularBufferSpec:
    block_size=4


@dataclass
class UniformTypeKVCacheSpecs:
    kv_cache_specs: dict

    @property
    def block_size(self):return next(iter(self.kv_cache_specs.values())).block_size


@dataclass
class Position:
    offset: int
    length: int


class Grid:
    def __init__(self,data):
        self.data=data

    def tolist(self):
        return list(self.data)


@dataclass
class ImageFeature:
    mm_position: Position
    data: dict
    modality: str="image"


def image(offset,length):
    return ImageFeature(Position(offset,length),{"image_grid_thw":NS(data=Grid([1,4,4]))})


def fake_specs():
    module=ModuleType("vllm.v1.kv_cache_interface")
    for cls in (FullAttentionSpec,MLAAttentionSpec,MambaSpec,CircularBufferSpec,UniformTypeKVCacheSpecs):
        setattr(module,cls.__name__,cls)
    return {"vllm.v1.kv_cache_interface":module}


def fixture(*,removed=((8,32),),features=()):
    request=NS(request_id="example-deadbeef",resumable=True,
        sampling_params=NS(extra_args={ee.MARKER:True,engine_patch.MARKER:True}),
        prompt_token_ids=list(range(62)),num_prompt_tokens=62,
        _output_token_ids=list(range(62,70)),_all_token_ids=list(range(70)),
        num_computed_tokens=69,num_in_flight_tokens=0,num_output_placeholders=0,
        num_stale_output_tokens=0,spec_token_ids=[],num_preemptions=0,
        status=NS(name="WAITING_FOR_STREAMING_REQ"),streaming_queue=[],
        mm_features=list(features),block_hashes=[],update_block_hashes=lambda:None)
    state=request._arc3_eviction_state=ee.LogicalState.create(request)
    state.paused(request)
    pages=[NS(block_id=i,ref_cnt=1) for i in range(1,7)]
    group0=NS(kv_cache_spec=UniformTypeKVCacheSpecs({
        "layer.attn":FullAttentionSpec(),
        "layer.compressed_key_cache":MLAAttentionSpec()}),
        layer_names=["layer.attn","layer.compressed_key_cache"])
    group1=NS(kv_cache_spec=MambaSpec(),layer_names=["layer.gdn"])
    managers=[NS(req_to_blocks={request.request_id:list(pages[:5])},num_cached_block={request.request_id:0},
                 _arc3_eviction_states={request.request_id:state},
                 _arc3_has_compressed=True,_arc3_table_capacity=80),
              NS(req_to_blocks={request.request_id:[pages[5]]},num_cached_block={request.request_id:0},
                 _arc3_eviction_states={request.request_id:state},
                 _arc3_has_compressed=False,_arc3_table_capacity=80)]
    events=[]
    released=[]
    def free_blocks(blocks):
        events.append("free")
        for block in blocks:
            if block.ref_cnt!=1:
                raise AssertionError("Page released twice")
            block.ref_cnt=0
            released.append(block)
    coordinator=NS(kv_cache_config=NS(kv_cache_groups=[group0,group1],
        _arc3_worker_kv_cache_groups=copy.deepcopy([group0,group1])),single_type_managers=managers)
    pool=NS(free_blocks=free_blocks)
    scheduler=NS(requests={request.request_id:request},kv_cache_manager=NS(
        coordinator=coordinator,block_pool=pool))
    core=NS(scheduler=scheduler,batch_queue=None,is_scheduler_paused=lambda:True)
    def collective_rpc(function,timeout,args):
        events.append("worker")
        assert function is ee.worker_compact
        assert request.num_computed_tokens==69
        assert request._all_token_ids==list(range(70))
        assert all(block.ref_cnt==1 for block in pages)
        assert args[0]==request.request_id
        wire=args[1]
        events.append("synchronized")
        return [{"plan_digest":wire["plan_digest"],"request_id":request.request_id,
                 "worker_synchronized":True,"copies":[]}]
    core.collective_rpc=collective_rpc
    plan=state.ledger.plan(removed,protected=((66,70),))
    payload=dict(operation_id="op1",expected_logical_tokens=70,
        logical_prefix_sha256=ee.digest(list(range(70))),
        logical_ranges=[list(x) for x in removed],expected_eviction_revision=0,
        resident_before=70,resident_after=plan.resident_after,
        prior_evicted=[],plan_digest=plan.digest)
    return NS(request=request,state=state,core=core,managers=managers,pages=pages,
              events=events,released=released,payload=payload)


class EngineTransactions(unittest.TestCase):
    def setUp(self):
        self.modules=patch.dict(sys.modules,fake_specs())
        self.modules.start()
        self.audit=patch.object(engine_patch,"_audit",lambda *args,**kwargs:None)
        self.audit.start()

    def tearDown(self):
        self.audit.stop()
        self.modules.stop()

    def test_aligned_delete_plans_main_and_descriptor_pages(self):
        f=fixture()
        tx=ee.prepare_transaction(f.core,"example",f.payload)
        self.assertEqual(tx["compact"].new_computed,45)
        self.assertEqual(tx["next_state"].deleted_tokens,24)
        self.assertEqual(tx["next_state"].deleted_groups,6)
        self.assertEqual(tx["next_state"].group_padding,0)
        copies=tx["groups"][0]["copies"]
        self.assertEqual([len(x["plan"].keep_rows) for x in copies],[45,11])
        self.assertEqual([b.block_id for b in tx["groups"][0]["freed_blocks"]],[4,5])
        self.assertEqual(tx["groups"][1]["copies"],[])
        self.assertEqual(tx["groups"][1]["freed_blocks"],())
        self.assertEqual(f.events,[])
        self.assertEqual(f.request._all_token_ids,list(range(70)))

    def test_partial_groups_keep_extra_descriptor_page(self):
        f=fixture(removed=((9,14),))
        tx=ee.prepare_transaction(f.core,"example",f.payload)
        self.assertEqual(tx["compact"].new_computed,64)
        self.assertEqual(tx["next_state"].group_padding,5)
        main,descriptors=tx["groups"][0]["copies"]
        self.assertEqual(len(main["plan"].keep_rows),64)
        self.assertEqual(len(descriptors["plan"].keep_rows),17)
        self.assertEqual(main["plan"].kept_block_ids,(1,2,3,4,5))
        self.assertEqual(tx["groups"][0]["freed_blocks"],())

    def test_page_release_follows_worker_ack_and_reuse_is_real(self):
        f=fixture()
        original_output=f.request._output_token_ids
        ack=ee.evict_resident(f.core,"example",f.payload)
        self.assertEqual(f.events,["worker","synchronized","free"])
        self.assertEqual(ack["freed_blocks_by_group"],[[4,5],[]])
        self.assertEqual((ack["resident_tokens"],ack["evicted_tokens"],ack["logical_tokens"]),(46,24,70))
        self.assertEqual(ack["eviction_revision"],1)
        self.assertIs(f.request._output_token_ids,original_output)
        self.assertEqual(original_output,list(range(62,70)))
        self.assertEqual(f.request.prompt_token_ids,list(range(8))+list(range(32,62)))
        self.assertEqual(f.request.num_computed_tokens,45)
        self.assertEqual(f.request._all_token_ids,list(range(8))+list(range(32,70)))
        state=f.request._arc3_eviction_state
        self.assertFalse(state.poisoned)
        self.assertEqual(state.logical_ids,list(range(70)))
        self.assertEqual(state.context(f.request)["logical_computed"],69)
        self.assertTrue(all(m._arc3_eviction_states[f.request.request_id] is state for m in f.managers))
        # Exercise the allocator ownership transition rather than counting a
        # masked-but-still-owned page as reclaimed memory.
        other_owner=f.released.pop(0)
        other_owner.ref_cnt=1
        f.managers[0].req_to_blocks["other"]=[other_owner]
        own_ids={p.block_id for p in f.managers[0].req_to_blocks[f.request.request_id]}
        self.assertNotIn(other_owner.block_id,own_ids)

    def test_uncertain_worker_outcome_poisoned_without_release(self):
        for failure in (TimeoutError("unknown RPC outcome"),"wrong-digest","not-synchronized"):
            with self.subTest(failure=failure):
                f=fixture()
                def rpc(*args,**kwargs):
                    if isinstance(failure,BaseException):
                        raise failure
                    return [{"plan_digest":"bad" if failure=="wrong-digest" else f.payload["plan_digest"],
                             "worker_synchronized":failure!="not-synchronized"}]
                f.core.collective_rpc=rpc
                with self.assertRaises((ee.EngineEvictionError,TimeoutError)):
                    ee.evict_resident(f.core,"example",f.payload)
                self.assertTrue(f.state.poisoned)
                self.assertEqual(f.released,[])
                self.assertEqual(f.request._all_token_ids,list(range(70)))

    def test_prepared_layout_is_audited_before_worker_rpc(self):
        f=fixture()
        audited=[]
        def audit(scheduler,event,request,**extra):
            if event=="evict_prepare":
                self.assertEqual(f.events,[])
                self.assertFalse(request._arc3_eviction_state.poisoned)
            audited.append((event,extra))
        with patch.object(engine_patch,"_audit",audit):
            ee.evict_resident(f.core,"example",f.payload)
        self.assertEqual([event for event,_ in audited],["evict_prepare","evict"])
        plan=audited[0][1]["eviction_plan"]
        self.assertEqual(plan["plan_digest"],f.payload["plan_digest"])
        self.assertEqual(plan["groups"][0]["copies"][1]["storage_block_size"],4)
        self.assertEqual(plan["groups"][0]["copies"][1]["kind"],"compressed")

    def test_cpu_commit_failure_is_poisoned(self):
        f=fixture()
        def fail_free(blocks):
            raise RuntimeError("simulated allocator failure")
        f.core.scheduler.kv_cache_manager.block_pool.free_blocks=fail_free
        with self.assertRaisesRegex(RuntimeError,"allocator"):
            ee.evict_resident(f.core,"example",f.payload)
        self.assertTrue(f.request._arc3_eviction_state.poisoned)

    def test_stale_and_invalid_plans_fail_before_rpc(self):
        for key,value in (("expected_logical_tokens",69),("logical_prefix_sha256","bad"),
                          ("expected_eviction_revision",1),("prior_evicted",[[1,2]]),
                          ("resident_after",45),("plan_digest","bad")):
            with self.subTest(key=key):
                f=fixture(); f.payload[key]=value
                with self.assertRaises(ee.EngineEvictionError):
                    ee.evict_resident(f.core,"example",f.payload)
                self.assertEqual(f.events,[])
                self.assertFalse(f.state.poisoned)

    def test_runtime_fences_reject_before_rpc(self):
        for reason in ("running","batch_queue","inflight","preempted","queued","shared","null"):
            with self.subTest(reason=reason):
                f=fixture()
                if reason=="running": f.core.is_scheduler_paused=lambda:False
                elif reason=="batch_queue": f.core.batch_queue=[object()]
                elif reason=="inflight": f.request.num_in_flight_tokens=1
                elif reason=="preempted": f.request.num_preemptions=1
                elif reason=="queued": f.request.streaming_queue=[object()]
                elif reason=="shared": f.pages[0].ref_cnt=2
                elif reason=="null": f.pages[0].block_id=0
                with self.assertRaises((ee.EngineEvictionError,engine_patch.PersistentStateError)):
                    ee.evict_resident(f.core,"example",f.payload)
                self.assertEqual(f.events,[])

    def test_preserve_mm_geometry_and_reindex_offsets(self):
        f=fixture(features=(image(2,4),image(12,4),image(40,4)))
        tx=ee.prepare_transaction(f.core,"example",f.payload)
        self.assertEqual([feature.mm_position.offset for feature in tx["new_features"]],[2,16])
        self.assertEqual([feature.mm_position.offset for feature in f.request.mm_features],[2,12,40])
        self.assertEqual([item["offset"] for item in tx["next_state"].logical_image_features],[2,12,40])
        f=fixture(removed=((9,14),),features=(image(12,4),))
        with self.assertRaisesRegex(ee.EngineEvictionError,"multimodal"):
            ee.evict_resident(f.core,"example",f.payload)
        self.assertEqual(f.events,[])

    def test_tail_and_current_output_cannot_be_evicted(self):
        f=fixture()
        f.payload["logical_ranges"]=[[66,68]]
        with self.assertRaises(ValueError):
            ee.prepare_transaction(f.core,"example",f.payload)
        f=fixture(removed=((62,64),))
        with self.assertRaisesRegex(ee.EngineEvictionError,"current output"):
            ee.prepare_transaction(f.core,"example",f.payload)

    def test_state_pause_after_compaction_keeps_original_output_ordinals(self):
        f=fixture()
        ee.evict_resident(f.core,"example",f.payload)
        state=f.request._arc3_eviction_state
        state.paused(f.request)
        self.assertEqual(state.logical_ids,list(range(70)))
        # Next request appends a new logical delta; generated output then grows
        # only the logical tail while the previously removed interval persists.
        state.logical_ids.extend([200,201])
        state.ledger.append(2)
        state.prompt_logical_length=72
        state.prompt_positions=tuple(state.ledger.resident_positions())
        f.request._output_token_ids[:]=[202,203]
        f.request._all_token_ids[:]=[state.logical_ids[p] for p in state.prompt_positions]+[202,203]
        state.paused(f.request)
        self.assertEqual(state.logical_ids,list(range(70))+[200,201,202,203])
        self.assertEqual(state.ledger.logical_tokens,74)
        self.assertEqual(state.ledger.resident_tokens,50)
        self.assertEqual(state.ledger.evicted,((8,32),))

    def test_plugin_component_install_order_and_failure_propagation(self):
        events=[]
        core=ModuleType("engine_eviction")
        worker=ModuleType("eviction_runner")
        diagnostics=ModuleType("eviction_diagnostics")
        def install_core(): events.append("core"); return {"patch":"core"}
        def install_worker(): events.append("worker"); return {"installed":True}
        def install_diagnostics(): events.append("diagnostics"); return {"installed":True}
        core.install=install_core; worker.install_runner_hooks=install_worker
        diagnostics.install_worker_diagnostics=install_diagnostics
        modules={"engine_eviction":core,"eviction_runner":worker,"eviction_diagnostics":diagnostics}
        with patch.dict(sys.modules,modules):
            receipt=engine_patch._install_eviction_components({"patch":"base"})
        self.assertEqual(events,["core","worker","diagnostics"])
        self.assertEqual(receipt["eviction_core"],{"patch":"core"})
        def fail_core(): raise RuntimeError("source guard mismatch")
        core.install=fail_core; events.clear()
        with patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(RuntimeError,"source guard"):
                engine_patch._install_eviction_components({"patch":"base"})
        self.assertEqual(events,[])

    def test_original_and_virtual_turn_limits_are_both_enforced(self):
        f=fixture(removed=((9,14),))
        f.core.scheduler.vllm_config=NS(model_config=NS(hf_text_config=NS(max_position_embeddings=128)))
        state=ee.prepare_transaction(f.core,"example",f.payload)["next_state"]
        self.assertEqual(state.group_padding,5)
        ee.validate_turn_capacity(f.core.scheduler,logical_tokens=70,resident_tokens=65,
                                  max_tokens=10,state=state)
        with self.assertRaisesRegex(ee.EngineEvictionError,"position limit"):
            ee.validate_turn_capacity(f.core.scheduler,logical_tokens=125,resident_tokens=65,
                                      max_tokens=4,state=state)
        with self.assertRaisesRegex(ee.EngineEvictionError,"page-table"):
            ee.validate_turn_capacity(f.core.scheduler,logical_tokens=76,resident_tokens=71,
                                      max_tokens=5,state=state)


if __name__=="__main__":
    unittest.main()
