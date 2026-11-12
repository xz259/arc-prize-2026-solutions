"""Regression for the real scheduler's lossy UniformType normalization."""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import engine_eviction as ee
from test_engine_eviction import (fixture,fake_specs,UniformTypeKVCacheSpecs,
                                 FullAttentionSpec,MLAAttentionSpec)

ROOT=Path(os.environ.get("ARC3_RUNTIME_SOURCE",str(
    Path(__file__).resolve().parent.parent/"persistent_state_20260912"/"source-snapshot"/"vllm")))


def actual_scheduler_generator():
    relative="v1/core/kv_cache_utils.py"
    data=(ROOT/relative).read_bytes()
    if hashlib.sha256(data).hexdigest()!=ee.SOURCE_HASHES[relative]:
        raise AssertionError("Owner-spec regression uses unpinned source")
    tree=ast.parse(data)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                  and node.name=="generate_scheduler_kv_cache_config")
    namespace={"copy":copy,"UniformTypeKVCacheSpecs":UniformTypeKVCacheSpecs}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
        str(ROOT/relative),"exec",flags=__import__("__future__").annotations.compiler_flag),namespace)
    return namespace[function.name]


class PhysicalOwnerSpecs(unittest.TestCase):
    def test_real_stock_collapse_is_preserved_but_physical_plans_remain_distinct(self):
        f=fixture()
        cfg=f.core.scheduler.kv_cache_manager.coordinator.kv_cache_config
        cfg.num_blocks=6
        del cfg._arc3_worker_kv_cache_groups
        original=actual_scheduler_generator()
        collapsed=original([cfg])
        # This is exactly the v2 mistake: both names receive the first full
        # owner's scheduler representative, despite physically different rows.
        self.assertIsInstance(collapsed.kv_cache_groups[0].kv_cache_spec,FullAttentionSpec)
        self.assertEqual(collapsed.kv_cache_groups[0].layer_names,
                         ["layer.attn","layer.compressed_key_cache"])
        with self.assertRaisesRegex(ee.EngineEvictionError,"not preserved"):
            ee.worker_owner_groups(collapsed)
        fixed=ee.preserve_worker_owner_specs(original)([cfg])
        self.assertIsInstance(fixed.kv_cache_groups[0].kv_cache_spec,FullAttentionSpec)
        owners=ee.worker_owner_groups(fixed)[0].kv_cache_spec.kv_cache_specs
        self.assertIsInstance(owners["layer.compressed_key_cache"],MLAAttentionSpec)
        f.core.scheduler.kv_cache_manager.coordinator.kv_cache_config=fixed
        with patch.dict(sys.modules,fake_specs()):
            tx=ee.prepare_transaction(f.core,"example",f.payload)
        main,compressed=tx["groups"][0]["copies"]
        self.assertEqual((main["kind"],main["plan"].storage_block_size),("main",16))
        self.assertEqual((compressed["kind"],compressed["plan"].storage_block_size),("compressed",4))
        self.assertEqual((len(main["plan"].keep_rows),len(compressed["plan"].keep_rows)),(45,11))
        self.assertEqual([b.block_id for b in tx["groups"][0]["freed_blocks"]],[4,5])

    def test_saved_worker_layout_is_deep_copied(self):
        f=fixture()
        cfg=f.core.scheduler.kv_cache_manager.coordinator.kv_cache_config;cfg.num_blocks=6
        fixed=ee.preserve_worker_owner_specs(actual_scheduler_generator())([cfg])
        physical=ee.worker_owner_groups(fixed)
        cfg.kv_cache_groups[0].kv_cache_spec.kv_cache_specs["layer.compressed_key_cache"].storage_block_size=999
        self.assertEqual(physical[0].kv_cache_spec.kv_cache_specs["layer.compressed_key_cache"].storage_block_size,4)
        self.assertEqual(fixed.kv_cache_groups[0].kv_cache_spec.storage_block_size,16)

    def test_owner_names_and_group_geometry_must_match(self):
        for failure in ("group_names","owner_names","block_size","group_count"):
            with self.subTest(failure=failure):
                f=fixture()
                cfg=f.core.scheduler.kv_cache_manager.coordinator.kv_cache_config;cfg.num_blocks=6
                fixed=ee.preserve_worker_owner_specs(actual_scheduler_generator())([cfg])
                physical=fixed._arc3_worker_kv_cache_groups
                if failure=="group_names":physical[0].layer_names.reverse()
                elif failure=="owner_names":physical[0].kv_cache_spec.kv_cache_specs.pop("layer.compressed_key_cache")
                elif failure=="block_size":fixed.kv_cache_groups[0].kv_cache_spec.storage_block_size=32
                else:physical.pop()
                with self.assertRaises(ee.EngineEvictionError):ee.worker_owner_groups(fixed)

    def test_multiple_workers_rejected_before_normalization(self):
        calls=[]
        wrapped=ee.preserve_worker_owner_specs(lambda configs:calls.append(configs))
        for configs in ([],[NS(),NS()]):
            with self.assertRaisesRegex(ee.EngineEvictionError,"one worker"):
                wrapped(configs)
        self.assertEqual(calls,[])


if __name__=="__main__":unittest.main()
