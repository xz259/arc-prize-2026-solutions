from __future__ import annotations

import copy
from types import ModuleType,SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import eviction_diagnostics as ed


class Cache:
    def __init__(self,shape,stride,dtype,pointer):
        self.shape=tuple(shape);self._stride=tuple(stride);self.dtype=dtype
        self.device="cuda:0";self._pointer=pointer
    def stride(self):return self._stride
    def storage_offset(self):return 0
    def data_ptr(self):return self._pointer


class ReadOnlyDiagnostics(unittest.TestCase):
    def test_bound_views_are_matched_without_reading_values(self):
        main=Cache((609,2,2768,296),(1638656,819328,296,1),"torch.uint8",1000)
        comp=Cache((609,692,1,128),(88576,128,128,1),"torch.bfloat16",2000)
        forward={"a.attn":NS(kv_cache=main),"a.compressed_key_cache":NS(kv_cache=comp)}
        spec=NS(block_size=2768,kv_cache_specs={
            "a.attn":NS(block_size=2768,storage_block_size=2768,dtype="torch.uint8"),
            "a.compressed_key_cache":NS(block_size=2768,storage_block_size=692,compress_ratio=4,
                                         dtype="torch.bfloat16")})
        config=NS(compilation_config=NS(static_forward_context=forward))
        runner=NS(vllm_config=config,compilation_config=config.compilation_config,
                  kv_caches=[main,comp],kernel_block_sizes=[2768],
                  kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=spec,layer_names=list(forward))]))
        before=copy.deepcopy((main.__dict__,comp.__dict__,list(forward),runner.kernel_block_sizes))
        result=ed.arc3_eviction_layout(NS(model_runner=runner))
        self.assertTrue(result["read_only"])
        self.assertFalse(result["tensor_values_read"])
        self.assertTrue(result["same_compilation_forward_context"])
        self.assertEqual(result["bound_owners"][0]["runner_cache_identity_indices"],[0])
        self.assertEqual(result["bound_owners"][1]["runner_cache_same_view_indices"],[1])
        self.assertEqual(result["groups"][0]["owners"][1]["spec"]["storage_block_size"],692)
        self.assertEqual(result["bound_owners"][0]["cache"]["stride"],list(main.stride()))
        self.assertEqual((main.__dict__,comp.__dict__,list(forward),runner.kernel_block_sizes),before)

    def test_non_tensor_and_state_lists_are_reported_without_assuming_dimensions(self):
        self.assertEqual(ed.describe_cache(None),{"type":"NoneType"})
        cache=Cache((3,4),(4,1),"torch.float32",100)
        result=ed.describe_cache([cache,None])
        self.assertEqual(result["items"][0]["shape"],[3,4])
        self.assertEqual(result["items"][1],{"type":"NoneType"})

    def test_worker_method_install_is_idempotent(self):
        class WorkerBase:pass
        module=ModuleType("vllm.v1.worker.worker_base");module.WorkerBase=WorkerBase
        with patch.dict(sys.modules,{module.__name__:module}):
            first=ed.install_worker_diagnostics()
            method=WorkerBase.arc3_eviction_layout
            second=ed.install_worker_diagnostics()
            self.assertIs(method,WorkerBase.arc3_eviction_layout)
            self.assertTrue(first["installed"])
            self.assertTrue(second["already_installed"])
            WorkerBase._arc3_eviction_layout_version="unknown"
            with self.assertRaises(RuntimeError):ed.install_worker_diagnostics()


if __name__=="__main__":unittest.main()
