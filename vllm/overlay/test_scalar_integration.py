"""Execute real runner publication code with CPU-only tensor/source fixtures."""
import ast
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import numpy as np
import eviction_runner as er
import scalar_publication as pub
from test_eviction_runner import context, runner_fixture


class ArrayTensor:
    def __init__(self, data, torch):
        self.data, self.torch = data, torch
        self.device, self.dtype, self.shape = "cpu", data.dtype, data.shape
    def data_ptr(self):
        return self.data.__array_interface__["data"][0]
    def stride(self):
        return tuple(n//self.data.itemsize for n in self.data.strides)
    def __getitem__(self, index):
        return ArrayTensor(self.data[index], self.torch)
    def __setitem__(self, index, value):
        self.torch.scalar_writes.append((self, index, value))
        self.data[index] = value
    def copy_(self, other):
        if self.torch.fail_copy:
            raise RuntimeError("lookup transfer error")
        self.data[...] = other.data
        return self


class Torch:
    int32, int64 = np.dtype("int32"), np.dtype("int64")
    def __init__(self):
        self.scalar_writes, self.fail_copy = [], False
    def zeros(self, size, *, dtype, device):
        return ArrayTensor(np.zeros(size, dtype=dtype), self)
    def full(self, size, value, *, dtype, device):
        return ArrayTensor(np.full(size, value, dtype=dtype), self)
    def tensor(self, value, *, dtype, device):
        return ArrayTensor(np.asarray(value, dtype=dtype), self)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {pub.FLAG: "1", "ARC3_EFFICIENCY_IDENTITY_LOOKUP": "1",
            "ARC3_EVICTION_GRAPH_MAPPING": "1", "ARC3_EVICTION_GRAPH_TRACE": "1"})
        env.start(); self.addCleanup(env.stop)
        self.torch = Torch()
        modules = patch.dict(sys.modules, {"torch": self.torch})
        modules.start(); self.addCleanup(modules.stop)
        self.runner = runner_fixture()
        self.runner.model_state.device = "cpu"
        self.runner.model_state.max_num_reqs = 2
        self.runner.cudagraph_manager = None
        er._allocate_buffers(self.runner)
        self.buffers = self.runner.model_state._arc3_eviction_buffers

    def stage(self, value, slot=0):
        data = NS(req_id="game", prefill_token_ids=[value["logical_token_ids"][p]
            for p in value["resident_positions"]],
            num_computed_tokens=sum(p < value["logical_computed"] for p in value["resident_positions"]))
        return er._stage_request(self.runner, slot, data, value)

    def test_identity_admission_and_continuation_real_stage_path_skip_writes(self):
        for length in (50, 60, 70):
            value = context(length=length, deleted=())
            rows, delta = self.stage(value)
            self.assertEqual(rows, [list(range(length))]*3)
            self.assertEqual(delta, 0)
        self.assertEqual(self.torch.scalar_writes, [])
        self.assertEqual(pub.status(self.buffers)["counts"]["published"], 3)

    def test_real_eviction_cpu_publish_does_not_touch_scalar_shadow(self):
        self.stage(context(deleted=()))
        value = context()
        prepared = er.prepare_worker_eviction(self.runner, "game", value, ((1, 2),))
        before = copy.deepcopy(pub.status(self.buffers))
        er.apply_worker_eviction(self.runner, "game", prepared)
        self.assertEqual(pub.status(self.buffers), before)
        rows, delta = self.stage(value)
        table, cutoff, padding = er.historical_virtual_lookup(value)
        self.assertEqual(self.buffers.lookup.data[0, :cutoff].tolist(), table)
        self.assertEqual(self.buffers.deleted.data.tolist(), [6, 0])
        self.assertEqual(self.buffers.padding.data.tolist(), [padding, 0])
        self.assertEqual(self.buffers.cutoffs.data.tolist(), [cutoff, 0])
        self.assertEqual(rows, [value["resident_positions"]]*3)
        self.assertEqual(delta, 6)

    def test_lookup_error_in_real_stage_forces_full_republication(self):
        value = context()
        self.torch.fail_copy = True
        with self.assertRaisesRegex(RuntimeError, "lookup transfer"):
            self.stage(value)
        self.assertIsNone(getattr(self.buffers, pub.STATE_ATTR).slots[0])
        self.assertEqual(self.runner._arc3_eviction_states, {})
        self.torch.fail_copy = False
        count = len(self.torch.scalar_writes)
        self.stage(value)
        self.assertEqual(len(self.torch.scalar_writes)-count, 3)

    def test_actual_stateless_hook_resets_reused_physical_slot(self):
        self.stage(context())
        tree = ast.parse(Path(er.__file__).read_text())
        install = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "install_runner_hooks")
        fn = copy.deepcopy(next(n for n in install.body if isinstance(n, ast.FunctionDef)
                                and n.name == "add_request"))
        fn.decorator_list = []
        namespace = dict(er.__dict__, original_add=lambda *a: "native-add")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                     "actual_stateless_hook", "exec"), namespace)
        self.runner._arc3_eviction_states.pop("game")
        result = namespace["add_request"](self.runner.model_state, 0,
            NS(req_id="new-stateless", sampling_params=NS(extra_args={})))
        self.assertEqual(result, "native-add")
        self.assertEqual([getattr(self.buffers, name).data[0] for name in pub.FIELDS], [0, 0, 0])
        count = len(self.torch.scalar_writes)
        self.stage(context(deleted=()))
        self.assertEqual(len(self.torch.scalar_writes), count)

    def test_graph_status_returns_only_cpu_shadow_counts(self):
        self.stage(context(deleted=()))
        status = er.arc3_eviction_graph_status(NS(model_runner=self.runner))
        self.assertTrue(status["scalar_publication"]["enabled"])
        self.assertEqual(status["scalar_publication"]["counts"]["issued_writes"], 0)
        self.assertEqual(status["scalar_publication"]["counts"]["skipped_writes"], 3)
        json.dumps(status)

    def test_control_mode_real_stage_remains_unconditional(self):
        with patch.dict(os.environ, {pub.FLAG: "0"}):
            self.stage(context(deleted=()))
            self.stage(context(length=60, deleted=()))
        self.assertEqual(len(self.torch.scalar_writes), 6)


class LaunchTests(unittest.TestCase):
    def test_proof_config_and_missing_key_defaults_to_zero(self):
        here = Path(__file__).resolve().parent
        config = json.loads((here/"optimization-config.json").read_text())
        self.assertEqual(config["label"], "batch-proof-v5")
        self.assertEqual([config[k] for k in ("skip_unchanged_scalars", "verify_bytes", "verify_allocator")], [1, 1, 1])
        tree = ast.parse((here/"launch_scaling.py").read_text())
        value = next(kw.value for n in ast.walk(tree) if isinstance(n, ast.Call)
            for kw in n.keywords if kw.arg == "ARC3_EFFICIENCY_SKIP_UNCHANGED_SCALARS")
        expression = compile(ast.Expression(body=value), "launch_flag", "eval")
        self.assertEqual(eval(expression, {"config": {}}), "0")
        self.assertEqual(eval(expression, {"config": config}), "1")


if __name__ == "__main__":
    unittest.main()
