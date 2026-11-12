"""Pinned native getter/validator contract; no runtime imports or plugin scans."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


ROOT = Path(os.environ.get("ARC3_RUNTIME_SOURCE", str(Path(__file__).resolve().parents[2]/
    "persistent_state_20260912/source-snapshot/vllm")))
SOURCE = ROOT/"config/vllm.py"
SOURCE_SHA256 = "b845db7f30bf535e7e1ed5257f5d6513f604f4ed56b5466e2d08986b92f53f53"


def native_contract():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise AssertionError("Native config source changed; review MRV2 pin contract again")
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmConfig")
    names = ("use_v2_model_runner", "_validate_v2_model_runner")
    functions = []
    for name in names:
        fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
        fn.decorator_list = []
        functions.append(fn)
    namespace = dict(envs=NS(VLLM_USE_V2_MODEL_RUNNER=None), HAS_TRITON=True,
                     logger=NS(warning_once=lambda *args: None))
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    post_init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    checks = [n for n in post_init.body if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
              and n.test.attr == "use_v2_model_runner"]
    assert len(checks) == 1
    startup = compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(checks[0])],
                      type_ignores=[])), "native_startup_v2_check", "exec")
    holder = type("ConfigContract", (), {
        "use_v2_model_runner": property(namespace["use_v2_model_runner"]),
        "_validate_v2_model_runner": namespace["_validate_v2_model_runner"]})()
    holder.parallel_config = NS(prefill_context_parallel_size=1)
    holder.speculative_config, holder.model_config = None, NS(is_diffusion=False)
    holder._dflash_needs_multi_kv_group = lambda: False
    holder._is_default_v2_model_runner_model = lambda: True
    holder._get_v2_model_runner_unsupported_features = Mock(return_value=[])
    return holder, namespace, startup, cls


class NativePinTests(unittest.TestCase):
    def test_auto_getter_rescans_but_explicit_pin_has_constant_path(self):
        config, namespace, _, _ = native_contract()
        for _ in range(3):
            self.assertTrue(config.use_v2_model_runner)
        self.assertEqual(config._get_v2_model_runner_unsupported_features.call_count, 3)
        namespace["envs"].VLLM_USE_V2_MODEL_RUNNER = True
        for _ in range(25):
            self.assertTrue(config.use_v2_model_runner)
        self.assertEqual(config._get_v2_model_runner_unsupported_features.call_count, 3)

    def test_pinned_startup_still_runs_unsupported_feature_validation_once(self):
        config, namespace, startup, _ = native_contract()
        namespace["envs"].VLLM_USE_V2_MODEL_RUNNER = True
        exec(startup, {"self": config})
        self.assertEqual(config._get_v2_model_runner_unsupported_features.call_count, 1)
        for _ in range(25):
            self.assertTrue(config.use_v2_model_runner)
        self.assertEqual(config._get_v2_model_runner_unsupported_features.call_count, 1)

    def test_pin_cannot_bypass_unsupported_features_or_triton_requirement(self):
        for missing_triton in (False, True):
            config, namespace, startup, _ = native_contract()
            namespace["envs"].VLLM_USE_V2_MODEL_RUNNER = True
            namespace["HAS_TRITON"] = not missing_triton
            config._get_v2_model_runner_unsupported_features.return_value = ["custom logits processors"]
            expected = "requires Triton" if missing_triton else "does not yet support"
            with self.assertRaisesRegex(ValueError, expected):
                exec(startup, {"self": config})

    def test_unsupported_scan_retains_plugin_entry_point_check(self):
        _, _, _, cls = native_contract()
        scan = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_get_v2_model_runner_unsupported_features")
        imports = [n for n in ast.walk(scan) if isinstance(n, ast.ImportFrom)
                   and n.module == "importlib.metadata"]
        self.assertEqual(len(imports), 1)
        self.assertIn("entry_points", [alias.name for alias in imports[0].names])
        calls = [n for n in ast.walk(scan) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "entry_points"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].keywords[0].value.value, "vllm.logits_processors")

    def test_launcher_pin_defaults_off_and_applies_before_runtime_import(self):
        here = Path(__file__).resolve().parent
        tree = ast.parse((here/"launch_scaling.py").read_text())
        conditional = next(n for n in tree.body if isinstance(n, ast.If)
                           and "pin_v2_runner" in ast.unparse(n.test))
        runtime_import = next(n for n in tree.body if isinstance(n, ast.Import)
                              and any(alias.name == "minimal_inference" for alias in n.names))
        self.assertLess(conditional.lineno, runtime_import.lineno)
        code = compile(ast.fix_missing_locations(ast.Module(body=[conditional], type_ignores=[])),
                       "launch_pin", "exec")
        env = {}
        exec(code, {"config": {}, "os": NS(environ=env)})
        self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", env)
        config = json.loads((here/"optimization-config.json").read_text())
        self.assertEqual(config["pin_v2_runner"], 1)
        exec(code, {"config": config, "os": NS(environ=env)})
        self.assertEqual(env["VLLM_USE_V2_MODEL_RUNNER"], "1")


if __name__ == "__main__":
    unittest.main()
