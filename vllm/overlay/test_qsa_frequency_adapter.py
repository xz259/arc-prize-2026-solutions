"""Phase 5: adapter config passthrough and memory reminder for qsa_frequency.

The adapter module pulls in the whole gameplay client at import time, which is
not present on the server. The two edited functions are pure, so they are
extracted with ast and executed in a clean namespace: this tests the code that
changed, not the client import chain.
"""
import ast, math, types, unittest
from pathlib import Path

SOURCE = (Path(__file__).resolve().parent / 'persistent_harness_adapter.py').read_text()
WANTED = {'HISTORY_MEMORY_REMINDER', 'history_eviction_config', '_pin_memory_reminder'}
module = ast.parse(SOURCE)
selected = [node for node in module.body
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in WANTED)
            or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANTED for t in node.targets))]
assert {getattr(n, 'name', None) or n.targets[0].id for n in selected} == WANTED, 'adapter functions not found'


class AppendOnlyViolation(Exception):
    pass


ns = {'math': math, 'AppendOnlyViolation': AppendOnlyViolation}
exec(compile(ast.Module(body=selected, type_ignores=[]), 'adapter-extract', 'exec'), ns)
history_eviction_config, _pin_memory_reminder, REMINDER = ns['history_eviction_config'], ns['_pin_memory_reminder'], ns['HISTORY_MEMORY_REMINDER']


class AdapterTests(unittest.TestCase):
    def test_config_passes_new_policy_and_recent_protection(self):
        c = history_eviction_config({"policy": "qsa_frequency", "fraction": 0.2, "protect_recent_fraction": 0.2})
        self.assertEqual(c, {"policy": "qsa_frequency", "fraction": 0.2, "protect_recent_fraction": 0.2})
        self.assertEqual(history_eviction_config({"policy": "oldest_interactions", "fraction": 0.1}),
                         {"policy": "oldest_interactions", "fraction": 0.1})
        self.assertEqual(history_eviction_config(None), {"policy": "off", "fraction": 0.1})
        for bad in ({"policy": "qsa_frequency", "fraction": 0.2, "protect_recent_fraction": 1.0},
                    {"policy": "qsa_frequency", "fraction": 0.2, "protect_recent_fraction": -0.5},
                    {"policy": "qsa_frequency", "fraction": 0.2, "extra": 1},
                    {"policy": "unknown", "fraction": 0.2}):
            with self.assertRaises(ValueError):
                history_eviction_config(bad)

    def test_memory_reminder_pinned_for_both_eviction_policies(self):
        for policy, expected in (("off", False), ("oldest_interactions", True), ("qsa_frequency", True)):
            agent = types.SimpleNamespace(_system_prompt="SYSTEM")
            _pin_memory_reminder(agent, {"policy": policy})
            self.assertEqual(REMINDER in agent._system_prompt, expected, policy)
            self.assertTrue(agent._system_prompt.startswith("SYSTEM"))
            _pin_memory_reminder(agent, {"policy": policy})      # idempotent on the pinned prompt
            agent._system_prompt = "CHANGED"
            with self.assertRaises(AppendOnlyViolation):
                _pin_memory_reminder(agent, {"policy": policy})


if __name__ == '__main__':
    unittest.main(verbosity=1)
