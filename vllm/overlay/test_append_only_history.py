"""CPU-only policy tests; they do not establish KV/GDN state correctness."""
import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from append_only_history import (AppendOnlyMessageState, AppendOnlyViolation,
                                CapacityExceeded, assert_token_append, observation_base)
from gameplay_config import GameplayConfig
from prolong_images import image_part
from prolong_program_suffix import BASE_KEY as PROGRAM_BASE, refresh_program_suffix
from prolong_snapshots import BASE_KEY as SNAPSHOT_BASE, refresh_snapshot_suffix


def workspace(n=16):
    def frame(step):
        return dict(ascii="RR\nRR", shape=[2, 2], step=step, level=1)
    actions = {i: dict(frame=frame(i), before_frame=frame(i-1), level_before=1,
                      level_after=1, attempt_before=1, attempt_after=1,
                      result=dict(action_display="RIGHT", action_num=i))
               for i in range(1, n+1)}
    programs = SimpleNamespace(value="PROGRAMS: initial_helper")
    programs.text = lambda: programs.value
    return SimpleNamespace(actions=actions, observations={}, current_level=1,
                           latest_action=n, scale=4, program_index=programs)


def observation(ws, *, identity="state-1"):
    return dict(role="user", _arc3_observation=True, _arc3_state_key=identity,
                _arc3_snapshot_upto=ws.latest_action, _arc3_snapshot_level=1,
                content=[dict(type="text", text="Current board; RIGHT available."),
                         image_part([[2, 2], [2, 2]], 4)])


def texts(message):
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(
        part["text"] for part in content if part["type"] == "text")


class AppendOnlyTests(unittest.TestCase):
    def setUp(self):
        self.ws = workspace()
        self.policy = AppendOnlyMessageState()
        self.messages = [dict(role="system", content="Solve all game levels."), observation(self.ws)]
        # Deliberately a fixture count. Serving integration must provide its exact
        # tokenizer/template/multimodal count; JSON length is never used as tokens.
        self.count = lambda messages: 100

    def prepare(self, messages=None, **kwargs):
        args = dict(workspace=self.ws, count=self.count, token_limit=40000,
                    renderer_fingerprint="pinned-renderer-and-tools-and-image-config")
        args.update(kwargs)
        return self.policy.prepare(self.messages if messages is None else messages, **args)

    def start(self):
        prepared = self.prepare()
        self.policy.commit(prepared)
        return prepared.messages()

    def test_initial_suffixes_use_existing_builders_in_original_order(self):
        prepared = self.prepare()
        expected = refresh_snapshot_suffix(
            refresh_program_suffix(self.messages, self.ws.program_index.text()), self.ws)
        self.assertEqual(prepared.messages(), expected)
        message = prepared.messages()[1]
        self.assertIn(PROGRAM_BASE, message)
        self.assertIn(SNAPSHOT_BASE, message)
        self.assertLess(texts(message).index("PROGRAMS"), texts(message).index("RECENT"))
        self.assertEqual(sum(p["type"] == "image_url" for p in message["content"]), 13)

    def test_prepare_does_not_commit_or_mutate_input(self):
        original = copy.deepcopy(self.messages)
        prepared = self.prepare()
        self.assertEqual(self.messages, original)
        self.assertEqual(self.policy.committed_messages(), [])
        self.assertEqual(prepared.newly_added_messages, 2)

    def test_library_updates_after_inspection_wait_until_next_observation(self):
        first = self.start()
        self.ws.program_index.value = "PROGRAMS: initial_helper new_function"
        exchange = [dict(role="assistant", content="Inspecting", tool_calls=[dict(
            id="call-1", type="function", function=dict(name="python", arguments='{"code":"pass"}'))]),
                    dict(role="tool", tool_call_id="call-1", content="Computed successfully.")]
        followup = self.prepare([*first, *exchange])
        self.assertEqual(followup.messages()[1], first[1])
        self.assertNotIn("new_function", texts(followup.messages()[1]))
        self.assertEqual(followup.messages()[2:], exchange)
        self.policy.commit(followup)
        later = self.prepare([*followup.messages(), observation(self.ws, identity="state-2")])
        self.assertEqual(later.messages()[:len(first)], first)
        self.assertIn("new_function", texts(later.messages()[-1]))
        self.assertEqual(sum(PROGRAM_BASE in m for m in later.messages()), 2)
        self.assertEqual(sum(SNAPSHOT_BASE in m for m in later.messages()), 2)

    def test_even_if_caller_refreshes_old_suffix_committed_version_wins(self):
        first = self.start()
        self.ws.program_index.value = "PROGRAMS: newly_discovered"
        from prolong_snapshots import remove_snapshot_suffix
        refreshed = refresh_snapshot_suffix(refresh_program_suffix(
            remove_snapshot_suffix(first), self.ws.program_index.text()), self.ws)
        self.assertNotEqual(refreshed[1], first[1])
        self.assertEqual(self.prepare(refreshed).messages(), first)

    def test_undecorated_old_observation_restores_exact_committed_copy(self):
        first = self.start()
        self.assertEqual(self.prepare(self.messages).messages(), first)

    def test_new_identical_board_with_new_identity_is_not_deduplicated(self):
        first = self.start()
        self.ws.program_index.value = "PROGRAMS: learned_reset"
        later = self.prepare([*first, observation(self.ws, identity="reset-state")])
        self.assertEqual(later.newly_added_messages, 1)
        self.assertIn("learned_reset", texts(later.messages()[-1]))
        self.assertEqual(later.messages()[1], first[1])

    def test_old_snapshot_is_not_rerendered_after_workspace_changes(self):
        first = self.start()
        self.ws.actions.clear()
        self.ws.latest_action = 0
        self.ws.current_level = 2
        self.assertEqual(self.prepare(first).messages()[1], first[1])

    def test_private_fields_are_not_sent_via_production_serializer(self):
        prepared = self.prepare()
        local = prepared.messages()
        self.assertIn("_arc3_state_key", local[1])
        wire = prepared.wire_messages()
        self.assertNotIn("_arc3_", json.dumps(wire))
        payload = GameplayConfig(min_p=0).chat_payload("model", local, [], max_tokens=None)
        self.assertEqual(payload["messages"], wire)
        self.assertNotIn("_arc3_", json.dumps(payload))

    def test_external_copy_mutation_does_not_change_prepared_or_committed_state(self):
        prepared = self.prepare()
        original = prepared.messages()
        detached = prepared.messages()
        detached[1]["content"][0]["text"] = "modified"
        detached[1]["content"][1]["image_url"]["url"] = "changed-image"
        self.assertEqual(prepared.messages(), original)
        self.policy.commit(prepared)
        committed = self.policy.committed_messages()
        committed[0]["content"] = "rewritten policy"
        self.assertEqual(self.policy.committed_messages(), original)

    def test_prepared_dataclass_is_frozen(self):
        prepared = self.prepare()
        with self.assertRaises(FrozenInstanceError):
            prepared.input_tokens = 1

    def test_failed_submission_does_not_freeze_unaccepted_suffix(self):
        failed = self.prepare()
        self.ws.program_index.value = "PROGRAMS: before_first_acceptance"
        retried = self.prepare()
        self.assertNotEqual(failed.messages(), retried.messages())
        self.policy.commit(retried)
        with self.assertRaises(AppendOnlyViolation):
            self.policy.commit(failed)

    def test_double_commit_rejected(self):
        prepared = self.prepare()
        self.policy.commit(prepared)
        with self.assertRaises(AppendOnlyViolation):
            self.policy.commit(prepared)

    def test_cross_game_commit_rejected(self):
        prepared = self.prepare()
        with self.assertRaises(AppendOnlyViolation):
            AppendOnlyMessageState().commit(prepared)

    def test_stale_concurrent_preparation_rejected(self):
        older, newer = self.prepare(), self.prepare()
        self.policy.commit(older)
        with self.assertRaises(AppendOnlyViolation):
            self.policy.commit(newer)

    def test_configuration_change_requires_explicit_new_session(self):
        first = self.start()
        with self.assertRaisesRegex(AppendOnlyViolation, "configuration changed"):
            self.prepare(first, renderer_fingerprint="different-tools-or-template")

    def test_system_rewrite_refused(self):
        first = self.start()
        first[0]["content"] += " Old context was removed."
        with self.assertRaisesRegex(AppendOnlyViolation, "message 0 changed"):
            self.prepare(first)

    def test_observation_board_text_or_identity_rewrite_refused(self):
        self.start()
        for alter in (lambda m: m[1]["content"][0].update(text="Changed board"),
                      lambda m: m[1].update(_arc3_state_key="another-state"),
                      lambda m: m[1]["content"][1]["image_url"].update(url="changed-image")):
            candidate = copy.deepcopy(self.messages)
            alter(candidate)
            with self.assertRaises(AppendOnlyViolation):
                self.prepare(candidate)

    def test_private_wrapper_cannot_hide_original_content_mutation(self):
        first = self.start()
        first[1]["content"][0]["text"] = "Different current board"
        with self.assertRaisesRegex(AppendOnlyViolation, "base no longer matches"):
            self.prepare(first)

    def test_assistant_normalization_or_argument_rewrite_refused(self):
        first = self.start()
        exchange = [dict(role="assistant", reasoning="exact reasoning", tool_calls=[dict(
            id="c", type="function", function=dict(name="python", arguments="malformed"))])]
        prepared = self.prepare([*first, *exchange])
        self.policy.commit(prepared)
        modified = prepared.messages()
        modified[-1]["tool_calls"][0]["function"]["arguments"] = "{}"
        with self.assertRaises(AppendOnlyViolation):
            self.prepare(modified)

    def test_open_reasoning_growth_requires_explicit_token_reconciliation(self):
        first = self.start()
        opened = self.prepare([*first, dict(role="assistant", content=None,
                                           reasoning="thinking", _arc3_open_reasoning=True)])
        self.policy.commit(opened)
        continued = opened.messages()
        continued[-1]["reasoning"] += " further"
        with self.assertRaises(AppendOnlyViolation):
            self.prepare(continued)

    def test_recovery_user_message_does_not_get_suffix(self):
        first = self.start()
        recovery = dict(role="user", content="Please use a valid call.", _arc3_recovery=True)
        prepared = self.prepare([*first, recovery])
        self.assertEqual(prepared.messages()[-1], recovery)
        self.assertEqual(prepared.messages()[1], first[1])

    def test_deleted_history_refused(self):
        self.start()
        with self.assertRaisesRegex(AppendOnlyViolation, "removed"):
            self.prepare(self.messages[:1])

    def test_multiple_unseen_observations_refused(self):
        with self.assertRaisesRegex(AppendOnlyViolation, "multiple unseen"):
            self.prepare([*self.messages, observation(self.ws, identity="second")])

    def test_full_count_receives_exact_candidate_with_images_and_private_control(self):
        seen = []
        prepared = self.prepare(count=lambda messages: seen.append(copy.deepcopy(messages)) or 123)
        self.assertEqual(seen, [prepared.messages()])
        self.assertEqual(prepared.input_tokens, 123)

    def test_counter_cannot_mutate_candidate(self):
        def count(messages):
            messages[0]["content"] = "bad counter mutation"
            return 100
        prepared = self.prepare(count=count)
        self.assertEqual(prepared.messages()[0], self.messages[0])

    def test_capacity_refusal_keeps_all_original_history_and_state(self):
        first = self.start()
        candidate = [*first, dict(role="tool", content="large output", tool_call_id="c")]
        original = copy.deepcopy(candidate)
        with self.assertRaises(CapacityExceeded) as raised:
            self.prepare(candidate, count=lambda messages: 40000)
        self.assertEqual(raised.exception.input_tokens, 40000)
        self.assertEqual(raised.exception.required_free_tokens, 1)
        self.assertEqual(self.policy.committed_messages(), first)
        self.assertEqual(candidate, original)

    def test_capacity_boundary_reserves_one_token_without_10_percent_headroom(self):
        self.assertEqual(self.prepare(count=lambda messages: 39999).input_tokens, 39999)
        self.assertEqual(self.prepare(count=lambda messages: 40000,
                                      required_free_tokens=0).input_tokens, 40000)

    def test_disabled_options_are_real_opt_in_choices(self):
        for program, snapshot in ((False, False), (True, False), (False, True)):
            policy = AppendOnlyMessageState(program_enabled=program, snapshot_enabled=snapshot)
            prepared = policy.prepare(self.messages, workspace=self.ws, count=self.count,
                                      token_limit=40000, renderer_fingerprint="fixed")
            self.assertEqual(PROGRAM_BASE in prepared.messages()[1], program)
            self.assertEqual(SNAPSHOT_BASE in prepared.messages()[1], snapshot)

    def test_existing_default_latest_only_helpers_are_unchanged(self):
        first = refresh_snapshot_suffix(refresh_program_suffix(self.messages, "OLD"), self.ws)
        default = refresh_snapshot_suffix(refresh_program_suffix(
            __import__("prolong_snapshots").remove_snapshot_suffix(
                [*first, observation(self.ws, identity="second")]), "NEW"), self.ws)
        self.assertNotIn(PROGRAM_BASE, default[1])
        self.assertNotIn(SNAPSHOT_BASE, default[1])
        self.assertEqual(observation_base(first[1]), default[1])

    def test_invalid_configuration_or_counter_rejected(self):
        for kwargs in (dict(token_limit=True), dict(required_free_tokens=-1),
                       dict(token_limit=0), dict(renderer_fingerprint=""),
                       dict(count=lambda messages: 1.5), dict(count=lambda messages: True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)


class ExactTokenPrefixTests(unittest.TestCase):
    def test_identical_prefix_returns_only_appended_ids(self):
        self.assertEqual(assert_token_append([1, 2, 3], [1, 2, 3, 4, 5]), (4, 5))

    def test_identical_stream_has_no_new_prefill(self):
        self.assertEqual(assert_token_append([1, 2], [1, 2]), ())

    def test_initial_stream_is_all_append(self):
        self.assertEqual(assert_token_append([], [1, 2]), (1, 2))

    def test_generation_token_must_be_part_of_acknowledged_prefix(self):
        with self.assertRaisesRegex(AppendOnlyViolation, "position 2"):
            assert_token_append([1, 2, 91], [1, 2, 92, 3])

    def test_shortening_or_noninteger_ids_rejected(self):
        with self.assertRaises(AppendOnlyViolation):
            assert_token_append([1, 2], [1])
        for bad in ([True], [-1], [1.0], ["1"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                assert_token_append([], bad)


class PinnedTextTemplateTests(unittest.TestCase):
    """Real local tokenizer/template coverage, deliberately text-only.

    Multimodal expansion and generated IDs require the live serving adapter.
    """

    @classmethod
    def setUpClass(cls):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        root = Path(__file__).resolve().parents[2]
        tokenizer = root / "evidence/optimization-20260909/kernel-checkpoint-v1/model-metadata/tokenizer.json"
        if not tokenizer.is_file():
            raise unittest.SkipTest("Pinned tokenizer evidence is not present")
        cls.tokenizer = Tokenizer.from_file(str(tokenizer))
        env = ImmutableSandboxedEnvironment()
        def fail(message):
            raise ValueError(message)
        env.globals["raise_exception"] = fail
        cls.template = env.from_string((root / "prolong_chat_template.jinja").read_text())

    def tokens(self, messages):
        payload = GameplayConfig(min_p=0).chat_payload("model", messages, [], max_tokens=None)
        rendered = self.template.render(messages=payload["messages"],
                                        **payload["chat_template_kwargs"],
                                        add_generation_prompt=True)
        return self.tokenizer.encode(rendered, add_special_tokens=False).ids

    def test_exact_text_render_prefix_survives_inspection_and_new_observation(self):
        ws = workspace()
        state = AppendOnlyMessageState(snapshot_enabled=False)
        obs = dict(role="user", content="Board described as text for this fixture.",
                   _arc3_observation=True, _arc3_state_key="one")
        args = dict(workspace=ws, count=lambda m: len(self.tokens(m)), token_limit=40000,
                    renderer_fingerprint="real-local-text-template-tokenizer")
        first = state.prepare([dict(role="system", content="Solve."), obs], **args)
        state.commit(first)
        first_tokens = self.tokens(first.messages())
        ws.program_index.value = "PROGRAMS: new_helper"
        exchange = [dict(role="assistant", reasoning_content="Inspect.", content="Check results."),
                    dict(role="tool", content="Function defined.", tool_call_id="one")]
        second = state.prepare([*first.messages(), *exchange], **args)
        self.assertEqual(second.input_tokens, len(self.tokens(second.messages())))
        self.assertTrue(assert_token_append(first_tokens, self.tokens(second.messages())))
        state.commit(second)
        later = state.prepare([*second.messages(), dict(role="assistant", content="Act now.",
                               reasoning_content="The result is useful."),
                               dict(obs, _arc3_state_key="two")], **args)
        self.assertTrue(assert_token_append(self.tokens(second.messages()), self.tokens(later.messages())))
        self.assertIn("new_helper", texts(later.messages()[-1]))
        self.assertEqual(later.messages()[1], first.messages()[1])

    def test_template_normalization_is_not_exact_generated_token_replay(self):
        messages = [dict(role="system", content="Solve."),
                    dict(role="user", content="Board.")]
        prefix = self.template.render(messages=messages, add_generation_prompt=True,
                                      enable_thinking=True, preserve_thinking=True)
        # A legal generated trace can retain whitespace that the historical
        # template strips. Reencoding the normalized message changes its tokens.
        actual = prefix + "  Inspect the board.  \n</think>\n\nDone.<|im_end|>\n"
        normalized = [*messages, dict(role="assistant", reasoning_content="  Inspect the board.  ",
                                     content="Done.")]
        actual_ids = self.tokenizer.encode(actual, add_special_tokens=False).ids
        reencoded_ids = self.tokens(normalized)
        with self.assertRaises(AppendOnlyViolation):
            assert_token_append(actual_ids, reencoded_ids)


if __name__ == "__main__":
    unittest.main()
