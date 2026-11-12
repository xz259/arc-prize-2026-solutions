"""Local tests of the ARC-2 v0 client (no server): tasks and scoring, the kernel and its helpers, the turn loop on the
fake session, the runner end to end, and the ARC-3 transport import with the ARC-2 texts.

    cd arc2_stack/harness && python -m unittest -v tests.test_arc2
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from arc2 import tasks as T                                       # noqa: E402
from arc2.agent import run_attempt                                # noqa: E402
from arc2.fake import FakeSession                                 # noqa: E402
from arc2.kernel import Kernel                                    # noqa: E402
from arc2.prompts import load_texts, opener_message, system_message  # noqa: E402

TRAIN_CH = T.DATA_DIR / "arc-agi_training_challenges.json"
TRAIN_SOL = T.DATA_DIR / "arc-agi_training_solutions.json"
FLIPLR = "67a3c6ac"          # training task solved by np.fliplr (found by scanning the training set)
ROT180 = "3c9b0459"


def train_task(task_id):
    return T.load_tasks(TRAIN_CH, TRAIN_SOL, [task_id])[0]


class TestTasks(unittest.TestCase):
    def test_eval_set_and_perfect_score(self):
        tasks = T.load_tasks(T.EVAL_CHALLENGES, T.EVAL_SOLUTIONS)
        self.assertEqual(len(tasks), 120)
        self.assertEqual(sum(len(t.test) for t in tasks), 172)
        perfect = {t.task_id: T.submission_entry(t, {1: t.solutions}) for t in tasks}
        total, per = T.score_submission(tasks, perfect)
        self.assertEqual(total, 120)
        empty = {t.task_id: T.placeholder_entry(t) for t in tasks}
        self.assertEqual(T.score_submission(tasks, empty)[0], 0)

    def test_competition_metric_weights_outputs(self):
        tasks = T.load_tasks(T.EVAL_CHALLENGES, T.EVAL_SOLUTIONS)
        multi = next(t for t in tasks if len(t.test) == 2)
        single = next(t for t in tasks if len(t.test) == 1)
        sub = {single.task_id: [{"attempt_1": T.PLACEHOLDER, "attempt_2": T.to_list(single.solutions[0])}],
               multi.task_id: [{"attempt_1": T.to_list(multi.solutions[0]), "attempt_2": T.PLACEHOLDER},
                               {"attempt_1": T.PLACEHOLDER, "attempt_2": T.PLACEHOLDER}]}
        self.assertEqual(T.score_outputs([single, multi], sub), (2, 3))
        self.assertEqual(T.score_submission([single, multi], sub)[0], 1.5)

    def test_second_attempt_and_partial_tasks(self):
        t = next(t for t in T.load_tasks(T.EVAL_CHALLENGES, T.EVAL_SOLUTIONS) if len(t.test) == 2)
        entry = T.submission_entry(t, {1: [[[0]], [[0]]], 2: [t.solutions[0], [[0]]]})
        self.assertEqual(T.score_task(t, entry), 0.5)

    def test_sandbox_payload_has_no_solutions(self):
        t = T.load_tasks(T.EVAL_CHALLENGES, T.EVAL_SOLUTIONS)[0]
        payload = json.dumps(t.sandbox_payload())
        for p in t.test:
            self.assertNotIn("output", json.dumps(p))
        self.assertNotIn(json.dumps(t.solutions[0]), payload)

    def test_grid_text_formats(self):
        self.assertEqual(T.grid_text([[0, 1], [9, 8]]), "Bb\nrS")
        self.assertEqual(T.grid_text([[0, 1], [9, 8]], "digits"), "01\n98")
        self.assertTrue(set(T.LETTERS) <= set("WwgGcBMPRbSYOrNp"))       # ARC-3's symbols only
        self.assertEqual(len(set(T.LETTERS)), 10)


class TestRepresent(unittest.TestCase):
    def test_represented_set_is_consistent_and_scored_in_its_frame(self):
        import numpy as np
        from arc2.represent import SYMMETRIES, apply_key, represent
        ch = json.load(open(T.EVAL_CHALLENGES))
        so = json.load(open(T.EVAL_SOLUTIONS))
        rch, rso, keys = represent(ch, so, seed=1)
        self.assertEqual(len(rch), 120)
        self.assertEqual(len(set(k["symmetry"] for k in keys.values()) - set(SYMMETRIES)), 0)
        for new, key in list(keys.items())[:20]:
            orig = key["original"]
            self.assertEqual(key["colour_map"][0], 0)
            self.assertEqual(sorted(key["colour_map"]), list(range(10)))
            self.assertEqual(rso[new][0], apply_key(so[orig][0], key))
            self.assertEqual(np.asarray(rch[new]["train"][0]["output"]).size, np.asarray(ch[orig]["train"][0]["output"]).size)
        again = represent(ch, so, seed=1)[2]
        self.assertEqual(keys, again)                                    # deterministic per task and seed
        with tempfile.TemporaryDirectory() as d:
            T.write_json(Path(d) / "c.json", rch)
            T.write_json(Path(d) / "s.json", rso)
            tasks = T.load_tasks(Path(d) / "c.json", Path(d) / "s.json")
            perfect = {t.task_id: T.submission_entry(t, {1: t.solutions}) for t in tasks}
            self.assertEqual(T.score_submission(tasks, perfect)[0], 120)
            orig_as_answer = {t.task_id: T.submission_entry(t, {1: so[keys[t.task_id]["original"]]}) for t in tasks}
            self.assertLess(T.score_submission(tasks, orig_as_answer)[0], 120)   # a memorized original answer is wrong


class TestKernel(unittest.TestCase):
    def setUp(self):
        self.task = train_task(FLIPLR)
        self.k = Kernel(self.task.sandbox_payload(), timeout_s=5)
        self.k.start()

    def tearDown(self):
        self.k.close()

    def payload(self, r):
        return json.loads(r.text)

    def test_state_persists_and_last_value_prints(self):
        self.k.run("x = 41")
        r = self.payload(self.k.run("x + 1"))
        self.assertEqual(r["stdout"], "42")
        self.assertEqual(r["returncode"], 0)

    def test_check_pass_and_diff_in_letters(self):
        r = self.payload(self.k.run("def t(g):\n    return np.fliplr(g)\ncheck(t)"))
        self.assertIn("demonstration pairs pass", r["stdout"])
        self.assertIn(f"{len(self.task.train)} of {len(self.task.train)}", r["stdout"])
        r = self.payload(self.k.run("def ident(g):\n    return g\ncheck(ident)"))
        self.assertIn("cells differ; first: (", r["stdout"])
        self.assertRegex(r["stdout"], r"expected [BbRNYgMOSr] got [BbRNYgMOSr]")

    def test_submit_function_grids_and_errors(self):
        self.k.run("def t(g):\n    return np.fliplr(g)")
        self.assertIn("Recorded attempt 1 (t)", self.payload(self.k.run("submit(t)"))["stdout"])
        self.assertIn("Recorded attempt 2 (grids)", self.payload(self.k.run("submit([[0, 1], [1, 0]], attempt=2)"))["stdout"])
        self.assertIn("nothing recorded", self.payload(self.k.run("submit(lambda g: g[0])"))["stdout"])
        self.assertIn("nothing recorded", self.payload(self.k.run("submit([['Q']])"))["stdout"])
        subs = self.k.submissions()
        import numpy as np
        self.assertEqual(subs[1][0], np.fliplr(np.array(self.task.test[0]["input"])).tolist())
        self.assertEqual(subs[2][0], [[0, 1], [1, 0]])

    def test_row_strings_and_digits_accepted(self):
        r = self.payload(self.k.run("def rows(g):\n    return [''.join(r[::-1]) for r in g]\ncheck(rows)"))
        self.assertIn(f"{len(self.task.train)} of {len(self.task.train)}", r["stdout"])

    def test_error_traceback_shows_model_line(self):
        r = self.payload(self.k.run("def f(g):\n    return 1 / 0\nf(None)"))
        self.assertIn("ZeroDivisionError", r["error"])
        self.assertIn("return 1 / 0", r["error"])

    def test_reserved_names_restored(self):
        r = self.payload(self.k.run("check = 3"))
        self.assertIn("check", r["note"])
        self.assertIn("demonstration pairs pass", self.payload(self.k.run("check(lambda g: g)"))["stdout"])

    def test_timeout_restarts_and_replays_library(self):
        self.k.run("def keep(g):\n    return np.fliplr(g)\nVALUE = 7\ny = 5")
        r = self.k.run("import time\ntime.sleep(30)")
        self.assertTrue(r.timed_out and r.restarted)
        self.assertIn("keep", json.loads(r.text)["error"])
        out = self.payload(self.k.run("print(VALUE, keep(test[0]['input']).shape)"))
        self.assertEqual(out["stdout"], f"7 {tuple(reversed(T.grid_shape(self.task.test[0]['input'])))[::-1]}")
        self.assertIn("NameError", self.payload(self.k.run("y"))["error"])

    def test_history_programs_source_show(self):
        self.k.run('def t(g):\n    """mirror"""\n    return np.fliplr(g)\ncheck(t)\nsubmit(t)')
        self.assertIn("check(t)", self.payload(self.k.run("history()"))["stdout"])
        self.assertIn("t: defined in call", self.payload(self.k.run("programs()"))["stdout"])
        self.assertIn("np.fliplr", self.payload(self.k.run("source('t')"))["stdout"])
        self.assertRegex(self.payload(self.k.run("show(test[0]['input'])"))["stdout"], r"^\d+x\d+\n[BbRNYgMOSr]+")

    def test_fallback_uses_best_checked_function(self):
        self.k.run("def a(g):\n    return g\ncheck(a)\ndef b(g):\n    return np.fliplr(g)\ncheck(b)")
        fb = self.k.fallback()
        self.assertEqual(fb["fn"], "b")


class TestAgentFake(unittest.TestCase):
    def run_script(self, task, script, budget=120_000, notice=8_000):
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        s = FakeSession(script)
        try:
            res = run_attempt(task, s, k, load_texts(), budget=budget, notice=notice)
        finally:
            k.close()
        return res, s

    def test_solves_with_scripted_program(self):
        task = train_task(ROT180)
        res, s = self.run_script(task, [
            ("pause", 500),
            ("code", "def t(g):\n    return np.rot90(g, 2)\ncheck(t)"),
            ("code", "submit(t)"),
            ("final", "Done."),
        ])
        self.assertEqual(res.end, "finished")
        self.assertEqual(res.first_full_pass_tokens, 500 + 400)
        entry = T.submission_entry(task, res.submissions)
        self.assertEqual(T.score_task(task, entry), 1.0)
        first = s.received[0]
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertTrue(first[1]["_arc3_observation"] and first[1]["_arc3_pin_parts"] == 1)
        self.assertEqual(s.received[1], [])                    # the length pause was continued in place
        tool = s.received[2][0]
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(json.loads(tool["content"])["tool"], "python")

    def test_nudge_then_no_answer_with_fallback(self):
        task = train_task(FLIPLR)
        res, s = self.run_script(task, [
            ("code", "def t(g):\n    return np.fliplr(g)\ncheck(t)"),
            ("final", "I think that is it."),
            ("final", "Still done."),
        ])
        self.assertEqual(res.end, "no_answer")
        self.assertEqual(res.nudges, {"no_submission": 1})
        self.assertEqual(res.fallback["fn"], "t")

    def test_budget_notice_and_budget_end(self):
        task = train_task(FLIPLR)
        res, s = self.run_script(task, [("code", "1")] * 10, budget=2_000, notice=1_000)
        self.assertEqual(res.end, "budget")
        self.assertIsNotNone(res.notice_at_tokens)
        notices = [m for ext in s.received for m in ext if m["role"] == "user" and "budget remain" in str(m["content"])]
        self.assertEqual(len(notices), 1)

    def test_turn_observations(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        s = FakeSession([("code", "1"), ("code", "2"), ("final", "x"), ("final", "y")])
        try:
            res = run_attempt(task, s, k, load_texts(), budget=1_300, notice=1_000, observe_turns=True)
        finally:
            k.close()
        appends = [ext for ext in s.received[1:] if ext]
        for ext in appends:                                  # at most one observation per operation, always last
            obs = [m for m in ext if m.get("_arc3_observation")]
            self.assertEqual(len(obs), 1)
            self.assertIs(ext[-1], obs[0])
        self.assertIn("turn 1 ended", appends[0][-1]["content"])
        self.assertIn("budget remain", appends[0][-1]["content"])   # the notice rides on the marker
        self.assertTrue(appends[2][0].get("_arc3_observation"))     # a harness note is a boundary too
        self.assertEqual(res.end, "no_answer")

    def test_deadline_ends_attempt(self):
        import time as _t
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        try:
            res = run_attempt(task, FakeSession([("code", "1")] * 5), k, load_texts(), budget=100_000, notice=1_000,
                              deadline=_t.monotonic() - 1)
        finally:
            k.close()
        self.assertEqual(res.end, "deadline")
        self.assertEqual(res.requests, 0)

    def test_bad_arguments_and_unknown_tool(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        s = FakeSession([("final", "x")])

        def bad_generate(external, _orig=s.generate):
            c = _orig(external)
            if len(s.received) == 1:
                c.message = {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "a", "function": {"name": "python", "arguments": "{not json"}},
                    {"id": "b", "function": {"name": "bash", "arguments": "{}"}}]}
            return c
        s.generate = bad_generate
        try:
            res = run_attempt(task, s, k, load_texts(), budget=50_000, notice=1_000)
        finally:
            k.close()
        tools = s.received[1]
        self.assertIn("could not be read", tools[0]["content"])
        self.assertIn("only tool is python", tools[1]["content"])
        self.assertEqual(res.tool_calls, 0)


class TestRunnerAndTransport(unittest.TestCase):
    def test_runner_fake_end_to_end(self):
        from arc2 import runner
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            runner.main(["--out", str(out), "--fake", "--lanes", "3", "--limit", "6", "--status-every", "1000",
                         "--timeline-every", "0", "--deadline-s", "3600"])
            score = json.loads((out / "score.json").read_text())
            sub = json.loads((out / "submission.json").read_text())
            self.assertEqual(score["attempted"], 6)
            self.assertEqual(len(sub), 6)
            self.assertEqual(score["ends"], {"finished": 6})
            self.assertTrue((out / "summary.md").exists())
            rows = [json.loads(l) for l in (out / "timeline.jsonl").read_text().splitlines()]
            self.assertTrue(rows and all("running" in r for r in rows))
            self.assertIn("timeline", score)
            lines = (out / "results.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 6)

    def test_timeline_stats(self):
        from arc2.runner import generation_counter, timeline_stats
        self.assertEqual(generation_counter('# HELP x\nvllm:generation_tokens_total{model_name="m"} 1234.0\n'), 1234.0)
        self.assertIsNone(generation_counter("other 1\n"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "timeline.jsonl"
            rows = [dict(t=60.0 * i, running=4 if i < 20 else 1, server_generation_tokens=60.0 * i * (1000 if i < 20 else 0)
                         + (0 if i < 20 else 0)) for i in range(30)]
            tok = 0.0
            for i, r in enumerate(rows):          # 1,000 tok/s with 4 lanes for 20 min, then 250 tok/s with 1 lane
                r["server_generation_tokens"] = tok
                tok += 60.0 * (1000 if i < 20 else 250)
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            st = timeline_stats(p, lanes=4)
            self.assertEqual(st["peak_10min_rate"], 1000.0)
            self.assertLess(st["mean_over_peak"], 1.0)
            self.assertAlmostEqual(st["share_below_full_lanes"], 10 / 30, places=3)

    def test_transport_import_pin_and_texts(self):
        from arc2.prompts import apply_transport_texts
        from arc2.shared import load_transport
        tr = load_transport()
        self.assertEqual(tr.ARC2_SHARED["status"], "pinned")
        replaced = apply_transport_texts(tr, load_texts())
        self.assertIn("TURN_BUDGET_CLOSE", replaced)
        self.assertNotIn("game", tr.TURN_BUDGET_CLOSE)
        d, i, t = tr.turn_cap_decision(16_000, False, 12_000, 16_000, True, check=8_000)
        self.assertIsNotNone(i)
        self.assertNotIn("game", i["text"])

    def test_reply_caps_off(self):
        from arc2.shared import CAPS_ENV_KEYS, configure_transport, load_transport
        saved = {k: os.environ.get(k) for k in CAPS_ENV_KEYS}
        try:
            tr = load_transport()
            rec = configure_transport(tr, texts=load_texts(), reply_caps="off", request_tokens=8000)
            self.assertEqual(rec["env"]["ARC3_ASSISTANT_TURN_CAPS"], "1000000,1000001")
            for cumulative in (0, 12_000, 16_000, 90_000):
                self.assertEqual(tr.turn_cap_decision(cumulative, False, 1_000_000, 1_000_001, False, check=8000),
                                 (8000, None, None))
            self.assertIsNone(tr.repetition_onset([7] * 400))
            s = tr.PersistentSession(base_url="http://127.0.0.1:9", model="m", tools=[], template_kwargs={},
                                     sampling={}, history_eviction={"policy": "qsa_frequency", "fraction": 0.1},
                                     im_end_ids=[1], decode=lambda ids: "")
            self.assertEqual((s.soft, s.hard, s.check, s.call_ext), (1_000_000, 1_000_001, 8000, 0))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_system_and_opener_render(self):
        texts = load_texts()
        sysm = system_message(texts, budget=120_000, notice=8_000)
        self.assertIn("B=black", sysm["content"])
        self.assertIn("120,000", sysm["content"])
        self.assertNotIn("{", sysm["content"].replace("{'", ""))
        task = T.load_tasks(T.EVAL_CHALLENGES)[0]
        op = opener_message(task, texts)
        self.assertTrue(op["content"][0]["text"].startswith(f"Puzzle {task.task_id}:"))
        dig = system_message(texts, budget=1, notice=1, fmt="digits")
        self.assertIn("integer from 0 to 9", dig["content"])



class TestV1(unittest.TestCase):
    """v1 pieces: grid verification at submit, the correction loop, the judge's pick, program rebuild and verify."""

    def kernel(self, task, payload=None):
        k = Kernel(payload or task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        return k

    def test_grid_submission_verified_by(self):
        task = train_task(FLIPLR)
        k = self.kernel(task)
        r = k.run("def t(g):\n    return np.fliplr(g)\ncheck(t)\nsubmit([t(x['input']) for x in test])")
        self.assertEqual([e.get("verified_by") for e in r.events if e["kind"] == "submit"], ["t"])
        r = k.run("submit([x['input'] for x in test], attempt=2)")
        self.assertEqual([e.get("verified_by") for e in r.events if e["kind"] == "submit"], [None])
        r = k.run("def u(g):\n    return g\nsubmit(u)")
        self.assertEqual([e.get("verified_by") for e in r.events if e["kind"] == "submit"], [None])

    def test_correction_once_then_finish(self):
        task = train_task(FLIPLR)
        k = self.kernel(task)
        s = FakeSession([("code", "def ident(g):\n    return g\ncheck(ident)\nsubmit(ident)"), ("final", "Done."),
                         ("final", "Still done.")])
        res = run_attempt(task, s, k, load_texts(), budget=120_000, notice=8_000, corrections=1)
        self.assertEqual(res.end, "finished")
        self.assertEqual(res.nudges, {"correction": 1})
        sent = [m["content"] for ext in s.received for m in ext if m["role"] == "user" and "verified" in str(m["content"])]
        self.assertEqual(len(sent), 1)
        self.assertIn("passes 0 of", sent[0])
        k2 = self.kernel(task)
        s2 = FakeSession([("code", "def t(g):\n    return np.fliplr(g)\ncheck(t)\nsubmit(t)"), ("final", "Done.")])
        res2 = run_attempt(task, s2, k2, load_texts(), budget=120_000, notice=8_000, corrections=1)
        self.assertEqual((res2.end, res2.nudges), ("finished", {}))          # a verified answer is never corrected

    def test_judge_pick_and_messages(self):
        from arc2.judge import Candidate, judge_messages, run_judge
        task = train_task(FLIPLR)
        right = [T.to_list(g) for g in task.solutions]
        wrong = [t["input"] for t in task.test]
        cands = [Candidate(outputs=wrong, source="def a(g):\n    return g", passed=0, total=len(task.train)),
                 Candidate(outputs=right)]
        msgs = judge_messages(task, cands, load_texts(), budget=32_000, notice=4_000)
        text = "".join(p["text"] for p in msgs[1]["content"])
        self.assertIn("Candidate 0: a program that passes 0 of", text)
        self.assertIn("Candidate 1: grids recorded without a program", text)
        s = FakeSession([("code", "print(len(candidates), candidates[1]['source'] == '')\npick(1, 0)"), ("final", "Done.")])
        res, order = run_judge(task, cands, s, load_texts(), kernel_kwargs=dict(timeout_s=10))
        self.assertEqual((res.end, order), ("finished", [1, 0]))
        s = FakeSession([("code", "pick(5)"), ("final", "Done."), ("final", "Done.")])
        res, order = run_judge(task, cands, s, load_texts(), kernel_kwargs=dict(timeout_s=10))
        self.assertEqual((res.end, order, res.nudges), ("no_answer", None, {"no_submission": 1}))

    def test_program_source_and_verify(self):
        from arc2.program import library, program_source, verify
        task = train_task(ROT180)
        lib = library(["import numpy as np\nK = 2\ndef helper(g):\n    return np.rot90(g, K)\n",
                       "def solve(g):\n    return helper(g)\nx = 5", "def other(g):\n    return g",
                       "def broken(:\n    pass"])
        src = program_source(lib, "solve")
        for part in ("import numpy as np", "K = 2", "def helper", "def solve"):
            self.assertIn(part, src)
        self.assertNotIn("def other", src)
        self.assertIsNone(program_source(lib, "missing"))
        v = verify(task.sandbox_payload(), src, "solve", timeout_s=10)
        self.assertEqual((v["passed"], v["total"], v["error"]), (len(task.train), len(task.train), None))
        self.assertEqual(v["outputs"], [T.to_list(g) for g in task.solutions])


class TestRunnerV1(unittest.TestCase):
    """The multi-pass scheduler, tail fill, the judge hook and the slot rule, on the fake session."""

    def run_main(self, argv, scripts=None):
        from arc2 import runner
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        out = Path(d) / "run"
        orig = runner.Session.make
        if scripts is not None:
            it = iter(scripts)
            runner.Session.make = lambda self, log: FakeSession(next(it), log=log)
        try:
            runner.main(["--out", str(out), "--fake", "--status-every", "1000", "--timeline-every", "0",
                         "--deadline-s", "3600"] + argv)
        finally:
            runner.Session.make = orig
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        return out, json.loads((out / "score.json").read_text()), rows

    def test_two_passes_and_tail_fill(self):
        out, score, rows = self.run_main(["--lanes", "3", "--limit", "4", "--passes", "2"])
        self.assertEqual(sorted(r["pass_no"] for r in rows), [1, 1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(set(score["by_pass"]), {"1", "2"})
        self.assertTrue(all((out / "tasks" / r["task_id"] / f"p{r['pass_no']}" / "log.jsonl").exists() for r in rows))
        out, score, rows = self.run_main(["--lanes", "3", "--limit", "4", "--passes", "1", "--max-passes", "3"])
        self.assertEqual(len(rows), 12)                    # the default script never verifies: every task stays band 1
        out, score, rows = self.run_main(["--lanes", "2", "--limit", "3", "--passes", "2", "--band-order", "0,1,3,2,4",
                                          "--confirm-order", "first_allpass_desc"])
        self.assertEqual(len(rows), 6)
        self.assertTrue(all({"band", "first_allpass"} <= set(r["sched"]) for r in rows))
        self.assertTrue(all(r["sched"]["band"] == (0 if r["pass_no"] == 1 else 1) for r in rows))

    def test_judge_orders_the_slots(self):
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        answer = "submit([np.full((1, 1), '{c}') for _ in test])"
        scripts = [[("code", answer.format(c="B")), ("final", "Done.")],
                   [("code", answer.format(c="b")), ("final", "Done.")],
                   [("code", "pick(0)"), ("final", "Done.")]]
        out, score, rows = self.run_main(["--lanes", "1", "--passes", "2", "--judge", "2plus",
                                          "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                                          "--ids", "@" + str(ids)], scripts)
        judge = [r for r in rows if r["kind"] == "judge"]
        self.assertEqual(len(judge), 1)
        self.assertEqual(judge[0]["n_candidates"], 2)
        first = judge[0]["order"][0]
        sub = json.loads((out / "submission.json").read_text())[FLIPLR]
        self.assertEqual(sub[0]["attempt_1"], [[first]])     # candidate k is pass k+1's grid: B (0), then b (1)
        self.assertEqual(score["judges"]["picked"], 1)

    def test_cycle_policy_order_and_stop(self):
        grids = "submit([x['input'] for x in test])"           # a grid answer: valid under the user's rule
        ident = "def f(g):\n    return g\ncheck(f)\nsubmit(f)"     # a function failing the demonstrations: not valid
        scripts = [[("code", grids), ("final", "Done.")],            # A pass 1: valid
                   [("code", ident), ("final", "Done.")],            # B pass 1: not valid
                   [("code", grids), ("final", "Done.")],            # A pass 2: valid -> A complete
                   [("code", grids), ("final", "Done.")],            # B pass 2: valid
                   [("code", grids), ("final", "Done.")]]            # B pass 3: valid -> B complete
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(f"{FLIPLR},{ROT180}")
        out, score, rows = self.run_main(["--lanes", "1", "--policy", "cycle", "--challenges", str(TRAIN_CH),
                                          "--solutions", str(TRAIN_SOL), "--ids", "@" + str(ids)], scripts)
        self.assertEqual([(r["task_id"], r["pass_no"]) for r in rows],
                         [(FLIPLR, 1), (ROT180, 1), (FLIPLR, 2), (ROT180, 2), (ROT180, 3)])
        self.assertEqual(score["tasks_by_valid_passes"], {"2": 2})
        sub = json.loads((out / "submission.json").read_text())[ROT180]
        task = train_task(ROT180)
        self.assertEqual(sub[0]["attempt_1"], T.to_list(task.test[0]["input"]))   # a valid grid, not the failing function's
        self.assertEqual(len(rows), 5)

    def test_rank_policy_easy_and_hard_lanes(self):
        """The user's rank order (2026-10-29): first passes FIFO; then, round by round, easy lanes take tasks with a valid
        answer before those without, fewest pass-1 tokens first, and hard lanes the reverse."""
        grids = "submit([x['input'] for x in test])"
        ident = "def f(g):\n    return g\ncheck(f)\nsubmit(f)"
        third = "00576224"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(f"{FLIPLR},{ROT180},{third}")
        A, B, C = FLIPLR, ROT180, third
        first = [[("pause", 5000), ("code", grids), ("final", "Done.")],    # A: valid, long
                 [("code", grids), ("final", "Done.")],                     # B: valid, short
                 [("code", ident), ("final", "Done.")]]                     # C: not valid
        later = [[("code", grids), ("final", "Done.")]] * 4
        common = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                  "--ids", "@" + str(ids)]
        out, score, rows = self.run_main(common, first + later)
        self.assertEqual([(r["task_id"], r["pass_no"]) for r in rows],
                         [(A, 1), (B, 1), (C, 1), (B, 2), (A, 2), (C, 2), (C, 3)])
        self.assertEqual(score["tasks_by_valid_passes"], {"2": 3})
        self.assertEqual({r["sched"]["lane"] for r in rows}, {"easy"})
        out, score, rows = self.run_main(common + ["--hard-lanes", "1"], first + later)
        self.assertEqual([(r["task_id"], r["pass_no"]) for r in rows],
                         [(A, 1), (B, 1), (C, 1), (C, 2), (A, 2), (B, 2), (C, 3)])
        self.assertEqual({r["sched"]["lane"] for r in rows}, {"hard"})

    def test_extra_pass_identical_long(self):
        """The user's phase 3 (2026-10-30): identical verified attempt-1 answers after a long pass 1 earn a third pass,
        scheduled with the open tasks; a short pass 1 does not."""
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        rot = "def rot(g):\n    return np.rot90(g, 2)\nsubmit(rot)"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(f"{FLIPLR},{ROT180}")
        A, B = FLIPLR, ROT180
        scripts = [[("pause", 5000), ("code", flip), ("final", "Done.")],     # A pass 1: long, verified
                   [("code", rot), ("final", "Done.")],                       # B pass 1: short, verified
                   [("code", rot), ("final", "Done.")],                       # B pass 2 (easy lane: shorter pass 1 first)
                   [("code", flip), ("final", "Done.")],                      # A pass 2: identical -> A earns pass 3
                   [("code", flip), ("final", "Done.")]]                      # A pass 3
        common = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                  "--ids", "@" + str(ids)]
        out, score, rows = self.run_main(common + ["--extra-pass-identical-long", "3000"], scripts)
        self.assertEqual([(r["task_id"], r["pass_no"]) for r in rows], [(A, 1), (B, 1), (B, 2), (A, 2), (A, 3)])
        out, score, rows = self.run_main(common, scripts)
        self.assertEqual([(r["task_id"], r["pass_no"]) for r in rows], [(A, 1), (B, 1), (B, 2), (A, 2)])

    def test_review_pass_and_p3(self):
        """v2l (the user, 2026-10-30): a long task whose passes 1 and 2 did not save identical verified answers gets one
        review after them; P3 keeps the votes' slot 1 and puts the reviewer's answer in slot 2. Without --review-long there
        is no review; the deadline guard skips a review that cannot finish (--no-phase3-guard starts it anyway)."""
        import numpy as np
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip, explanation='mirror')"
        rot = "def rot(g):\n    return np.rot90(g, 2)\nsubmit(rot)"
        grids = "submit([x['input'] for x in test])"
        review = "print(len(previous))\nsubmit([np.rot90(x['input'], 2) for x in test])"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(f"{FLIPLR},{ROT180}")
        A, B = FLIPLR, ROT180
        scripts = [[("pause", 5000), ("code", flip), ("final", "Done.")],     # A pass 1: long, verified
                   [("code", rot), ("final", "Done.")],                       # B pass 1: short, verified
                   [("code", rot), ("final", "Done.")],                       # B pass 2: identical, but pass 1 short
                   [("code", grids), ("final", "Done.")],                     # A pass 2: grids, not identical -> review
                   [("code", review), ("final", "Done.")]]                    # A's review
        common = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                  "--ids", "@" + str(ids)]
        out, score, rows = self.run_main(common + ["--review-long", "3000"], scripts)
        self.assertEqual([(r["task_id"], r["kind"], r.get("pass_no")) for r in rows],
                         [(A, "solve", 1), (B, "solve", 1), (B, "solve", 2), (A, "solve", 2), (A, "review", None)])
        self.assertEqual(score["reviews"]["n"], 1)
        task = {t.task_id: t for t in T.load_tasks(TRAIN_CH, TRAIN_SOL)}[A]
        x = np.array(task.test[0]["input"])
        entry = json.loads((out / "submission.json").read_text())[A][0]
        self.assertEqual(entry["attempt_1"], np.fliplr(x).tolist())                  # the votes' slot 1 stays
        self.assertEqual(entry["attempt_2"], np.rot90(x, 2).tolist())                # the reviewer's answer in slot 2
        self.assertEqual({p["label"] for p in rows[-1]["previous"]}, {"A", "B"})
        self.assertTrue((out / "tasks" / A / "review" / "attempt.json").exists())
        out, score, rows = self.run_main(common, scripts)
        self.assertEqual([r["kind"] for r in rows], ["solve"] * 4)
        out, score, rows = self.run_main(common + ["--review-long", "3000", "--deadline-s", "600"], scripts)
        self.assertEqual([r["kind"] for r in rows], ["solve"] * 4)                    # 600 s x 60 tok/s < 42,000
        out, score, rows = self.run_main(common + ["--review-long", "3000", "--deadline-s", "600", "--no-phase3-guard"],
                                         scripts)
        self.assertEqual([r["kind"] for r in rows], ["solve"] * 4 + ["review"])

    def test_phase3_priority_third_more_review(self):
        """v2m (the user, 2026-10-31: "third pass on identical answers > additional pass > review"): after two passes the round
        starts its third passes, then the further passes of tasks short of two valid passes, then the reviews; a 4th pass comes
        after every review. v2l's order ('rank') for the same scripts: by the rank key, with a task's own review before its
        further pass. --phase3-rule valid (the user: two valid answers, then an extra pass if identical, verified or not, and a
        review otherwise): no review for a task without two valid passes, and identical unverified grids earn the extra pass."""
        import numpy as np  # noqa: F401
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        ident = "def f(g):\n    return g\ncheck(f)\nsubmit(f)"
        grids = "submit([x['input'] for x in test])"
        rgrids = "submit([np.rot90(x['input'], 2) for x in test])"
        review = "print(len(previous))\nsubmit([np.rot90(x['input'], 2) for x in test])"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(f"{FLIPLR},{ROT180},00576224")
        A, B, C = FLIPLR, ROT180, "00576224"
        S = {k: [("code", v), ("final", "Done.")] for k, v in dict(flip=flip, ident=ident, grids=grids, rgrids=rgrids,
                                                                     review=review).items()}
        first = [[("pause", 5000)] + S["flip"],              # A1: long, verified
                 [("pause", 5000)] + S["ident"],             # B1: long, not valid
                 [("pause", 6000)] + S["grids"]]             # C1: longer, a valid grid answer
        second = [S["flip"], S["rgrids"], S["ident"]]        # A2 identical verified (third), C2 other grids (review), B2 invalid
        common = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                  "--ids", "@" + str(ids), "--extra-pass-identical-long", "3000", "--review-long", "3000", "--max-passes", "4"]
        seen = lambda rows: [(r["task_id"], r["kind"], r.get("pass_no")) for r in rows]
        head = [(A, "solve", 1), (B, "solve", 1), (C, "solve", 1), (A, "solve", 2), (C, "solve", 2), (B, "solve", 2)]
        out, score, rows = self.run_main(common + ["--phase3-priority", "third,more,review"],
                                         first + second + [S["flip"], S["ident"], S["review"], S["review"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (B, "solve", 3), (C, "review", None), (B, "review", None),
                                             (B, "solve", 4)])
        out, score, rows = self.run_main(common, first + second + [S["flip"], S["review"], S["review"], S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (C, "review", None), (B, "review", None), (B, "solve", 3),
                                             (B, "solve", 4)])
        valid = common + ["--phase3-priority", "third,more,review", "--phase3-rule", "valid"]
        out, score, rows = self.run_main(valid, first + second + [S["flip"], S["ident"], S["review"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (B, "solve", 3), (C, "review", None), (B, "solve", 4)])
        same = [S["flip"], S["grids"], S["ident"]]           # C2 repeats C1's (unverified) grids
        out, score, rows = self.run_main(valid, first + same + [S["flip"], S["grids"], S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (C, "solve", 3), (B, "solve", 3), (B, "solve", 4)])
        out, score, rows = self.run_main(common + ["--phase3-priority", "third,more,review"],
                                         first + same + [S["flip"], S["ident"], S["review"], S["review"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (B, "solve", 3), (C, "review", None), (B, "review", None),
                                             (B, "solve", 4)])                 # v2l's rule: unverified grids get a review
        out, score, rows = self.run_main(valid + ["--phase3-length", "mean", "--extra-pass-identical-long", "5300",
                                                  "--review-long", "5300"],
                                         first + second + [S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(B, "solve", 3), (B, "solve", 4)])    # means below 5,300: no phase 3
        long2 = [[("pause", 9000)] + S["flip"], S["grids"], S["ident"]]       # A2 long: A's mean 7,0xx, C's mean ~3,0xx
        out, score, rows = self.run_main(valid + ["--phase3-length", "mean", "--rank-length", "mean", "--extra-pass-identical-long",
                                                  "1000", "--review-long", "1000"],
                                         first + long2 + [S["grids"], S["flip"], S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(C, "solve", 3), (A, "solve", 3), (B, "solve", 3), (B, "solve", 4)])
        out, score, rows = self.run_main(valid + ["--phase3-length", "mean", "--extra-pass-identical-long", "1000",
                                                  "--review-long", "1000"],
                                         first + long2 + [S["flip"], S["grids"], S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (C, "solve", 3), (B, "solve", 3), (B, "solve", 4)])  # pass 1 order
        with self.assertRaises(SystemExit):
            self.run_main(common + ["--phase3-priority", "third,review"], first)

    def test_slot_rule_votes_and_verification(self):
        from arc2 import slots as S
        def row(outs1, ver1, outs2=None):
            subs = {"1": outs1}
            submits = [dict(attempt=1, fn="f" if ver1 else "grids", train_passed=3 if ver1 else None, train_total=3,
                            verified_by="f" if ver1 else None)]
            if outs2 is not None:
                subs["2"] = outs2
                submits.append(dict(attempt=2, fn="grids", train_passed=None, train_total=3, verified_by=None))
            return dict(submissions=subs, submits=submits)
        A, B, C = [[[1]]], [[[2]]], [[[3]]]
        rows = [row(A, False, C), row(B, True), row(B, True)]
        self.assertEqual(S.slot_outputs(rows, 1), {1: [[[2]]], 2: [[[1]]]})
        self.assertEqual(S.band(rows), 4)
        self.assertEqual(S.band([row(A, True), row(B, True)]), 2)
        self.assertEqual(S.band([row(A, True)]), 3)
        self.assertEqual(S.band([row(A, False)]), 1)
        self.assertEqual(S.band([dict(submissions={}, submits=[])]), 0)
        fallback = dict(submissions={}, submits=[], fallback=dict(outputs=A))           # not valid
        failing = dict(submissions={"1": A}, submits=[dict(attempt=1, fn="f", train_passed=1, train_total=3,
                                                           verified_by=None)])      # a failing function: not valid
        for first in (fallback, failing):
            self.assertEqual(S.slot_outputs([first, row(B, False), row(C, False)], 1), {1: [[[2]]], 2: [[[3]]]})
        self.assertEqual([S.pass_valid(r) for r in (fallback, failing, row(B, False), row(C, True))],
                         [False, False, True, True])
        # equal votes: pass 2's attempt 1 (C) outranks pass 1's attempt 2 (B)
        self.assertEqual(S.slot_outputs([row(A, True, B), row(C, True)], 1), {1: [[[1]]], 2: [[[3]]]})
        fulls = S.full_answers([row(A, False), row(C, False)])
        self.assertEqual(S.slot_outputs([row(A, False), row(C, False)], 1, judge_order=[1]), {1: [[[3]]], 2: [[[1]]]})
        self.assertEqual([f["outputs"] for f in fulls], [A, C])


class TestV2(unittest.TestCase):
    """v2 (the user's design, 2026-10-29): note / explanation / merged history, the keep-going loop, the idle and
    repetition limits, and the in-reply status cue with the hard close."""

    def kernel(self, task):
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        return k

    def test_note_explanation_and_history(self):
        k = self.kernel(train_task(FLIPLR))
        k.run("note('mirror image left-right')\ndef t(g):\n    return np.fliplr(g)\ncheck(t)\n"
              "submit([t(x['input']) for x in test], explanation='flip each row')")
        r = json.loads(k.run("history()").text)
        for part in ("note: mirror image left-right", "explanation: flip each row", "the outputs of t",
                     "functions defined: t"):
            self.assertIn(part, r["stdout"])
        self.assertEqual([e["kind"] for e in k.records], ["note", "check", "submit"])

    def test_keep_going_until_valid(self):
        from arc2.agent import run_attempt
        task = train_task(FLIPLR)
        k = self.kernel(task)
        s = FakeSession([("final", "Thinking out loud."),
                         ("code", "def f(g):\n    return g\ncheck(f)\nsubmit(f)"), ("final", "Done?"),
                         ("code", "def t(g):\n    return np.fliplr(g)\ncheck(t)\nsubmit(t)"), ("final", "Done.")])
        res = run_attempt(task, s, k, load_texts(), budget=120_000, notice=8_000, no_call="keep_going",
                          prompt_key="system_v2", reply_cap=32_000)
        self.assertEqual((res.end, res.nudges), ("finished", {"keep_going": 2}))
        sent = [m["content"] for ext in s.received for m in ext if m["role"] == "user" and "Keep working" in str(m["content"])]
        self.assertIn("passes 0 of", sent[1])
        system = s.received[0][0]["content"]
        self.assertIn("note(text)", system)
        self.assertIn("32,000 tokens", system)
        k2 = self.kernel(task)
        s2 = FakeSession([("final", "x")] * 10)
        res2 = run_attempt(task, s2, k2, load_texts(), budget=120_000, notice=8_000, no_call="keep_going", idle_limit=3)
        self.assertEqual((res2.end, res2.nudges), ("stalled", {"keep_going": 3}))
        k3 = self.kernel(task)
        s3 = FakeSession([("code", "submit([x['input'] for x in test])"), ("final", "Done.")])
        res3 = run_attempt(task, s3, k3, load_texts(), budget=120_000, notice=8_000, no_call="keep_going")
        self.assertEqual((res3.end, res3.nudges), ("finished", {}))        # a grid answer is valid under the rule

    def test_v2_decision_status_cue_and_close(self):
        from arc2.shared import CAP_STATE, v2_decision
        calls = []

        def original(cumulative, think_closed, soft, hard, cued, **kw):
            calls.append((cumulative, soft, hard, kw.get("call_extension")))
            return 1, {"kind": "hard_close", "close": True, "text": "x"}, None
        d = v2_decision(original, close=32000, ext=4096, status_every=12000, status_text=load_texts()["status_cue"])
        CAP_STATE.used, CAP_STATE.budget = 50000, 200000
        self.assertEqual(d(0, False, 31999, 32000, False, check=8000), (8000, None, None))
        self.assertEqual(d(8000, False, 31999, 32000, False, check=8000)[0], 4000)       # aligned to the status point
        limit, inj, _ = d(12000, False, 31999, 32000, False, check=8000)
        self.assertEqual(limit, 8000)
        self.assertIn("12,000 tokens; 50,000 of this puzzle's 200,000", inj["text"])
        self.assertFalse(inj["close"])
        self.assertIsNone(d(12000, True, 31999, 32000, False, check=8000)[1])            # not after the reasoning closed
        self.assertIsNone(d(24000, False, 31999, 32000, False, check=8000, call_open=True)[1])
        self.assertEqual(d(24000, False, 31999, 32000, False, check=8000)[0], 8000)       # next stop: the close
        self.assertEqual(d(32000, False, 31999, 32000, False, check=8000)[1]["kind"], "hard_close")
        self.assertEqual(calls[-1], (32000, 0, 32000, 4096))                               # ARC-3's close, soft cue off
        d(5000, False, 31999, 32000, False, check=8000, repeat_onset=3)
        self.assertEqual(calls[-1][0], 5000)                                               # repetition: ARC-3's logic


class TestV2b(unittest.TestCase):
    def test_status_ended_replies_do_not_count_as_idle(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)

        class CueSession(FakeSession):                     # every reply without a tool call carried a status note
            def generate(self, external):
                c = super().generate(external)
                if not c.message.get("tool_calls"):
                    c.cues = ("soft_nudge",)
                return c
        s = CueSession([("final", "x")] * 8 + [("code", "submit([x['input'] for x in test])"), ("final", "Done.")])
        res = run_attempt(task, s, k, load_texts(), budget=120_000, notice=8_000, no_call="keep_going", idle_limit=3,
                          idle_skip_status=True)
        self.assertEqual((res.end, res.nudges), ("finished", {"keep_going": 8}))
        from arc2.shared import v2_decision
        d = v2_decision(lambda *a, **k: (1, None, None), close=32000, ext=4096, status_every=12000,
                        status_text=load_texts()["status_cue_v2b"])
        self.assertIn("does not end your reply", d(12000, False, 31999, 32000, False, check=8000)[1]["text"])

class TestV2c(unittest.TestCase):
    """v2c (the user, 2026-10-29): a 160K pass budget, a status note every 10K tokens of the pass, the tokens spent in every
    turn marker, and from 120K a wrap-up reminder of rising urgency in every note and user turn."""

    def test_wrap_tiers_and_status_note(self):
        from arc2.shared import CAP_STATE, v2c_decision, wrap_text
        texts = load_texts()
        self.assertEqual(wrap_text(texts, 119_999, 160_000, 120_000), "")
        self.assertEqual(wrap_text(texts, 150_000, 160_000, None), "")
        self.assertIn("40,000 tokens remain: plan", wrap_text(texts, 120_000, 160_000, 120_000))
        self.assertIn("30,000 tokens remain: record", wrap_text(texts, 130_000, 160_000, 120_000))
        self.assertIn("Only about 15,000", wrap_text(texts, 145_000, 160_000, 120_000))
        self.assertIn("nearly spent", wrap_text(texts, 155_000, 160_000, 120_000))
        calls = []

        def original(cumulative, think_closed, soft, hard, cued, **kw):
            calls.append((cumulative, soft, hard, kw.get("call_extension")))
            return 1, {"kind": "hard_close", "close": True, "text": "x"}, None
        d = v2c_decision(original, close=32000, ext=4096, status_every=10000, status_text=texts["status_cue_v2c"],
                         texts=texts, wrap_at=120000)
        CAP_STATE.used, CAP_STATE.budget = 25000, 160000
        self.assertEqual(d(5000, False, 31999, 32000, False, check=8000), (5000, None, None))   # to the pass's 30K point
        CAP_STATE.used = 30000
        limit, inj, _ = d(10000, False, 31999, 32000, False, check=8000)
        self.assertEqual(limit, 8000)
        self.assertIn("30,000 of this puzzle's 160,000 tokens are spent. This note does not end your reply", inj["text"])
        self.assertFalse(inj["close"])
        CAP_STATE.used = 130000
        self.assertIn("30,000 tokens remain", d(3000, False, 31999, 32000, False, check=8000)[1]["text"])
        self.assertIsNone(d(3000, True, 31999, 32000, False, check=8000)[1])            # not after the reasoning closed
        self.assertIsNone(d(3000, False, 31999, 32000, False, check=8000, call_open=True)[1])
        self.assertEqual(d(32000, False, 31999, 32000, False, check=8000)[1]["kind"], "hard_close")
        self.assertEqual(calls[-1], (32000, 0, 32000, 4096))

    def test_marker_and_user_turns_carry_the_wrap_up(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        s = FakeSession([("pause", 1000), ("code", "x = 1"), ("final", "thinking"),
                         ("code", "submit([x['input'] for x in test])"), ("final", "Done.")])
        res = run_attempt(task, s, k, load_texts(), budget=4000, notice=0, no_call="keep_going", observe_turns=True,
                          marker_key="turn_marker_v2c", wrap_at=1000)
        self.assertEqual(res.end, "finished")
        users = [m["content"] for ext in s.received for m in ext if m.get("role") == "user" and isinstance(m["content"], str)]
        self.assertTrue(any("turn 1 ended; 1,400 of this puzzle's 4,000 tokens are spent. The budget is nearly spent "
                            "(about 2,600 tokens remain)" in u for u in users))
        self.assertTrue(any(u.startswith("(Harness: no valid solution is saved yet") and
                            u.endswith("1,800 of this puzzle's 4,000 tokens are spent. The budget is nearly spent (about "
                                       "2,200 tokens remain): record your answer with submit(...) now.)") for u in users))
        k2 = Kernel(task.sandbox_payload(), timeout_s=10)
        k2.start()
        self.addCleanup(k2.close)
        s2 = FakeSession([("final", "thinking"), ("code", "submit([x['input'] for x in test])"), ("final", "Done.")])
        run_attempt(task, s2, k2, load_texts(), budget=4000, notice=0, no_call="keep_going", observe_turns=True)
        users = [m["content"] for ext in s2.received for m in ext if m.get("role") == "user" and isinstance(m["content"], str)]
        self.assertTrue(any(u.endswith("400 of this puzzle's 4,000 tokens are spent.)") for u in users))   # v2b unchanged
        self.assertIn("(Harness: turn 1 ended.)", users)


class TestV2d(unittest.TestCase):
    """v2d (the user, 2026-10-29): no budget in the system prompt; every harness message gives the tokens spent, naming the
    budget from 100K; from 120K the wrap-up reminder and the in-reply status note (every 10K of the pass)."""

    def test_late_cue_and_spent_clause(self):
        from arc2.shared import CAP_STATE, spent_text, v2c_decision
        texts = load_texts()
        self.assertEqual(spent_text(texts, 99_999, 160_000, 100_000), "99,999 tokens are spent so far")
        self.assertEqual(spent_text(texts, 100_000, 160_000, 100_000), "100,000 of this puzzle's 160,000 tokens are spent")
        self.assertEqual(spent_text(texts, 5, 160_000, None), "")
        d = v2c_decision(lambda *a, **k: (1, None, None), close=32000, ext=4096, status_every=10000,
                         status_text=texts["status_cue_v2c"], texts=texts, wrap_at=120000, cue_from=120000)
        CAP_STATE.used, CAP_STATE.budget = 25000, 160000
        self.assertEqual(d(5000, False, 31999, 32000, False, check=8000), (8000, None, None))     # no stop at 30K
        CAP_STATE.used = 100000
        self.assertEqual(d(3000, False, 31999, 32000, False, check=8000), (8000, None, None))     # no note at 100K
        CAP_STATE.used = 115000
        self.assertEqual(d(3000, False, 31999, 32000, False, check=8000)[0], 5000)                # stops at 120K
        CAP_STATE.used = 120000
        inj = d(8000, False, 31999, 32000, False, check=8000)[1]
        self.assertIn("120,000 of this puzzle's 160,000 tokens are spent. About 40,000 tokens remain", inj["text"])

    def test_messages_and_system_prompt(self):
        texts = load_texts()
        self.assertIn("budget of", texts["system_v2"])
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        s = FakeSession([("code", "x = 1"), ("pause", 600), ("code", "y = 2"), ("final", "thinking"),
                         ("code", "submit([x['input'] for x in test])"), ("final", "Done.")])
        res = run_attempt(task, s, k, texts, budget=4000, notice=0, no_call="keep_going", observe_turns=True,
                          prompt_key="system_v2d", reply_cap=32000, marker_key="turn_marker_v2d", wrap_at=2000,
                          limit_at=1200, text_keys=dict(keep_going="keep_going_v2d", recovery_repeat="recovery_repeat_v2d",
                                                        nudge_format="nudge_format_v2d"))
        self.assertEqual(res.end, "finished")
        system = s.received[0][0]["content"]
        self.assertNotIn("4,000", system)
        self.assertIn("The harness reports how many tokens are spent on this puzzle.", system)
        users = [m["content"] for ext in s.received for m in ext if m.get("role") == "user" and isinstance(m["content"], str)]
        self.assertIn("(Harness: turn 1 ended; 400 tokens are spent so far.)", users)
        self.assertIn("(Harness: turn 2 ended; 1,400 of this puzzle's 4,000 tokens are spent.)", users)
        self.assertTrue(any(u.startswith("(Harness: no valid solution") and
                            u.endswith("explanation. 1,800 of this puzzle's 4,000 tokens are spent.)") for u in users))
        self.assertIn("(Harness: turn 3 ended; 2,200 of this puzzle's 4,000 tokens are spent. The budget is nearly spent "
                      "(about 1,800 tokens remain): record your answer with submit(...) now.)", users)

    def test_runner_flags(self):
        out, score, rows = TestRunnerV1.run_main(self, ["--lanes", "1", "--limit", "1", "--prompt", "v2d", "--marker", "v2d",
                                                        "--reply-caps", "v2c", "--cue-from", "120000", "--budget",
                                                        "160000", "--notice", "0", "--no-call", "keep_going",
                                                        "--observations", "turns", "--policy", "rank",
                                                        "--max-passes", "1"])
        self.assertEqual([r["end"] for r in rows], ["stalled"])      # the default script's identity never verifies


class TestV2e(unittest.TestCase):
    """v2e (the user, 2026-10-29): the system prompt says the preloaded names are used inside python; the unknown-tool
    message says how to call them; one history() reminder in the first harness message from 120K."""

    def test_prompts_unknown_tool_and_reminder(self):
        from arc2.fake import FakeCompletion
        texts = load_texts()
        self.assertIn("to use inside your python code (they are not separate tools)", texts["system_v2e"])
        self.assertNotIn("budget of", texts["system_v2e"])
        self.assertIn("budget of", texts["system_v2e_budget"])
        self.assertIn("to use inside your python code", texts["system_v2e_budget"])

        class NamedSession(FakeSession):                   # ("named", name): a call to a tool other than python
            def generate(self, external):
                if self.script and self.script[0][0] == "named":
                    step = self.script.pop(0)
                    self.received.append(external)
                    self.calls += 1
                    n = self.tokens_per_reply
                    msg = {"role": "assistant", "content": "",
                           "tool_calls": [{"id": f"call_{self.calls}", "type": "function",
                                           "function": {"name": step[1], "arguments": json.dumps({"answer": []})}}]}
                    return FakeCompletion(message=msg, finish_reason="tool_calls",
                                          usage=dict(prompt_tokens=0, completion_tokens=n), token_ids=[0] * n,
                                          assistant_token_ids=[0] * n, pending=False)
                return super().generate(external)
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        s = NamedSession([("code", "x = 1"), ("pause", 600), ("code", "y = 2"), ("named", "submit"),
                          ("code", "submit([x['input'] for x in test])"), ("final", "Done.")])
        res = run_attempt(task, s, k, texts, budget=4000, notice=0, no_call="keep_going", observe_turns=True,
                          marker_key="turn_marker_v2d", limit_at=100000, recall_at=1000,
                          text_keys=dict(keep_going="keep_going_v2d", recovery_repeat="recovery_repeat_v2d",
                                         nudge_format="nudge_format_v2d", unknown_tool="unknown_tool_v2e"))
        self.assertEqual(res.end, "finished")
        msgs = [m for ext in s.received for m in ext]
        tools = [m["content"] for m in msgs if m.get("role") == "tool"]
        self.assertIn("Unknown tool 'submit'; the only tool is python. check, submit, note, history, source and show are "
                      "Python functions: call them inside your python code.", tools)
        users = [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m["content"], str)]
        self.assertIn("(Harness: turn 1 ended; 400 tokens are spent so far.)", users)
        self.assertIn("(Harness: turn 2 ended; 1,400 tokens are spent so far. Earlier turns may no longer be in your "
                      "context; history() shows your checks, submissions, notes and functions so far.)", users)
        self.assertEqual(sum("history() shows" in u for u in users), 1)

    def test_runner_flags(self):
        out, score, rows = TestRunnerV1.run_main(self, ["--lanes", "1", "--limit", "1", "--prompt", "v2e", "--marker", "v2e",
                                                        "--reply-caps", "v2c", "--cue-from", "120000", "--budget",
                                                        "160000", "--notice", "0", "--no-call", "keep_going",
                                                        "--observations", "turns", "--policy", "rank",
                                                        "--max-passes", "1"])
        self.assertEqual([r["end"] for r in rows], ["stalled"])


class TestAdmission(unittest.TestCase):
    """Pass-start admission on the server's KV-pool gauge (the user, 2026-10-29: 32 lanes with admission)."""

    def test_gauge_parse_wait_and_unreadable(self):
        from arc2.runner import PoolGauge, kv_usage
        text = ('# HELP x\nvllm:num_requests_running{engine="0"} 26.0\n'
                'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.8712\n')
        self.assertAlmostEqual(kv_usage(text), 0.8712)
        self.assertIsNone(kv_usage("# fetch failed: URLError()\n"))
        seq = iter([0.91, 0.88, 0.70])
        g = PoolGauge("u", 0.85, max_age_s=0.0,
                      fetcher=lambda: f'vllm:kv_cache_usage_perc{{engine="0"}} {next(seq)}\n')
        waited = g.wait(step_s=0.01)
        self.assertGreater(waited, 0.0)
        self.assertEqual((g.waits, g.value), (1, 0.70))
        self.assertTrue(PoolGauge("u", 0.85, fetcher=lambda: "# fetch failed\n").admits())
        self.assertEqual(PoolGauge("u", 0.85, fetcher=lambda: 1 / 0).wait(step_s=0.01), 0.0)   # errors admit


class TestStartRefusal(unittest.TestCase):
    """v2g (2026-10-29, run08a's incident): a refused session start is not a pass; open sessions close on SIGTERM."""

    def test_refused_start_is_retried_not_recorded(self):
        from arc2 import runner

        class PersistentProtocolError(Exception):
            pass

        class Refusing(FakeSession):
            def generate(self, external):
                raise PersistentProtocolError("429 Client Error: Too Many Requests for url: http://x/arc3/persistent/start")
        made = []

        def make(self, log):
            made.append(1)
            return Refusing(log=log) if len(made) == 1 else FakeSession(log=log)
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        out = Path(d) / "run"
        orig = runner.Session.make
        runner.Session.make = make
        try:
            runner.main(["--out", str(out), "--fake", "--status-every", "1000", "--timeline-every", "0", "--deadline-s", "3600",
                         "--lanes", "1", "--limit", "1", "--policy", "rank", "--max-passes", "1"])
        finally:
            runner.Session.make = orig
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual([(r["pass_no"], r["end"]) for r in rows], [(1, "finished")])
        tl = [json.loads(l) for l in (out / "timeline.jsonl").read_text().splitlines()]
        self.assertEqual(tl[-1]["start_refusals"], 1)
        self.assertEqual(len(made), 2)

    def test_close_open(self):
        from arc2.runner import close_open
        closed = []

        class Obj:
            def __init__(self, name, fail=False):
                self.name, self.fail = name, fail

            def close(self):
                if self.fail:
                    raise RuntimeError("x")
                closed.append(self.name)
        items = {1: (Obj("s1"), Obj("k1")), 2: (Obj("s2", fail=True), Obj("k2")), 3: (Obj("s3"), Obj("k3"), dict(kind="solve"))}
        self.assertEqual(close_open(items), 2)                         # v2n: items may carry a context
        self.assertEqual(sorted(closed), ["k1", "k2", "k3", "s1", "s3"])


class TestV2n(unittest.TestCase):
    """v2n (the reviewer session's input 22; the user, 2026-11-01: "fold it in"): a pass open at SIGTERM keeps the answers
    it saved; under --phase3-rule valid the reviewer is shown the pair review_open judged."""

    def test_provisional_row(self):
        from arc2.runner import provisional_row
        import numpy as np
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        self.assertIsNone(provisional_row(task, k, dict(kind="solve", pass_no=2, sched=None)))     # nothing saved yet
        k.run("def flip(g):\n    return np.fliplr(g)\nsubmit(flip)")
        row = provisional_row(task, k, dict(kind="solve", pass_no=2, sched={"lane": "easy"}), tokens=1234)
        x = np.array(task.test[0]["input"])
        self.assertEqual((row["kind"], row["pass_no"], row["end"], row["provisional"], row["generated_tokens"]),
                         ("solve", 2, "killed", True, 1234))
        self.assertEqual(row["submissions"][1][0], np.fliplr(x).tolist())
        self.assertEqual(row["submits"][0]["verified_by"], "flip")
        self.assertEqual(row["programs"]["1"]["fn"], "flip")
        self.assertEqual(row["score"], 1.0)
        rv = provisional_row(task, k, dict(kind="review", sched=None, labels={"A": 1}, previous=[{"label": "A"}]))
        self.assertEqual((rv["kind"], rv["labels"], rv["previous"]), ("review", {"A": 1}, [{"label": "A"}]))
        self.assertNotIn("pass_no", rv)

    def test_sigterm_records_saved_answers(self):
        """A pass that saved a verified answer and is still generating when the SIGTERM comes: its row (end 'killed') and
        its answer reach results.jsonl and submission.json before the exit (v2m lost both)."""
        import signal as sg
        import threading
        import time
        import numpy as np
        from arc2 import runner
        saved = threading.Event()
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"

        class Blocking(FakeSession):
            def generate(self, external):
                if self.script and self.script[0][0] == "block":
                    self.script.pop(0)
                    saved.set()
                    time.sleep(20)                  # in flight when the SIGTERM comes
                    raise RuntimeError("session closed")
                return super().generate(external)
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        d = tempfile.mkdtemp()
        out = Path(d) / "run"
        orig_make, orig_h = runner.Session.make, sg.getsignal(sg.SIGTERM)
        runner.Session.make = lambda self, log: Blocking([("code", flip), ("block", 0)], log=log)

        def kill():
            saved.wait(30)
            time.sleep(0.5)
            os.kill(os.getpid(), sg.SIGTERM)
        threading.Thread(target=kill, daemon=True).start()
        try:
            with self.assertRaises(SystemExit) as cm:
                runner.main(["--out", str(out), "--fake", "--status-every", "1000", "--timeline-every", "0", "--deadline-s", "3600",
                             "--lanes", "1", "--policy", "rank", "--max-passes", "1", "--challenges", str(TRAIN_CH),
                             "--solutions", str(TRAIN_SOL), "--ids", "@" + str(ids)])
            self.assertEqual(cm.exception.code, 143)
        finally:
            runner.Session.make = orig_make
            sg.signal(sg.SIGTERM, orig_h)
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual([(r["task_id"], r["pass_no"], r["end"], r["provisional"]) for r in rows], [(FLIPLR, 1, "killed", True)])
        x = np.array(train_task(FLIPLR).test[0]["input"])
        self.assertEqual(json.loads((out / "submission.json").read_text())[FLIPLR][0]["attempt_1"], np.fliplr(x).tolist())
        self.assertIn("recorded the saved answers of 1 open passes", (out / "terminated.txt").read_text())

    def test_sigterm_records_a_review_in_flight(self):
        """The reviewer session's ask (2026-11-01, on v2g's flush): a run killed while its review is generating, after the
        review saved an answer: the review's row (end 'killed') reaches results.jsonl and its answer reaches submission.json
        in P3's slot 2, beside the passes' slot 1."""
        import signal as sg
        import threading
        import time
        import numpy as np
        from arc2 import runner
        saved = threading.Event()
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        grids = "submit([x['input'] for x in test])"
        review = "print(len(previous))\nsubmit([np.rot90(x['input'], 2) for x in test])"

        class Blocking(FakeSession):
            def generate(self, external):
                if self.script and self.script[0][0] == "block":
                    self.script.pop(0)
                    saved.set()
                    time.sleep(20)                  # in flight when the SIGTERM comes
                    raise RuntimeError("session closed")
                return super().generate(external)
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        out = Path(tempfile.mkdtemp()) / "run"
        it = iter([[("pause", 5000), ("code", flip), ("final", "Done.")],         # pass 1: long, verified
                   [("code", grids), ("final", "Done.")],                          # pass 2: valid, distinct -> the review
                   [("code", review), ("block", 0)]])                              # the review: saves, then in flight
        orig_make, orig_h = runner.Session.make, sg.getsignal(sg.SIGTERM)
        runner.Session.make = lambda self, log: Blocking(next(it), log=log)

        def kill():
            saved.wait(60)
            time.sleep(0.5)
            os.kill(os.getpid(), sg.SIGTERM)
        threading.Thread(target=kill, daemon=True).start()
        try:
            with self.assertRaises(SystemExit) as cm:
                runner.main(["--out", str(out), "--fake", "--status-every", "1000", "--timeline-every", "0", "--deadline-s", "3600",
                             "--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                             "--ids", "@" + str(ids), "--review-long", "3000"])
            self.assertEqual(cm.exception.code, 143)
        finally:
            runner.Session.make = orig_make
            sg.signal(sg.SIGTERM, orig_h)
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        self.assertEqual([(r["kind"], r.get("pass_no"), r["end"], r.get("provisional", False)) for r in rows],
                         [("solve", 1, "finished", False), ("solve", 2, "finished", False), ("review", None, "killed", True)])
        self.assertEqual(sorted(p["pass_no"] for p in rows[-1]["previous"]), [1, 2])
        x = np.array(train_task(FLIPLR).test[0]["input"])
        entry = json.loads((out / "submission.json").read_text())[FLIPLR][0]
        self.assertEqual(entry["attempt_1"], np.fliplr(x).tolist())                  # the votes' slot 1
        self.assertEqual(entry["attempt_2"], np.rot90(x, 2).tolist())                # the killed review's saved answer
        self.assertIn("recorded the saved answers of 1 open passes", (out / "terminated.txt").read_text())

    def test_context_removal_fact(self):
        """v2n (the user, 2026-11-01): the solver and reviewer system prompts say that older turns leave the context."""
        texts = load_texts()
        fact = ("When this conversation grows long, older turns are removed from your context; history() keeps your checks, "
                "submissions, notes and functions.")
        for key in ("system_v2j", "system_v2j_budget", "system_review_v2j"):
            self.assertEqual(texts[key].count(fact), 1, key)
        self.assertTrue(texts["system_review_v2j"].startswith(texts["system_v2j"]))

    def test_review_sees_the_valid_pair(self):
        """--phase3-rule valid: pass 1 invalid, passes 2 and 3 valid and distinct: the reviewer sees passes 2 and 3 (v2m
        showed passes 1 and 2), and the length thresholds read the same pair."""
        ident = "def f(g):\n    return g\ncheck(f)\nsubmit(f)"
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        grids = "submit([x['input'] for x in test])"
        review = "print(len(previous))\nsubmit([np.rot90(x['input'], 2) for x in test])"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        scripts = [[("pause", 5000), ("code", ident), ("final", "Done.")],       # pass 1: long, not valid
                   [("pause", 5000), ("code", flip), ("final", "Done.")],        # pass 2: long, verified
                   [("code", grids), ("final", "Done.")],                        # pass 3: valid grids, distinct
                   [("code", review), ("final", "Done.")]]                       # the review
        argv = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                "--ids", "@" + str(ids), "--review-long", "3000", "--max-passes", "4", "--phase3-rule", "valid",
                "--phase3-priority", "third,more,review"]
        out, score, rows = TestRunnerV1.run_main(self, argv, scripts)
        self.assertEqual([(r["kind"], r.get("pass_no")) for r in rows], [("solve", 1), ("solve", 2), ("solve", 3), ("review", None)])
        self.assertEqual(sorted(p["pass_no"] for p in rows[-1]["previous"]), [2, 3])


class TestV2o(unittest.TestCase):
    """v2o (the reviewer session's input 23, item 2; the user, 2026-11-01: "add it as a flag"): --pass2-frame runs pass 2 on
    a colour-permuted (optionally also rotated or reflected) copy of the task and maps its answers back."""

    def test_frame_round_trip(self):
        import numpy as np
        from arc2 import represent as RP
        g = np.random.RandomState(0).randint(0, 10, size=(3, 5)).tolist()
        cmap = RP.task_key("x", 2)["colour_map"]
        for sym in sorted(RP.SYMMETRIES) + ["identity"]:
            key = dict(symmetry=sym, colour_map=cmap)
            self.assertEqual(RP.from_frame(RP.to_frame(g, key), key), g, sym)
        k = RP.pass_key(FLIPLR, 2)
        self.assertEqual((k["symmetry"], k["colour_map"][0]), ("identity", 0))
        self.assertNotEqual(k["colour_map"], list(range(10)))
        self.assertNotEqual(RP.pass_key(FLIPLR, 2, symmetry=True)["symmetry"], "identity")
        self.assertIsNone(RP.from_frame(None, k))
        self.assertEqual(RP.from_frame([[1, 2], [3]], k), [[1, 2], [3]])            # not a grid: unchanged
        self.assertEqual(RP.from_frame([[1, 12]], k), [[1, 12]])
        task = train_task(FLIPLR)
        shown = RP.framed_task(task, k)
        self.assertEqual((shown.task_id, shown.solutions, len(shown.train), len(shown.test)),
                         (task.task_id, None, len(task.train), len(task.test)))
        self.assertNotEqual(shown.sandbox_payload()["test"], task.sandbox_payload()["test"])
        self.assertEqual(RP.from_frame(shown.test[0]["input"], k), task.test[0]["input"])

    def run_pass2(self, scripts, extra, ids=(FLIPLR,)):
        f = Path(tempfile.mkdtemp()) / "ids.txt"
        f.write_text(",".join(ids))
        argv = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                "--ids", "@" + str(f), "--passes", "2", "--max-passes", "2"] + extra
        return TestRunnerV1.run_main(self, argv, scripts)

    def test_pass2_runs_in_the_frame_and_maps_back(self):
        import numpy as np
        flip = "def flip(g):\n    return np.fliplr(g)\ncheck(flip)\nsubmit(flip)"
        grids = "submit([x['input'] for x in test])"
        x = train_task(FLIPLR).test[0]["input"]
        for mode in ("colour", "colour_symmetry"):
            out, score, rows = self.run_pass2([[("code", flip), ("final", "Done.")], [("code", grids), ("final", "Done.")]],
                                              ["--pass2-frame", mode])
            solves = {r["pass_no"]: r for r in rows if r["kind"] == "solve"}
            self.assertEqual(sorted(solves), [1, 2])
            self.assertNotIn("frame", solves[1])
            self.assertEqual(solves[1]["score"], 1.0)
            p2 = solves[2]
            self.assertEqual(p2["frame"]["mode"], mode)
            self.assertEqual(p2["frame"]["symmetry"] == "identity", mode == "colour")
            self.assertEqual(p2["submissions"]["1"][0], x)                # the framed test input, mapped back
            self.assertEqual(p2["programs"], {})
            events = [json.loads(l) for l in (out / "tasks" / FLIPLR / "p2" / "log.jsonl").read_text().splitlines()]
            self.assertEqual([e for e in events if e.get("kind") == "frame"][0]["payload"]["mode"], mode)
            sub = json.loads((out / "submission.json").read_text())[FLIPLR][0]
            self.assertEqual(sub["attempt_1"], np.fliplr(np.array(x)).tolist())
        out, score, rows = self.run_pass2([[("code", flip), ("final", "Done.")], [("code", flip), ("final", "Done.")]],
                                          ["--pass2-frame", "colour"])
        p2 = [r for r in rows if r["kind"] == "solve" and r["pass_no"] == 2][0]
        self.assertEqual(p2["score"], 1.0)                                    # fliplr commutes with a colour map
        self.assertEqual(p2["programs_framed"]["1"]["fn"], "flip")
        self.assertEqual(p2["submits"][0]["verified_by"], "flip")

    def test_frame_tasks_parity_and_default_off(self):
        import zlib
        from arc2 import runner
        self.assertEqual(runner.parse(["--out", "x", "--fake"]).pass2_frame, "off")
        ids = (FLIPLR, ROT180, "00576224")
        framed = {}
        for half in ("odd", "even"):
            out, score, rows = self.run_pass2(None, ["--pass2-frame", "colour", "--pass2-frame-tasks", half], ids)
            p2 = {r["task_id"]: r for r in rows if r["kind"] == "solve" and r["pass_no"] == 2}
            self.assertEqual(sorted(p2), sorted(ids))
            framed[half] = {t for t, r in p2.items() if r.get("frame")}
            self.assertEqual(framed[half], {t for t in ids if (zlib.crc32(t.encode()) % 2 == 1) == (half == "odd")})
            self.assertFalse(any(r.get("frame") for r in rows if r.get("pass_no") == 1))
        self.assertEqual(framed["odd"] | framed["even"], set(ids))
        self.assertFalse(framed["odd"] & framed["even"])
        out, score, rows = self.run_pass2(None, [], ids)
        self.assertFalse(any(r.get("frame") or "programs_framed" in r for r in rows))

    def test_provisional_row_maps_back(self):
        from arc2 import represent as RP
        from arc2.runner import provisional_row
        task = train_task(FLIPLR)
        key = RP.pass_key(FLIPLR, 2)
        k = Kernel(RP.framed_task(task, key).sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        k.run("submit([x['input'] for x in test])")
        row = provisional_row(task, k, dict(kind="solve", pass_no=2, sched=None, frame=key, frame_mode="colour"))
        self.assertEqual({int(a): v for a, v in row["submissions"].items()}[1][0], task.test[0]["input"])
        self.assertEqual((row["programs"], row["frame"]["mode"], row["provisional"]), ({}, "colour", True))

    def test_review_shows_a_framed_answer_without_its_program(self):
        """The reviewer session's ask (2026-11-01): a framed pass is shown with its mapped-back grids, its explanation and its
        verified status; no program block, and the kind says why (no 'a program that passes ...' without a program)."""
        from arc2 import review as RV
        d = Path(tempfile.mkdtemp())
        ev = {"kind": "submit", "attempt": 1, "explanation": "mirror it"}
        for k in (1, 2):
            (d / f"p{k}.jsonl").write_text(json.dumps({"kind": "tool", "payload": {"events": [ev]}}) + "\n")
        sub = dict(attempt=1, fn="flip", train_passed=3, train_total=3, verified_by="flip")
        prog = {"1": dict(fn="flip", source="def flip(g):\n    return np.fliplr(g)", passed=3, total=3)}
        p1 = dict(pass_no=1, submissions={"1": [[[1]]]}, submits=[sub], programs=prog)
        p2 = dict(pass_no=2, submissions={"1": [[[2]]]}, submits=[sub], programs=prog,
                  frame=dict(symmetry="identity", colour_map=list(range(10)), mode="colour"))
        previous, labels = RV.previous_from_rows(p1, p2, lambda k: d / f"p{k}.jsonl", seed="t")
        by = {x["pass_no"]: x for x in previous}
        self.assertEqual((by[1]["source"] != "", by[1]["explanation"], by[1]["frame"]), (True, "mirror it", None))
        self.assertEqual((by[2]["source"], by[2]["explanation"], by[2]["verified"], by[2]["outputs"], by[2]["frame"]),
                         ("", "mirror it", True, [[[2]]], "colour"))
        texts = load_texts()
        previous = [dict(x, kind_text=RV.answer_kind(texts, x["verified"], x["fn"], x["passed"], x["total"], x.get("frame")))
                    for x in previous]
        block = RV.review_block(previous, texts)
        lab = {x["pass_no"]: x["label"] for x in previous}
        self.assertIn(f"Attempt {lab[2]}, answer 1: " + texts["review_kind_framed"].format(kind=texts["review_kind_verified"]),
                      block)
        self.assertIn(f"Attempt {lab[1]}, answer 1: " + texts["review_kind_verified"] + ".", block)
        self.assertIn(f"Attempt {lab[1]}, answer 1, program:", block)
        self.assertNotIn(f"Attempt {lab[2]}, answer 1, program:", block)
        self.assertTrue(block.endswith(texts["review_tail_framed"]))
        self.assertEqual(RV.answer_kind(texts, False, "grids", None, 3, "colour_symmetry"),
                         texts["review_kind_framed_symmetry"].format(kind=texts["review_kind_grids"]))
        plain = [dict(x, frame=None, kind_text=RV.answer_kind(texts, x["verified"], x["fn"], x["passed"], x["total"]))
                 for x in previous]
        self.assertTrue(RV.review_block(plain, texts).endswith(texts["review_tail"]))


class TestV2p(unittest.TestCase):
    """v2p (the user, 2026-11-01, by gain per token in six eval runs): phase 3 in the order third (identical answers), more1 (one
    valid pass), review (two distinct valid answers), more0 (no valid pass); --phase3-priority third,more,review is unchanged."""

    def test_phase3_priority_split(self):
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        ident = "def f(g):\n    return g\ncheck(f)\nsubmit(f)"
        grids = "submit([x['input'] for x in test])"
        rgrids = "submit([np.rot90(x['input'], 2) for x in test])"
        review = "print(len(previous))\nsubmit([np.rot90(x['input'], 2) for x in test])"
        A, B, C, D = FLIPLR, ROT180, "00576224", "007bbfb7"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(",".join([A, B, C, D]))
        S = {k: [("code", v), ("final", "Done.")] for k, v in dict(flip=flip, ident=ident, grids=grids, rgrids=rgrids,
                                                                     review=review).items()}
        first = [[("pause", 5000)] + S["flip"],              # A1: long, verified
                 [("pause", 5000)] + S["ident"],             # B1: long, not valid
                 [("pause", 6000)] + S["grids"],             # C1: a valid grid answer
                 [("pause", 7000)] + S["grids"]]             # D1: a valid grid answer
        second = [S["flip"], S["rgrids"], S["ident"], S["ident"]]   # pass 2 in rank order A, C, D, B: A identical (third), C
        #                                                             distinct (review), D one valid (more1), B none (more0)
        common = ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions", str(TRAIN_SOL),
                  "--ids", "@" + str(ids), "--extra-pass-identical-long", "3000", "--review-long", "3000", "--max-passes", "4",
                  "--phase3-rule", "valid", "--phase3-length", "mean"]
        seen = lambda rows: [(r["task_id"], r["kind"], r.get("pass_no")) for r in rows]
        head = [(A, "solve", 1), (B, "solve", 1), (C, "solve", 1), (D, "solve", 1),
                (A, "solve", 2), (C, "solve", 2), (D, "solve", 2), (B, "solve", 2)]
        out, score, rows = TestRunnerV1.run_main(self, common + ["--phase3-priority", "third,more1,review,more0"], first + second + [
            S["flip"], S["ident"], S["review"], S["ident"], S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (D, "solve", 3), (C, "review", None), (B, "solve", 3),
                                             (D, "solve", 4), (B, "solve", 4)])
        out, score, rows = TestRunnerV1.run_main(self, common + ["--phase3-priority", "third,more,review"], first + second + [
            S["flip"], S["ident"], S["ident"], S["review"], S["ident"], S["ident"]])
        self.assertEqual(seen(rows), head + [(A, "solve", 3), (D, "solve", 3), (B, "solve", 3), (C, "review", None),
                                             (D, "solve", 4), (B, "solve", 4)])        # v2m: more (D, B by rank) before review
        for bad in ("third,more1,review", "third,more,more1,review,more0", "third,more1,review,more"):
            with self.assertRaises(SystemExit):
                TestRunnerV1.run_main(self, common + ["--phase3-priority", bad], first)



class TestV2q(unittest.TestCase):
    """v2q (the user, 2026-11-01: "Can we add a recovery (and maybe stop and park in the queue if it does not recover)?"):
    replies whose tool calls all fail count in a row; 4 add the recovery note, 8 end the pass ('stalled') and the task takes
    its next pass from the queue."""

    def run1(self, scripts, passes="1"):
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        return TestRunnerV1.run_main(self, ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH), "--solutions",
                                            str(TRAIN_SOL), "--ids", "@" + str(ids), "--passes", passes, "--valid-target", "1",
                                            "--max-passes", passes], scripts)

    def test_recovery_note_then_normal_finish(self):
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        out, score, rows = self.run1([[("args", "{}")] * 4 + [("code", flip), ("final", "Done.")]])
        self.assertEqual([(r["pass_no"], r["end"]) for r in rows], [(1, "finished")])
        self.assertEqual(rows[0]["nudges"].get("invalid_recovery"), 1)
        self.assertNotIn("invalid_stop", rows[0]["nudges"])
        self.assertEqual(score["score"], 1.0)

    def test_three_failed_calls_do_nothing(self):
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        out, score, rows = self.run1([[("args", "{}")] * 3 + [("code", flip), ("final", "Done.")]])
        self.assertEqual(rows[0]["end"], "finished")
        self.assertNotIn("invalid_recovery", rows[0]["nudges"])

    def test_stop_at_eight_and_next_pass_from_the_queue(self):
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        out, score, rows = self.run1([[("args", "{}")] * 3 + [("args", "not json")] + [("args", "{}")] * 4 + [("code", flip)],
                                      [("code", flip), ("final", "Done.")]], passes="2")
        self.assertEqual([(r["pass_no"], r["end"]) for r in rows], [(1, "stalled"), (2, "finished")])
        self.assertEqual((rows[0]["nudges"].get("invalid_recovery"), rows[0]["nudges"].get("invalid_stop")), (1, 1))
        self.assertEqual(rows[0]["tool_calls"], 0)
        self.assertEqual(score["score"], 1.0)

    def test_recovery_text_reaches_the_model(self):
        from arc2 import runner
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        made = []
        orig = runner.Session.make
        def make(self, log):
            made.append(FakeSession([("args", "{}")] * 4 + [("code", flip), ("final", "Done.")], log=log))
            return made[-1]
        runner.Session.make = make
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        try:
            runner.main(["--out", str(Path(d) / "run"), "--fake", "--status-every", "1000", "--timeline-every", "0",
                         "--deadline-s", "3600", "--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH),
                         "--solutions", str(TRAIN_SOL), "--ids", "@" + str(ids), "--passes", "1", "--valid-target", "1",
                         "--max-passes", "1", "--observations", "turns"])
        finally:
            runner.Session.make = orig
        texts = [json.dumps(m) for ext in made[0].received for m in ext]
        hits = [i for i, x in enumerate(texts) if "ran nothing" in x]
        self.assertEqual(len(hits), 1)
        self.assertEqual(sum(1 for ext in made[0].received for m in ext if m.get("role") == "user" and "ran nothing" in json.dumps(m)), 1)

    def test_off_with_zero(self):
        flip = "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"
        ids = Path(tempfile.mkdtemp()) / "ids.txt"
        ids.write_text(FLIPLR)
        out, score, rows = TestRunnerV1.run_main(self, ["--lanes", "1", "--policy", "rank", "--challenges", str(TRAIN_CH),
                                                        "--solutions", str(TRAIN_SOL), "--ids", "@" + str(ids), "--passes", "1",
                                                        "--valid-target", "1", "--max-passes", "1",
                                                        "--invalid-recover", "0", "--invalid-limit", "0"],
                                                  [[("args", "{}")] * 9 + [("code", flip), ("final", "Done.")]])
        self.assertEqual(rows[0]["end"], "finished")
        self.assertNotIn("invalid_recovery", rows[0]["nudges"])

class TestV2h(unittest.TestCase):
    """v2h (the user, 2026-10-29: fix the two mismatches; production values elsewhere)."""

    def test_check_shows_test_grids_and_tool_description(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        r = json.loads(k.run("def flip(g):\n    return np.fliplr(g)\ncheck(flip)").text)
        self.assertIn("test input 0 -> 3x3:\nbMO\nMOM\nRRM", r["stdout"])
        r = json.loads(k.run("def bad(g):\n    return 1 / 0\ncheck(bad)").text)
        self.assertIn("test input 0: error", r["stdout"])
        from arc2 import runner
        d = runner.TOOL["function"]["description"]
        self.assertIn("note", d)
        self.assertNotIn("programs", d)
        self.assertEqual(runner.parse(["--out", "x", "--fake"]).evict_fraction, 0.2)


class TestForecastAdmission(unittest.TestCase):
    """Length-aware admission (the user, 2026-10-29): the pool forecast over the horizon decides a pass start."""

    def test_projection_admits_and_blocks(self):
        import math
        from arc2.runner import Forecast
        fc = Forecast(pool=755, reserve=8, margin=30, horizon_s=900, z=0.84, sig_task=0.6, sig_prior=0.8,
                      prior_tokens=45000, rho=1.2, opener=5000, cap_pages=30, speed=60.0)
        self.assertEqual(fc.pages(60000), 23)
        self.assertEqual(fc.pages(500000), 30)
        long_ = dict(g=50000.0, res=60000.0, mu=math.log(150000), sig=0.6)       # grows to the cap: +7 pages
        short = dict(g=15000.0, res=28000.0, mu=math.log(20000), sig=0.6)        # ends within the horizon: -13 pages
        new = fc.mu_sig([])                                                      # first pass: the prior
        self.assertEqual(new, (math.log(45000), 0.8))
        # worst point is 1/4 of the horizon (13,500 tokens): the short pass still runs (+5), the long one +5, the new
        # pass 11 pages: 621; at the full horizon 600 + 7 - 13 + 26 = 620
        self.assertEqual(fc.projected(600, [long_, short], new), 621)
        self.assertTrue(fc.admits(600, [long_, short], new))
        self.assertFalse(fc.admits(700, [long_, short], new))                    # 721 > 755 - 8 - 30
        past = dict(g=120000.0, res=82000.0, mu=math.log(60000), sig=0.6)       # past its cautious length: runs on
        self.assertEqual(fc.length(past["mu"], past["sig"], past["g"]), 180000.0)
        self.assertEqual(fc.mu_sig([20000, 80000])[0], (math.log(20000) + math.log(80000)) / 2)
        # a waiting hard pass reserves room: the same easy pass no longer fits
        self.assertTrue(fc.admits(690, [long_, short], new))                    # 711 <= 717
        self.assertFalse(fc.admits(690, [long_, short], new, pending=[fc.mu_sig([150000])]))   # 736 > 717

    def test_progress_callback(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        seen = []
        run_attempt(task, FakeSession([("code", "submit([x['input'] for x in test])"), ("final", "Done.")]), k,
                    load_texts(), budget=4000, notice=0, progress=lambda g, res: seen.append((g, res)))
        self.assertEqual(seen, [(400, 0), (800, 0)])


class TestV2j(unittest.TestCase):
    """v2j (the user, 2026-10-29: build A-D from the review of the 13 failing tasks)."""

    def kernel(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        return task, k

    def test_hidden_verification_and_submit_grids(self):
        task, k = self.kernel()
        r = json.loads(k.run("def flip(g):\n    return np.fliplr(g)\nsubmit(flip)").text)
        self.assertIn("demonstrations: 3 of 3 pass", r["stdout"])
        self.assertIn("test input 0 -> 3x3:\nbMO\nMOM\nRRM", r["stdout"])                    # C
        look = ("def look(g):\n    for p in train:\n        if np.array_equal(p['input'], g):\n"
                "            return p['output']\n    return np.fliplr(g)[::-1]\ncheck(look)")
        r = json.loads(k.run(look).text)
        self.assertIn("0 of 3 demonstration pairs pass", r["stdout"])                         # D: a lookup
        self.assertIn("passes only when the function can read this pair from train", r["stdout"])
        r = json.loads(k.run("def peek(g):\n    return np.fliplr(g) if len(test) else g\ncheck(peek)").text)
        self.assertIn("0 of 3 demonstration pairs pass", r["stdout"])                         # D: reads test
        r = json.loads(k.run("def t0(g):\n    _ = test[0]\n    return np.fliplr(g)\ncheck(t0)").text)
        self.assertIn("IndexError", r["stdout"])                                               # the error path
        self.assertIn("passes only when the function can read this pair from train", r["stdout"])
        r = json.loads(k.run("submit(look, attempt=2)").text)
        self.assertIn("demonstrations: 0 of 3 pass", r["stdout"])
        recs = [x for x in k.records if x["kind"] == "submit"]
        self.assertEqual([bool(x.get("verified_by")) for x in recs], [True, False])

    def test_note_guard_and_earlier_answers(self):
        from arc2.agent import earlier_answers
        from arc2 import slots as S
        task, k = self.kernel()

        class NoteSession(FakeSession):                    # the 2nd reply without a call ends right after a note
            n = 0

            def generate(self, external):
                c = super().generate(external)
                if not c.message.get("tool_calls"):
                    NoteSession.n += 1
                    if NoteSession.n == 1:
                        c.cues = ("soft_nudge",)
                return c
        grids = "submit([x['input'] for x in test])"
        s = NoteSession([("code", grids), ("code", "submit(lambda g: np.fliplr(g))"), ("final", "a"), ("final", "b")],
                        tokens_per_reply=40)
        res = run_attempt(task, s, k, load_texts(), budget=4000, notice=0, no_call="keep_going", observe_turns=True,
                          note_guard=True)
        self.assertEqual(res.end, "finished")
        self.assertEqual(res.nudges, {"note_continue": 1})
        users = [m["content"] for ext in s.received for m in ext if m.get("role") == "user" and isinstance(m["content"], str)]
        self.assertTrue(any(u.startswith("(Harness: the previous reply ended right after a harness note") for u in users))
        self.assertFalse(any(u.startswith("(Harness: attempt 2 is") for u in users))           # reverted (2026-10-30)
        self.assertEqual(len(res.earlier), 1)                                                 # the replaced grid answer
        row = dict(submissions={str(a): v for a, v in res.submissions.items()}, submits=res.submits, earlier=res.earlier)
        outs = S.slot_outputs([row], 1)
        self.assertEqual(outs[1], res.submissions[1])
        self.assertEqual(outs[2], [res.earlier[0]["outputs"][0]])


class TestV2j2(unittest.TestCase):
    """v2j2 (the reviewer, the user's OK): a function that reads test (directly or through a helper) does not verify; the
    system prompt's check() line states the hidden-pair rule."""

    def test_reads_test_rule_and_prompt_clause(self):
        task = train_task(FLIPLR)
        k = Kernel(task.sandbox_payload(), timeout_s=10)
        k.start()
        self.addCleanup(k.close)
        loop = ("def via_loop(g):\n    for t in test:\n        pass\n    return np.fliplr(g)\ncheck(via_loop)")
        r = json.loads(k.run(loop).text)
        self.assertIn("3 of 3 demonstration pairs pass", r["stdout"])
        self.assertIn("does not count as passing: via_loop reads the test inputs", r["stdout"])
        self.assertTrue(r["stdout"].rstrip().endswith("False"))
        helper = ("def helper(g):\n    return len(test)\ndef outer(g):\n    helper(g)\n    return np.fliplr(g)\n"
                  "submit(outer)")
        r = json.loads(k.run(helper).text)
        self.assertIn("does not count as passing: it reads the test inputs", r["stdout"])
        r = json.loads(k.run("def clean(g):\n    return np.fliplr(g)\nsubmit(clean, attempt=2)").text)
        self.assertNotIn("does not count as passing", r["stdout"])
        recs = [x for x in k.records if x["kind"] == "submit"]
        self.assertEqual([bool(x.get("verified_by")) for x in recs], [False, True])
        texts = load_texts()
        for key in ("system_v2j", "system_v2j_budget"):
            self.assertIn("Each pair is checked with only the other pairs in train and with test empty", texts[key])
        self.assertNotIn("budget of", texts["system_v2j"])
        self.assertIn("budget of", texts["system_v2j_budget"])


class TestV2l(unittest.TestCase):
    """v2l (the user, 2026-10-30): v2j2 with the v2k changes reverted; the reviewer pass's P3 rule and answer set; a start
    that generated nothing is not a pass; review sessions' own status-note thresholds."""

    def test_reverted_texts_and_flags(self):
        import contextlib
        import io
        from arc2 import runner
        texts = load_texts()
        for k in ("system_v2k", "keep_going_v2k", "grid_followup_v2k", "grid_window_v2k", "attempt2_note",
                  "system_review_v2k"):
            self.assertNotIn(k, texts)
        for flag in (["--grid-followup", "100"], ["--attempt2-note"], ["--prompt", "v2k"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                runner.parse(["--out", "x"] + flag)

    def test_p3_reviewed_outputs(self):
        from arc2 import slots as S
        outs = {1: [[[1]], [[5]]], 2: [[[2]], None]}
        got = S.reviewed_outputs(outs, {"1": [[[3]], [[5]]], "2": [[[4]], [[6]]]})
        self.assertEqual(got, {1: [[[1]], [[5]]], 2: [[[3]], [[6]]]})   # attempt 1 when it differs, else attempt 2
        self.assertEqual(S.reviewed_outputs(outs, {1: [[[1]], [[5]]]}), outs)        # nothing new: the votes' slot 2
        self.assertEqual(S.reviewed_outputs(outs, None), outs)

    def test_previous_from_rows(self):
        from arc2.review import previous_from_rows
        d = Path(tempfile.mkdtemp())
        (d / "p1.jsonl").write_text(json.dumps(dict(kind="tool", payload=dict(
            events=[dict(kind="submit", attempt=1, explanation="mirror")]))) + "\n")
        p1 = dict(pass_no=1, submissions={"1": [[[1]]]}, programs={"1": dict(source="def flip(g): pass")},
                  submits=[dict(attempt=1, fn="flip", verified_by="flip", train_passed=3, train_total=3)])
        p2 = dict(pass_no=2, submissions={1: [[[2]]], 2: [[[3]]]}, programs={},
                  submits=[dict(attempt=1, fn="grids"), dict(attempt=2, fn="grids")])
        prev, labels = previous_from_rows(p1, p2, lambda k: d / f"p{k}.jsonl", seed="x")
        self.assertEqual(sorted(labels.values()), [1, 2])
        self.assertEqual(len(prev), 3)
        a = next(x for x in prev if x["pass_no"] == 1)
        self.assertEqual((a["explanation"], a["verified"], a["source"]), ("mirror", True, "def flip(g): pass"))

    def test_start_refused_rule(self):
        from arc2.agent import AttemptResult
        from arc2.runner import start_refused
        r = AttemptResult("t")
        r.end, r.generated_tokens = "error", 0
        self.assertTrue(start_refused(r))                    # e.g. a connection error while the server restarts
        r.end = "finished"
        self.assertFalse(start_refused(r))
        r.end, r.generated_tokens = "error", 10
        self.assertFalse(start_refused(r))

    def test_review_status_note_thresholds(self):
        from arc2.shared import CAP_STATE, v2c_decision
        texts = load_texts()

        def original(cumulative, think_closed, soft, hard, cued, **kw):
            return 1, None, None
        d = v2c_decision(original, close=32000, ext=4096, status_every=10000, status_text=texts["status_cue_v2c"],
                         texts=texts, wrap_at=120000, cue_from=120000)

        def reset():
            CAP_STATE.cue_from = CAP_STATE.wrap_at = CAP_STATE.used = CAP_STATE.budget = None
        self.addCleanup(reset)
        reset()
        CAP_STATE.used, CAP_STATE.budget = 50000, 60000
        self.assertIsNone(d(4000, False, 31999, 32000, False, check=8000)[1])          # the passes' 120K: no note
        CAP_STATE.cue_from, CAP_STATE.wrap_at = 45000, 45000                          # a review's thresholds
        inj = d(4000, False, 31999, 32000, False, check=8000)[1]
        self.assertIn("50,000 of this puzzle's 60,000 tokens are spent", inj["text"])
        self.assertIn("10,000 tokens remain", inj["text"])


class TestReview(unittest.TestCase):
    """The reviewer session (the user's design, 2026-10-30): the opener carries both attempts' answers, the kernel has them
    as previous, and the review saves its own attempts like a pass."""

    def test_review_session(self):
        from arc2.review import answer_kind, review_messages, run_review
        texts = load_texts()
        task = train_task(FLIPLR)
        flip = [[list(reversed(r)) for r in t["input"]] for t in task.test]
        same = [t["input"] for t in task.test]
        previous = [dict(label="A", attempt=1, outputs=flip, source="def flip(g):\n    return np.fliplr(g)",
                         explanation="mirror left-right", verified=True, fn="flip", passed=3, total=3),
                    dict(label="B", attempt=1, outputs=same, source="", explanation="", verified=False, fn="grids",
                         passed=None, total=None)]
        for p in previous:
            p["kind_text"] = answer_kind(texts, p["verified"], p["fn"], p["passed"], p["total"])
        msgs = review_messages(task, previous, texts, budget=60_000, reply_cap=32_000)
        self.assertIn("This is a review.", msgs[0]["content"])
        # the reverted base (the user, 2026-10-30): v2j's system prompt, without v2k's two sentences
        self.assertTrue(texts["system_review_v2j"].startswith(texts["system_v2j"]))
        self.assertNotIn("two people", msgs[0]["content"])
        self.assertNotIn("last resort", msgs[0]["content"])
        body = msgs[1]["content"][-1]["text"]
        for frag in ("Two independent attempts, A and B", "Attempt A, answer 1: a program that passes every demonstration",
                     "Its explanation: mirror left-right", "Attempt B, answer 1: output grids without a program",
                     "Its explanation: (none recorded)", "Attempt A, answer 1, program:", "preloaded in python as previous"):
            self.assertIn(frag, body)
        self.assertEqual(msgs[1]["_arc3_pin_parts"], 1)
        s = FakeSession([("code", "print(len(previous), previous[0]['label'], previous[1]['source'] == '')"),
                         ("code", "def flip(g):\n    return np.fliplr(g)\nsubmit(flip)"), ("final", "done")],
                        tokens_per_reply=500)
        res, kernel = run_review(task, previous, s, texts, budget=60_000, no_call="keep_going", observe_turns=True,
                                 reply_cap=32_000)
        self.addCleanup(kernel.close)
        self.assertEqual(res.end, "finished")
        self.assertTrue(res.submits[-1]["verified_by"])
        tool_texts = [m["content"] for ext in s.received for m in ext if m.get("role") == "tool"]
        self.assertTrue(any("2 A True" in x for x in tool_texts))

class TestImages(unittest.TestCase):
    """Grid images as an aid beside the text (the user, 2026-10-29)."""

    def test_render_sizes_and_pixels(self):
        import io
        from PIL import Image
        from arc2 import images as I
        self.assertEqual(I.canvas_size(30, 30), (480, 480))
        self.assertEqual(I.canvas_size(3, 3), (256, 256))
        self.assertEqual(I.canvas_size(2, 20), (320, 256))
        self.assertEqual(I.canvas_size(17, 17), (288, 288))                # 272 px rounded up to 32-px tokens
        self.assertEqual(I.image_tokens([[0] * 30] * 30), 225)
        img = Image.open(io.BytesIO(I.grid_png([[1, 2], [3, 9]])))
        self.assertEqual(img.size, (256, 256))
        self.assertEqual(img.getpixel((0, 0)), I.PALETTE[1])
        self.assertEqual(img.getpixel((15, 15)), I.PALETTE[1])
        self.assertEqual(img.getpixel((16, 0)), I.PALETTE[2])
        self.assertEqual(img.getpixel((31, 31)), I.PALETTE[9])
        self.assertEqual(img.getpixel((32, 0)), I.PAD)

    def test_opener_and_prompt(self):
        texts = load_texts()
        task = train_task(FLIPLR)
        plain = opener_message(task, texts)
        self.assertEqual((len(plain["content"]), plain["_arc3_pin_parts"]), (2, 1))
        msg = opener_message(task, texts, image_cell=16)
        kinds = [p["type"] for p in msg["content"]]
        n_pairs, n_tests = len(task.train), len(task.test)
        n_img = 2 * n_pairs + n_tests
        self.assertEqual(len(kinds), 2 + 2 * n_img)                           # the 2 text parts, then caption + image each
        self.assertEqual(kinds[:2], ["text", "text"])
        self.assertEqual(kinds[2:], ["text", "image_url"] * n_img)
        self.assertEqual((msg["_arc3_pin_parts"], msg["_arc3_pin_split"]), (len(kinds), 2))   # v7.7: text pin + board pin
        self.assertEqual(msg["content"][:2], plain["content"])                # the same text parts as without images
        off = system_message(texts, budget=200_000, notice=8_000, key="system_v2", reply_cap=32_000)["content"]
        on = system_message(texts, budget=200_000, notice=8_000, key="system_v2", reply_cap=32_000, image_cell=16)["content"]
        self.assertNotIn("image", off)
        self.assertIn("16 by 16 pixel block", on)
        self.assertIn("never an image", on)
        self.assertEqual(on.replace(texts["images_note"].format(cell=16), ""), off)

if __name__ == "__main__":
    unittest.main()
