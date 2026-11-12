"""Bounded experimental persistent-state gameplay using the existing runner.

Example: --persistent-base-url http://127.0.0.1:1234 --games ID1,ID2
--total-seconds 300 --context 40000 --expected-sequences 36 --expected-mtp 0
--context-profile nvfp4 --expected-gpu-memory-utilization 0.95 --active-games 0
--request-slots 0 --clock-mode gameplay --shutdown-margin 0 --no-token-limit
--no-response-cap --response-headroom-fraction 0.10 --min-p 0 --reasoning-effort
xhigh --image-scale 4 --stdout-token-limit 1024 --output PATH [--execute]

No model generation occurs without the inherited explicit --execute flag.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace


def _option(argv, name, default=None):
    matches = [i for i, arg in enumerate(argv) if arg == name or arg.startswith(name + "=")]
    if len(matches) > 1:
        raise ValueError(f"Repeated {name} is not allowed")
    if not matches:
        return default
    i = matches[0]
    if "=" in argv[i]:
        return argv[i].split("=", 1)[1]
    if i + 1 >= len(argv) or argv[i+1].startswith("--"):
        raise ValueError(f"{name} needs a value")
    return argv[i+1]


def validate_scope(argv):
    seconds = float(_option(argv, "--total-seconds", "nan"))
    games = _option(argv, "--games", "all").split(",")
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("This first persistent control requires explicit --total-seconds in (0, 600]")
    if "all" in games or not 1 <= len(games) <= 4 or len(set(games)) != len(games) or not all(games):
        raise ValueError("Select one to four distinct public games explicitly with --games")
    if _option(argv, "--replicas", "1") != "1":
        raise ValueError("The first persistent control uses unique games, without replicas")
    if _option(argv, "--expected-mtp", "0") != "0":
        raise ValueError("Persistent state currently supports only MTP0")
    if _option(argv, "--clock-mode", "entry") != "gameplay":
        raise ValueError("Use --clock-mode gameplay so preparation is outside this bounded run")
    if "--no-token-limit" not in argv or "--no-response-cap" not in argv:
        raise ValueError("This control requires --no-token-limit and --no-response-cap")
    if _option(argv, "--output") is None:
        raise ValueError("Choose an explicit new --output directory for this experiment")
    if _option(argv, "--expected-mtp") is None:
        argv = [*argv, "--expected-mtp", "0"]
    return argv


def _history_settings(policy="oldest_interactions", fraction=0.10):
    if (policy not in {"off", "oldest_interactions"} or type(fraction) not in (float,int)
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("Require off/oldest_interactions and finite --kv-evict-fraction in (0,1)")
    return dict(policy=policy, fraction=float(fraction))


def apply_persistent_settings(settings, *, base_url, history_eviction=None):
    config = _history_settings(**(history_eviction or {}))
    settings["transport"] = "experimental persistent session HTTP; external message deltas after start"
    settings["reasoning_continuation"] = "exact server assistant-token ledger; cumulative dispatch parse"
    settings["persistent_state"] = dict(
        phase="oldest_interaction_eviction" if config["policy"]!="off" else "managed_suffix_eviction",
        base_url=base_url, prefix_caching=False, history_eviction=config,
        text_truncation=False, kv_eviction=True,
        suffixes="frozen once per real observation; previous suffix evicted at next observation",
        capacity=("oldest complete interactions evicted at resident capacity; active block and system protected"
                  if config["policy"]!="off" else "authoritative resident ledger; chronological eviction disabled"),
        memory_reminder="fixed system suffix before first prefill" if config["policy"]!="off" else "disabled",
        capacity_failure="stop explicitly if protected/active content leaves no capacity; no text-trimming fallback",
        response_headroom_applied=False, close="owned session closed on game exit or request failure",
        raw_wire_log="raw-harness/*.jsonl persistent-request/persistent-response",
        parsed_history="standard request snapshots remain display/dispatch views, not replayed input")
    return settings


def archive_experiment_sources(output):
    source = Path(__file__).resolve().parent
    names = ("run_persistent_gameplay.py", "persistent_harness_adapter.py", "append_only_history.py",
             "managed_suffix_history.py", "HARNESS_SUFFIX_DESIGN.md", "test_managed_suffix_history.py",
             "test_append_only_history.py", "test_persistent_harness_adapter.py",
             "HARNESS_HISTORY_DESIGN.md", "test_history_harness.py")
    manifest_path = output / "code-sha256.json"
    manifest = json.loads(manifest_path.read_text())
    for name in names:
        raw = (source/name).read_bytes()
        relative = Path("persistent-eviction") / name
        target = output / "implementation" / relative
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(raw)
        manifest[str(relative)] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_persistent_validation(output, *, runner_completed):
    """Supplement baseline validation with stateful failures and close receipts."""
    allocation_path = output / "allocation-summary.json"
    if not allocation_path.exists():
        return None
    allocation = json.loads(allocation_path.read_text())
    attempted, closed, errors = set(), set(), []
    for path in sorted((output / "raw-harness").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            payload = event["payload"]
            if event["kind"] == "persistent-request" and payload["route"] == "/arc3/persistent/start":
                attempted.add(payload["body"]["session_id"])
            elif event["kind"] == "persistent-close":
                closed.add(payload["session_id"])
            elif event["kind"] == "persistent-close-error":
                errors.append(payload)
    failures = [dict(game_id=row["game_id"], stop_reason=row["stop_reason"])
                for row in allocation["games"] if row["stop_reason"] in
                {"persistent_request_error", "persistent_capacity", "token_accounting_violation"}]
    result = dict(valid=bool(runner_completed and not failures and not errors
                            and attempted == closed and allocation["accounting_valid"]
                            and allocation["active_requests"] == 0),
                  runner_completed=runner_completed, attempted_sessions=sorted(attempted),
                  acknowledged_closed_sessions=sorted(closed), unclosed_sessions=sorted(attempted-closed),
                  game_failures=failures, close_errors=errors,
                  scope="harness/session receipt validation; engine KV/GDN parity is checked separately")
    (output/"persistent-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--persistent-base-url", required=True)
    parser.add_argument("--kv-eviction-policy", choices=("off", "oldest_interactions"),default="oldest_interactions")
    parser.add_argument("--kv-evict-fraction",type=float,default=0.10)
    arguments=sys.argv[1:] if argv is None else argv
    try:
        for name in ("--kv-eviction-policy", "--kv-evict-fraction"):
            _option(arguments,name)
    except ValueError as exc:
        parser.error(str(exc))
    options, remaining = parser.parse_known_args(arguments)
    try:
        history_config = _history_settings(options.kv_eviction_policy, options.kv_evict_fraction)
        remaining = validate_scope(remaining)
    except ValueError as exc:
        parser.error(str(exc))
    # Keep heavyweight harness imports after CLI validation. This also leaves the
    # existing runner in control of offline setup, preflight, game selection,
    # accounting, deadline, monitors, transcripts and scoring.
    import run_prolong

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = builtins.__import__(name, globals, locals, fromlist, level)
        if name == "prolong_duck" and "configure_solver" in fromlist:
            def configure(solver, **kwargs):
                module.configure_solver(solver, **kwargs)
                from persistent_harness_adapter import configure_persistent_solver
                return configure_persistent_solver(solver, base_url=options.persistent_base_url,
                                                    history_eviction=history_config)
            return SimpleNamespace(configure_solver=configure)
        return module

    def arm(settings, solver, **kwargs):
        apply_persistent_settings(settings, base_url=options.persistent_base_url, history_eviction=history_config)
        archive_experiment_sources(Path(settings["output"]))
        return run_prolong.arm_run_clock(settings, solver, **kwargs)

    def print_settings(*args, **kwargs):
        # The dry plan must not falsely advertise production stateless transport.
        if len(args) == 1 and isinstance(args[0], str):
            try:
                settings = json.loads(args[0])
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(settings, dict) and "context_policy" in settings and "gameplay" in settings:
                    apply_persistent_settings(settings, base_url=options.persistent_base_url, history_eviction=history_config)
                    args = (json.dumps(settings, indent=2),)
        builtins.print(*args, **kwargs)

    # main imports Duck only after establishing its environment settings. Keep
    # that ordering with an import wrapper bound solely to this function's
    # builtins; do not monkeypatch shared modules or Python's global importer.
    scope = dict(run_prolong.main.__globals__, arm_run_clock=arm, print=print_settings,
                 __builtins__=dict(vars(builtins), __import__=scoped_import))
    bound_main = FunctionType(run_prolong.main.__code__, scope, run_prolong.main.__name__,
                              run_prolong.main.__defaults__, run_prolong.main.__closure__)
    bound_main.__kwdefaults__ = run_prolong.main.__kwdefaults__
    completed = False
    try:
        result = bound_main(remaining)
        completed = True
        return result
    finally:
        if "--execute" in remaining:
            validation = write_persistent_validation(Path(_option(remaining, "--output")),
                                                       runner_completed=completed)
            if completed and validation is not None and not validation["valid"]:
                raise RuntimeError("Persistent harness/cleanup validation failed; inspect preserved evidence")


if __name__ == "__main__":
    main()
