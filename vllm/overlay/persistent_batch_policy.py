"""Opt-in native batch-queue coalescing for active persistent owners.

Patch only scheduling and queue-fill early-return conditions. Keep the native
queue capacity, Future handling, output/abort processing and request guards.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import os
from pathlib import Path

VERSION = "arc3-persistent-single-batch-v1"
FLAG = "ARC3_PERSISTENT_SINGLE_BATCH"
CORE_SHA256 = "f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad"
POLICY_NAME = "_arc3_single_persistent_batch"
HELPER_NAME = "_arc3_persistent_batch_policy_active"


class BatchPolicyError(RuntimeError):
    pass


def enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise BatchPolicyError(f"{FLAG} must be 0 or 1")
    return value == "1"


def active(core, *, is_persistent, paused_status):
    import one_lookahead
    if any(one_lookahead.marked(req) and req.status != paused_status and not req.is_finished()
           for req in core.scheduler.requests.values()):
        return False
    active_owner = any(is_persistent(req) and req.status != paused_status
                       and not req.is_finished() for req in core.scheduler.requests.values())
    if active_owner:
        config = core.vllm_config
        pc = config.parallel_config
        if (pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1
                or pc.data_parallel_size != 1 or config.num_speculative_tokens != 0):
            raise BatchPolicyError("Persistent single-batch control is scoped to TP1/PP1/DP1 MTP0")
    return active_owner


def transform(source):
    """Return the exact native method with two condition changes and a bool."""
    if hashlib.sha256(source.encode()).hexdigest() != CORE_SHA256:
        raise BatchPolicyError("Native engine core source changed; review queue policy again")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                            and n.name == "step_with_batch_queue"))
    schedule_anchor = ast.parse("self.scheduler.has_requests()", mode="eval").body
    schedule_nodes = [n for n in fn.body if isinstance(n, ast.If)
        and ast.dump(n.test, include_attributes=False) == ast.dump(schedule_anchor, include_attributes=False)]
    early_anchor = ast.parse("len(batch_queue) < self.batch_queue_size and (model_executed or self.scheduler.has_requests())",
                             mode="eval").body
    early_nodes = [n for n in ast.walk(fn) if isinstance(n, ast.If)
        and ast.dump(n.test, include_attributes=False) == ast.dump(early_anchor, include_attributes=False)]
    if len(schedule_nodes) != 1 or len(early_nodes) != 1:
        raise BatchPolicyError("Expected exactly one native scheduling and early-return condition")
    schedule_nodes[0].test = ast.BoolOp(op=ast.And(), values=[schedule_nodes[0].test,
        ast.parse(f"not ({POLICY_NAME} and batch_queue)", mode="eval").body])
    early_nodes[0].test = ast.BoolOp(op=ast.And(), values=[early_nodes[0].test,
        ast.parse(f"not {POLICY_NAME}", mode="eval").body])
    fn.body.insert(1, ast.parse(f"{POLICY_NAME} = {HELPER_NAME}(self)").body[0])
    return ast.fix_missing_locations(fn)


def build(source, namespace, predicate):
    fn = transform(source)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    scope = dict(namespace)
    scope[HELPER_NAME] = predicate
    exec(compile(ast.fix_missing_locations(tree), "<arc3-native-persistent-batch-policy>", "exec"), scope)
    return scope[fn.name]


def install():
    """General-plugin install before constructing EngineCore and its step_fn."""
    if not enabled():
        return {"version": VERSION, "installed": False}
    import engine_patch
    from vllm.v1.engine import core as module
    from vllm.v1.request import RequestStatus
    cls = module.EngineCore
    if getattr(cls, "_arc3_batch_policy", None) == VERSION:
        return {"version": VERSION, "installed": True, "already_installed": True}
    path = Path(module.__file__).resolve()
    source = path.read_text()
    original = cls.step_with_batch_queue
    # Refuse to overwrite an independently wrapped method or a previous patch.
    native = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    method = next(n for n in native.body if isinstance(n, ast.FunctionDef) and n.name == original.__name__)
    if Path(original.__code__.co_filename).resolve() != path or original.__code__.co_firstlineno != method.lineno:
        raise BatchPolicyError("Existing queue method is not the reviewed native implementation")
    predicate = functools.partial(active, is_persistent=engine_patch.is_persistent,
                                  paused_status=RequestStatus.WAITING_FOR_STREAMING_REQ)
    cls.step_with_batch_queue = functools.wraps(original)(build(source, module.__dict__, predicate))
    cls._arc3_batch_policy = VERSION
    return {"version": VERSION, "installed": True, "native_source_sha256": CORE_SHA256,
            "queue_capacity_unchanged": True, "request_frontier_guards_unchanged": True}
