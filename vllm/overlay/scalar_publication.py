"""Skip unchanged QSA scalar publications, with a physical-slot CPU shadow.

No GPU reads, new GPU buffers, asynchronous transfers or kernel changes. The
allocation factory owns the exact torch.zeros calls that justify initial zeros.
Every writer must use publish_scalars; direct writes require invalidation first.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os


VERSION = "arc3-skip-unchanged-scalars-v1"
FLAG = "ARC3_EFFICIENCY_SKIP_UNCHANGED_SCALARS"
STATE_ATTR = "_arc3_scalar_publication"
FIELDS = ("deleted", "padding", "cutoffs")


class ScalarPublicationError(RuntimeError):
    pass


def enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise ScalarPublicationError(f"{FLAG} must be 0 or 1")
    return value == "1"


def _layout(tensor):
    # Tensor allocation metadata is CPU-visible; this does not read its data.
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


@dataclass
class _Shadow:
    tensors: tuple
    layouts: tuple
    slots: list
    active: set = field(default_factory=set)
    counts: dict = field(default_factory=lambda: dict(
        calls=0, published=0, failed=0, issued_writes=0,
        skipped_writes=0, unknown_republications=0))


def allocate_zero_fields(torch_module, rows, *, device):
    """Return the existing three zero tensors plus an optional proven shadow.

    Splat into the normal buffer namespace, leaving lookup allocation unchanged.
    Called only at initial buffer creation; arbitrary live tensors are never
    accepted as an allegedly zero allocation.
    """
    use_shadow = enabled()
    if type(rows) is not int or rows <= 0:
        raise ScalarPublicationError("Scalar buffer capacity must be a positive integer")
    tensors = tuple(torch_module.zeros(rows, dtype=torch_module.int64, device=device)
                    for _ in FIELDS)
    result = dict(zip(FIELDS, tensors))
    if use_shadow:
        result[STATE_ATTR] = _Shadow(tensors=tensors,
            layouts=tuple(_layout(tensor) for tensor in tensors), slots=[(0, 0, 0)]*rows)
    return result


def _state(buffers):
    state = getattr(buffers, STATE_ATTR, None)
    if not isinstance(state, _Shadow):
        raise ScalarPublicationError("Scalar shadow must originate from allocate_zero_fields")
    for name, tensor, layout in zip(FIELDS, state.tensors, state.layouts):
        current = getattr(buffers, name)
        if current is not tensor or _layout(current) != layout:
            raise ScalarPublicationError("Graph-owned scalar tensor identity/layout changed")
    return state


def _slot(state, slot):
    if type(slot) is not int or not 0 <= slot < len(state.slots):
        raise ScalarPublicationError("Physical scalar slot is outside allocated capacity")


def invalidate(buffers, slot=None):
    """Required before an independently reviewed out-of-band test/write.

    There are no such serving writers in the reviewed source. Closing an owner
    does not invalidate or zero its shadow: it describes the physical GPU slot.
    """
    state = _state(buffers)
    indices = range(len(state.slots)) if slot is None else (slot,)
    for index in indices:
        _slot(state, index)
        if index in state.active:
            raise ScalarPublicationError("Cannot invalidate an active scalar publication")
    for index in indices:
        state.slots[index] = None


@contextmanager
def publish_scalars(buffers, slot, deleted, padding, cutoff):
    """Publish scalars, committing shadow only after the caller's lookup copy.

    A failure in any assignment or the with-body leaves the slot unknown, so
    the next attempt cannot skip any of the three scalar writes. This does not
    authorize retrying failed eviction transactions or changing existing fences.
    """
    values = (deleted, padding, cutoff)
    if not enabled():
        # Preserve the existing unconditional assignment order in control mode.
        # If toggled after initialization, discard the shadow so later enabling
        # cannot trust values changed while the optimization was disabled.
        if isinstance(getattr(buffers, STATE_ATTR, None), _Shadow):
            invalidate(buffers, slot)
        for name, value in zip(FIELDS, values):
            getattr(buffers, name)[slot] = value
        yield
        return
    state = _state(buffers)
    _slot(state, slot)
    if any(type(value) is not int or value < 0 for value in values):
        raise ScalarPublicationError("Scalar metadata values must be nonnegative integers")
    if slot in state.active:
        raise ScalarPublicationError("Nested publication targets an active physical slot")
    previous = state.slots[slot]
    state.slots[slot] = None
    state.active.add(slot)
    state.counts["calls"] += 1
    state.counts["unknown_republications"] += int(previous is None)
    try:
        for index, (name, value) in enumerate(zip(FIELDS, values)):
            if previous is not None and previous[index] == value:
                state.counts["skipped_writes"] += 1
                continue
            # Count issued operations, including a submission that raises.
            state.counts["issued_writes"] += 1
            getattr(buffers, name)[slot] = value
        yield
    except BaseException:
        state.counts["failed"] += 1
        raise
    else:
        state.slots[slot] = values
        state.counts["published"] += 1
    finally:
        state.active.remove(slot)


def status(buffers):
    state = getattr(buffers, STATE_ATTR, None)
    return {"version": VERSION, "enabled": enabled(), "initialized": isinstance(state, _Shadow),
            "known_slots": sum(value is not None for value in state.slots) if isinstance(state, _Shadow) else 0,
            "counts": dict(state.counts) if isinstance(state, _Shadow) else None}
