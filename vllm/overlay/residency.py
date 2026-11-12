"""Logical token identity and resident storage accounting for exact span eviction.

No token text, tensors, model execution or allocator mutation lives here. A plan
is pure; commit is only allowed after the engine acknowledges the same plan.
Logical positions never change when resident rows are removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


class ResidencyError(ValueError):
    pass


def _integer(value):
    return type(value) is int and value >= 0


def normalize_ranges(ranges, *, limit):
    """Validate nonempty, disjoint half-open ranges; coalesce adjacency only."""
    if not _integer(limit):
        raise ResidencyError("Invalid logical limit")
    rows = []
    for row in ranges:
        if (not isinstance(row, (list, tuple)) or len(row) != 2
                or not all(_integer(v) for v in row) or not row[0] < row[1] <= limit):
            raise ResidencyError("Invalid logical eviction range")
        rows.append(tuple(row))
    rows.sort()
    out = []
    for start, end in rows:
        if out and start < out[-1][1]:
            raise ResidencyError("Overlapping eviction ranges")
        if out and start == out[-1][1]:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return tuple(out)


def _reject_overlap(requested, blocked, message):
    """Both inputs are normalized (sorted, disjoint). A request [start, end) overlaps a blocked range iff the last
    blocked range starting before `end` ends after `start` (2026-09-23 fast path; the pairwise loop was quadratic)."""
    if not requested or not blocked:
        return
    from bisect import bisect_left
    starts = [left for left, _ in blocked]
    for start, end in requested:
        k = bisect_left(starts, end)
        if k and blocked[k - 1][1] > start:
            raise ResidencyError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                    sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class EvictionPlan:
    logical_tokens: int
    prior_evicted: tuple[tuple[int, int], ...]
    ranges: tuple[tuple[int, int], ...]
    survivor_intervals: tuple[tuple[int, int], ...]
    resident_before: int
    resident_after: int
    removed_tokens: int
    digest: str

    def wire(self):
        return {"logical_tokens": self.logical_tokens,
                "prior_evicted": [list(r) for r in self.prior_evicted],
                "ranges": [list(r) for r in self.ranges],
                "survivor_intervals": [list(r) for r in self.survivor_intervals],
                "resident_before": self.resident_before,
                "resident_after": self.resident_after,
                "removed_tokens": self.removed_tokens, "digest": self.digest}


class ResidentLedger:
    def __init__(self, logical_tokens=0, evicted=()):
        if not _integer(logical_tokens):
            raise ResidencyError("Invalid logical token count")
        self.logical_tokens = logical_tokens
        self.evicted = normalize_ranges(evicted, limit=logical_tokens)

    @property
    def resident_tokens(self):
        return self.logical_tokens - sum(end-start for start, end in self.evicted)

    def append(self, count):
        if not _integer(count):
            raise ResidencyError("Appended token count must be nonnegative")
        self.logical_tokens += count

    def survivor_intervals(self, evicted=None):
        removed = self.evicted if evicted is None else evicted
        out, begin = [], 0
        for start, end in removed:
            if begin < start:
                out.append((begin, start))
            begin = end
        if begin < self.logical_tokens:
            out.append((begin, self.logical_tokens))
        return tuple(out)

    def resident_positions(self):
        spans = self.survivor_intervals()   # 2026-09-24 stage E: numpy spans (same list)
        if not spans:
            return []
        import numpy as np
        return np.concatenate([np.arange(b, e, dtype=np.int64) for b, e in spans]).tolist()

    def plan(self, ranges, *, protected=()):
        requested = normalize_ranges(ranges, limit=self.logical_tokens)
        protected = normalize_ranges(protected, limit=self.logical_tokens)
        _reject_overlap(requested, self.evicted, "Requested span is already partially or wholly evicted")
        _reject_overlap(requested, protected, "Eviction overlaps protected tokens")
        all_removed = normalize_ranges((*self.evicted, *requested), limit=self.logical_tokens)
        survivors = self.survivor_intervals(all_removed)
        removed = sum(end-start for start, end in requested)
        data = dict(logical_tokens=self.logical_tokens, prior_evicted=self.evicted,
                    ranges=requested, survivor_intervals=survivors,
                    resident_before=self.resident_tokens,
                    resident_after=self.resident_tokens-removed, removed_tokens=removed)
        return EvictionPlan(**data, digest=_digest(data))

    def commit(self, plan, *, acknowledged_digest):
        if not isinstance(plan, EvictionPlan):
            raise ResidencyError("Expected an eviction plan")
        if (plan.logical_tokens != self.logical_tokens or plan.prior_evicted != self.evicted
                or acknowledged_digest != plan.digest):
            raise ResidencyError("Stale or unacknowledged eviction plan")
        fresh = self.plan(plan.ranges)
        if fresh != plan:
            raise ResidencyError("Eviction plan was altered")
        self.evicted = normalize_ranges((*self.evicted, *plan.ranges), limit=self.logical_tokens)
