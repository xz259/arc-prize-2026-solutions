"""Service-side managed suffix ownership, separate from prompt text and KV."""
from __future__ import annotations

from dataclasses import dataclass
import json

from residency import ResidencyError


@dataclass(frozen=True)
class SuffixTransition:
    revision: int
    previous_live: tuple
    registrations: tuple  # (observation_id, logical_range_or_none, exact_record_json)
    retire_ids: tuple[str, ...]
    logical_offset: int

    def acknowledgement(self):
        return {"registered_observation_ids": [r[0] for r in self.registrations],
                "evicted_observation_ids": list(self.retire_ids)}


class SuffixRegistry:
    def __init__(self):
        self.revision = 0
        self.seen = set()
        self.live = {}
        self.records = {}

    def plan(self, records, retire_ids, *, logical_offset, delta_tokens):
        if (type(logical_offset) is not int or logical_offset < 0
                or type(delta_tokens) is not int or delta_tokens < 0):
            raise ResidencyError("Invalid suffix append coordinates")
        if not isinstance(records, list) or not isinstance(retire_ids, list):
            raise ResidencyError("Expected suffix record and retirement lists")
        if len(records) > 1:
            raise ResidencyError("Only one new observation is supported per append")
        expected = list(self.live) if records else []
        if retire_ids != expected:
            raise ResidencyError("Retire exactly the preceding live suffix on a new observation")
        registrations = []
        for record in records:
            if not isinstance(record, dict):
                raise ResidencyError("Invalid rendered suffix record")
            oid = record.get("observation_id")
            if not isinstance(oid, str) or not oid or len(oid) > 160 or oid in self.seen:
                raise ResidencyError("Observation identity missing or reused")
            span = record.get("local_token_range")
            absolute = None
            if span is not None:
                if (not isinstance(span, (list, tuple)) or len(span) != 2
                        or any(type(v) is not int for v in span)
                        or not 0 <= span[0] < span[1] <= delta_tokens):
                    raise ResidencyError("Managed token span must lie in the newly rendered input")
                absolute = (logical_offset + span[0], logical_offset + span[1])
            registrations.append((oid, absolute, json.dumps(record, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)))
        return SuffixTransition(self.revision, tuple(self.live.items()),
                                tuple(registrations), tuple(retire_ids), logical_offset)

    def eviction_ranges(self, transition):
        self._check(transition)
        return [list(self.live[oid]) for oid in transition.retire_ids]

    def _check(self, transition):
        if (not isinstance(transition, SuffixTransition)
                or transition.revision != self.revision
                or transition.previous_live != tuple(self.live.items())):
            raise ResidencyError("Stale suffix ownership transition")

    def commit(self, transition):
        self._check(transition)
        for oid in transition.retire_ids:
            del self.live[oid]
        for oid, span, record in transition.registrations:
            if oid in self.seen:
                raise ResidencyError("Observation identity was already committed")
            self.seen.add(oid)
            self.records[oid] = {"logical_token_range": span, "rendered_record": json.loads(record)}
            if span is not None:
                self.live[oid] = span
        self.revision += 1
