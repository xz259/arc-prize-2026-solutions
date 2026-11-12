"""Request FIFO with conserved, redistributable generated-token allowances.

Reservations are part of each allowance, not additional spending. Ambiguous
submitted requests charge their upper bound separately from measured usage.
"""
from collections import deque
from contextlib import contextmanager
from decimal import Decimal, ROUND_FLOOR
import json
from pathlib import Path
import random
import threading
import time


def shuffled_games(games, seed):
    games = list(games)
    random.Random(seed).shuffle(games)
    return games


def calibrated_token_budget(throughput, seconds):
    rate, duration = Decimal(str(throughput)), Decimal(str(seconds))
    if not rate.is_finite() or not duration.is_finite() or min(rate, duration) <= 0:
        raise ValueError("Finite positive throughput and available duration required")
    return int((rate * duration).to_integral_value(rounding=ROUND_FLOOR))


class AllocationStopped(RuntimeError):
    pass


class Allocation:
    def __init__(self, game_ids, *, slots, deadline, output, token_cap=None,
                 total_tokens=None, redistribute=True, active_games=None, unlimited_tokens=False):
        game_ids = list(game_ids)
        if not game_ids or len(set(game_ids)) != len(game_ids) or type(slots) is not int or slots <= 0:
            raise ValueError("Distinct nonempty games and positive integer slots required")
        if unlimited_tokens and (token_cap is not None or total_tokens is not None):
            raise ValueError("Deadline-only mode cannot also specify a token budget")
        if not unlimited_tokens and total_tokens is None:
            if type(token_cap) is not int or token_cap <= 0:
                raise ValueError("Positive integer token cap or total budget required")
            total_tokens = token_cap * len(game_ids)
        if not unlimited_tokens and (type(total_tokens) is not int or total_tokens < len(game_ids)):
            raise ValueError("Total budget must provide at least one token per game")
        if active_games is not None and (type(active_games) is not int or active_games <= 0):
            raise ValueError("Active-game limit must be a positive integer or None")
        self.active_game_limit = active_games
        count = len(game_ids) if active_games is None else min(active_games, len(game_ids))
        self.active_game_ids = set(game_ids[:count])
        self.waiting_game_ids = deque(game_ids[count:])
        self.activation_order = list(game_ids[:count])
        self.slots, self.token_cap, self.deadline = slots, token_cap, deadline
        self.unlimited_tokens = bool(unlimited_tokens)
        self.total_tokens, self.redistribute = total_tokens, redistribute and not self.unlimited_tokens
        self.unallocated_tokens = 0
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cv = threading.Condition()
        self.active = self.max_active = 0
        self.queue = deque((gid, 0) for gid in game_ids)
        share, remainder = (0, 0) if self.unlimited_tokens else divmod(total_tokens, len(game_ids))
        self.rows = {}
        for i, gid in enumerate(game_ids):
            allowance = None if self.unlimited_tokens else share + int(i < remainder)
            self.rows[gid] = dict(game_id=gid, shuffled_index=i, requests=0,
                initial_allowance_tokens=allowance, allowance_tokens=allowance,
                received_tokens=0, returned_tokens=0, reserved_tokens=0,
                active_request=None, submitted=False, settled=False, queued=False,
                generated_tokens=0, charged_tokens=0, prompt_tokens=0,
                cached_tokens=0, cached_usage_missing=0, uncertain_requests=0,
                failed_requests=0, consecutive_failures=0, queue_seconds=0.,
                serving_seconds=0., stop_reason=None, finished=False, finalizing=False,
                activation_epoch=time.time() if gid in self.active_game_ids else None)
        self.event("configured", game_ids=game_ids, slots=slots, token_cap=token_cap,
                   total_tokens=total_tokens, redistribute=self.redistribute, deadline_epoch=deadline,
                   unlimited_tokens=self.unlimited_tokens,
                   active_game_limit=active_games, active_game_ids=game_ids[:count])

    def _next_ticket(self):
        # Waiting games retain their first-request placeholders, but cannot block
        # a continuing active game. Admission remains FIFO within the active group.
        return next((ticket for ticket in self.queue if ticket[0] in self.active_game_ids), None)

    def event(self, event, **data):
        with (self.output / "allocation.jsonl").open("a") as f:
            f.write(json.dumps(dict(event=event, epoch=time.time(), **data)) + "\n")

    def stopped(self, gid):
        with self.cv:
            row = self.rows[gid]
            if time.time() >= self.deadline and row["stop_reason"] is None:
                row["stop_reason"] = "global_deadline"
            # Do not stop merely at exact allowance after a response: its valid
            # actions still execute. The next admission enforces exhaustion.
            return row["stop_reason"] is not None or row["finished"] or row["finalizing"]

    @contextmanager
    def admit(self, gid, *, max_tokens=None):
        queued_at = time.monotonic()
        with self.cv:
            row = self.rows[gid]
            if row["active_request"] is not None or row["queued"]:
                raise RuntimeError("At most one outstanding request per game")
            if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
                raise ValueError("Positive integer response cap required")
            if self.unlimited_tokens and max_tokens is None:
                raise ValueError("Deadline-only admission requires an explicit response cap")
            ticket = (gid, row["requests"])
            if not self.unlimited_tokens and row["charged_tokens"] >= row["allowance_tokens"]:
                row["stop_reason"] = row["stop_reason"] or "generated_token_cap"
            if ticket[1] > 0:
                self.queue.append(ticket)
            row["queued"] = True
            self.event("queued", game_id=gid, request=ticket[1])
            try:
                while True:
                    if self.stopped(gid):
                        raise AllocationStopped(row["stop_reason"] or "finished")
                    if self.active < self.slots and self._next_ticket() == ticket:
                        break
                    self.cv.wait(timeout=min(.2, max(.001, self.deadline - time.time())))
                self.queue.remove(ticket)
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                remaining = None if self.unlimited_tokens else row["allowance_tokens"] - row["charged_tokens"]
                reservation = max_tokens if self.unlimited_tokens else min(remaining, max_tokens or remaining)
                row.update(active_request=ticket[1], reserved_tokens=reservation,
                           submitted=False, settled=False)
                row["requests"] += 1
                wait = time.monotonic() - queued_at
                row["queue_seconds"] += wait
                request = dict(game_id=gid, request=ticket[1], queue_seconds=wait,
                               remaining_tokens=remaining, reserved_tokens=reservation,
                               admitted_epoch=time.time())
                self.event("admitted", **request, active=self.active)
                self.cv.notify_all()
            except BaseException:
                if ticket in self.queue:
                    self.queue.remove(ticket)
                row["queue_seconds"] += time.monotonic() - queued_at
                self.cv.notify_all()
                raise
            finally:
                row["queued"] = False
        try:
            yield request
        finally:
            with self.cv:
                if not row["settled"]:
                    uncertain = row["submitted"]
                    self.record(gid, dict(request=ticket[1], completion_tokens=None,
                        charged_tokens=row["reserved_tokens"] if uncertain else 0, uncertain=uncertain,
                        serving_seconds=0., error="unsettled_request_exit"))
                row.update(active_request=None, reserved_tokens=0, submitted=False)
                self.active -= 1
                if row["finalizing"]:
                    self._finalize(gid)
                self.cv.notify_all()

    def bound_reservation(self, gid, upper_bound):
        """Narrow a request's accounting bound before submission, not its HTTP cap."""
        with self.cv:
            row = self.rows[gid]
            if row["active_request"] is None or row["settled"] or row["submitted"]:
                raise RuntimeError("Reservation bound requires an unsubmitted live request")
            if type(upper_bound) is not int or upper_bound <= 0:
                raise ValueError("Positive integer accounting bound required")
            row["reserved_tokens"] = min(row["reserved_tokens"], upper_bound)
            self.event("reservation_bounded", game_id=gid, request=row["active_request"],
                       reserved_tokens=row["reserved_tokens"])
            return row["reserved_tokens"]

    def mark_submitted(self, gid):
        with self.cv:
            row = self.rows[gid]
            if row["active_request"] is None or row["settled"]:
                raise RuntimeError("Submission requires a live reservation")
            row["submitted"] = True

    def record(self, gid, record):
        with self.cv:
            row = self.rows[gid]
            if row["active_request"] is None or row["settled"]:
                raise RuntimeError("Missing reservation or duplicate settlement")
            if record.get("request", row["active_request"]) != row["active_request"]:
                raise RuntimeError("Settlement request identity mismatch")
            charged = record["charged_tokens"]
            observed = record.get("completion_tokens")
            if type(charged) is not int or charged < 0 or (observed is not None and
                    (type(observed) is not int or observed < 0 or observed > charged)):
                raise ValueError("Invalid measured usage/charge")
            if charged > row["reserved_tokens"]:
                row["stop_reason"] = "token_accounting_violation"
                self.event("token_accounting_violation", game_id=gid, record=record)
                raise RuntimeError("Server usage exceeds request reservation")
            row["generated_tokens"] += observed or 0
            row["charged_tokens"] += charged
            row["reserved_tokens"] = 0
            row["settled"] = True
            row["prompt_tokens"] += record.get("prompt_tokens") or 0
            cached = record.get("cached_tokens")
            row["cached_tokens"] += cached or 0
            row["cached_usage_missing"] += int(cached is None)
            row["serving_seconds"] += record["serving_seconds"]
            row["uncertain_requests"] += int(record.get("uncertain", False))
            failed = record.get("error") is not None
            row["failed_requests"] += int(failed)
            row["consecutive_failures"] = row["consecutive_failures"] + 1 if failed else 0
            if row["consecutive_failures"] >= 3:
                row["stop_reason"] = row["stop_reason"] or "consecutive_request_failures"
            self.event("request_finished", game_id=gid, **record)

    def finish(self, gid, **result):
        with self.cv:
            row = self.rows[gid]
            if row["finished"] or row["finalizing"]:
                return
            # Result metadata cannot overwrite ledger fields.
            for key in ("state", "score", "actions", "levels_completed", "number_of_levels"):
                if key in result:
                    row[key] = result[key]
            row["finalizing"] = True
            row["stop_reason"] = row["stop_reason"] or (
                "global_deadline" if time.time() >= self.deadline else
                "generated_token_cap" if not self.unlimited_tokens and row["charged_tokens"] >= row["allowance_tokens"]
                and result.get("state") != "won" else result.get("state", "finished"))
            self.queue = deque(ticket for ticket in self.queue if ticket[0] != gid)
            if row["active_request"] is None:
                self._finalize(gid)
            self.cv.notify_all()

    def _finalize(self, gid):
        row = self.rows[gid]
        if row["finished"] or row["active_request"] is not None:
            return
        surplus = 0 if self.unlimited_tokens else row["allowance_tokens"] - row["charged_tokens"]
        if not self.unlimited_tokens:
            row["allowance_tokens"] -= surplus
        row["returned_tokens"] += surplus
        row.update(finished=True, finalizing=False)
        self.active_game_ids.discard(gid)
        self.waiting_game_ids = deque(g for g in self.waiting_game_ids if g != gid)
        promoted = []
        if self.active_game_limit is not None:
            while (len(self.active_game_ids) < self.active_game_limit and self.waiting_game_ids
                   and time.time() < self.deadline):
                replacement = self.waiting_game_ids.popleft()
                if self.stopped(replacement):
                    continue
                self.active_game_ids.add(replacement)
                self.activation_order.append(replacement)
                self.rows[replacement]["activation_epoch"] = time.time()
                promoted.append(replacement)
        # Replacement and redistribution happen under the same condition lock,
        # before any new request can reserve tokens from its updated allowance.
        cohort = [g for g in self.rows if g in self.active_game_ids]
        recipients = [r for r in self.rows.values() if r["game_id"] in self.active_game_ids
                      and not self.stopped(r["game_id"])] if self.redistribute else []
        self.event("cohort_updated", finished_game_id=gid, promoted=promoted,
                   active_game_ids=cohort, eligible_game_ids=[r['game_id'] for r in recipients])
        increments = {}
        if recipients:
            share, remainder = divmod(surplus, len(recipients))
            for i, recipient in enumerate(recipients):
                amount = share + int(i < remainder)  # Stable shuffled-order remainder.
                recipient["allowance_tokens"] += amount
                recipient["received_tokens"] += amount
                increments[recipient["game_id"]] = amount
        else:
            self.unallocated_tokens += surplus
        self.event("budget_redistributed", game_id=gid, surplus=surplus,
                   increments=increments, unallocated_tokens=self.unallocated_tokens)
        self.event("game_finished", **row)

    def save(self):
        with self.cv:
            charged = sum(r["charged_tokens"] for r in self.rows.values())
            reserved = sum(r["reserved_tokens"] for r in self.rows.values())
            remaining = None if self.unlimited_tokens else sum(
                r["allowance_tokens"] - r["charged_tokens"] - r["reserved_tokens"] for r in self.rows.values())
            conserved = None if self.unlimited_tokens else (
                charged + reserved + remaining + self.unallocated_tokens == self.total_tokens)
            accounting_valid = (all(0 <= r["generated_tokens"] <= r["charged_tokens"]
                                    and r["reserved_tokens"] >= 0 for r in self.rows.values())
                and self.active == sum(r["active_request"] is not None for r in self.rows.values())
                and 0 <= self.active <= self.slots and (self.unlimited_tokens or conserved))
            data = dict(max_active_requests=self.max_active, active_requests=self.active,
                unlimited_tokens=self.unlimited_tokens, accounting_valid=accounting_valid,
                token_cap=self.token_cap, total_tokens=self.total_tokens, redistribute=self.redistribute,
                active_game_limit=self.active_game_limit,
                active_game_ids=[g for g in self.rows if g in self.active_game_ids],
                waiting_game_ids=list(self.waiting_game_ids), activation_order=self.activation_order,
                charged_tokens=charged, reserved_tokens=reserved, remaining_tokens=remaining,
                unallocated_tokens=self.unallocated_tokens, conserved=conserved,
                games=list(self.rows.values()),
                never_admitted=[gid for gid, row in self.rows.items() if not row["requests"]])
            target = self.output / "allocation-summary.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            temporary.replace(target)
            return data
