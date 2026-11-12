"""CPU-only long-position policy/client tests; synthetic journals are labeled fixtures."""
import asyncio
from contextlib import redirect_stdout, redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from raw_gpu_checks import ContractFailure
from raw_history_long_position_checks import (
    BUDGET, CONTEXT, LOGICAL_TARGET, audit_long_positions, exercise, main,
    plan_description,
)
from raw_history_eviction_checks import intervals
from session_service import SessionManager
from test_raw_history_eviction_checks import Backend, LocalClient


class LongPositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.loop = asyncio.new_event_loop(); self.backend = Backend()
        self.service = self.root/'service.jsonl'; self.engine = self.root/'engine.jsonl'
        self.engine.write_text('')
        self.manager = SessionManager(self.backend, max_context=CONTEXT, max_sessions=1,
                                      audit_path=str(self.service))
        self.client = LocalClient('http://127.0.0.1:1234', self.root, 30, 'long-test',
            deadline=time.monotonic()+60, manager=self.manager, loop=self.loop)

    def tearDown(self):
        self.loop.run_until_complete(self.manager.close_all())
        self.loop.close(); self.temp.cleanup()

    @staticmethod
    def prompt(count, offset=0):
        return [100+(i+offset)%1000 for i in range(count)]

    def panel(self):
        self.result = exercise(self.client, self.prompt)
        return self.client

    def test_real_policy_accepts_48128_logical_with_only_6144_resident(self):
        c = self.panel()
        self.assertEqual(len(c.oracle.logical), LOGICAL_TARGET)
        self.assertEqual(len(c.oracle.resident_positions), 6144)
        self.assertEqual(len(c.oracle.holes), 41984)
        self.assertEqual(len(self.backend.evictions), 14)
        self.assertEqual(len(self.backend.submissions), 16)  # exact retry is not a new turn.
        self.assertTrue(self.result['idempotency']['status_unchanged'])
        self.assertEqual([t['resident_tokens'] for t in c.turns], [3008, 6016]+[6144]*14)
        self.assertEqual([t['logical_tokens'] for t in c.turns if t['logical_tokens'] > CONTEXT],
                         [42112, 45120, 48128])
        self.assertTrue(all(len(ids) == 3000 for _, ids in self.backend.submissions))
        self.assertTrue(all(p in c.oracle.resident_positions for p in range(128)))
        self.assertEqual(c.oracle.history_state()['blocks'][-1]['start'], 45120)

    def test_short_source_rejected_before_submission(self):
        with self.assertRaisesRegex(ContractFailure, 'exact planned delta'):
            exercise(self.client, lambda count, offset: [1]*(count-1))
        self.assertEqual(self.backend.submissions, [])

    def test_nonfresh_or_wrong_budget_rejected(self):
        self.client.oracle.budget = 6000
        with self.assertRaisesRegex(ContractFailure, 'fresh 7K'):
            exercise(self.client, self.prompt)
        self.assertEqual(self.backend.submissions, [])

    def test_plan_is_bounded_and_dry_run_cannot_execute(self):
        plan = plan_description()
        self.assertLess(plan['logical_target'], plan['logical_ceiling_exclusive'])
        self.assertEqual(plan['max_seconds'], 600)
        with patch('raw_history_long_position_checks.run', side_effect=AssertionError('No run')), redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 0)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--execute'])
        required = ['--execute', '--output', 'x', '--expected-hostname', 'h',
                    '--expected-boot-id', 'b', '--engine-events', 'e', '--service-events', 's']
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(required+['--max-seconds', '601'])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(required+['--expected-context', '50000'])

    def synthetic_journal(self):
        """Add synthetic engine byte/page/position evidence to actual CPU service rows."""
        c = self.panel(); c.close(c.sid)
        service = [json.loads(line) for line in self.service.read_text().splitlines()]
        rid = service[0]['engine_request_id']; engine = []; blocks = [[1, 2], [100]]
        next_block = 3; holes = set()
        for t in c.turns:
            i = t['turn']
            if t['removed_tokens']:
                sc = next(r for r in service if r['event'] == 'evict_commit' and r['operation_id'] == t['request_id'])
                ack = sc['acknowledgement']; freed = [[blocks[0].pop()], []]
                planned = [dict(layer_name=f'{kind}-{j}', kind=kind, retained_rows=32)
                           for kind in ('main', 'compressed') for j in range(12)]
                proof = dict(sha256='a'*64, rows=32, bytes=1024)
                copies = [dict(layer_name=r['layer_name'], copied_rows=32, retained_rows=32, row_bytes=32,
                               byte_verification=dict(before=proof, after=copy.deepcopy(proof), equal=True)) for r in planned]
                ack.update(freed_blocks_by_group=freed,
                    worker_receipts=[dict(worker_synchronized=True, byte_verification_enabled=True, copies=copies)],
                    allocator_reacquisition=dict(kind='actual_allocator_reacquisition_without_forward', attempted=True,
                        released_block_ids=freed[0], reacquired_released_block_ids=freed[0], reacquired_ref_cnt=1,
                        free_count_before=10, free_count_after=10, free_order_before_sha256='b'*64,
                        free_order_after_sha256='b'*64, queue_order_restored=True, other_live_owners_unchanged=True,
                        all_temporary_owners_released=True, model_forward_performed=False))
                engine.append(dict(event='evict_prepare', request_id=rid,
                                   eviction_plan=dict(plan_digest=ack['plan_digest'], groups=[dict(copies=planned)])))
                engine.append(dict(event='evict', request_id=rid, eviction=ack, resident_tokens=t['resident_after_eviction'],
                    tokens_sha256=t['after_eviction_sha256'], computed_tokens=t['resident_after_eviction']-1,
                    block_ids=copy.deepcopy(blocks)))
            engine.append(dict(event='start' if i == 0 else 'resume', request_id=rid, turn=i,
                               computed_tokens=t['expected_resume_computed'], block_ids=copy.deepcopy(blocks)))
            if i:
                blocks[0].append(next_block); next_block += 1
            for start, end in t['history_eviction']['combined_ranges']:
                holes.update(range(start, end))
            groups = sum(all(p in holes for p in range(g*4, g*4+4)) for g in {p//4 for p in holes})
            engine.append(dict(event='pause', request_id=rid, turn=i, block_ids=copy.deepcopy(blocks),
                resident_tokens=t['resident_tokens'], tokens_sha256=t['resident_sha256'],
                computed_tokens=t['resident_tokens']-1, computed_prefix_sha256=t['computed_prefix_sha256'],
                in_flight_tokens=0, output_placeholders=0, preemptions=0,
                logical_eviction_state=dict(logical_tokens=t['logical_tokens'], logical_sha256=t['logical_sha256'],
                    eviction_revision=t['eviction_revision'], deleted_tokens=len(holes), deleted_groups=groups,
                    group_padding=len(holes)-4*groups, evicted_ranges=intervals(holes), poisoned=False),
                turn_work=dict(prefill_tokens=t['expected_prefill_tokens'], decode_tokens=7,
                    scheduled_tokens=t['expected_prefill_tokens']+7, resume_computed_tokens=t['expected_resume_computed'],
                    minimum_scheduled_position=t['expected_resume_computed'])))
        engine.append(dict(event='close_released', request_id=rid, block_ids=[[], []], request_still_registered=False))
        self.write_journal(engine, service)
        return c, engine, service

    def write_journal(self, engine, service):
        self.engine.write_text(''.join(json.dumps(r)+'\n' for r in engine))
        self.service.write_text(''.join(json.dumps(r)+'\n' for r in service))

    def test_complete_synthetic_audit_and_exact_work_totals(self):
        c, _, _ = self.synthetic_journal()
        audit = audit_long_positions(self.engine, self.service, c.sid, c.turns)
        self.assertTrue(audit['valid'], (audit['errors'], audit['physical_audit']['errors']))
        self.assertEqual(audit['logical_tokens'], 48128)
        self.assertEqual(audit['resident_tokens'], 6144)
        self.assertEqual(audit['highest_computed_original_position'], 48126)
        self.assertEqual(audit['expected_total_prefill_tokens'], 48015)
        self.assertEqual(audit['expected_total_decode_tokens'], 112)
        self.assertEqual(audit['physical_release_transactions'], 14)

    def test_rebased_original_position_journal_is_rejected(self):
        c, engine, service = self.synthetic_journal()
        ep = [r for r in engine if r['event'] == 'pause'][-1]
        ep['logical_eviction_state']['logical_tokens'] = ep['resident_tokens']
        self.write_journal(engine, service)
        audit = audit_long_positions(self.engine, self.service, c.sid, c.turns)
        self.assertFalse(audit['valid'])
        self.assertIn('exact_original_position_ledger', [r['check'] for r in audit['errors']])

    def test_replayed_historical_prefill_is_rejected(self):
        c, engine, service = self.synthetic_journal()
        ep = [r for r in engine if r['event'] == 'pause'][-1]
        ep['turn_work']['minimum_scheduled_position'] = 0
        ep['turn_work']['prefill_tokens'] = ep['resident_tokens']-8
        self.write_journal(engine, service)
        audit = audit_long_positions(self.engine, self.service, c.sid, c.turns)
        self.assertFalse(audit['valid'])
        self.assertIn('only_new_input_and_pending_final_forwarded',
                      [r['check'] for r in audit['physical_audit']['errors']])

    def test_incomplete_run_cannot_claim_crossed_context(self):
        c = self.client
        c.submit('short', self.prompt(3000), observation=('a', 128))
        c.close(c.sid)
        audit = audit_long_positions(self.engine, self.service, c.sid, c.turns)
        self.assertFalse(audit['valid'])
        self.assertEqual(audit['completed_turns_beyond_context'], [])
        self.assertIn('exact_48128_logical_tokens', [r['check'] for r in audit['errors']])


if __name__ == '__main__':
    unittest.main()
