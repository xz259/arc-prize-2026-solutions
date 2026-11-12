from __future__ import annotations

import copy
from dataclasses import replace
import math
import random
import unittest

from history_eviction import (HistoryPolicy,HistoryRegistry,HistoryValidationError,
                              HistoryCapacityError)
from residency import ResidentLedger


def setup(starts,logical,*,evicted=(),budget=None,fraction=0.10,policy="oldest_interactions"):
    registry=HistoryRegistry()
    transition=registry.plan_registration([
        {"observation_id":f"obs{i}","local_start_token":start} for i,start in enumerate(starts)],
        logical_offset=0,delta_tokens=logical)
    registry.commit(transition)
    ledger=ResidentLedger(logical,evicted)
    config=HistoryPolicy.from_payload({"policy":policy,"fraction":fraction,
                                      "resident_token_budget":budget or logical},max_context=max(logical,budget or 1))
    return registry,ledger,config


class PolicyValidation(unittest.TestCase):
    def test_default_off_and_budget_defaults_to_server(self):
        config=HistoryPolicy.from_payload(None,40000)
        self.assertEqual((config.policy,config.fraction,config.resident_token_budget),("off",0.1,40000))
        self.assertEqual(HistoryPolicy.from_payload({"policy":"oldest_interactions"},32000).resident_token_budget,32000)

    def test_invalid_fraction_policy_and_budget(self):
        for fraction in (0,1,-0.1,1.1,True,None,"0.1",float("nan"),float("inf"),10**1000):
            with self.subTest(fraction=str(fraction)):
                with self.assertRaises(HistoryValidationError):
                    HistoryPolicy.from_payload({"fraction":fraction},40000)
        for config in ({"policy":"importance"},{"unknown":1},{"resident_token_budget":40001},
                       {"resident_token_budget":0},{"resident_token_budget":True},[],"off"):
            with self.subTest(config=config):
                with self.assertRaises(HistoryValidationError):HistoryPolicy.from_payload(config,40000)

    def test_decimal_target_avoids_binary_round_up(self):
        config=HistoryPolicy.from_payload({"policy":"oldest_interactions","fraction":0.07},100)
        self.assertGreater(0.07*100,7)
        self.assertEqual(config.target_tokens(100),7)
        registry,ledger,config=setup([10,17,40],100,fraction=0.07)
        decision=registry.plan_eviction(config,ledger)
        self.assertEqual(decision.receipt["selected_block_ids"],["obs0"])
        self.assertEqual(decision.receipt["history_removed_tokens"],7)


class Registration(unittest.TestCase):
    def test_optional_proof_is_validated_but_not_geometric_state(self):
        registry=HistoryRegistry()
        record={"observation_id":"a","local_start_token":2}
        plain=registry.plan_registration([record],logical_offset=0,delta_tokens=10)
        proved=registry.plan_registration([{**record,"proof":{"rendered_prefix_sha256":"evidence"}}],
                                          logical_offset=0,delta_tokens=10)
        self.assertEqual(plain,proved)
        for proof in (None,[],"text",True):
            with self.assertRaises(HistoryValidationError):
                registry.plan_registration([{**record,"proof":proof}],logical_offset=0,delta_tokens=10)
        registry.commit(proved)
        self.assertEqual(registry.snapshot()["blocks"][0]["start"],2)

    def test_first_prefix_and_next_observation_commit_boundary(self):
        registry=HistoryRegistry()
        pending=registry.plan_registration([{"observation_id":"first","local_start_token":30}],
                                           logical_offset=0,delta_tokens=100)
        self.assertEqual(registry.snapshot()["blocks"],[])
        registry.commit(pending)
        self.assertEqual(registry.snapshot()["protected_prefix_tokens"],30)
        pending=registry.plan_registration([{"observation_id":"second","local_start_token":5}],
                                           logical_offset=200,delta_tokens=20)
        self.assertEqual(registry.snapshot()["blocks"],[{"observation_id":"first","start":30,"end":None}])
        registry.commit(pending)
        self.assertEqual(registry.snapshot()["blocks"],[{"observation_id":"first","start":30,"end":205},
                                                        {"observation_id":"second","start":205,"end":None}])
        self.assertEqual(registry.snapshot()["active_observation_id"],"second")

    def test_recovery_tool_and_empty_continuation_do_not_start_blocks(self):
        registry=HistoryRegistry()
        registry.commit(registry.plan_registration([{"observation_id":"a","local_start_token":10}],
                                                   logical_offset=0,delta_tokens=30))
        registry.commit(registry.plan_registration([],logical_offset=100,delta_tokens=25))
        registry.commit(registry.plan_registration([],logical_offset=150,delta_tokens=0))
        self.assertEqual(registry.snapshot()["blocks"],[{"observation_id":"a","start":10,"end":None}])
        self.assertEqual(registry.registered_through,150)

    def test_registration_validation_and_atomic_rejection(self):
        registry=HistoryRegistry()
        registry.commit(registry.plan_registration([{"observation_id":"a","local_start_token":10}],
                                                   logical_offset=0,delta_tokens=30))
        before=registry.snapshot()
        invalid=[None,{},[{"observation_id":"a","local_start_token":0}],
                 [{"observation_id":"b","local_start_token":10}],
                 [{"observation_id":"b","local_start_token":-1}],
                 [{"observation_id":"b","local_start_token":True}],
                 [{"observation_id":"b","local_start_token":1,"extra":"unreviewed"}],
                 [{"observation_id":"b","local_start_token":1},{"observation_id":"c","local_start_token":1}],
                 [{"observation_id":"b","local_start_token":2},{"observation_id":"c","local_start_token":1}],
                 [{"observation_id":"","local_start_token":0}]]
        for records in invalid:
            with self.subTest(records=records):
                with self.assertRaises(HistoryValidationError):
                    registry.plan_registration(records,logical_offset=40,delta_tokens=10)
                self.assertEqual(registry.snapshot(),before)
        with self.assertRaises(HistoryValidationError):registry.plan_registration([],logical_offset=29,delta_tokens=10)

    def test_stale_or_modified_transition_cannot_commit(self):
        registry=HistoryRegistry()
        first=registry.plan_registration([{"observation_id":"a","local_start_token":2}],
                                         logical_offset=0,delta_tokens=10)
        competing=registry.plan_registration([{"observation_id":"b","local_start_token":3}],
                                             logical_offset=0,delta_tokens=10)
        registry.commit(first)
        before=registry.snapshot()
        for transition in (first,competing,replace(competing,revision=1)):
            with self.assertRaises(HistoryValidationError):registry.commit(transition)
            self.assertEqual(registry.snapshot(),before)


class WholeObservationPlanning(unittest.TestCase):
    def test_oldest_six_thousand_block_rounds_up(self):
        registry,ledger,policy=setup([1000,7000,20000,35000],40000)
        before=(registry.snapshot(),ledger.evicted,ledger.logical_tokens)
        decision=registry.plan_eviction(policy,ledger)
        self.assertEqual(decision.ranges,((1000,7000),))
        self.assertEqual(decision.receipt["target_tokens"],4000)
        self.assertEqual(decision.receipt["history_removed_tokens"],6000)
        self.assertEqual(decision.receipt["selected_block_ids"],["obs0"])
        self.assertEqual((registry.snapshot(),ledger.evicted,ledger.logical_tokens),before)

    def test_fifteen_hundred_plus_three_thousand_requires_both(self):
        registry,ledger,policy=setup([1000,2500,5500,35000],40000)
        decision=registry.plan_eviction(policy,ledger)
        self.assertEqual(decision.ranges,((1000,5500),))
        self.assertEqual(decision.receipt["history_removed_tokens"],4500)
        self.assertEqual(decision.receipt["selected_block_ids"],["obs0","obs1"])

    def test_already_evicted_suffix_holes_do_not_count_twice(self):
        registry,ledger,policy=setup([1000,5500,8500,35000],43000,
            evicted=((2000,4000),),budget=41000)
        decision=registry.plan_eviction(policy,ledger)
        self.assertEqual(decision.receipt["target_tokens"],4100)
        self.assertEqual(decision.receipt["history_removed_tokens"],5500)
        self.assertEqual(decision.history_ranges,((1000,2000),(4000,8500)))
        self.assertEqual(decision.eviction_plan.prior_evicted,((2000,4000),))

    def test_pending_suffix_precedes_denominator_and_is_excluded_from_history(self):
        registry,ledger,policy=setup([1000,7000,14000,35000],40000,budget=37000)
        decision=registry.plan_eviction(policy,ledger,pending_suffix_ranges=((2000,5000),),incoming_tokens=0)
        self.assertEqual(decision.receipt["resident_tokens_after_suffix"],37000)
        self.assertEqual(decision.receipt["target_tokens"],3700)
        self.assertEqual(decision.receipt["suffix_removed_tokens"],3000)
        self.assertEqual(decision.receipt["history_removed_tokens"],10000)
        self.assertEqual(decision.history_ranges,((1000,2000),(5000,14000)))
        self.assertEqual(decision.ranges,((1000,14000),))
        self.assertEqual(decision.eviction_plan.removed_tokens,13000)

    def test_pending_active_suffix_allowed_but_active_history_protected(self):
        registry,ledger,policy=setup([1000,7000,35000],40000,budget=39000)
        decision=registry.plan_eviction(policy,ledger,pending_suffix_ranges=((36000,37000),))
        self.assertEqual(decision.receipt["selected_block_ids"],["obs0"])
        self.assertEqual(decision.history_ranges,((1000,7000),))
        self.assertEqual(decision.suffix_ranges,((36000,37000),))

    def test_zero_resident_blocks_skipped(self):
        registry,ledger,policy=setup([1000,4000,8000,18000],20000,evicted=((1000,4000),),budget=17000)
        decision=registry.plan_eviction(policy,ledger)
        self.assertEqual(decision.receipt["skipped_zero_resident_block_ids"],["obs0"])
        self.assertEqual(decision.receipt["selected_block_ids"],["obs1"])
        self.assertEqual(decision.ranges,((4000,8000),))

    def test_current_unfinished_block_cannot_be_cut(self):
        registry,ledger,policy=setup([1000],40000)
        before=(registry.snapshot(),ledger.evicted)
        with self.assertRaises(HistoryCapacityError) as caught:registry.plan_eviction(policy,ledger)
        self.assertEqual(caught.exception.status_code,413)
        self.assertEqual(caught.exception.receipt["eligible_history_tokens"],0)
        self.assertEqual((registry.snapshot(),ledger.evicted),before)

    def test_incoming_observation_does_not_close_active_before_forwarding(self):
        registry,ledger,policy=setup([1000],39000,budget=40000)
        transition=registry.plan_registration([{"observation_id":"new","local_start_token":0}],
                                             logical_offset=39000,delta_tokens=1000)
        with self.assertRaises(HistoryCapacityError):
            registry.plan_eviction(policy,ledger,incoming_tokens=1000)
        self.assertEqual(registry.snapshot()["active_observation_id"],"obs0")
        self.assertEqual(len(transition.registrations),1)

    def test_additional_blocks_selected_when_input_needs_more_than_nominal_target(self):
        registry,ledger,policy=setup([1000,6000,12000,18000,30000],40000)
        decision=registry.plan_eviction(policy,ledger,incoming_tokens=9000)
        self.assertEqual(decision.receipt["target_tokens"],4000)
        self.assertEqual(decision.receipt["capacity_needed_tokens"],9001)
        self.assertEqual(decision.receipt["selected_block_ids"],["obs0","obs1"])
        self.assertEqual(decision.receipt["history_removed_tokens"],11000)
        self.assertEqual(decision.receipt["minimum_output_capacity"],2000)

    def test_insufficient_nominal_target_fails_even_if_smaller_cut_would_fit(self):
        registry,ledger,policy=setup([1000,1500],40000)
        with self.assertRaises(HistoryCapacityError) as caught:registry.plan_eviction(policy,ledger)
        receipt=caught.exception.receipt
        self.assertEqual((receipt["target_tokens"],receipt["capacity_needed_tokens"],receipt["eligible_history_tokens"]),(4000,1,500))
        self.assertEqual(receipt["selected_block_ids"],[])
        self.assertFalse(receipt["eviction_committed"])

    def test_completed_block_touching_last_four_is_wholly_ineligible(self):
        registry,ledger,policy=setup([1,999],1000)
        with self.assertRaises(HistoryCapacityError) as caught:registry.plan_eviction(policy,ledger)
        self.assertEqual(caught.exception.receipt["tail_protected_block_ids"],["obs0"])

    def test_first_observation_is_evictable_system_prefix_is_not(self):
        registry,ledger,policy=setup([1000,7000,35000],40000)
        decision=registry.plan_eviction(policy,ledger)
        self.assertEqual(decision.receipt["selected_block_ids"],["obs0"])
        self.assertEqual(decision.receipt["protected_prefix_tokens"],1000)
        self.assertEqual(decision.eviction_plan.survivor_intervals[0],(0,1000))
        for invalid in (((1,500),),((39997,39999),)):
            with self.assertRaises(HistoryValidationError):
                registry.plan_eviction(policy,ledger,pending_suffix_ranges=invalid)

    def test_no_pressure_and_suffix_only_noop(self):
        registry,ledger,policy=setup([1000,7000,35000],39000,budget=40000)
        decision=registry.plan_eviction(policy,ledger,incoming_tokens=500)
        self.assertFalse(decision.receipt["triggered"])
        self.assertEqual(decision.ranges,())
        self.assertEqual(decision.receipt["minimum_output_capacity"],500)
        decision=registry.plan_eviction(policy,ledger,pending_suffix_ranges=((36000,37000),),incoming_tokens=1500)
        self.assertFalse(decision.receipt["triggered"])
        self.assertEqual(decision.ranges,((36000,37000),))
        self.assertEqual(decision.receipt["history_removed_tokens"],0)

    def test_empty_initial_registry_is_allowed_when_input_fits(self):
        registry=HistoryRegistry();ledger=ResidentLedger()
        policy=HistoryPolicy.from_payload({"policy":"oldest_interactions"},40000)
        decision=registry.plan_eviction(policy,ledger,incoming_tokens=2000)
        self.assertEqual(decision.ranges,())
        self.assertEqual(decision.receipt["resident_tokens_after_input"],2000)
        self.assertEqual(registry.snapshot()["blocks"],[])
        with self.assertRaises(HistoryCapacityError):registry.plan_eviction(policy,ledger,incoming_tokens=40000)

    def test_policy_off_does_not_implicitly_evict_history(self):
        registry,ledger,policy=setup([1000,7000,35000],40000,policy="off")
        with self.assertRaises(HistoryCapacityError) as caught:registry.plan_eviction(policy,ledger)
        self.assertEqual(caught.exception.reason,"history_eviction_disabled")
        decision=registry.plan_eviction(policy,ledger,pending_suffix_ranges=((36000,37000),))
        self.assertEqual(decision.history_ranges,())
        self.assertEqual(decision.ranges,((36000,37000),))

    def test_repeated_plans_follow_resident_basis_without_rebased_registry(self):
        registry,ledger,policy=setup([1000,2500,5500,15000,35000],40000)
        first=registry.plan_eviction(policy,ledger)
        ledger.commit(first.eviction_plan,acknowledged_digest=first.eviction_plan.digest)
        ledger.append(4500)
        second=registry.plan_eviction(policy,ledger)
        self.assertEqual(second.receipt["target_tokens"],4000)
        self.assertEqual(second.receipt["selected_block_ids"],["obs2"])
        self.assertEqual(second.ranges,((5500,15000),))
        self.assertEqual(registry.snapshot()["blocks"][0]["start"],1000)

    def test_small_random_cases_match_independent_resident_set_oracle(self):
        rng=random.Random(77)
        for _ in range(80):
            logical=200
            starts=[10,*sorted(rng.sample(range(20,175),5)),180]
            # Remove random interior cells without changing protected prefix/tail.
            holes=sorted(rng.sample(range(12,178),10))
            ranges=[]
            for p in holes:
                if ranges and ranges[-1][1]==p:ranges[-1]=(ranges[-1][0],p+1)
                else:ranges.append((p,p+1))
            registry,ledger,policy=setup(starts,logical,evicted=ranges,budget=190)
            decision=registry.plan_eviction(policy,ledger)
            resident=set(range(logical))-set(holes)
            selected=[];removed=set()
            for i,(start,end) in enumerate(zip(starts,starts[1:])):
                subset=set(range(start,end))&resident
                if not subset:continue
                selected.append(f"obs{i}");removed|=subset
                if len(removed)>=19:break
            actual={p for a,b in decision.history_ranges for p in range(a,b)}
            self.assertEqual(actual,removed)
            self.assertEqual(decision.receipt["selected_block_ids"],selected)
            self.assertEqual(decision.receipt["history_removed_tokens"],len(removed))


if __name__=="__main__":unittest.main()
