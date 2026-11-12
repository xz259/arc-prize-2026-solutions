"""CPU checks for system-prompt KV sharing (prefix_share_v1; stage 15, 2026-09-27).

Registry and swap semantics on a pool with vLLM BlockPool ref counting (and on the real BlockPool when vLLM imports):
the first session registers its leading attention block (registry reference), later sessions with the same text prefix
swap their private block for it (private copy freed), mismatching / multimodal / short / non-exclusive sessions are left
alone, a repeated append is a no-op, closing sessions never frees the shared block, other groups are untouched, and the
switch off changes nothing. Eviction (engine_eviction.prepare_transaction on the eviction tests' fixture): a shared
leading block with ref_cnt > 1 is accepted when the plan keeps its rows in place, and a plan moving any of its rows
(main or compressed owner) is refused; an unregistered shared page is still refused."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefix_share_v1 as ps
import test_engine_eviction as te
import engine_eviction as ee
import engine_patch
try:   # imported before any test patches vllm.v1.kv_cache_interface with the eviction tests' fakes
    from vllm.v1.core.block_pool import BlockPool as REAL_BLOCK_POOL
except Exception as _exc:  # noqa: BLE001 - vLLM is optional here
    REAL_BLOCK_POOL = None; _REAL_POOL_ERROR = repr(_exc)

BLOCK = 16


class FakeUniform(te.UniformTypeKVCacheSpecs):
    @property
    def first_spec(self):
        return next(iter(self.kv_cache_specs.values()))


class Pool:
    """vLLM BlockPool ref counting: touch +1 (leaving the free queue at 0), free -1 (joining it at 0)."""
    def __init__(self, n):
        self.blocks = [NS(block_id=i, ref_cnt=0, is_null=(i == 0)) for i in range(n)]
        self.free = [b for b in self.blocks[1:]]

    def get(self, k):
        out = [self.free.pop() for _ in range(k)]
        for b in out:
            b.ref_cnt += 1
        return out

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            b.ref_cnt -= 1
            if b.ref_cnt == 0 and not b.is_null:
                self.free.append(b)


def scheduler(pool):
    attn = NS(kv_cache_spec=FakeUniform({'layer.attn': te.FullAttentionSpec(), 'layer.compressed_key_cache': te.MLAAttentionSpec()}),
              layer_names=['layer.attn', 'layer.compressed_key_cache'])
    gdn = NS(kv_cache_spec=te.MambaSpec(), layer_names=['layer.gdn'])
    managers = [NS(req_to_blocks={}), NS(req_to_blocks={})]
    return NS(kv_cache_manager=NS(block_pool=pool, coordinator=NS(kv_cache_config=NS(kv_cache_groups=[attn, gdn]),
                                                                    single_type_managers=managers)))


def session(sched, pool, rid, tokens, *, computed=None, images=(), attn_blocks=3):
    s = NS(request_id=rid, _all_token_ids=list(tokens), num_computed_tokens=len(tokens) - 1 if computed is None else computed,
           mm_features=[NS(mm_position=te.Position(o, l)) for o, l in images])
    sched.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[rid] = pool.get(attn_blocks)
    sched.kv_cache_manager.coordinator.single_type_managers[1].req_to_blocks[rid] = pool.get(1)
    return s


class Registry(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, te.fake_specs()); self.modules.start()
        self.env = patch.dict(os.environ, {ps.FLAG: '1', ps.BLOCKS_FLAG: '1'}); self.env.start()
        ps.reset_for_tests()

    def tearDown(self):
        ps.reset_for_tests(); self.env.stop(); self.modules.stop()

    def test_register_share_skip_and_close(self):
        pool = Pool(64); sch = scheduler(pool); prefix = list(range(100, 100 + BLOCK))
        mgr = sch.kv_cache_manager.coordinator.single_type_managers
        a = session(sch, pool, 'a', prefix + [1, 2, 3, 4, 5])
        a0 = mgr[0].req_to_blocks['a'][0]; free0 = len(pool.free)
        r = ps.on_append(sch, a)
        self.assertEqual(r['action'], 'registered'); self.assertEqual(a0.ref_cnt, 2); self.assertEqual(len(pool.free), free0)
        b = session(sch, pool, 'b', prefix + [9, 9, 9]); b0 = mgr[0].req_to_blocks['b'][0]; b_rest = mgr[0].req_to_blocks['b'][1:]
        b_gdn = list(mgr[1].req_to_blocks['b']); free1 = len(pool.free)
        r = ps.on_append(sch, b)
        self.assertEqual(r['action'], 'shared'); self.assertEqual(r['old_block_ids'], [b0.block_id])
        self.assertIs(mgr[0].req_to_blocks['b'][0], a0); self.assertEqual(mgr[0].req_to_blocks['b'][1:], b_rest)
        self.assertEqual(a0.ref_cnt, 3); self.assertEqual(b0.ref_cnt, 0); self.assertIn(b0, pool.free)
        self.assertEqual(len(pool.free), free1 + 1); self.assertEqual(mgr[1].req_to_blocks['b'], b_gdn)
        self.assertIsNone(ps.on_append(sch, b)); self.assertEqual(a0.ref_cnt, 3)                     # already shared
        c = session(sch, pool, 'c', [7] + prefix[1:] + [1])                                         # different prefix
        self.assertIsNone(ps.on_append(sch, c)); self.assertEqual(ps.STATS['skipped_mismatch'], 1)
        d = session(sch, pool, 'd', prefix + [1], images=[(BLOCK - 2, 4)])                            # image inside the prefix
        self.assertIsNone(ps.on_append(sch, d)); self.assertEqual(ps.STATS['skipped_mm'], 1)
        e = session(sch, pool, 'e', prefix[:10])                                                    # shorter than a block
        self.assertIsNone(ps.on_append(sch, e)); self.assertEqual(ps.STATS['skipped_short'], 1)
        f = session(sch, pool, 'f', prefix + [1], computed=BLOCK - 1)                               # prefix not computed yet
        self.assertIsNone(ps.on_append(sch, f)); self.assertEqual(ps.STATS['skipped_short'], 2)
        for rid in ('a', 'b'):                                                                       # close both sessions
            pool.free_blocks(mgr[0].req_to_blocks.pop(rid))
        self.assertEqual(a0.ref_cnt, 1); self.assertNotIn(a0, pool.free)                             # registry keeps it
        g = session(sch, pool, 'g', prefix + [4]); self.assertEqual(ps.on_append(sch, g)['action'], 'shared')
        self.assertEqual(a0.ref_cnt, 2); self.assertEqual(ps.registry()['block_ids'], [a0.block_id])

    def test_switch_off_is_inert(self):
        pool = Pool(16); sch = scheduler(pool); prefix = list(range(BLOCK))
        with patch.dict(os.environ, {ps.FLAG: '0'}):
            a = session(sch, pool, 'a', prefix + [1]); b = session(sch, pool, 'b', prefix + [2])
            self.assertIsNone(ps.on_append(sch, a)); self.assertIsNone(ps.on_append(sch, b)); self.assertIsNone(ps.registry())
        with patch.dict(os.environ, {ps.FLAG: 'yes'}):                                          # bad value: off, never raises
            self.assertEqual(ps.on_append(sch, a)['action'], 'disabled'); self.assertIsNone(ps.on_append(sch, a))

    def test_worker_owner_specs_identify_the_group(self):
        """The stock scheduler normalizes group specs; the overlay's preserved worker specs name the attention group."""
        pool = Pool(32); sch = scheduler(pool); cfg = sch.kv_cache_manager.coordinator.kv_cache_config
        cfg._arc3_worker_kv_cache_groups = list(cfg.kv_cache_groups)
        cfg.kv_cache_groups = [NS(kv_cache_spec=te.MLAAttentionSpec(), layer_names=g.layer_names) if i == 0 else g
                               for i, g in enumerate(cfg.kv_cache_groups)]
        cfg.kv_cache_groups[0].kv_cache_spec.storage_block_size = BLOCK // 4    # representative block size stays 16
        self.assertEqual(ps.attention_group(cfg), (0, BLOCK))
        prefix = list(range(BLOCK)); a = session(sch, pool, 'a', prefix + [1]); b = session(sch, pool, 'b', prefix + [2])
        self.assertEqual(ps.on_append(sch, a)['action'], 'registered'); self.assertEqual(ps.on_append(sch, b)['action'], 'shared')

    def test_missing_attention_group_disables_without_raising(self):
        pool = Pool(16); sch = scheduler(pool); cfg = sch.kv_cache_manager.coordinator.kv_cache_config
        cfg.kv_cache_groups = [NS(kv_cache_spec=te.MambaSpec(), layer_names=['x']), NS(kv_cache_spec=te.MambaSpec(), layer_names=['y'])]
        a = session(sch, pool, 'a', list(range(BLOCK)) + [1]); before = list(sch.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks['a'])
        r = ps.on_append(sch, a)
        self.assertEqual(r['action'], 'disabled'); self.assertIn('found 0', r['reason']); self.assertIsNone(ps.on_append(sch, a))
        self.assertEqual(sch.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks['a'], before)
        self.assertIn('disabled', ps.registry())

    def test_unused_registration_is_replaced(self):
        pool = Pool(32); sch = scheduler(pool); mgr = sch.kv_cache_manager.coordinator.single_type_managers
        p1, p2 = list(range(BLOCK)), list(range(50, 50 + BLOCK))
        a = session(sch, pool, 'a', p1 + [1]); ps.on_append(sch, a); a0 = mgr[0].req_to_blocks['a'][0]
        b = session(sch, pool, 'b', p2 + [1]); self.assertIsNone(ps.on_append(sch, b))    # p1 still in use by a
        pool.free_blocks(mgr[0].req_to_blocks.pop('a')); self.assertEqual(a0.ref_cnt, 1)  # a closes; registry only
        r = ps.on_append(sch, b); b0 = mgr[0].req_to_blocks['b'][0]
        self.assertEqual(r['action'], 'replaced'); self.assertEqual(r['released_block_ids'], [a0.block_id])
        self.assertEqual(a0.ref_cnt, 0); self.assertIn(a0, pool.free); self.assertEqual(b0.ref_cnt, 2)
        c = session(sch, pool, 'c', p2 + [3]); self.assertEqual(ps.on_append(sch, c)['action'], 'shared'); self.assertEqual(b0.ref_cnt, 3)

    def test_non_exclusive_owner_is_not_registered(self):
        pool = Pool(16); sch = scheduler(pool); prefix = list(range(BLOCK))
        a = session(sch, pool, 'a', prefix + [1]); pool.touch([sch.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks['a'][0]])
        self.assertIsNone(ps.on_append(sch, a)); self.assertEqual(ps.STATS['skipped_not_exclusive'], 1); self.assertIsNone(ps.registry())

    def test_real_block_pool_semantics(self):
        if REAL_BLOCK_POOL is None:
            self.skipTest(f'vLLM unavailable: {_REAL_POOL_ERROR}')
        real = REAL_BLOCK_POOL(num_gpu_blocks=12, enable_caching=False, hash_block_size=BLOCK)
        sch = scheduler(real); prefix = list(range(BLOCK)); mgr = sch.kv_cache_manager.coordinator.single_type_managers
        for rid, tail in (('a', 1), ('b', 2)):
            mgr[0].req_to_blocks[rid] = real.get_new_blocks(2); mgr[1].req_to_blocks[rid] = real.get_new_blocks(1)
        free0 = real.get_num_free_blocks()
        a = NS(request_id='a', _all_token_ids=prefix + [1], num_computed_tokens=BLOCK, mm_features=[])
        b = NS(request_id='b', _all_token_ids=prefix + [2], num_computed_tokens=BLOCK, mm_features=[])
        self.assertEqual(ps.on_append(sch, a)['action'], 'registered'); self.assertEqual(ps.on_append(sch, b)['action'], 'shared')
        self.assertEqual(real.get_num_free_blocks(), free0 + 1); self.assertEqual(mgr[0].req_to_blocks['a'][0].ref_cnt, 3)


class Churn(unittest.TestCase):
    """Kaggle runs 110 games through fewer lanes: sessions start, join the shared block at their first turn boundary,
    rebase (restart with fresh private blocks), pause and close in random interleavings, and the first registrant has an
    odd prefix. After every event each block's ref_cnt equals its holders (sessions + registry) and the free queue holds
    exactly the unreferenced blocks; while the odd session lives, production sessions are refused (its block is in use),
    and once it closes the next production session takes the registry over and every later one shares."""
    def setUp(self):
        self.modules = patch.dict(sys.modules, te.fake_specs()); self.modules.start()
        self.env = patch.dict(os.environ, {ps.FLAG: '1', ps.BLOCKS_FLAG: '1'}); self.env.start()
        ps.reset_for_tests()

    def tearDown(self):
        ps.reset_for_tests(); self.env.stop(); self.modules.stop()

    def test_110_games_through_eight_lanes(self):
        import collections, random
        rnd = random.Random(3); pool = Pool(600); sch = scheduler(pool); mgr = sch.kv_cache_manager.coordinator.single_type_managers
        prod = list(range(1000, 1000 + BLOCK)); odd = [5] + prod[1:]
        live, turns, prefix_of = {}, collections.Counter(), {}
        def start(rid, prefix):
            live[rid] = session(sch, pool, rid, prefix + [rnd.randrange(100) for _ in range(rnd.randrange(1, 40))],
                                attn_blocks=rnd.randrange(2, 6)); prefix_of[rid] = prefix
        def close(rid):
            live.pop(rid); pool.free_blocks(mgr[0].req_to_blocks.pop(rid)); pool.free_blocks(mgr[1].req_to_blocks.pop(rid))
        def check():
            holders = collections.Counter(id(b) for rid in live for g in (0, 1) for b in mgr[g].req_to_blocks[rid])
            if ps._REGISTRY:
                holders.update(id(b) for b in ps._REGISTRY['blocks'])
            free = {id(b) for b in pool.free}
            for b in pool.blocks[1:]:
                self.assertEqual(b.ref_cnt, holders[id(b)], b.block_id); self.assertEqual(id(b) in free, b.ref_cnt == 0, b.block_id)
        start('g0', odd); self.assertEqual(ps.on_append(sch, live['g0'])['action'], 'registered'); check()
        games, refused_while_odd_lives, actions = 1, 0, collections.Counter()
        while games < 110 or live:
            if games < 110 and len(live) < 8 and rnd.random() < 0.5:
                start(f'g{games}', prod); games += 1; check(); continue
            if not live:
                continue
            rid = rnd.choice(sorted(live)); u = rnd.random()
            if rid == 'g0':                                                         # the odd registrant: lives until
                if turns[rid] >= 3 and refused_while_odd_lives >= 3:                # production sessions were refused
                    close(rid)
                else:
                    ps.on_append(sch, live[rid]); turns[rid] += 1
            elif u < 0.08:
                close(rid)                                                          # game over / budget exhausted
            elif u < 0.12:
                prefix = prefix_of[rid]; close(rid); start(rid, prefix)             # rebase: fresh private blocks
            else:
                r = ps.on_append(sch, live[rid]); turns[rid] += 1
                actions[r['action'] if r else 'none'] += 1
                if 'g0' in live and prefix_of[rid] is prod and r is None:
                    refused_while_odd_lives += 1
            check()
        self.assertEqual(ps.STATS['replaced'], 1); self.assertEqual(ps._REGISTRY['tokens'], tuple(prod))
        self.assertGreater(refused_while_odd_lives, 0); self.assertGreater(actions['shared'], 50)
        self.assertEqual([b.ref_cnt for b in ps._REGISTRY['blocks']], [1])          # all games closed; registry keeps it
        self.assertEqual(len(pool.free), len(pool.blocks) - 2)                       # null block + the registered block


class Eviction(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, te.fake_specs()); self.modules.start()
        self.audit = patch.object(engine_patch, '_audit', lambda *a, **k: None); self.audit.start()
        ps.reset_for_tests()

    def tearDown(self):
        ps.reset_for_tests(); self.audit.stop(); self.modules.stop()

    def share_first_page(self, f):
        first = f.managers[0].req_to_blocks[f.request.request_id][0]
        first.ref_cnt = 3                                          # session + another session + registry
        ps._REGISTRY = dict(group_id=0, block_size=BLOCK, blocks=[first], tokens=tuple(range(BLOCK)), digest='x', owner='other', when=0)
        return first

    def test_shared_page_accepted_when_rows_stay(self):
        f = te.fixture(removed=((40, 50),)); first = self.share_first_page(f)
        tx = ee.prepare_transaction(f.core, 'example', f.payload)
        self.assertIs(tx['groups'][0]['kept_blocks'][0], first); self.assertNotIn(first, tx['groups'][0]['freed_blocks'])

    def test_moving_shared_rows_is_refused(self):
        f = te.fixture(removed=((8, 32),)); self.share_first_page(f)
        with self.assertRaisesRegex(ee.EngineEvictionError, 'shared prefix block'):
            ee.prepare_transaction(f.core, 'example', f.payload)

    def test_partial_eviction_inside_shared_page_is_refused(self):
        f = te.fixture(removed=((14, 20),)); self.share_first_page(f)   # main rows 14-15 and compressed row 3 would move
        with self.assertRaisesRegex(ee.EngineEvictionError, 'shared prefix block'):
            ee.prepare_transaction(f.core, 'example', f.payload)

    def test_unregistered_shared_page_is_still_refused(self):
        f = te.fixture(removed=((40, 50),)); f.managers[0].req_to_blocks[f.request.request_id][0].ref_cnt = 2
        with self.assertRaisesRegex(ee.EngineEvictionError, 'not exclusively owned'):
            ee.prepare_transaction(f.core, 'example', f.payload)


if __name__ == '__main__':
    unittest.main()
