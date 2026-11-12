"""Synthetic receipt construction tests; no claim of native state readout."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
schema=importlib.import_module(HERE.name+'.receipt_schema')
strict=importlib.import_module(HERE.name+'.strict_receipt')
ContractFailure,compare_diagnostics,token_digest=strict.ContractFailure,strict.compare_diagnostics,strict.token_digest


def fixture(ids=None, prompt=511, prefill=511):
    ids = list(range(512)) if ids is None else list(ids)
    components = [dict(x, sha256=hashlib.sha256((x['owner']+x['category']).encode()).hexdigest())
                  for x in schema.expected_components(len(ids))]
    source = dict(schema='test-source-manifest', own_files={'worker.py':'1'*64}, runtime_files={}, helper_modules={})
    canonical = schema.canonical_state(ids, components, schema.expected_logical_metadata(ids, prompt, prefill))
    worker = dict(schema=schema.SCHEMA, scope=schema.SCOPE,
        **dict.fromkeys(schema.FLAGS, True), source_manifest=source,
        source_manifest_sha256=schema.json_digest(source), components=components,
        canonical=canonical, computed_tokens=len(ids),
        computed_token_ids_sha256=token_digest(ids), logical_positions_sha256=token_digest(range(len(ids))),
        owner_inventory_sha256=schema.json_digest(schema.expected_owners()),
        coverage_manifest_sha256=schema.coverage_digest(len(ids)),
        coverage=schema.expected_coverage(len(ids)), comparison_digest=schema.json_digest(canonical),
        ownership=dict(request_id='actual-request', slot=0, core_block_ids=[[i+1] for i in range(6)],
            worker_block_ids=[[i+1] for i in range(6)], block_tables_match=True,
            exclusive_pages=True, frontier_verified=True))
    return dict(computed_tokens=len(ids), worker=worker)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(512))
        self.good = fixture(self.ids)

    def check(self, receipt, ids=None, prompt=511, prefill=511):
        return schema.validate_receipt(receipt, expected_ids=self.ids if ids is None else ids,
            expected_source_manifest_sha256=self.good['worker']['source_manifest_sha256'],
            expected_prompt_tokens=prompt, expected_prefill_len=prefill)

    def test_full_categories_and_bounds(self):
        schema.verify_inventory()
        for c in (511, 512, 513, 514, 1024, 2048):
            ids = list(range(c)); r = fixture(ids)
            self.check(r, ids=ids)
            self.assertEqual(len(r['worker']['components']), 121)
            self.assertEqual(len(r['worker']['canonical']['owner_inventory']), 73)
            self.assertLessEqual(r['worker']['coverage']['selected_bytes'], schema.MAX_BYTES)
        self.assertEqual(schema.expected_coverage(2048)['selected_bytes'], schema.MAX_BYTES)

    def test_physical_identity_excluded_from_equality(self):
        other = copy.deepcopy(self.good); w = other['worker']
        w['ownership'].update(request_id='other-request', slot=9,
            core_block_ids=[[v+40 for v in row] for row in w['ownership']['core_block_ids']],
            worker_block_ids=[[v+40 for v in row] for row in w['ownership']['worker_block_ids']])
        self.check(other)
        result = compare_diagnostics(self.good, other, expected_computed_ids=self.ids)
        self.assertTrue(result['cache_gdn_bytes_equal'])

    def test_missing_duplicate_unknown_and_ordered_components(self):
        variants = []
        for category in schema.CATEGORY_COUNTS:
            r = copy.deepcopy(self.good)
            index = next(i for i, x in enumerate(r['worker']['components']) if x['category'] == category)
            r['worker']['components'].pop(index); variants.append(r)
        for mutation in ('duplicate', 'unknown', 'reverse'):
            r = copy.deepcopy(self.good); cs = r['worker']['components']
            if mutation == 'duplicate': cs[1] = copy.deepcopy(cs[0])
            elif mutation == 'unknown': cs[0]['owner'] = 'unknown'
            else: cs.reverse()
            variants.append(r)
        for r in variants:
            with self.assertRaises(ContractFailure): self.check(r)

    def test_each_category_byte_change_cannot_match(self):
        for category in schema.CATEGORY_COUNTS:
            other = copy.deepcopy(self.good); w = other['worker']
            changed = next(x for x in w['components'] if x['category'] == category)
            changed['sha256'] = hashlib.sha256(b'changed bytes').hexdigest()
            # deepcopy preserves alias between canonical and mirrored components.
            w['comparison_digest'] = schema.json_digest(w['canonical'])
            self.check(other)
            with self.assertRaises(ContractFailure):
                compare_diagnostics(self.good, other, expected_computed_ids=self.ids)

    def test_geometry_coordinate_dtype_and_numeric_type_guards(self):
        for key, value in [('bytes', 0), ('dtype', 'float32'), ('shape', [512]),
                           ('coordinates', dict(kind='token_positions', start=True, stop=512, step=1))]:
            r = copy.deepcopy(self.good); r['worker']['components'][0][key] = value
            with self.assertRaises(ContractFailure): self.check(r)
        r = copy.deepcopy(self.good); r['worker']['coverage']['owner_count'] = 73.0
        with self.assertRaises(ContractFailure): self.check(r)

    def test_source_flags_empty_coverage_and_malformed_hash(self):
        for flag in schema.FLAGS:
            r = copy.deepcopy(self.good); r['worker'][flag] = 1
            with self.assertRaises(ContractFailure): self.check(r)
        for key, value in [('source_manifest', {}), ('source_manifest_sha256', '0'*64),
                           ('coverage', {}), ('comparison_digest', None), ('components', [])]:
            r = copy.deepcopy(self.good); r['worker'][key] = value
            with self.assertRaises(ContractFailure): self.check(r)

    def test_bad_frontier_independent_ids_and_logical_metadata(self):
        for ids in ([0]*510, [0]*2049, [True]*512, [-1]*512, list(range(511))+[999]):
            with self.assertRaises(ContractFailure): self.check(self.good, ids=ids)
        for key in self.good['worker']['canonical']['logical_metadata']:
            r = copy.deepcopy(self.good); r['worker']['canonical']['logical_metadata'][key] = None
            r['worker']['comparison_digest'] = schema.json_digest(r['worker']['canonical'])
            with self.assertRaises(ContractFailure): self.check(r)

    def test_ownership_and_block_table_checks(self):
        for key, value in [('slot', True), ('request_id', ''), ('exclusive_pages', False),
                           ('frontier_verified', False), ('block_tables_match', False),
                           ('core_block_ids', [[0]]*6), ('worker_block_ids', [[999]]*6)]:
            r = copy.deepcopy(self.good); r['worker']['ownership'][key] = value
            with self.assertRaises(ContractFailure): self.check(r)

    def test_analytic_mrope_bytes_and_ple_tail(self):
        md = schema.expected_logical_metadata(self.ids, 511, 511)
        import struct
        expected = b''.join(struct.pack('<i', p) for _ in range(3) for p in range(511))
        self.assertEqual(md['mrope_prefill_positions_sha256'], hashlib.sha256(expected).hexdigest())
        self.assertEqual(md['ple_predecessor_ids_sha256'], token_digest([510, 511]))

    def test_strict_helper_is_unchanged(self):
        p = Path(__file__).with_name('strict_receipt.py')
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),
                         '266462bc16b705baf821aa4ee706c71716fe4f344e50c90856fb4280da425995')


if __name__ == '__main__':
    unittest.main()
