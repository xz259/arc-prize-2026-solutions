"""Pure-stdlib, independently reconstructed scalar state receipt contract.

No serving imports, install hooks or state readout. A valid receipt is not proof
that the readout implementation is correct; that implementation needs review and
native validation separately. Physical allocation identity never enters equality.
"""
import hashlib
import json
from pathlib import Path
import re
import struct

if __package__:
    from .strict_receipt import ContractFailure, token_digest
else:
    from strict_receipt import ContractFailure, token_digest

SCHEMA = 'arc3-one-lookahead-snapshot-v1'
CANONICAL_SCHEMA = 'arc3-one-lookahead-canonical-state-v1'
SCOPE = 'text-only-no-eviction-mtp0'
MIN_COMPUTED = 511
MAX_COMPUTED = 2048
MAX_BYTES = 75154560
MAX_CHUNK_BYTES = 1048576
INVENTORY_SHA256 = 'c1f431ab8de8185b1ce872bd75654d84ee85f3243322bbc68418cfa9e2d75bce'
CATEGORY_COUNTS = dict(main_nvfp4=12, qsa_compressed=12, qsa_raw_key=12,
    qsa_raw_rope_position=12, gdn_conv=36, gdn_ssm=36, ple_short_conv=1)
FLAGS = ('complete', 'read_only', 'source_admitted', 'owner_coverage_complete',
         'worker_fenced', 'metadata_stable')


def require(ok, message):
    if not ok:
        raise ContractFailure(message)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def json_digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def same(left, right):
    """JSON type-sensitive equality; bool/float must not impersonate integers."""
    return json_bytes(left) == json_bytes(right)


def digest_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def expected_owners():
    result = []
    prefix = 'language_model.model.layers.'
    for layer in range(48):
        base = prefix + str(layer)
        if layer % 4 == 3:
            result.extend([
                dict(owner=base+'.self_attn.attn', group_id=0, categories=['main_nvfp4']),
                dict(owner=base+'.self_attn.indexer.compressed_key_cache', group_id=0,
                     categories=['qsa_compressed']),
                dict(owner=base+'.self_attn.indexer.raw_key_cache', group_id=1,
                     categories=['qsa_raw_key', 'qsa_raw_rope_position']),
            ])
        else:
            result.append(dict(owner=base+'.linear_attn', group_id=2+layer % 4,
                               categories=['gdn_conv', 'gdn_ssm']))
    result.append(dict(owner=prefix+'1.ple', group_id=5, categories=['ple_short_conv']))
    return sorted(result, key=lambda x: x['owner'])


def verify_inventory():
    path = Path(__file__).with_name('expected_inventory.json')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == INVENTORY_SHA256,
            'Pinned owner inventory file changed')
    data = json.loads(path.read_text())
    actual = {name: group['group_id'] for group in data['groups'] for name in group['owner_names']}
    require(actual == {x['owner']: x['group_id'] for x in expected_owners()},
            'Analytic owner inventory differs from pinned source')
    require(same(data['component_counts'], CATEGORY_COUNTS) and data['owners'] == 73,
            'Pinned component inventory changed')


def expected_components(computed):
    require(type(computed) is int and MIN_COMPUTED <= computed <= MAX_COMPUTED,
            'Computed frontier outside scalar oracle scope')
    components = []
    for owner in expected_owners():
        for category in owner['categories']:
            if category == 'main_nvfp4':
                dtype, shape = 'uint8', [computed, 2, 296]
                coords = dict(kind='token_positions', start=0, stop=computed, step=1)
                size = computed*2*296
            elif category == 'qsa_compressed':
                dtype, shape = 'bfloat16', [computed//4, 128]
                coords = dict(kind='compression_groups', start=0, stop=computed//4, step=1, compress_ratio=4)
                size = computed//4*128*2
            elif category in ('qsa_raw_key', 'qsa_raw_rope_position'):
                dtype, shape, size = (('bfloat16', [4, 128], 4*128*2) if category == 'qsa_raw_key'
                                      else ('int64', [4, 3], 4*3*8))
                coords = dict(kind='token_positions', start=computed-4, stop=computed, step=1)
            else:
                dtype = 'bfloat16'
                shape, index = {'gdn_conv': ([3, 10240], 0),
                    'gdn_ssm': ([48, 128, 128], 1), 'ple_short_conv': ([9, 10240], 0)}[category]
                size = 2
                for dimension in shape:
                    size *= dimension
                coords = dict(kind='state_component', index=index)
            components.append(dict(owner=owner['owner'], category=category,
                dtype=dtype, shape=shape, coordinates=coords, bytes=size))
    return sorted(components, key=lambda x: (x['owner'], x['category']))


def expected_logical_metadata(ids, original_prompt_tokens, prefill_len):
    computed = len(ids)
    require(type(original_prompt_tokens) is int and type(prefill_len) is int
            and 1 <= original_prompt_tokens <= prefill_len <= computed,
            'Invalid independently expected prompt/prefill frontier')
    row = b''.join(struct.pack('<i', p) for p in range(prefill_len))
    return dict(original_prompt_tokens=original_prompt_tokens, prefill_len=prefill_len,
        mrope_axes=3, mrope_delta=0,
        mrope_prefill_positions_sha256=hashlib.sha256(row*3).hexdigest(),
        ple_context_len=2, ple_predecessor_ids_sha256=token_digest(ids[-2:]))


def expected_coverage(computed):
    owners = [x['owner'] for x in expected_owners()]
    components = expected_components(computed)
    return dict(expected_owners=owners, observed_owners=owners, owner_count=73,
        component_count=121, category_counts=dict(CATEGORY_COUNTS), missing=[], unknown=[],
        selected_bytes=sum(x['bytes'] for x in components), max_selected_bytes=MAX_BYTES,
        max_copy_chunk_bytes=MAX_CHUNK_BYTES)


def coverage_digest(computed):
    return json_digest(dict(owner_inventory=expected_owners(), components=expected_components(computed)))


def canonical_state(ids, components, logical_metadata):
    return dict(schema=CANONICAL_SCHEMA, scope=SCOPE, computed_tokens=len(ids),
        computed_token_ids_sha256=token_digest(ids),
        logical_positions_sha256=token_digest(range(len(ids))),
        owner_inventory=expected_owners(), components=components, logical_metadata=logical_metadata)


def validate_receipt(receipt, *, expected_ids, expected_source_manifest_sha256,
                     expected_prompt_tokens, expected_prefill_len):
    verify_inventory()
    require(type(expected_ids) in (list, tuple) and all(type(x) is int and x >= 0 for x in expected_ids),
            'Expected computed IDs must be independent nonnegative integers')
    computed = len(expected_ids)
    descriptors = expected_components(computed)
    require(digest_ok(expected_source_manifest_sha256), 'Missing pinned source manifest digest')
    require(type(receipt) is dict and type(receipt.get('worker')) is dict,
            'Missing core/worker snapshot objects')
    worker = receipt['worker']
    require(worker.get('schema') == SCHEMA and worker.get('scope') == SCOPE, 'Wrong snapshot schema/scope')
    require(all(worker.get(flag) is True for flag in FLAGS), 'Incomplete/unproven snapshot')
    require(type(worker.get('source_manifest')) is dict and worker['source_manifest'], 'Missing source manifest')
    require(worker.get('source_manifest_sha256') == expected_source_manifest_sha256
            and json_digest(worker['source_manifest']) == expected_source_manifest_sha256,
            'Snapshot source identity differs from independent pin')
    require(same(worker.get('coverage'), expected_coverage(computed)), 'Incomplete or altered owner coverage')
    require(worker.get('owner_inventory_sha256') == json_digest(expected_owners()), 'Owner inventory digest mismatch')
    require(worker.get('coverage_manifest_sha256') == coverage_digest(computed), 'Coverage manifest digest mismatch')
    components = worker.get('components')
    require(type(components) is list and len(components) == 121, 'Missing snapshot components')
    for actual, expected in zip(components, descriptors):
        require(type(actual) is dict and set(actual) == set(expected) | {'sha256'}, 'Malformed component record')
        require(digest_ok(actual['sha256']), 'Missing component byte digest')
        require(same({k:v for k,v in actual.items() if k != 'sha256'}, expected),
                'Component geometry, coordinates or ordering mismatch')
    logical = expected_logical_metadata(expected_ids, expected_prompt_tokens, expected_prefill_len)
    canonical = canonical_state(expected_ids, components, logical)
    require(same(worker.get('canonical'), canonical), 'Canonical state differs from independent logical manifest')
    for key in ('computed_tokens', 'computed_token_ids_sha256', 'logical_positions_sha256'):
        require(same(worker.get(key), canonical[key]), 'Snapshot field disagrees with canonical state: '+key)
    require(worker.get('comparison_digest') == json_digest(canonical), 'Comparison hash does not cover canonical state')
    ownership = worker.get('ownership')
    require(type(ownership) is dict and type(ownership.get('request_id')) is str
            and ownership['request_id'] and type(ownership.get('slot')) is int and ownership['slot'] >= 0,
            'Missing worker ownership identity')
    for key in ('block_tables_match', 'exclusive_pages', 'frontier_verified'):
        require(ownership.get(key) is True, 'Unverified ownership: '+key)
    blocks = ownership.get('core_block_ids')
    require(type(blocks) is list and len(blocks) == 6
            and all(type(group) is list and group and all(type(i) is int and i > 0 for i in group)
                    for group in blocks), 'Invalid core block ownership lists')
    require(same(blocks, ownership.get('worker_block_ids')), 'Core/worker ownership lists disagree')
    return worker
