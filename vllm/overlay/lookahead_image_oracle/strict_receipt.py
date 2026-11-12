"""Strict CPU comparison only; no snapshot, native import, or installation."""
import hashlib
import json
import re


class ContractFailure(RuntimeError):
    pass


def token_digest(ids):
    # Same canonical token encoding as the frozen raw_gpu_checks.py helper.
    return hashlib.sha256(json.dumps(list(ids), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def compare_diagnostics(a, b, *, expected_computed_ids):
    if type(expected_computed_ids) not in (tuple, list):
        raise ContractFailure('Expected computed token IDs must be an independent list or tuple')
    if len(expected_computed_ids) > 40000 or any(type(x) is not int or x < 0 for x in expected_computed_ids):
        raise ContractFailure('Invalid expected computed token prefix')
    count = len(expected_computed_ids)
    digest = token_digest(expected_computed_ids)
    workers = []
    for label, receipt in (('left', a), ('right', b)):
        if type(receipt) is not dict or type(receipt.get('worker')) is not dict:
            raise ContractFailure(label + ': expected core and worker receipt objects')
        worker = receipt['worker']
        for where, value in (('core', receipt.get('computed_tokens')),
                             ('worker', worker.get('computed_tokens'))):
            if type(value) is not int or value < 0 or value != count:
                raise ContractFailure(label + ': ' + where + ' computed count disagrees with expected prefix')
        for key in ('comparison_digest', 'computed_token_ids_sha256'):
            value = worker.get(key)
            if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
                raise ContractFailure(label + ': invalid ' + key)
        if worker['computed_token_ids_sha256'] != digest:
            raise ContractFailure(label + ': computed IDs disagree with expected prefix')
        workers.append(worker)
    left, right = workers
    if left['comparison_digest'] != right['comparison_digest']:
        raise ContractFailure('Valid computed cache/GDN digests differ')
    return dict(computed_tokens_equal=True, computed_ids_equal=True, cache_gdn_bytes_equal=True,
                expected_computed_tokens=count, expected_computed_token_ids_sha256=digest,
                left=left, right=right)
