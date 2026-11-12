"""Score fixed assistant-history tokens while retaining original image features.

Use the chat endpoint's prompt logprobs on original replay messages, tools,
reasoning, and images. Freeze the last 512 rendered assistant-span tokens per
stream (including message/tool markup). This is a paired likelihood diagnostic
on observed histories, not generated-target or puzzle-score noninferiority.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import uuid
from urllib.request import Request, urlopen

from benchmark_serving_config import BASE, idle, post
from gameplay_config import GameplayConfig, tokenization_payload
from minimal_inference import ROOT, IDENTITY, MODEL_NAME, matches, save


IMAGE_RESET_ENDPOINTS = ['/reset_mm_cache', '/reset_encoder_cache']


def image_cache_policy(cold):
    return dict(mode='cold_before_each_score' if cold else 'inherited',
                before_each_score=bool(cold),
                reset_endpoints=list(IMAGE_RESET_ENDPOINTS) if cold else [],
                same_server_bracketed=bool(cold), idle_required=bool(cold))


def validate_cold_image_report(report, panel_sha256, expected_root=None):
    """Validate recorded policy/receipts only; never contact a server."""
    assert report['complete'] is True and report['stage'] == 'complete'
    assert report['panel_sha256'] == panel_sha256
    assert report['repeat_max_logprob_difference'] == 0
    assert report.get('image_cache_policy') == image_cache_policy(True)
    rows = report['rows']
    index = {(row['stream'], row['repeat']): row for row in rows}
    assert rows and len(index) == len(rows)
    assert {key[1] for key in index} == {0, 1}
    roots, receipt_paths = [], set()
    for row in rows:
        stream, repeat = row['stream'], row['repeat']
        assert (stream, 1-repeat) in index
        assert row['logprobs'] == index[stream, 1-repeat]['logprobs']
        assert row['target_tokens'] == len(row['logprobs'])
        receipt = row['image_cache_reset']
        expected_name = f'stream{stream}-r{repeat}.json'
        assert receipt['complete'] is True and receipt['request_output'] == expected_name
        assert receipt['server_root_before'] == receipt['server_root_after']
        assert receipt['idle_before'] is True and receipt['idle_after'] is True
        assert receipt['started_epoch'] <= receipt['finished_epoch']
        assert [x['endpoint'] for x in receipt['resets']] == IMAGE_RESET_ENDPOINTS
        for reset in receipt['resets']:
            assert reset['status'] == 200 and isinstance(reset['body'], str)
            assert receipt['started_epoch'] <= reset['epoch'] <= receipt['finished_epoch']
        name = row['image_cache_reset_file']
        assert name == expected_name[:-5]+'.image-cache-reset.json'
        assert name not in receipt_paths
        receipt_paths.add(name)
        # save() uses this exact encoding, also retained beside each raw response.
        raw = (json.dumps(receipt, indent=2)+'\n').encode()
        assert row['image_cache_reset_sha256'] == hashlib.sha256(raw).hexdigest()
        roots.append(receipt['server_root_before'])
    assert all(root == roots[0] for root in roots)
    if expected_root is not None:
        assert roots[0] == expected_root, 'Cold likelihood came from another server'
    return dict(passed=True, image_cache_policy=image_cache_policy(True),
                scored_rows=len(rows), reset_calls=2*len(rows), server_root=roots[0],
                panel_sha256=panel_sha256, scope='Recorded per-score cold-cache receipts and exact repeats')


def server():
    identity = json.loads(IDENTITY.read_text())
    assert matches(identity['root'])
    assert '--no-enable-prefix-caching' in identity['argv']
    assert identity['environment']['ARC3_MOE_FINALIZE'] == 'unfused'
    return identity


def image_count(messages):
    return sum(part.get('type') == 'image_url' for message in messages
               if isinstance(message.get('content'), list) for part in message['content'])


def rendered(payload, messages=None, generation=True):
    request = tokenization_payload(payload)
    request['add_generation_prompt'] = generation
    if messages is not None:
        request['messages'] = messages
    return post('/tokenize', request)['tokens']


def freeze_stream(index, request, config):
    payload = config.chat_payload(MODEL_NAME, request['messages'], request['tools'],
                                  max_tokens=1, seed=8100+index)
    payload.update(temperature=0., min_p=0., prompt_logprobs=0, return_token_ids=True)
    tokens = rendered(payload)
    spans = []
    for i, message in enumerate(payload['messages']):
        if message['role'] != 'assistant':
            continue
        assert not isinstance(message.get('content'), list), 'Assistant targets must be text/tool markup'
        before = rendered(payload, payload['messages'][:i], generation=False)
        after = rendered(payload, payload['messages'][:i+1], generation=False)
        assert tokens[:len(before)] == before and tokens[:len(after)] == after, \
            'Prefix rendering changed; do not infer assistant token boundaries'
        assert 0 < len(before) < len(after) <= len(tokens)
        spans.append(dict(message=i, start=len(before), end=len(after)))
    positions = [position for span in spans for position in range(span['start'], span['end'])][-512:]
    assert len(positions) >= 32 and positions == sorted(set(positions))
    return dict(stream=index, payload=payload, prompt_token_ids=tokens,
                assistant_spans=spans, target_positions=positions,
                images=image_count(payload['messages']))


def reset_image_caches(raw_output, expected):
    """Reset only the owned idle server's image caches; retain partial failures."""
    from run_decode_profile_candidate import same_server
    path = raw_output.with_name(raw_output.stem+'.image-cache-reset.json')
    receipt = dict(complete=False, request_output=raw_output.name,
                   server_root_before=expected['root'], resets=[], started_epoch=time.time())
    save(path, receipt)
    try:
        same_server(expected)
        before_metrics = idle()
        same_server(expected)
        (path.parent/(path.stem+'-before-metrics.txt')).write_text(before_metrics)
        receipt['idle_before'] = True
        save(path, receipt)
        for endpoint in IMAGE_RESET_ENDPOINTS:
            same_server(expected)
            idle()
            same_server(expected)
            with urlopen(Request(BASE+endpoint, data=b''), timeout=30) as response:
                receipt['resets'].append(dict(endpoint=endpoint, status=response.status,
                    body=response.read().decode(), epoch=time.time()))
            save(path, receipt)
            assert receipt['resets'][-1]['status'] == 200, endpoint
            same_server(expected)
        after_metrics = idle()
        current = same_server(expected)
        (path.parent/(path.stem+'-after-metrics.txt')).write_text(after_metrics)
        receipt.update(complete=True, idle_after=True, server_root_after=current['root'],
                       finished_epoch=time.time())
        save(path, receipt)
        return receipt
    except Exception as exc:
        receipt.update(error=f'{type(exc).__name__}: {exc}', finished_epoch=time.time())
        save(path, receipt)
        raise


def score(row, raw_output, *, cold_image_cache=False, expected=None):
    idle()
    if cold_image_cache:
        if expected is None:
            expected = server()
        reset_image_caches(raw_output, expected)
        from run_decode_profile_candidate import same_server
        same_server(expected)
    response = post('/v1/chat/completions', dict(row['payload'], cache_salt=uuid.uuid4().hex))
    save(raw_output, response)
    if cold_image_cache:
        same_server(expected)
    assert response['prompt_token_ids'] == row['prompt_token_ids']
    assert response['usage']['prompt_tokens'] == len(row['prompt_token_ids'])
    multimodal = (response['usage'].get('prompt_tokens_details') or {}).get('multimodal_tokens') or {}
    if row['images']:
        assert multimodal.get('image', 0) > 0, 'Image features were not accounted for by the served request'
    logprobs = response['prompt_logprobs']
    assert len(logprobs) == len(row['prompt_token_ids'])
    values = []
    for position in row['target_positions']:
        token = row['prompt_token_ids'][position]
        item = logprobs[position][str(token)]
        value = item['logprob']
        assert math.isfinite(value)
        values.append(value)
    return response, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--cold-image-cache', action='store_true',
                        help='Reset idle multimodal and encoder caches before every score; default preserves caches')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = server()
    save(args.output/'server-identity.json', identity)
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    context = int(identity['argv'][identity['argv'].index('--max-model-len')+1])
    report = dict(stage='freeze', complete=False, started_epoch=time.time(), rows=[],
                  format=identity['environment'].get('ARC3_QSA_KV_FORMAT', 'bf16'),
                  image_cache_policy=image_cache_policy(args.cold_image_cache),
                  scope=__doc__.strip())

    def update(**fields):
        report.update(**fields)
        save(args.output/'result.json', report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['rows', 'scope']}), flush=True)

    update()
    if args.panel:
        raw = args.panel.read_bytes()
        panel = json.loads(raw)
        assert panel['model'] == identity['model']
    else:
        assert report['format'] in ('bf16', 'nvfp4'), report['format']  # stage 0: reference panel built on the NVFP4 serving stack
        source = (ROOT/'replay-panel-v1.json').read_bytes()
        config = GameplayConfig()
        panel = dict(model=identity['model'], baseline_server=identity,
                     source_sha256=hashlib.sha256(source).hexdigest(),
                     gameplay=config.manifest(), target_rule='Last 512 rendered assistant-span tokens, including markup',
                     streams=[])
        for i, stream in enumerate(json.loads(source)['streams']):
            panel['streams'].append(freeze_stream(i, stream['steps'][-1]['request'], config))
            save(args.output/'panel-building.json', panel)
        assert sum(row['images'] for row in panel['streams']) > 0
        raw = (json.dumps(panel, indent=2)+'\n').encode()
    (args.output/'panel.json').write_bytes(raw)
    update(stage='score', panel_sha256=hashlib.sha256(raw).hexdigest())
    for repeat in range(2):
        for row in panel['streams']:
            assert server()['root'] == identity['root']
            assert len(row['prompt_token_ids'])+1 <= context
            raw_output = args.output/f"stream{row['stream']}-r{repeat}.json"
            response, values = score(row, raw_output,
                                     cold_image_cache=args.cold_image_cache, expected=identity)
            assert server()['root'] == identity['root']
            scored = dict(stream=row['stream'], repeat=repeat,
                context_tokens=len(row['prompt_token_ids']), target_tokens=len(values),
                images=row['images'], logprobs=values, mean_nll=-statistics.mean(values))
            if args.cold_image_cache:
                receipt_path = raw_output.with_name(raw_output.stem+'.image-cache-reset.json')
                receipt_raw = receipt_path.read_bytes()
                scored.update(image_cache_reset=json.loads(receipt_raw),
                              image_cache_reset_file=receipt_path.name,
                              image_cache_reset_sha256=hashlib.sha256(receipt_raw).hexdigest())
            report['rows'].append(scored)
            update(stage='score', last_stream=row['stream'], repeat=repeat)
    report['mean_nll'] = -statistics.mean(value for row in report['rows'] for value in row['logprobs'])
    size = len(panel['streams'])
    report['repeat_max_logprob_difference'] = max(abs(x-y)
        for a,b in zip(report['rows'][:size], report['rows'][size:])
        for x,y in zip(a['logprobs'], b['logprobs']))
    update(stage='validate_repeat')
    assert report['repeat_max_logprob_difference'] == 0., 'Within-configuration prompt-logprob repeats differ'
    if args.reference:
        reference = json.loads(args.reference.read_text())
        assert reference['complete'] and reference['panel_sha256'] == report['panel_sha256']
        before = {row['stream']:row for row in reference['rows'] if row['repeat'] == 0}
        deltas = []
        report['paired_changes'] = []
        for after in report['rows'][:size]:
            original = before[after['stream']]['logprobs']
            assert len(original) == len(after['logprobs'])
            values = [a-b for a,b in zip(original, after['logprobs'])]
            deltas.extend(values)
            report['paired_changes'].append(dict(stream=after['stream'],
                mean_nll_increase=statistics.mean(values), max_abs_logprob_change=max(map(abs, values))))
        report['mean_nll_increase'] = statistics.mean(deltas)
    if args.cold_image_cache:
        validate_cold_image_report(dict(report, stage='complete', complete=True),
                                   report['panel_sha256'], expected_root=identity['root'])
    update(stage='complete', complete=True, finished_epoch=time.time())


if __name__ == '__main__':
    main()
