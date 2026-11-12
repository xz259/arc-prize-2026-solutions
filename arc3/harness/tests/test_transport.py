"""v6 transport checks without a server:
  T1 the response parser accepts real persistent responses recorded in g8b1 (the control's protocol), replayed in
     order through one session (start, appends, continuations), and its pending/finish bookkeeping matches;
  T2 request bodies: start carries chat/messages/tools/sampling + history_eviction + the observation sidecar; an
     append carries expected revision/prefix, external messages, sampling, continuation;
  T3 HC1b: a 413 persistent_session_error with eviction_committed false becomes CapacityRefusal with
     I_max = B - 1 - R + E, the ledger discards the plan, and the agent retries once with N6 tool content;
  T4 caps: CAPS 8000,12000 / CHECK 8000 -> max_tokens 8000 first, soft cue at the 8K pause (think open), hard close.
usage: PYTHONPATH=src:. python3 tests/test_transport.py <g8b1 raw-harness game jsonl>"""
import json
import os
import sys

os.environ['ARC3_ASSISTANT_TURN_CAPS'] = '8000,12000'
os.environ['ARC3_ASSISTANT_TURN_CHECK'] = '8000'
from v6 import transport as tp  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('%-72s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


def t1_replay(path):
    rows = [json.loads(l) for l in open(path)]
    reqs = [r for r in rows if r['kind'] == 'persistent-request']
    resps = [r for r in rows if r['kind'] == 'persistent-response']
    start = reqs[0]['payload']['body']
    he = start['history_eviction']
    s = tp.PersistentSession(base_url='http://x', model='m', tools=[], template_kwargs={}, sampling={},
                             history_eviction=he, im_end_ids=(248046,), decode=lambda ids: '')
    s.session_id = start['session_id']
    ok, n, pend = True, 0, 0
    for req, resp in zip(reqs, resps):
        body = req['payload']['body']
        cont = bool(body.get('continuation'))
        try:
            c = s._parse(resp['payload'], continuation=cont, seconds=0.0)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print('  parse failed at request', n, repr(exc)[:200])
            break
        s.revision = c.persistent['revision']
        s.assistant_ids = c.assistant_token_ids
        s.pending = c.pending
        s.last_finish = c.finish_reason
        pend += c.pending
        n += 1
    check('T1 parser accepts %d recorded g8b1 responses (%d pauses)' % (n, pend), ok and n == len(resps), (n, len(resps)))


def t2_bodies():
    sent = []

    class FakeHTTP:
        def start(self, body, timeout):
            sent.append(('start', body))
            return _resp(body['session_id'], 1)

        def append(self, sid, body, timeout):
            sent.append(('append', body))
            return _resp(sid, body['expected_revision'] + 1)

        def close(self, sid, timeout=15):
            return {}

        def shutdown(self):
            pass

    def _resp(sid, rev):
        obs = sent[-1][1]['managed_suffixes']['observations']
        return dict(choices=[dict(message=dict(role='assistant', content='x'), finish_reason='stop', token_ids=[5, 248046])],
                    usage=dict(prompt_tokens=10, completion_tokens=2),
                    persistent=dict(session_id=sid, revision=rev, prefix_sha256='a' * 64,
                                    history_eviction=dict(policy='qsa_frequency', fraction=0.1),
                                    managed_suffixes=dict(registered_observation_ids=[o['observation_id'] for o in obs],
                                                          evicted_observation_ids=[])))
    he = dict(fraction=0.1, policy='qsa_frequency', protect_recent_tokens=5484, target_basis_tokens=30000, unit='split')
    s = tp.PersistentSession(base_url='http://x', model='m', tools=[{'t': 1}], template_kwargs={'reasoning_effort': 'xhigh'},
                             sampling=dict(temperature=1.0, top_p=0.95, top_k=20, max_tokens=12000),
                             history_eviction=he, im_end_ids=(248046,), decode=lambda ids: '')
    s.transport = FakeHTTP()
    s.generate([dict(role='system', content='S'), dict(role='user', content='O1', _arc3_observation=True)])
    s.generate([dict(role='tool', tool_call_id='c', content='T'), dict(role='user', content='O2', _arc3_observation=True)])
    k0, b0 = sent[0]
    k1, b1 = sent[1]
    check('T2 start body', k0 == 'start' and b0['chat']['messages'][0]['role'] == 'system'
          and b0['chat']['max_tokens'] == 8000 and b0['history_eviction'] == he
          and b0['managed_suffixes']['observations'][0]['message_index'] == 1
          and '_arc3_observation' not in json.dumps(b0['chat']['messages']), b0['chat'].get('max_tokens'))
    check('T2 append body', k1 == 'append' and b1['expected_revision'] == 1 and b1['expected_prefix_sha256'] == 'a' * 64
          and [m['role'] for m in b1['external_messages']] == ['tool', 'user'] and b1['continuation'] is False
          and b1['managed_suffixes']['observations'][0]['observation_id'] == 'obs-00000002', b1.get('expected_revision'))


def t3_hc1b():
    receipt = dict(capacity_fallback='tail_exhausted', eviction_committed=False, resident_token_budget=35484,
                   resident_tokens_after_suffix=34000, eligible_history_tokens=900, triggered=True)
    err = tp.PersistentProtocolError('413', status_code=413,
                                     error_payload=dict(error=dict(type='persistent_session_error', message='Context capacity',
                                                                   history_eviction=receipt)))
    s = tp.PersistentSession(base_url='http://x', model='m', tools=[], template_kwargs={}, sampling={},
                             history_eviction=dict(policy='qsa_frequency', fraction=0.1), im_end_ids=(1,), decode=lambda i: '')
    r = s._capacity_refusal(err)
    check('T3 refusal parsed with I_max = B - 1 - R + E', isinstance(r, tp.CapacityRefusal) and r.i_max == 35484 - 1 - 34000 + 900,
          getattr(r, 'i_max', None))
    err2 = tp.PersistentProtocolError('413', status_code=413, error_payload=dict(error=dict(
        type='persistent_session_error', message='Context capacity reached after planned eviction')))
    r2 = s._capacity_refusal(err2)
    check('T3 receipt-less 413 is a refusal with I_max None', isinstance(r2, tp.CapacityRefusal) and r2.i_max is None)
    from v6.agent import V6Agent
    calls = []

    class S:
        def generate(self, external):
            calls.append(json.loads(json.dumps(external)))
            if len(calls) == 1:
                raise tp.CapacityRefusal('x', receipt=receipt, i_max=2383)
            return 'ok'
    a = V6Agent.__new__(V6Agent)
    a._v6_session, a._v6_count = S(), (lambda t: len(t) // 4)
    a._v6_stats = dict(capacity_refusals=0)
    a._v6_log = lambda k, p: None
    out = a._v6_generate([dict(role='tool', tool_call_id='c', content='X' * 40000), dict(role='user', content='O')])
    check('T3 agent retries once with N6 in place of the tool result', out == 'ok' and len(calls) == 2
          and calls[1][0]['content'].startswith('This output was too large to keep in context (') and calls[1][1]['content'] == 'O',
          calls[1][0]['content'][:80] if len(calls) > 1 else None)


def t4_caps():
    soft, hard, check_ = tp.turn_caps_from_env()
    a = tp.turn_cap_decision(0, False, soft, hard, False, check=check_)
    b = tp.turn_cap_decision(8000, False, soft, hard, False, check=check_)
    c = tp.turn_cap_decision(12000, True, soft, hard, True, check=check_)
    d = tp.turn_cap_decision(8000, True, soft, hard, False, check=check_)
    check('T4 first request max_tokens 8000', a == (8000, None, None), a)
    check('T4 soft cue at 8K with think open (closes think), then 4,000 more', b[0] == 4000 and b[1]['kind'] == 'soft_budget'
          and b[1]['text'].endswith('</think>\n\n'), b[:1])
    check('T4 hard close at 12K', c[0] == 1 and c[1]['kind'] == 'hard_close' and c[1]['close'], c[1]['kind'])
    check('T4 no cue when think already closed', d == (4000, None, None), d)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        t1_replay(sys.argv[1])
    t2_bodies()
    t3_hc1b()
    t4_caps()
    print('TEST_TRANSPORT', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    sys.exit(1 if FAIL else 0)
