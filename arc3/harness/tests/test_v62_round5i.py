"""v6.2 round 5i unit tests (user 2026-10-07 ~04:40, run 390 at +80 min: "soft cap nudge at 8k then 12k with hard cap at
16k"; V62_BUILD item 95): ARC3_ASSISTANT_TURN_CAPS='nudge,soft,hard' adds a nudge in the model's voice that leaves the
reasoning open, ahead of the closing cue at the soft cap and the close at the hard cap; 'soft,hard' is unchanged."""
import os
import v62.transport as T
fails = []


def check(name, got, want):
    ok = got == want
    print('%-66s -> %-44r %s' % (name, got if not isinstance(got, str) else got[:44], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def env(caps, check_='8000'):
    os.environ['ARC3_ASSISTANT_TURN_CAPS'], os.environ['ARC3_ASSISTANT_TURN_CHECK'] = caps, check_


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


# --- parsing ---------------------------------------------------------------------------------------------------------
env('12000,16000')
check('two values: caps unchanged', T.turn_caps_from_env(), (12000, 16000, 8000))
check('two values: no nudge', T.turn_nudge_from_env(), 0)
env('0,16000')
check('0,16000 (387b): no nudge', (T.turn_caps_from_env()[:2], T.turn_nudge_from_env()), ((0, 16000), 0))
env('8000,12000,16000')
check('three values: soft and hard are the last two', T.turn_caps_from_env(), (12000, 16000, 8000))
check('three values: the nudge is the first', T.turn_nudge_from_env(), 8000)
env('12000,8000,16000')
check('nudge above the soft cap refused', raises(T.turn_nudge_from_env), True)
env('8000,0,16000')
check('nudge without a closing cue refused', raises(T.turn_nudge_from_env), True)
env('4000,8000,12000,16000')
check('four values refused', (raises(T.turn_caps_from_env), raises(T.turn_nudge_from_env)), (True, True))

# --- the schedule: nudge 8K, closing cue 12K, close 16K, requests of at most 8,000 ---------------------------------------
D = lambda cum, closed=False, cued=False, nudged=False, nudge=8000: T.turn_cap_decision(
    cum, closed, 12000, 16000, cued, check=8000, nudge=nudge, nudged=nudged)
check('first request lands on the nudge point', D(0), (8000, None, None))
lim, inj, tail = D(8000)
check('at 8K (reasoning open): the nudge, then up to the soft cap', (lim, inj['kind'], inj['close'], tail), (4000, 'soft_nudge', False, None))
check('the nudge leaves the reasoning open', '</think>' in inj['text'], False)
check('the nudge text (user 05:05; names the python call after 391\'s empty replies)', inj['text'], "\n\nThis turn's "
      "reasoning has used 8,000 tokens. I should conclude with the best plan so far and act on it with a python call.\n\n")
check('after the nudge: no second nudge, run to the soft cap', D(8040, nudged=True), (3960, None, None))
lim, inj, _ = D(12040, nudged=True)
check('at 12K: the closing cue, closes the reasoning', (lim, inj['kind'], inj['close'], inj['text'].endswith('\n</think>\n\n')),
      (3960, 'soft_budget', False, True))
check('the closing cue text (user 05:05: names the 4,000 left before the close)', inj['text'], "\n\nThis turn's reasoning "
      "budget is spent, and about 4,000 tokens remain for my reply before the turn is closed; the game itself continues and "
      "the next turn has a fresh budget. I will conclude now with the best plan so far and act.\n</think>\n\n")
check('8000,12000 (386 schedule): the cue names 4,000 too', '4,000 tokens remain' in T.turn_cap_decision(
    8000, False, 8000, 12000, False, check=8000)[1]['text'], True)
check('0,16000: no cue at all', T.turn_cap_decision(12000, False, 0, 16000, False, check=8000), (4000, None, None))
lim, inj, _ = D(16040, closed=True, cued=True, nudged=True)
check('at 16K: the hard close as before', (lim, inj['kind'], inj['close']), (1, 'hard_close', True))
check('reasoning already closed at 8K: no nudge', D(8000, closed=True), (4000, None, None))
check('after a repeat cue (cued): no nudge', D(8000, cued=True), (8000, None, None))
lim, inj, tail = T.turn_cap_decision(8000, False, 12000, 16000, False, check=8000, repeat_onset=7900, nudge=8000)
check('a repetition at the nudge point takes precedence', (inj['kind'], tail['reason']), ('repeat_cue', 'repeated_tokens'))
check('nudge off: the 12K/16K schedule unchanged (8K pause, no injection)', (D(0, nudge=0), D(8000, nudge=0)),
      ((8000, None, None), (4000, None, None)))
check('nudge off: closing cue at 12K', D(12000, nudge=0)[1]['kind'], 'soft_budget')

# --- through a session: the flags, the request sizes and the reset when the message ends ---------------------------------
IM_END = 248046


class FakeHTTP:
    def __init__(self, finishes, refuse_at=None):
        self.finishes, self.ids, self.sent, self.refuse_at = list(finishes), [], [], refuse_at

    def _resp(self, body, sid, rev, n, continuation):
        inj = body.get('assistant_injection')
        injected = [20000 + i for i in range(40)] if inj else []     # distinct ids: no repetition cue in the test
        finish = self.finishes.pop(0)
        prev = (self.ids + injected) if continuation else []
        ids = [30000 + len(prev) + i for i in range(n - 1)] + ([IM_END] if finish == 'stop' else [30000 + len(prev) + n])
        self.ids = prev + ids
        obs = body['managed_suffixes']['observations']
        persistent = dict(session_id=sid, revision=rev, prefix_sha256='a' * 64, injected_token_ids=injected,
                          history_eviction=dict(policy='qsa_frequency', fraction=0.1),
                          managed_suffixes=dict(registered_observation_ids=[o['observation_id'] for o in obs],
                                                evicted_observation_ids=[]))
        if inj:
            persistent['assistant_injection'] = inj
        choice = dict(message=dict(role='assistant', content='x'), finish_reason=finish, token_ids=ids)
        if continuation:
            choice['persistent_parse'] = dict(scope='full_assistant', prefix_tokens=len(prev), assistant_token_ids=self.ids)
        return dict(choices=[choice], usage=dict(prompt_tokens=10, completion_tokens=len(ids)), persistent=persistent)

    def start(self, body, timeout):
        self.sent.append(body)
        return self._resp(body, body['session_id'], 1, body['chat']['max_tokens'], False)

    def append(self, sid, body, timeout):
        self.sent.append(body)
        if self.refuse_at is not None and len(self.sent) == self.refuse_at:
            raise T.PersistentProtocolError('413', status_code=413, error_payload=dict(error=dict(
                type='persistent_session_error', message='Context capacity')))
        return self._resp(body, sid, body['expected_revision'] + 1, body['sampling']['max_tokens'], body['continuation'])

    def close(self, sid, timeout=15):
        return {}

    def shutdown(self):
        pass


def session(fake):
    logs = []
    s = T.PersistentSession(base_url='http://x', model='m', tools=[], template_kwargs={}, sampling=dict(max_tokens=12000),
                            history_eviction=dict(policy='qsa_frequency', fraction=0.1), im_end_ids=(IM_END,),
                            decode=lambda ids: '', log=lambda k, p: logs.append((k, p)))
    s.transport = fake
    return s, logs


env('8000,12000,16000')
fake = FakeHTTP(['length', 'length', 'length', 'stop', 'length', 'stop'])
s, logs = session(fake)
s.generate([dict(role='system', content='S'), dict(role='user', content='O1', _arc3_observation=True)])
s.generate([])
nudged_mid = s.nudged
s.generate([])
r = s.generate([])
sizes = [b['chat']['max_tokens'] if 'chat' in b else b['sampling']['max_tokens'] for b in fake.sent]
kinds = [(b.get('assistant_injection') or {}).get('kind') for b in fake.sent]
check('message 1: request sizes 8000, 4000, 3960, 1', sizes, [8000, 4000, 3960, 1])
check('message 1: injections none, nudge, cue, close', kinds, [None, 'soft_nudge', 'soft_budget', 'hard_close'])
check('nudged set after the nudge', nudged_mid, True)
check('closed by the hard close; flags reset', (r.closed_by, s.nudged, s.cued), ('hard_close', False, False))
check('the completion lists the cues sent into it', r.cues, ('soft_nudge', 'soft_budget', 'hard_close'))
check('injection log carries the nudge cap', [p.get('nudge_cap') for k, p in logs if k == 'persistent-injection'], [8000, 8000, 8000])
s.generate([dict(role='tool', tool_call_id='c', content='T'), dict(role='user', content='O2', _arc3_observation=True)])
r2 = s.generate([])
check('message 2: the nudge again (8000, then 4000 with the nudge)', ([b['sampling']['max_tokens'] for b in fake.sent[4:]],
      [(b.get('assistant_injection') or {}).get('kind') for b in fake.sent[4:]], r2.pending), ([8000, 4000], [None, 'soft_nudge'], False))
check('message 2: its own cues only', r2.cues, ('soft_nudge',))
fake = FakeHTTP(['length'], refuse_at=2)
s, _ = session(fake)
s.generate([dict(role='system', content='S'), dict(role='user', content='O1', _arc3_observation=True)])
try:
    s.generate([])
    refused = False
except T.CapacityRefusal:
    refused = True
check('a refused nudge request does not count as nudged', (refused, s.nudged, s.message_cues), (True, False, []))
env('12000,16000')
fake = FakeHTTP(['length', 'length', 'stop'])
s, _ = session(fake)
s.generate([dict(role='system', content='S'), dict(role='user', content='O1', _arc3_observation=True)])
s.generate([])
s.generate([])
check('12000,16000 through a session: 8000, 4000, then the cue (no nudge)', ([b['chat']['max_tokens'] if 'chat' in b else
      b['sampling']['max_tokens'] for b in fake.sent], [(b.get('assistant_injection') or {}).get('kind') for b in fake.sent]),
      ([8000, 4000, 4000], [None, None, 'soft_budget']))
print('FAILS:', fails)
