"""Dry-run stand-in for v6.transport.PersistentSession (run_v6.py --fake-model): every request is answered with one
python tool call that picks a random valid action (parsed from the latest opener), occasionally an inspection call
or a reply without a tool call, so the solver, games, auto-RESET, logs and scoring run end to end without a server."""
import json
import random
import re
import os
import time

from v62.transport import Completion, PersistentProtocolError

STRESS = os.environ.get("ARC3_V61_FAKE_STRESS", "0") == "1"
SCRIPT = (json.load(open(os.environ["ARC3_V61_FAKE_SCRIPT"])) if os.environ.get("ARC3_V61_FAKE_SCRIPT") else [])
# round 5d dry run: per-request probability of the 380 g047 failure (a turn that ended on <|endoftext|>)
PROTOCOL_ERROR = float(os.environ.get("ARC3_V62_FAKE_PROTOCOL_ERROR", "0") or 0)


class FakeSession:
    def __init__(self, log=None, seed=None):
        self.log = log or (lambda k, p: None)
        self.rng = random.Random(seed)
        self.rebased_away, self.blocks, self.evicted_ranges = set(), [], []
        self.closed = False
        self.valid = ['ACTION1']
        self.n = 0

    def span_evicted(self, a, b):
        return 0

    def generate(self, external):
        self.n += 1
        for m in external:
            if m.get('role') == 'user':
                c = m.get('content')
                text = c if isinstance(c, str) else ' '.join(p.get('text', '') for p in c if p.get('type') == 'text')
                found = re.search(r'Valid actions: ([^.\n]+)\.', text)
                if found:
                    self.valid = [v.strip() for v in found.group(1).split(',') if v.strip() and v.strip() != 'none']
        self.log('persistent-request', dict(fake=True, external=len(external)))
        if PROTOCOL_ERROR and self.rng.random() < PROTOCOL_ERROR:
            raise PersistentProtocolError("Completed assistant lacks actual generated im_end")
        if SCRIPT:                                  # v6.1 scripted test: items in order, then random play
            item = SCRIPT.pop(0)
            if item.get('empty'):
                return self._call('{}', raw_text=item.get('raw'))
            return self._call(json.dumps(dict(code=item['code'])))
        asked = self._summary_asked(external)       # round 5l dry run: answer the summary and notes requests
        if asked is not None and self.rng.random() < 0.8:
            kind, n = asked
            head = 'Notes L%d' % n if kind == 'notes' else 'Rule L%d' % n
            a = self.rng.choice([v for v in (self.valid or ['ACTION1']) if v.upper() != 'MOUSE'] or ['ACTION1'])
            code = ('save_summary(%r)\naction([%r])' % ('%s:\n- fake item one about the level\n- fake item two about the '
                                                        'level' % head, a))
            return self._call(json.dumps(dict(code=code)))
        r = self.rng.random()
        if STRESS:                                  # v6.1 dry run: exercise the new paths
            special = self._stress(r)
            if special is not None:
                return special
        if r < 0.1:
            msg = dict(role='assistant', content='Thinking only.', reasoning_content='Let me call python now.')
            tokens = 50
        else:
            if r < 0.3:
                code = 'print(current_frame.shape, len(history), programs())'
            else:
                a = self.rng.choice(self.valid or ['ACTION1'])
                if a.upper() == 'MOUSE':
                    code = "action([{'action': 'MOUSE', 'row': %d, 'col': %d}])" % (self.rng.randrange(64), self.rng.randrange(64))
                else:
                    code = "action(['%s'])" % a
            msg = dict(role='assistant', content='', reasoning_content='fake',
                       tool_calls=[dict(id='call-%d' % self.n, type='function',
                                        function=dict(name='python', arguments=json.dumps(dict(code=code))))])
            tokens = self.rng.randint(200, 1500)
        time.sleep(0.05)
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=100, completion_tokens=tokens),
                          token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False, closed_by=None,
                          persistent={}, seconds=0.05)

    @staticmethod
    def _summary_asked(external):
        """('rule', n) after a level-up request, ('notes', n) in a notes turn, else None (round 5l dry runs)."""
        for m in reversed(external):
            if m.get('role') != 'user':
                continue
            c = m.get('content')
            text = c if isinstance(c, str) else ' '.join(p.get('text', '') for p in c if p.get('type') == 'text')
            found = re.search(r'begins with the line "(Rule|Notes) L(\d+):"', text)
            if found:
                return ('notes' if found.group(1) == 'Notes' else 'rule'), int(found.group(2))
            return None
        return None

    def _reply(self, msg, tokens):
        time.sleep(0.05)
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=100, completion_tokens=tokens),
                          token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False, closed_by=None,
                          persistent={}, seconds=0.05)

    def _call(self, arguments, raw_text=None):
        self.n += 1
        msg = dict(role='assistant', content='', reasoning_content=raw_text or 'fake',
                   tool_calls=[dict(id='call-%d' % self.n, type='function',
                                    function=dict(name='python', arguments=arguments))])
        return self._reply(msg, self.rng.randint(200, 900))

    def _stress(self, r):
        """v6.1 paths: a 45-action batch (per-call cap), RESET as a first action, a call without `code` whose raw
        text still holds the body (re-parse), one without any body (format error, breaker), an invalid action."""
        valid = [v for v in (self.valid or ['UP']) if v.upper() not in ('MOUSE',)] or ['UP']
        if r < 0.06:
            batch = [self.rng.choice(valid) for _ in range(45)]
            return self._call(json.dumps(dict(code='action(%r)' % batch)))
        if r < 0.09:
            return self._call(json.dumps(dict(code="action(['RESET'])")))
        if r < 0.12:
            body = "print('reparsed ok', len(history))"
            raw = 'fake</think>\n<tool_call>\n<function=python>\n%s\n</parameter>\n</function>\n</tool_call>' % body
            return self._call('{}', raw_text=raw)
        if r < 0.15:
            return self._call('{}')
        if r < 0.17:
            return self._call(json.dumps(dict(code="action(['NOT_AN_ACTION'])")))
        return None

    def close(self):
        self.closed = True
