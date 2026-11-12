"""Dry-run stand-in for v6.transport.PersistentSession (run_v6.py --fake-model): every request is answered with one
python tool call that picks a random valid action (parsed from the latest opener), occasionally an inspection call
or a reply without a tool call, so the solver, games, auto-RESET, logs and scoring run end to end without a server."""
import json
import random
import re
import time

from v6.transport import Completion


class FakeSession:
    def __init__(self, log=None, seed=0):
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
        r = self.rng.random()
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

    def close(self):
        self.closed = True
