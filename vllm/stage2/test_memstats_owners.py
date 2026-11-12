"""CPU test of memstats: the owners trace attributes lists to their holders; the collector monitor writes lines; the only=<pid>
filter; the sessmem counters in readings (2026-10-08)."""
import importlib.util, json, os, sys, tempfile, time, gc, random
D = tempfile.mkdtemp(); os.environ['ARC3_S2_MEMSTATS_DIR'] = D
spec = importlib.util.spec_from_file_location('memstats', sys.argv[1]); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class RequestState:                     # an instance attribute holding a token list
    def __init__(self, n):
        self.prompt_token_ids = [random.randrange(300, 9000) for _ in range(n)]
        self.output_token_ids = [random.randrange(300, 9000) for _ in range(n // 2)]


class Processor:
    def __init__(self):
        self.request_states = {}        # plain dict keyed by request id -> instances


class Slotted:
    __slots__ = ('ids',)

    def __init__(self, n):
        self.ids = [random.randrange(300, 9000) for _ in range(n)]


P = Processor()
for i in range(40):
    P.request_states['req-%08x-%d' % (i * 7919, i)] = RequestState(3000)
ledger = {'arc3-persistent-%032x' % i: [random.randrange(300, 9000) for _ in range(2500)] for i in range(30)}   # module-level plain dict
holder_of_ledger = Processor(); holder_of_ledger.request_states = ledger
slots = [Slotted(1500) for _ in range(20)]
M.start()
for _ in range(3):
    junk = [[{} for _ in range(1000)] for _ in range(50)]; gc.collect(); del junk
M._reading('t1', 'owners')
r = json.load(open(os.path.join(D, 'memstats-t1-%d.json' % os.getpid())))
o = r['owners']
paths = {p: v for p, v in o['holders']}
print('gc state keys:', sorted(r['gc'])[:5], 'enabled', r['gc'].get('enabled'))
print('lists %d, traced %d, of big %d' % (o['lists'], o['traced'], o['of_big']))
for p, v in o['holders'][:6]:
    print('  %-80s lists %3d elements %7d' % (p[:80], v['lists'], v['elements']))
ok = [any('RequestState.prompt_token_ids' in p for p in paths), any('RequestState.output_token_ids' in p for p in paths),
      any('Processor.request_states' in p and 'dict[' in p for p in paths), any(p.endswith('Slotted') or 'Slotted' in p for p in paths)]
M._gc_line()
lines = [json.loads(l) for l in open(os.path.join(D, 'gcprof-%d.jsonl' % os.getpid()))]
ok.append(lines[-1]['gc']['n'][2] >= 3)
print('gcprof line:', {k: lines[-1][k] for k in ('gc', 'blocks', 'rss_kb')})
# 2026-10-08: "only=<pid>" limits a triggered reading to one process
trig = os.path.join(D, 'trigger')
open(trig, 'w').write('t2 light only=999999')
time.sleep(5)
ok.append(not os.path.exists(os.path.join(D, 'memstats-t2-%d.json' % os.getpid())))
open(trig, 'w').write('t3 light only=%d' % os.getpid())
for _ in range(10):
    time.sleep(1)
    if os.path.exists(os.path.join(D, 'memstats-t3-%d.json' % os.getpid())):
        break
ok.append(os.path.exists(os.path.join(D, 'memstats-t3-%d.json' % os.getpid())))
print('only=: other pid skipped %s, own pid read %s' % (ok[-2], ok[-1]))
# 2026-10-08: a process with the sessmem part loaded records its counters in every reading; others do not
r3 = json.load(open(os.path.join(D, 'memstats-t3-%d.json' % os.getpid())))
ok.append('sessmem' not in r3)
import types
sys.modules['stage2_v1.sessmem'] = types.SimpleNamespace(STATS=dict(closed_freed=3, freed_collectors=3, errors=0))
M._reading('t4', 'light')
r4 = json.load(open(os.path.join(D, 'memstats-t4-%d.json' % os.getpid())))
ok.append(r4.get('sessmem') == dict(closed_freed=3, freed_collectors=3, errors=0) and r4['gc'].get('threshold') == list(gc.get_threshold()))
print('sessmem counters: absent without the part %s, recorded with it %s' % (ok[-2], ok[-1]))
print('TEST_MEMSTATS_OWNERS', 'PASS' if all(ok) else 'FAIL %s' % ok)
