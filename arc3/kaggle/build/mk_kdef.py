"""mk_kdef.py A.json B.json REQ80.json OUT.json (2026-10-24, Claude main session 099768dc; the Kaggle port's guard-free launcher).
Builds the launch definition kaggle_run.py renders (KDEF.json) from two live captures of the same stack (cap_env.py: the p77 pair,
arm a 89,516 x 25 + 2 on GPU 0 / port 1377, arm b 98,828 x 23 + 2 on GPU 1 / port 1378; p77_common.sh = p76_common.sh apart from the
two profiles), so nothing is retyped from the shell chain:
  - every variable that differs between the arms is explained by the label, the GPU, the port or one profile knob (NL sessions,
    NP pool lanes, KVG KV GiB, CTX context, BASIS eviction basis, KEEP rebase keep, CAPL capture list, GRAPH its maximum, AGECAP the
    age cap), or the build stops; the AOT root is left to the launcher (it derives it);
  - rendering arm a and arm b back reproduces both captures exactly (all variables kept);
  - rendering the 80K profile reproduces p76s16a01's launch-time ARC3_* / VLLM_* environment (REQ80 = its requested-config.json,
    the 80K arm of p76) apart from the label, the keys launch_scaling sets later (server_environment) and the AOT root.
The definition drops what belongs to the host or the shell session (HOME, PATH, LD_LIBRARY_PATH, LANG, SSH_*, PWD, ...), the server's
PYTHONPATH (runtime_python.sh and the launcher build it) and the variables launch_scaling_s2i.py requires absent (ARC3_DENSE_EXPERIMENT,
ARC3_PRIVATE_MOE_SERVING, ARC3_PLE_HANDOFF_RECEIPTS, ARC3_PERSISTENT_ONE_LOOKAHEAD(_IMAGE_DIAGNOSTIC), ARC3_PROFILE_DIR/STEPS,
ARC3_ENGINE_TIMING_ENABLE); it sets them itself."""
import hashlib
import json
import os
import sys
import time

A, B, REQ, OUT = [json.load(open(p)) for p in sys.argv[1:4]] + [sys.argv[4]]
LA, LB = A['label'], B['label']
PROFILES = {   # p76_common.sh (80K = p76 arm a, 70K = p76 arm b) and p77_common.sh (89K = p77 arm a, 98K = p77 arm b) CONF_OF
    '80k': dict(K=4, NL=29, NP=27, KVG='18.31', CTX=80204, BASIS=64204, KEEP=40102, CAPL='8,16,24,32,36,40,80,120,130,135,140,145', AGECAP=120000, TAIL=16000),
    '70k': dict(K=4, NL=32, NP=30, KVG='18.33', CTX=70892, BASIS=54892, KEEP=35446, CAPL='8,16,24,32,36,40,80,120,150,155,160', AGECAP=120000, TAIL=16000),
    '89k': dict(K=4, NL=27, NP=25, KVG='18.70', CTX=89516, BASIS=73516, KEEP=44758, CAPL='8,16,24,32,36,40,80,120,125,130,135', AGECAP=120000, TAIL=16000),
    '98k': dict(K=4, NL=25, NP=23, KVG='18.82', CTX=98828, BASIS=82828, KEEP=49414, CAPL='8,16,24,32,36,40,80,110,115,120,125', AGECAP=128000, TAIL=16000),
}
for p in PROFILES.values():
    p['GRAPH'] = max(int(x) for x in p['CAPL'].split(','))
ARMS = {LA: dict(PROFILES['89k'], GPU='0', PORT='1377'), LB: dict(PROFILES['98k'], GPU='1', PORT='1378')}
HOST = {'HOME', 'PATH', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL', 'LOGNAME', 'OLDPWD', 'PWD', 'SHELL', 'SHLVL', 'SSH_CLIENT', 'SSH_CONNECTION',
        'SSH_TTY', 'TERM', 'USER', 'MAIL', 'HOSTNAME', '_'}
ABSENT = {'ARC3_PROFILE_DIR', 'ARC3_PROFILE_STEPS', 'ARC3_DENSE_EXPERIMENT', 'ARC3_PRIVATE_MOE_SERVING', 'ARC3_PLE_HANDOFF_RECEIPTS',
          'ARC3_ENGINE_TIMING_ENABLE', 'ARC3_PERSISTENT_ONE_LOOKAHEAD', 'ARC3_PERSISTENT_ONE_LOOKAHEAD_IMAGE_DIAGNOSTIC'}
DERIVED = {'ARC3_DENSE_AOT_ROOT'}          # launch_scaling_s2i.py derives it from the profile (the cache tag)
SHELL_ONLY = {'FREEZE_DEF', 'NL'}          # the p7x definition path (per arm letter); lc_fin.sh reuses NL for the NLL panel line
TAIL_KEYS = {'TAIL', 'TAIL_V', 'PROTECT_TOKENS'}   # equal in both arms (16,000): mapped by name
K_KEYS = {'K', 'MTPK', 'ARC3_S2_MTP_TOKENS', 'ARC3_MTP_TOKENS'}   # equal in both arms (4): mapped by name


def knob_of(part, k, va, vb):
    if va == vb:
        return None
    if va.replace(LA, '@LABEL@') == vb.replace(LB, '@LABEL@'):
        return 'LABEL'
    hits = [n for n in ('NL', 'NP', 'KVG', 'CTX', 'BASIS', 'KEEP', 'CAPL', 'GRAPH', 'AGECAP', 'GPU', 'PORT')
            if (va, vb) == (str(ARMS[LA][n]), str(ARMS[LB][n]))]
    assert len(hits) == 1, '%s %s: %r vs %r matches %s' % (part, k, va, vb, hits)
    return hits[0]


DEF = dict(built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), builder='mk_kdef.py (Claude main session 099768dc)',
           label_token='@LABEL@', profiles=PROFILES, sources={}, knobs={}, env={}, dropped={})
for name, cap in ((LA, A), (LB, B)):
    DEF['sources'][name] = dict(server_pid=cap['server_pid'], client_pid=cap['client_pid'], md5=hashlib.md5(json.dumps(cap, sort_keys=True).encode()).hexdigest())
for part in ('server_serve', 'client_run_v6'):
    ea, eb = A[part], B[part]
    assert set(ea) == set(eb), (part, sorted(set(ea) ^ set(eb)))
    knobs, env, drop = {}, {}, []
    for k in sorted(ea):
        if k in HOST or k in ABSENT or k in DERIVED or k in SHELL_ONLY or (part == 'server_serve' and k == 'PYTHONPATH'):
            drop.append(k)
            continue
        kn = knob_of(part, k, ea[k], eb[k])
        if kn is None and k in TAIL_KEYS:
            kn = 'TAIL'
        if kn is None and k in K_KEYS:
            kn = 'K'
        if kn is not None:
            knobs[k] = kn
        env[k] = ea[k].replace(LA, '@LABEL@')
    DEF['knobs'][part], DEF['env'][part], DEF['dropped'][part] = knobs, env, drop
DEF['client_cwd'] = A['client_cwd']
DEF['server_argv_a'] = A['server_argv']
DEF['client_argv_a'] = A['client_argv']


def render(part, prof, label, gpu, port):
    vals = dict(prof, GPU=str(gpu), PORT=str(port))
    out = {}
    for k, v in DEF['env'][part].items():
        kn = DEF['knobs'][part].get(k)
        out[k] = str(vals[kn]) if kn not in (None, 'LABEL') else v.replace('@LABEL@', label)
    return out


# check 1: both arms render back to their captures (every kept variable)
for name, cap in ((LA, A), (LB, B)):
    for part in ('server_serve', 'client_run_v6'):
        r = render(part, ARMS[name], name, ARMS[name]['GPU'], ARMS[name]['PORT'])
        bad = [k for k in r if r[k] != cap[part][k]]
        assert not bad, ('render', name, part, bad[:5])
# check 2: the 80K profile against p76s16a01's launch-time environment (ARC3_* / VLLM_* at launch_scaling's m.save)
req, l80 = REQ['environment'], 'p76s16a01'
r80 = render('server_serve', PROFILES['80k'], l80, '0', '1377')
later = {'VLLM_PLE_CPU_OFFLOAD', 'VLLM_PLE_OFFLOAD_READY_TIMEOUT', 'VLLM_SERVER_DEV_MODE'}   # minimal_inference.server_environment, after m.save
mine = {k: v for k, v in r80.items() if k.startswith(('ARC3_', 'VLLM_')) or k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
diff = sorted(k for k in set(mine) | set(req) if k not in later | DERIVED | ABSENT | {'PYTHONPATH'} and mine.get(k) != req.get(k))
assert not diff, ('80K vs p76s16a01', [(k, mine.get(k), req.get(k)) for k in diff[:10]])
assert set(req) - set(mine) <= DERIVED | ABSENT | {'PYTHONPATH'}, sorted(set(req) - set(mine))
DEF['checks'] = dict(arms_render_back=True, p76a_80k_launch_env=dict(keys=len(req), equal=len(req) - len(set(req) & (DERIVED | ABSENT | {'PYTHONPATH'})),
                                                                     not_compared=sorted(set(req) & (DERIVED | ABSENT | {'PYTHONPATH'}))))
with open(OUT + '.tmp', 'w') as fh:
    json.dump(DEF, fh, indent=1, sort_keys=True)
os.replace(OUT + '.tmp', OUT)
print('wrote %s: server %d kept (%d knobs, %d dropped), client %d kept (%d knobs, %d dropped); checks %s' % (
    OUT, len(DEF['env']['server_serve']), len(DEF['knobs']['server_serve']), len(DEF['dropped']['server_serve']),
    len(DEF['env']['client_run_v6']), len(DEF['knobs']['client_run_v6']), len(DEF['dropped']['client_run_v6']), DEF['checks']))
