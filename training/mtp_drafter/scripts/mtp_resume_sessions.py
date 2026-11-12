"""mtp_resume_sessions.py (2026-10-13, Claude fork 5d20b23b): which sessions must the MTP full run replay again when it
resumes (a crash, a deliberate stop, or a move to another pod)? The trainer's state sidecar (<state>.json, written by
train_mtp_sp.py's StateSaver) holds the stream state at the save: the consumed segments (session id + segment number)
and the trained windows of partly used segments. A session needs replaying unless every one of its segments is consumed,
or (same pod, --stream-dir) is consumed or still present unconsumed in the stream directory. Segments trained after
the save were deleted, so their sessions are replayed again; the trainer then deletes re-written consumed segments
unseen and skips trained windows. Sessions that failed in an earlier replay attempt are skipped (user, 10-13: failed
sessions are skipped with a short note), and listed.

Segment count per session: ceil(rows / SEGMENT), rows = the larger of the corpus stream length (sessions.jsonl "n") and
any replayed "stream_tokens" (an over-count only causes an extra replay; an under-count could lose a segment).

usage: python mtp_resume_sessions.py --sessions LIST.json --sidecar STATE.pt.json --driver D1.jsonl [D2.jsonl ...]
         [--stream-dir DIR] [--streams /workspace/arc3-inference/dflash_streams_20261011] --out RESUME.json
       Writes RESUME.json (the sessions to replay, in the list's format and order) and RESUME.summary.json."""
import argparse
import json
import math
import os
import re

SEG_RE = re.compile(r'(arc3-[0-9a-f]{32})-[^_/]*_s(\d+)\.safetensors$')
KEY_RE = re.compile(r'^(arc3-[0-9a-f]{32})_s(\d+)$')


def key_of(name):
    m = SEG_RE.search(name)
    if m:
        return m.group(1), int(m.group(2))
    m = KEY_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', required=True)
    ap.add_argument('--sidecar', default='', help="the resume state's sidecar (absent: nothing consumed yet)")
    ap.add_argument('--driver', nargs='*', default=[], help='driver.jsonl of every earlier replay attempt')
    ap.add_argument('--stream-dir', default='', help='same-pod resume: the stream directory as it is now')
    ap.add_argument('--streams', default='/workspace/mtp_refit_r1/streams_all')   # 10-22 MTP refit (was fr1's dflash_streams_20261011)
    ap.add_argument('--segment', type=int, default=65536)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    sessions = json.load(open(a.sessions))
    rows = {}
    for line in open(os.path.join(a.streams, 'sessions.jsonl')):
        d = json.loads(line)
        rows[d['session_id']] = int(d.get('n') or 0)
    status, stream_tokens = {}, {}
    for p in a.driver:
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:
                continue                                  # a partly written last line
            if 'status' in d and 'session_id' in d:
                sid = d['session_id']
                if not status.get(sid, '').startswith('ok'):  # an ok in any attempt wins
                    status[sid] = d['status']
                if d.get('stream_tokens'):
                    stream_tokens[sid] = max(stream_tokens.get(sid, 0), int(d['stream_tokens']))
    consumed, partial = {}, {}
    if a.sidecar:
        st = json.load(open(a.sidecar)).get('stream') or {}
        for k in st.get('consumed', []):
            kk = key_of(k)
            if kk:
                consumed.setdefault(kk[0], set()).add(kk[1])
        for k in st.get('trained_w', {}):
            kk = key_of(k)
            if kk:
                partial.setdefault(kk[0], set()).add(kk[1])
    present = {}
    if a.stream_dir and os.path.isdir(a.stream_dir):
        for n in os.listdir(a.stream_dir):
            kk = key_of(n)
            if kk:
                present.setdefault(kk[0], set()).add(kk[1])

    out, failed, done, buffered, unknown_len = [], [], 0, 0, 0
    for s in sessions:
        sid = s['session_id']
        st = status.get(sid, '')
        if st and not st.startswith('ok'):
            failed.append(dict(session_id=sid, run=s.get('run'), status=st, stream_tokens=stream_tokens.get(sid, 0)))
            continue
        n = max(rows.get(sid, 0), stream_tokens.get(sid, 0))
        if n <= 0:
            unknown_len += 1
            out.append(s)
            continue
        need = set(range(math.ceil(n / a.segment)))
        have_c = consumed.get(sid, set())
        if need <= have_c:
            done += 1
            continue
        if st.startswith('ok') and need <= (have_c | present.get(sid, set())):
            buffered += 1                                 # replayed completely; the rest waits in the stream directory
            continue
        out.append(s)
    json.dump(out, open(a.out, 'w'), indent=0)
    tok = sum(max(rows.get(s['session_id'], 0), stream_tokens.get(s['session_id'], 0)) for s in out)
    summ = dict(sessions=len(sessions), done=done, buffered=buffered, replay=len(out), replay_tokens=tok,
                failed_skipped=len(failed), failed_tokens=sum(f['stream_tokens'] for f in failed), failed=failed,
                unknown_length=unknown_len, consumed_segments=sum(len(v) for v in consumed.values()),
                partial_segments=sum(len(v) for v in partial.values()), sidecar=a.sidecar, drivers=a.driver,
                stream_dir=a.stream_dir)
    json.dump(summ, open(os.path.splitext(a.out)[0] + '.summary.json', 'w'), indent=1)
    print('resume list: %d of %d sessions to replay (%.1fM tokens); done %d, buffered %d, failed and skipped %d '
          '(%.2fM tokens), unknown length %d -> %s' % (len(out), len(sessions), tok / 1e6, done, buffered, len(failed),
                                                       summ['failed_tokens'] / 1e6, unknown_len, a.out))


if __name__ == '__main__':
    main()
