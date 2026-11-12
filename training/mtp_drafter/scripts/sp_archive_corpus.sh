#!/bin/bash
# sp_archive_corpus.sh (2026-10-15, Claude fork 5d20b23b; side pod; user 10-14 ~00:0x: "archive this corpus since we will be
# using a new one", option (1) approved): moves the old corpus text, its stream indexes and the superseded corpus versions into
# /mnt/shared/arc3-inference/archive/corpus_v7_20261015/ (a rename on the shared volume; nothing deleted), writes MANIFEST.json
# there (original path, archive path, bytes, files, role) and repoints the side pod's /workspace/arc3-inference links to the
# archive paths, so session lists and scripts that name the old paths still resolve. Run only after fr1's replay has stopped
# (it reads the token files) and once the harness pod's dependencies are cleared. DRY=1: the plan only.
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SRC=/mnt/shared/arc3-inference
A=$SRC/archive/corpus_v7_20261015
W=/workspace/arc3-inference
DIRS="dflash_corpus_20261006 dflash_corpus_v2_20261007 dflash_corpus_v3_20261007 dflash_corpus_v3_build dflash_corpus_v4_20261008
dflash_corpus_v5_20261008 dflash_corpus_v6_20261009 dflash_corpus_v7_20261011 dflash_streams_20261010 dflash_streams_20261011
dflash_streams_v7new_20261011"
for d in $DIRS; do
  [ -d $SRC/$d ] && [ ! -L $SRC/$d ] || { echo "not a directory on the volume: $SRC/$d"; exit 1; }
  [ -e $A/$d ] && { echo "already in the archive: $A/$d"; exit 1; }
done
if pgrep -f "[p]hase6_replay_arm|[r]eplay_driver" > /dev/null; then echo "a replay is running: stop it first"; exit 1; fi
mkdir -p $A
python3 - $SRC $A $DIRS <<'PYEOF'
import json, os, sys, time
src, arc, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
roles = {'dflash_corpus_v7_20261011': 'corpus v7: the MTP full run fr1 (2,480 training sessions; 865M stream tokens) and DFlash Phase 6',
         'dflash_streams_20261011': 'stream index of corpus v7 (sessions.jsonl: stream length n per session; fr1 STEPS / trainer --streams)',
         'dflash_streams_v7new_20261011': 'stream index of the v7 additions (redbluepill 236 + additional 166 games, job 404)'}
m = []
for d in dirs:
    p, nb, nf = os.path.join(src, d), 0, 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                nb += os.lstat(os.path.join(root, f)).st_size; nf += 1
            except OSError:
                pass
    m.append(dict(name=d, original=p, archive=os.path.join(arc, d), bytes=nb, files=nf,
                  role=roles.get(d, 'superseded corpus / stream version (DFlash, 10-06 .. 10-10)')))
    print('%-32s %8.2f GB %7d files' % (d, nb / 1e9, nf))
json.dump(dict(archived_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), by='Claude fork 5d20b23b (user-approved 10-15)',
               reason='new corpus on the fixed-numerics stack (GDN state FP16 etc.); the old corpus is kept, not used',
               side_pod_links_repointed=True, deleted_with_it='sidepod_20261012/features/sp_mtptrain20m (493 GB) and sp_val50mtp '
               '(76 GB): user-approved deletion 10-15; sp_valrbp50mtp (70 GB) kept', dirs=m),
          open(os.path.join(arc, 'MANIFEST.json'), 'w'), indent=1)
print('total %.2f GB' % (sum(x['bytes'] for x in m) / 1e9))
PYEOF
[ -n "${DRY:-}" ] && { echo "DRY: would move the above into $A and repoint $W links"; rm -f $A/MANIFEST.json; rmdir $A 2>/dev/null; exit 0; }
for d in $DIRS; do
  mv $SRC/$d $A/$d || { echo "move failed: $d"; exit 1; }
  if [ -L $W/$d ]; then ln -sfn $A/$d $W/$d; fi
done
echo "archived to $A; links: $(for d in $DIRS; do readlink $W/$d; done | grep -c "$A")/$(echo $DIRS | wc -w) repointed"
