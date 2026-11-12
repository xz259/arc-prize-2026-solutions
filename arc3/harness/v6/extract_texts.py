"""Extract harness v6's model-facing texts verbatim from the canonical spec (milestone2_solutions/HARNESS_V6_SPEC.md,
§6.1-6.4 and §7) into v6/texts.json, with the spec's sha256, so the harness never drifts from the agreed wording.
Keys:
  system              §6.1 system prompt (static; the first 3,104 rendered tokens are the shared page)
  tool_description    §6.2
  summary_trace       §6.3 action summary + trace templates (two lines)
  first_turn          §6.3 first turn of a game
  reminder            §6.3 standing reminder
  levelup             §6.3 level-up block
  restart             §6.3 restart block (with its three image captions)
  noop_stop           §6.3 no-op stop block
  animation           §6.3 the three animation lines
  program_list        §6.3 program list block
  resumption          §6.3 resumption opener
  nudges              §6.4 (raw block; parsed into named entries below)
  N1 N1_plan N3 N4 N5 N6 truncation_marker board_caption board_same board_more retention_saved retention_rejected
  retention_removed timeout
  A1 A3 A4_levelup A4_rules A5   §7 additions (not used by v6.0)
usage: python3 extract_texts.py <spec.md> <out texts.json>"""
import hashlib
import json
import re
import sys


def blocks_after(lines, marker):
    """The first ```text block after the line containing marker."""
    i = next(k for k, l in enumerate(lines) if marker in l)
    while not lines[i].startswith('```text'):
        i += 1
    j = i + 1
    while not lines[j].startswith('```'):
        j += 1
    return '\n'.join(lines[i + 1:j])


def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    raw = open(spec_path, 'rb').read()
    lines = raw.decode().split('\n')
    t = {}
    t['system'] = blocks_after(lines, '### 6.1 System prompt')
    t['tool_description'] = blocks_after(lines, '### 6.2 Tool description')
    t['summary_trace'] = blocks_after(lines, '**Action summary and trace:**')
    t['first_turn'] = blocks_after(lines, '**First turn of a game:**')
    t['reminder'] = blocks_after(lines, '**Standing reminder**')
    t['levelup'] = blocks_after(lines, '**Level-up block:**')
    t['restart'] = blocks_after(lines, '**Restart block**')
    t['noop_stop'] = blocks_after(lines, '**No-op stop block:**')
    t['animation'] = blocks_after(lines, '**Animation line**')
    t['program_list'] = blocks_after(lines, '**Program list block**')
    t['resumption'] = blocks_after(lines, '**Resumption opener**')
    nudges = blocks_after(lines, '### 6.4 Nudges')
    t['nudges'] = nudges
    paras = {}
    cur = None
    for line in nudges.split('\n'):
        m = re.match(r'^(N\d|Soft-cap cue|Budget-close|Truncation marker|Board conversion caption|Retention notes|Python timeout)', line)
        if m:
            cur = m.group(1)
            paras[cur] = []
            continue
        if cur is not None and line.strip():
            paras[cur].append(line)
    n1 = paras['N1']
    t['N1'] = n1[0]
    t['N1_plan'] = re.sub(r'^\[When the reply.s reasoning ended in a plan, add:\]\s*', '', n1[1])
    t['N3'] = paras['N3'][0]
    t['N4'] = paras['N4'][0]
    t['N5'] = paras['N5'][0]
    t['N6'] = paras['N6'][0]
    t['truncation_marker'] = paras['Truncation marker'][0]
    caps = re.findall(r'\[[^\]]*\]', paras['Board conversion caption'][0])
    t['board_caption'], t['board_same'], t['board_more'] = caps
    ret = [s.strip() for s in paras['Retention notes'][0].split('|')]
    t['retention_saved'], t['retention_rejected'], t['retention_removed'] = ret
    t['timeout'] = paras['Python timeout'][0]
    t['A1'] = blocks_after(lines, '**A1. Priors and probing.**')
    t['A3'] = blocks_after(lines, '**A3. Life-length line**')
    out = dict(spec_sha256=hashlib.sha256(raw).hexdigest(), spec_path=spec_path, texts=t)
    json.dump(out, open(out_path, 'w'), indent=1, ensure_ascii=False)
    print('extracted %d texts from %s (sha256 %s)' % (len(t), spec_path, out['spec_sha256'][:16]))


if __name__ == '__main__':
    main()
