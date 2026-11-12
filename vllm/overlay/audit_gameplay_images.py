"""Compare actual wire board/recap PNG pixels with their logged source frames."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image


def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def audit(gameplay, repo, scale):
    duck = repo / 'arc3_codex_handoff/references/duck_harness/original_tufa_duck_harness/src/ARC3-Inference/inference'
    colors = literal(duck/'agent/vision_context.py', 'ARC_COLOR_MAP')
    alphabet = literal(duck/'utils/grid_utils.py', 'ARC_COLOR_CHARS')
    checked = []
    user_image_count = 0
    for wire in sorted((gameplay/'raw-harness').glob('*.jsonl')):
        root = gameplay/'prolong'/wire.stem
        journal = [json.loads(s) for s in (root/'log.jsonl').read_text().splitlines() if s]
        events = {row['seq']: row for row in journal}
        initial = next(row for row in journal if row['kind'] == 'observation')

        def compare(part, ref, *, kind, event_id, request_id):
            frame = json.loads((root/ref['ref']).read_text()) if 'ref' in ref else ref
            grid = frame['ascii'].splitlines()
            assert grid and all(len(row) == len(grid[0]) for row in grid)
            expected = Image.new('RGB', (len(grid[0]), len(grid)))
            expected.putdata([colors[alphabet.index(c)] for row in grid for c in row])
            expected = expected.resize((expected.width*scale, expected.height*scale), Image.Resampling.NEAREST)
            url = part['image_url']['url']
            assert url.startswith('data:image/png;base64,')
            raw = base64.b64decode(url.partition(',')[2], validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                actual = source.convert('RGB')
            checked.append(dict(game=wire.stem, request_id=request_id, kind=kind, event=event_id,
                dimensions=list(actual.size), png_sha256=hashlib.sha256(raw).hexdigest(),
                exact_pixels=actual.size == expected.size and actual.tobytes() == expected.tobytes()))

        for line in wire.read_text().splitlines():
            row = json.loads(line)
            if row['kind'] != 'persistent-request':
                continue
            body = row['payload']['body']
            messages = body['chat']['messages'] if 'chat' in body else body['external_messages']
            for message in messages:
                if message['role'] != 'user':
                    continue
                parts = message['content']
                user_image_count += sum(part['type'] == 'image_url' for part in parts)
                text = next(part['text'] for part in parts if part['type'] == 'text')
                match = re.search(r'Latest executed action event: (\d+)', text)
                current = events[int(match.group(1))] if match else initial
                image = next(part for part in parts if part['type'] == 'image_url')
                compare(image, current['payload']['frame'], kind='current', event_id=current['seq'], request_id=body['request_id'])
                for index, part in enumerate(parts):
                    if part['type'] != 'text':
                        continue
                    recent = re.match(r'E(\d+) after .+; L(\d+)/attempt(\d+)', part['text'])
                    prior = re.match(r'workspace\.(get_event|action_record)\((\d+)\)(.*); previous L(\d+)/attempt(\d+);', part['text'])
                    if not recent and not prior:
                        continue
                    event_id = int(recent.group(1) if recent else prior.group(2))
                    key = 'before_frame' if prior and 'before_frame' in prior.group(3) else 'frame'
                    compare(parts[index+1], events[event_id]['payload'][key], kind='snapshot', event_id=event_id, request_id=body['request_id'])
    assert checked, 'No actual wire board images checked'
    assert len(checked) == user_image_count, 'Some user-message images lack a recognized source-frame pointer'
    return dict(valid=all(row['exact_pixels'] for row in checked), images=checked,
        scope='Actual persistent-request current-board and caption-indexed recap PNG pixels; arbitrary Python outputs and model understanding excluded',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gameplay', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.gameplay, args.repo, args.scale)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(valid=result['valid'], checked=len(result['images']),
        current=sum(row['kind']=='current' for row in result['images']),
        snapshots=sum(row['kind']=='snapshot' for row in result['images'])), indent=2))
    raise SystemExit(0 if result['valid'] else 1)
