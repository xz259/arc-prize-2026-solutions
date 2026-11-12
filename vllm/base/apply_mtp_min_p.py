"""Hash-guarded, reversible MTP min-p support for this pinned runtime.

Feature is opt-in via ARC3_MTP_MIN_P=1. QSA/MoE/weights are untouched. Existing
ordinary min-p is reused for bonus tokens; verification expands thresholds per
request before top-k/top-p, using the same probability-domain comparison.
"""
import argparse
import json
import os
from pathlib import Path
from apply_determinism_diagnostics import sha, once

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'vllm/sampling_params.py': '73e5bf250cb13803e4ce390407ae059b497898c8b81f9985d2b5c055d4db8f7d',
    'vllm/v1/sample/logits_processor/__init__.py': '1f9b02dae8cf6d44484638037123d3a2dec6b7b9a2bcc9afceef1a9bf3eaedec',
    'vllm/v1/sample/rejection_sampler.py': '4bb87d7984d967be9c7e5b6a6b489c02571ef2f2913211cb3f1eea74b2e1f32f',
}


def patched_source(path, original):
    s = original.decode()
    if path.endswith('/sampling_params.py'):
        s = once(s, '        if self.min_p > _SAMPLING_EPS or self.logit_bias:\n',
                 '        import os\n'
                 '        min_p_supported = os.environ.get("ARC3_MTP_MIN_P") == "1"\n'
                 '        if (self.min_p > _SAMPLING_EPS and not min_p_supported) or self.logit_bias:\n')
    elif path.endswith('/__init__.py'):
        s = once(s, '        logger.warning(\n'
                    '            "min_p and logit_bias parameters won\'t work with speculative decoding."\n'
                    '        )\n'
                    '        return LogitsProcessors(\n'
                    '            [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]\n'
                    '        )\n',
                 '        import os\n'
                 '        processors = [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]\n'
                 '        if os.environ.get("ARC3_MTP_MIN_P") == "1":\n'
                 '            processors.append(MinPLogitsProcessor(vllm_config, device, is_pin_memory))\n'
                 '            logger.info("ARC3 speculative min-p enabled; logit_bias remains unsupported.")\n'
                 '        else:\n'
                 '            logger.warning("min_p and logit_bias parameters won\'t work with speculative decoding.")\n'
                 '        return LogitsProcessors(processors)\n')
    else:
        s = once(s, 'from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor\n',
                 'from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor, MinPLogitsProcessor\n')
        s = once(s, '    # Get expanded top_k and top_p tensors.\n',
                 '    # Match ordinary sampling: temperature -> min-p -> top-k -> top-p.\n'
                 '    # Bonus tokens already use Sampler and the same MinP processor.\n'
                 '    # These filtered target logits feed both acceptance and recovery.\n'
                 '    for processor in sampling_metadata.logitsprocs.argmax_invariant:\n'
                 '        if isinstance(processor, MinPLogitsProcessor) and processor.min_p_count:\n'
                 '            min_p = expand_batch_to_tokens(\n'
                 '                processor.min_p.view(-1), cu_num_draft_tokens, num_tokens\n'
                 '            ).unsqueeze(1)\n'
                 '            probability_values = torch.nn.functional.softmax(logits, dim=-1)\n'
                 '            max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)\n'
                 '            adjusted_min_p = max_probabilities.mul_(min_p)\n'
                 '            logits.masked_fill_(probability_values < adjusted_min_p, -float("inf"))\n'
                 '\n'
                 '    # Get expanded top_k and top_p tensors.\n')
    compile(s, path, 'exec')
    return s.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', type=Path, default=ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--revert', action='store_true')
    args = p.parse_args()
    assert not (args.apply and args.revert)
    if args.apply or args.revert:
        from minimal_inference import IDENTITY, owned_snapshot, matches, port_open
        assert not port_open(), 'Stop the owned server before changing source'
        if IDENTITY.exists():
            assert not any(matches(x) for x in owned_snapshot(json.loads(IDENTITY.read_text())))
    changes = []
    for rel, expected in SOURCES.items():
        target = args.site/rel
        backup = ROOT/'patches/mtp-min-p-original'/rel
        current = target.read_bytes()
        original = current if sha(current) == expected else backup.read_bytes()
        assert sha(original) == expected, rel
        patched = patched_source(rel, original)
        assert sha(current) in {expected, sha(patched)}, rel
        changes.append((target, backup, original, patched, dict(path=rel,original_sha256=expected,patched_sha256=sha(patched))))
    if args.apply or args.revert:
        for target, backup, original, patched, row in changes:
            backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists(): assert sha(backup.read_bytes()) == row['original_sha256']
            else: backup.write_bytes(original)
            temporary = target.with_name(target.name+'.arc3-minp-tmp')
            temporary.write_bytes(original if args.revert else patched)
            os.replace(temporary,target)
        manifest = ROOT/'patches/mtp-min-p-manifest.json'
        if args.revert: manifest.unlink(missing_ok=True)
        else: manifest.write_text(json.dumps(dict(patch='mtp-min-p-v1',files=[c[4] for c in changes]),indent=2)+'\n')
    print(json.dumps(dict(applied=args.apply,reverted=args.revert,files=[c[4] for c in changes]),indent=2))


if __name__ == '__main__': main()
