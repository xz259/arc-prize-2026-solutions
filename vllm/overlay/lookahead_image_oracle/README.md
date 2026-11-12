# Bounded image/text state oracle

This is a fresh diagnostic derivative of `lookahead_snapshot_v2`; the frozen scalar package is unchanged. It does not enable lookahead, change scheduling, render inputs, run inference, reclaim cache, or install itself. The isolated stage first installs its reviewed image-aware `one_lookahead`, then calls this package's `core.install_diagnostic_only()` explicitly. The package may be named `lookahead_image_oracle`; executable imports are relative.

## What is compared

The original 73 cache owners and all 121 components remain mandatory: complete selected NVFP4 rows including scales, compressed QSA rows, raw QSA key/position rings, GDN convolution and recurrent states, and PLE convolution. Tensor/view admission, physical page lookup, row extraction and raw byte copies are unchanged from scalar V2. Only the expected QSA position-ring values now use the actual three-axis MRoPE coordinates.

Computed frontiers remain 511–2048, with exactly one owned page in each of six allocator groups, no preemption, no history deletion, no poisoned owner, no prefix caching, MTP0 and single-worker execution. Every owner must be paused and native queues/stop transactions empty before a snapshot. This is not a 40K or eviction correctness gate.

Each image has independently checked `{identifier, offset, length, grid_thw}`. Offsets are in accepted token space. At most two unpruned T=1 images, spatial merge 2, and at most 512 visual tokens per image are admitted. The actual model's inherited Qwen3VL position method and an independent analytic implementation must agree; stored worker prefill positions, signed delta, and raw QSA ring positions must match them. The image bridge restricts actual native fixtures further to processed 256×256 images with grid `[1,16,16]` and 64 visual tokens.

A paused, fully consumed image must no longer hold a core encoder input reference, but its actual worker feature association and encoded BF16 `[visual_tokens,2560]` output must still be present. Encoder output bytes enter the canonical digest, along with image layout and MRoPE/PLE metadata. Missing or pending-free outputs fail closed. Global encoder cache retention is allowed; this package makes no claim about all possible encoder eviction/cache policies.

The cache extraction budget remains 75,154,560 bytes per readout with copies at most 1 MiB. Encoder outputs have a separate bound of 5,242,880 bytes per pass, and are read twice around the cache snapshot to detect instability (at most 10 MiB extra transfer). These synchronizing diagnostic reads are outside performance timing.

## API for the native control

After the explicit install, existing utility dispatch can call:

```python
receipt = await engine_core.call_utility_async(
    'arc3_one_lookahead_diagnostic', external_request_id)
```

The core envelope is unchanged (`computed_tokens`, `accepted_tokens`, `delta`, `settlement`, `worker`). The worker schema is now `arc3-one-lookahead-image-snapshot-v1`. The comparison canonical contains the old logical/cache state plus `image_layout` and `encoder_outputs`. Physical page/slot/pointer ownership remains outside equality.

Validate each receipt using independently reconstructed computed IDs, the query's actual prompt/prefill lengths, the externally pinned source-manifest digest, and the independently verified bridge input receipt:

```python
from lookahead_image_oracle.receipt_schema import validate_receipt
from lookahead_image_oracle.strict_receipt import compare_diagnostics

validate_receipt(receipt,
    expected_ids=expected_computed_ids,
    expected_source_manifest_sha256=ORACLE_SOURCE_SHA,
    expected_prompt_tokens=expected_prompt_length,
    expected_prefill_len=expected_prefill_length,
    expected_image_layout=verified_input_receipt['image_layout'])
# Validate both sides before comparing. This unchanged strict helper includes
# image/encoder bytes through the canonical comparison_digest.
compare_diagnostics(reference, candidate,
    expected_computed_ids=expected_computed_ids)
```

Omitting `expected_image_layout` preserves scalar/text expectation (`[]`); it cannot admit a receipt with images. Do not derive expected IDs, image offsets or query lengths from the state receipt under test. The processor/PNG/pixel-tensor evidence is supplied by the separate prepared-MM bridge, not manufactured by the oracle.

For a delta-zero stop, reference accepted length is one greater than candidate length. Reference appends prepared input `D`; candidate appends the exact sampled token `[u] + D` without decode/re-encode. The bridge's accepted-base plus shifted relative image offsets must therefore agree across the pair, as must actual input tensors and final computed token prefix.

After closing **all** owners, a second read-only utility proves request-specific encoder release while permitting global retained outputs:

```python
closed = await engine_core.call_utility_async(
    'arc3_one_lookahead_encoder_closed',
    actual_native_request_id, verified_input_receipt['image_layout'])
from lookahead_image_oracle.closed import validate_closed_receipt
validate_closed_receipt(closed, request_id=actual_native_request_id,
    image_layout=verified_input_receipt['image_layout'],
    source_sha=ORACLE_SOURCE_SHA)
```

Use the actual native request ID from the admitted earlier snapshot. This utility requires empty core/worker request maps and no native batch/future/pending stop; it never flushes or repairs state. Pending native cleanup can cause rejection until ordinary cleanup completes. It proves reference/association release, not that reusable global encoder bytes were erased.

## Local and native validation

Run construction tests with the project NumPy-enabled Python and explicit repository fixture root when copied elsewhere:

```sh
ARC3_SNAPSHOT_TEST_REPO=<project> \
  inference_stack/.venv-harness/bin/python -I -B -m unittest discover \
  -s inference_stack/experiments/host_recovery_20260916/lookahead_image_v1/oracle -p 'test_*.py' -v
```

CPU tests are construction/source tests, not CUDA validation. Native gates must verify the exact runtime/helper/processor source pins, exercise text regression, image prefill and nonempty append at delta one and real stop/drain delta zero, compare repeated full state/logprob readouts, and prove request/encoder cleanup. None of those image-native gates has been run by this local package. Failure is evidence to inspect, never permission to omit a component or repin blindly.
