# Read-only scalar frontier oracle, local derivative v2

This package is implemented and CPU tested, **not deployed or natively admitted**. It changes no frozen candidate, model, scheduler, kernel, cache content, sampling state or production source. Importing the package installs nothing. No benchmark or process controller is included.

The first scope is the frozen scalar matched-frontier control: text only, no eviction or preemption, MRV2, TP1/DP1/PP1, MTP0, no DBO/prefix caching/transfer connectors, Mamba mode `none`, and 511–2048 computed input tokens. Sampling and state comparisons belong to the separate `lookahead_control_v1` driver. A pass would cover its tested frontiers only; it would not admit gameplay overlap, mixed cohorts, stochastic sampling, near-40K contexts, multimodal inputs or eviction.

`core.core_diagnostic(core, external)` is the replacement read-only utility for the existing diagnostic API. It requires every resident owner to be acknowledged and paused, no in-flight/placeholders/stale tokens, empty native batch and future queues, no pending stop transaction/append, and exclusive non-null target pages. Core ownership is checked again after the synchronous RPC. Timeout/error is propagated without retry or repair.

`worker.worker_snapshot(worker, wire)` performs explicit all-stream fences and bounded byte copies **outside timed execution**. It verifies actual GPU token counters and accepted IDs, reciprocal request slots, persistent worker block tables, owner-specific specs, both runtime layer bindings, and no pending staged writes. It never calls model forward, staged-write application, recurrent reset, zero-cache, eviction or allocator cleanup.

The complete inventory is 73 owners and 121 components:

| Category | Count | Valid content |
|---|---:|---|
| Main NVFP4 | 12 | Computed rows `[0,C)`, including all packed data and scale bytes; shape `[C,2,296]` |
| Compressed QSA | 12 | Complete groups `[0,C//4)`, BF16 shape `[C//4,128]` |
| Raw QSA keys | 12 | Four initialized rows `[C-4,C)` in chronological order, physical offset modulo 4 |
| Raw QSA positions | 12 | The same four rows' packed three-axis int64 positions, analytically checked |
| GDN convolution / SSM | 36 + 36 | Typed current-state views, excluding page padding |
| PLE short convolution | 1 | Typed current recurrent view `[9,10240]` |

Actual per-owner `KVCacheTensor` allocation size/offset/block stride is checked. Packed pools can share base storage and repeat cache base pointers across groups; the request's owned physical page regions must be disjoint. Pool block count is runtime-derived, not the historical 621. The narrow oracle rejects expanded worker block IDs (`blocks_per_kv_block != 1`) rather than guessing a different namespace. C<=2048 needs exactly one owned block in each pinned group; this is a deliberate scope guard.

Byte hashing reinterprets tensors as uint8 before CPU transfer, without numeric conversion. Each copy is at most 1 MiB, and selected payload is at most 75,154,560 bytes. No entire pool or model weight read is performed. Before/after metadata checks include layer object identity, view addresses/strides and token/block state. These checks do not replace the driver's independent same-owner repeated snapshot.

The canonical hash includes the complete ordered owner/component descriptors, component hashes, computed IDs/logical positions and analytic text MRoPE/PLE boundary metadata. Physical block IDs, slot IDs, pointer addresses, accepted-only uncomputed tails and request identities remain diagnostic fields outside state equality. MRoPE prefill positions are read as actual little-endian int32 bytes; PLE predecessors come from actual accepted token storage. The unused shared batch scratch and offloaded weights are excluded.

## V1 native rejection and exact V2 correction

V1 failed before its first state comparison on the unmarked 511-token reference: `Unknown inner cache stride`. The separate read-only runtime layout receipt (`host-recovery-lookahead-scalar-layout-20260916-v1.json`, SHA256 `4cd8eb50038dea35ff4cfb019441081854596ef732cc077327ecbc9ed0516167`) shows all 12 main NVFP4 owners as logical `[633,2,2768,296]` with strides `[1638656,296,592,1]`. This is token-major NHD physical storage. V1 incorrectly required logical inner contiguity, rejecting token stride 592 against expected 296. The failed state and result are preserved; no numerical comparison passed at that point.

The pinned `Qwen3_8FlashNextQSAFlashAttentionBackend` in `qsa.py` inherits FlashAttention's stride order and overrides its logical shape to use `2*side_bytes(256, 'nvfp4') = 296`. FlashAttention NHD selects `(0,2,1,3)`. The pinned allocator lays out physical `[B,N,H,D]` and inverse-permutes to logical `[B,H,N,D]`, yielding the observed strides. V2 adds exact source pins for `qsa.py` and `flash_attn.py` to the existing allocator/helper pins.

Only the exact main-NVFP4 category now requires uint8, tail `[2,2768,296]`, and inner strides `[296,592,1]`. Dense HND, other transposes, partial/overlapping strides and alternate geometry are rejected. Every other category retains V1's original strict dense checks, including singleton strides. A main page touches exactly `296 + 2767*592 + 295 + 1 = 1,638,656` bytes; the original allocation size, offset, page stride, storage extent and owned-region nonoverlap checks remain intact. Canonical token/head/296-byte extraction is unchanged, including all packed payload and scale bytes.

Five additional CPU tests cover this exact admission, packed page gaps, forbidden layouts/overlap/out-of-allocation views, dense categories, canonical NHD byte order at C511/C2048, scale-byte mutations, uncomputed rows/padding, page relocation and the 1 MiB copy split. AST parity checks bind the executable delta to the category guard and sole call-site keyword. NumPy/metadata construction does not replace a native rerun.

The same scalar driver schema, source-manifest parameter, 73-owner/121-component coverage, fences and cleanup requirements apply. Only a fresh diagnostic stage may import V2; changing already-imported V1 files or live method bindings is outside this package. No scheduler, model, kernel, sampling or serving behavior is modified.

## Source authority and local validation

`receipt_schema.py`, `strict_receipt.py` and `expected_inventory.json` are byte-identical copies of the frozen shared contract owned by `lookahead_control_v1`. `SOURCE_MANIFEST.json` pins nine own/runtime-contract files, 29 installed vLLM source files and five helper modules. Each snapshot verifies those bytes. The driver must receive the **canonical JSON digest** of this source-manifest object as its independent `--expected-source-manifest-sha256`; this is different from the formatted file's SHA256. `VALIDATION.json` records both.

Run the construction tests using the bundled Python with NumPy:

```sh
ARC3_SNAPSHOT_TEST_REPO=/absolute/path/to/ARC-AGI-3 \
  python3 -I -B -m unittest discover -s /absolute/path/to/lookahead_snapshot_v2 -p 'test_*.py' -v
```

The 15 inherited tests cover actual core guard bodies, uncertain-RPC no-retry behavior, all seven component extraction categories with a byte-preserving NumPy adapter, all four ring phases, page relocation, invalid layouts, selected-byte budgets, missing PLE, excluded padding/uncomputed data, packed allocation ownership and exact local source pins. The NumPy adapter is construction evidence, **not Torch/native proof**. One test reads the existing pinned source tree through the explicit repo path; it imports no serving modules and runs no GPU operation.

## Future isolated-stage seam, not a launch recipe

After separate root authorization and a reviewed stage composition, make this package importable in both core and worker processes. The frozen candidate installer runs first; then explicitly call `lookahead_snapshot_v2.core.install_diagnostic_only()`. It accepts only the exact original frozen `EngineCore.arc3_one_lookahead_diagnostic` method and changes that diagnostic method alone. It does not install or change lookahead scheduling. The unchanged endpoint then reaches the stricter core utility and pickles the package's top-level worker callable through the existing collective RPC.

Before a native claim, validate the source manifest against the actual stage, confirm the source-pinned physical layout/stride/worker-block guards admit the current runtime, and run the independent driver with saved raw HTTP receipts. Required evidence includes unchanged-paused-owner repeat equality, every category covered, actual dropped-u equality, independently expected computed prefixes before/after append, exact resumed tokens and all eight nonempty finite logprob rows, and owned-session cleanup. Any failed or uncertain RPC requires owner inspection; this module performs no recovery.

Multimodal state needs original three-axis positions and encoder/image ownership. Eviction needs retained original positions, QSA descriptor identity, protected PLE predecessors and an oracle preserving retained-state approximation; `range(C)` and `C//4` are invalid substitutes after deletion. These extensions remain unimplemented and mandatory before gameplay promotion.
