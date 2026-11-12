# Stage-2 parts

The third layer of the engine. Each *part* is a set of in-memory patches to vLLM or to the [overlay](../overlay/), applied by an import hook right after the patched module is imported and only if the module's source hash matches the hash the part was written against. Nothing on disk is changed. On the pods this directory is the package `stage2_v1`.

`install.py` reads the list of parts from `ARC3_STAGE2` and installs them; `bootstrap/sitecustomize.py` runs it in every server process. The header of `install.py` describes most parts, and each part's module docstring gives the design, the measurements behind it and its switches.

## The parts that ran

Both ARC-AGI-3 packages install the 43 parts below; the list is the `STAGE2` entry of each profile in [`KDEF.json`](../../arc3/kaggle/kaggle_defs/KDEF.json). A part's code is in the module of the same name unless noted.

### Speculative decoding and sampling

| Part | What it does |
|---|---|
| `guards` | Lets the eviction code's request staging accept a server that runs with a drafter (in `install.py`). |
| `rollback` | One recurrent state per lane instead of one per draft position: a rejected draft is undone by recomputing the state through the accepted tokens at the next step (kernels in `rb_kernels.py`). |
| `wy` | The same replay and the verify step in the chunked (WY) matrix form (`wy_kernels.py`). |
| `persist` | Speculative decoding inside persistent sessions: stops, the count of kept tokens, and the commit of a paused lane's pending tokens when it resumes (`persist_kernels.py`). |
| `draftqsa` | Keeps the drafter consistent with eviction: its decode steps use the evicted layout, and its positions stay the original ones. |
| `drafthead` | The drafter scores a fixed subset of 32,000 tokens (`draft_vocab_32k.json`, which is not in the repository; see [`../launch/`](../launch/)) instead of the full vocabulary, and truncates its own distribution to the target's sampling settings (top-k 20, top-p 0.95). |
| `draftwin` | Limits the drafter's attention selection to its newest tokens (16,384 in the shipped profiles), so its cache can be a short ring (`draftring`). |
| `vsampler` | Draft verification through a sampler that works on at most 32 candidates per row (`cand_verify.py`). It carries the two fixes of the write-up's "exact speculative sampling" and block verification. |
| `topkfix` | Exact top-k and top-p masking in place of vLLM's Triton kernel, which truncates rows with many tied logits wrongly. |

### KV cache and recurrent state

| Part | What it does |
|---|---|
| `kvrot` | The 4-bit KV format: keys and values centered, Hadamard-rotated and stored as codebook indices with block scales chosen by squared error (`lutc`; reader and writer in `kvlut_kernels.py`). |
| `kvring` | The BF16 window over the newest tokens of every QSA layer (`kvring_kernels.py`, the faster two-kernel reader `kvring_fast.py`). With `ARC3_S2_KVRING_EXACT` it also keeps the exact set: BF16 copies of the first prompt tokens and of the most-selected older tokens (`kvring_exact.py`). |
| `draftring` | Moves the drafter's KV cache and the PLE state out of vLLM's shared KV pool into a small ring per session. |

The recurrent-state format is not a part: `ARC3_S2_GDN_STATE=scaled16sr` selects FP16 with an FP32 scale per 32 state rows and stochastic rounding (`s16.py`), which the rollback kernels then use.

### Eviction, rebase and positions

| Part | What it does |
|---|---|
| `agecap` | The backstop of eviction: whole interactions older than 120,000 tokens go, whatever their selection counts. |
| `agestat` | Records, at each eviction, how the selection counts are distributed over token age. |
| `hollowgroup` | Treats an assistant turn and the tool output that follows it as one block when eviction decides whether a mostly evicted turn should go whole. |
| `sumpin` | Pins the notes and the summary that open a conversation until they rank among the least-selected tokens for several evictions in a row. |
| `padbudget` | Counts the padding of the 4-token groups in a session's budget. |
| `rope2m` | Extends the position table beyond the trained 262,144 positions (to 1,048,576 in the shipped profiles) without changing the trained rows. |
| `rebaseobs` | Makes a rebase keep only the system prompt and the newest part of the conversation, back to a whole observation. |
| `rebase74` | The server side of the harness's rebase and parking: a rebase on request, suspending a session (its KV pages are freed, its token history stays) and resuming it, and the 60,000-token cap on what a rebase keeps. |
| `promptmove` | Re-rotates the system prompt's cached keys so that the prompt stays just behind the oldest cached token. |

### Scheduling and overcommit

| Part | What it does |
|---|---|
| `schedfast` | Constant-time KV accounting in the scheduler's allocation path. |
| `overlap` | Lets the engine admit resumed sessions and start evictions while the GPU runs the current step. |
| `prestage` | Stages a resumed session on the worker during the previous step. |
| `resumefast` | Removes redundant work from the worker's handling of a resumed session. |
| `poolpress` | Runs two more lanes than the KV pool is sized for: when the pool is about to fill, the fullest sessions evict earlier. |
| `stallguard` | Keeps overcommit free of preemptions: a lane that cannot get a page waits instead of preempting another lane, and ends its turn if the wait lasts a second. |
| `deadlinefix` | Answers an append that arrives after its session was aborted, instead of failing the engine. |

### Memory and host-side work

| Part | What it does |
|---|---|
| `ctxfree` | Keeps the API server, the engine process and the PLE helper from creating CUDA contexts (in `memopt.py`). |
| `topkshare` | One index buffer for the QSA selection instead of one per layer (in `memopt.py`). |
| `embedhost` | Keeps the input embedding table in pinned host memory (1.18 GiB less on the GPU). |
| `pleconv` | Bounds the memory of the PLE convolution's prefill. |
| `splitfix` | Runs vLLM's startup memory profile under the allocator's split limit (a port of vLLM PR 58430). |
| `ctxtail` | Makes the per-turn work on a session's history depend on its cached and new tokens, not on its whole history. |
| `mmstrip` | Stops resumed sessions from re-sending images the server has already processed. |
| `mmonly` | Puts vLLM's image cache in the mode that cannot drift between the API process and the engine. |
| `sessmem` | Frees what a closed session held in the API server. |
| `fqfast` | Computes the selection counts for eviction from arrays instead of Python objects. |
| `agefix` | Removes audit work that grew with a session's history. |

### Checkpoint, determinism and startup

| Part | What it does |
|---|---|
| `b1inc` | Loads Intel's AutoRound W4A16 experts in this stack. |
| `marlindet` | Runs the Marlin MoE kernel in a fixed token order, so that outputs repeat bit for bit. |
| `loadskip` | Makes each startup reader open only the checkpoint files it needs. |
| `ckptfetch` | Reads the weight shards into the page cache in parallel ahead of the loader. |

The profile `82kx1632a30p16` (package v16) adds `eprune`: the experts listed in `eprune_k16.json`, the 16 least-used of each layer, are never loaded.

## Other modules

These are not on the shipped list.

- **Diagnostics:** `steptimer.py`, `memdiag.py`, `memstats.py`, `hostprof.py` and `canary.py`, switched on by their own environment variables, and the parts `stepev`, `draftgraph`, `pwgraph` and `seloverlap`.
- **Study tools and levers that were measured and set aside:** `kvq4` (block scales shared by groups of four tokens), `kvfq`, `kvwin`, `kvevict`, `idxfq`, `headonly`, `precmode`, `attnprof`, `foldc` (replaced by `draftring`), `kvlut_vec.py`, `qsareq_kernels.py` and `lowm_kernels.py`.
- **Export tools for the drafter's training data:** `replayfx` replays recorded sessions exactly, and `dflashtaps` writes the target's hidden states (see [`../../training/`](../../training/)).
- `test_kvring_kernels.py` and `test_memstats_owners.py` are tests.
