# Private gameplay profiling v1

`run_profiled_gameplay.py` adds an explicit `public25` (25 official unique games,
900 gameplay seconds) or `smoke` (four explicit official IDs, 150 gameplay
seconds) contract to the reviewed PRO-LONG v5.22 runner. Both use all games as
active players, request slots 0 (resolved to the game count), 40,000 context,
NVFP4, MTP0, expected engine ceiling 30, utilization 0.95, temperature 1,
min-p 0, xhigh, scale 4, stdout threshold 1024, enabled programs/snapshots,
no response/token/turn cap, and 10% oldest complete interaction eviction.

No network/generation without `--execute`. The root controller supplies host,
boot, lock, deadline and independent server identity/health guards; the original
runner retains its own owned-process/config/idle/template/tokenizer preflight.
The overlay's exact 111-file release manifest and four adapter module hashes
are verified. No original/frozen source or shared importer is patched.

The controller's CLI is the executable example. Extra required arguments:

```
--harness-root /workspace/arc3-inference/runs/prolong-v5.22-release-20260912-v1/overlay
--harness-manifest /workspace/arc3-inference/runs/prolong-v5.22-release-20260912-v1/source-sha256.json
--profile-contract public25
--persistent-base-url http://127.0.0.1:1234
```

The original runner creates the fresh output directory and starts/resets its
memory monitor before arming gameplay. Profiling activates only at clock arm,
after source archives are written; its journal opens lazily. New wrapper
sources are preserved under `implementation/gameplay-profile/` and included in
`code-sha256.json`. The two unchanged design notes needed by the inherited
archiver came from `persistent_history_eviction_20260913`.

## Journal

`gameplay-harness-profile.jsonl` is private, event-only JSONL. Common fields:
`version`, `event`, `pid`, `thread_id`, and contextual `game_id`, `session_id`,
`operation_id`, `action_num`, `tool_id`, `model_tool_call_id`, `tool_name` when
known. Session/operation IDs appear once allocated; earlier user construction
can only be correlated by game and action. No telemetry enters prompts, model
tool outputs, workspace logs or image pixels.

- `span`: `phase`, `span_id`, `parent_span_id`, `wall_start/end`,
  `mono_start/end`, `seconds`, `error_type`. These are **inclusive nested
  scopes**, never additive global GPU-stall measurements.
- `operation_start`: exact persistent session/op IDs with wall/monotonic start,
  before preparation and HTTP dispatch.
- `completion_result`: `accepted=true` only after existing adapter token/state
  receipt validation, exact `completion_tokens`, `prompt_tokens`,
  `resident_tokens`, `logical_tokens` (nullable), `appended_prompt_tokens`,
  `finish_reason`, and end clocks. On exception, `accepted=false`, nullable
  known received count, `error_type`, `deadline_observed`. A known failed count
  requires received exact token IDs matching usage. Null is unknown, not zero
  or a reserved budget. Engine-generated deadline tails are analyzed separately.
- `run_clock`: actual `gameplay_started_epoch`, `deadline_epoch`, game IDs,
  total seconds, and wall/monotonic start. Preparation is outside this clock.
- `run_finished`: runner completion/error and end clocks, retaining the actual
  start/deadline. This event is after inherited monitor/scoring/cleanup, so use
  the original deadline for gameplay exposure, not this end-to-end duration.

Phases cover external/persistent staging, program/snapshot/user assembly,
display request snapshots, raw/transcript journals, HTTP serialization,
response parse/normalization, Python lifecycle/subprocess/action callbacks,
output filtering, PNG construction and runtime assets. `http_transport`
includes the **entire request generation and queue wait**. `python_subprocess`
includes its parent action callbacks. Neither is an isolated overhead number.
Display payloads remain diagnostic views; append requests still contain only
new external messages and the exact persistent session ledger.

`harness-profile-status.json` has attempted/written/dropped counts and `valid`.
`profiled-gameplay-result.json` has `status=passed|failed`, persistent cleanup
validation, recorder status, actual start/deadline and finish. Journal write
failure cannot alter the model/tool call, but fails profile acceptance afterward.

## Local evidence and limits

Focused CPU tests compare original and profiled actual two-call Python loops,
including exact persistent request bodies, tool result text and PNG bytes;
they reject historical tokenizer/stateless transport use. They also check
thread correlation, lazy output creation, source pins, class/super/deepcopy
semantics, source archival, contract rejection and clock order. The sealed
overlay excludes model assets, so the CPU fixture's tokenizer lookup alone is
rebound to the existing local metadata asset. Production modules are loaded
from the verified overlay and their import paths are checked after tests.

A real source-pinned dry plan resolves 25 official unique games, 25 request
slots, no cap and the intended sampling without creating output or contacting
the server. Live behavior and instrumentation cost still require the smoke
and gameplay runs; local tests do not establish GPU performance.
