# Harness v6.0 build plan (2026-10-03, main session `fa00db0a`)

- **Spec:** `milestone2_solutions/HARNESS_V6_SPEC.md` (draft 1 plus Harness Design's 10-03 updates). This file covers how it is built.
- **Owner:**
  - this session implements the port, deploys it and runs the screens (user decision O3, relayed 04:00, consistent with the user's 04:12 go-ahead);
  - Harness Design reviews.

## Base and layout

- **`src/`:** dfranzen's `ARC3-Inference/inference` and `tufa-arc-agi-framework/src/taaf` at `10882e3`, unmodified, with his license (`LICENSE.dfranzen`, Apache-2.0).
  - The pod's `venv-duck` already has `arcengine 0.9.3` (his pin).
  - v6 runs put this `src/` first on `PYTHONPATH`; the control's older `duck/src` copy is not touched.
- **`v6/`:** our code.
  - Subclasses and overrides; no edits to `src/`.
  - The `ToolAgent` at `10882e3` keeps the hooks our control's adapters patch: `_chat_completion`, `_trim_messages_for_context`, `analyze`, `_build_user_prompt`, `_run_python_tool`, `_render_tool_payload`.
- **`tests/`:** CPU unit tests (spec §11 step 12) and an offline replay of `g8b1` frames.

## Components (spec section → module)

1. **Transport** (§3, §5.7, §5.8) → `v6/transport.py`. A persistent-session client, adapted from the control adapter (`persistent_harness_adapter.py`, `f48e6ee3`):
   - start / append / continuation; observation registration for every opener;
   - per-completion caps: soft 8,000, hard 12,000, check 8,000, with the adapter's `turn_cap_decision`, `repetition_onset` and cue texts;
   - **`length` = pause → continue in place**; nothing is appended while an assistant is pending;
   - **HC1b:** any 413 `persistent_session_error` with `eviction_committed: false` → one retry with N6 sized from I_max = B − 1 − R + E; no receipt → the minimal notice;
   - receipt bookkeeping: merged `history_ranges` (reset at a rebase), `history_state.blocks`;
   - raw-harness JSONL (`model-turn` with usage, `persistent-request` / `persistent-response`) in the control's format, so `g7_common.py` and `paired_scores.py` read it unchanged.
2. **Turn loop** (§2) → `v6/agent.py`. `V6Agent(ToolAgent)` replaces `analyze` for append-only sessions:
   - the opener (plus any pending tool result) is the append, then generate;
   - an acting call ends the turn;
   - **yield** at 2,048 generated tokens without an action, checked only at a clean boundary: a resumption opener appended with the tool result;
   - recovery N1, N3, N4, N5 appended, never stubbed; `TURN_BUDGET_RECOVERY` after a hard close;
   - no trimming, no rollback, no summaries, no memory sections or world model.
3. **Texts** (§6.1–6.4, the 32-token "Older turns…" line, the 10-s tool rule) → `v6/prompts.py`. The test asserts that the static system turn renders to ≥ 3,104 tokens and that its first 3,104 tokens are identical across two games.
4. **Images** (§5.1) → `v6/images.py`:
   - the board at 8× first, then the change image (navy `(16, 24, 48)` for unchanged cells; omitted on a level's first turn, after a restart, and when nothing changed);
   - death: the reset board, the death frame and the fatal-step change.
5. **Sandbox** (§5.2, §5.3) → dfranzen's `python_tool_sandbox` with:
   - the **10-s** limit;
   - nice 10, `RLIMIT_AS`;
   - library and seed restore;
   - `history[i].animation_frames`;
   - the **transient-pixel NO-OP exemption**;
   - per-call CPU, overhead against history length, and timeout logging.
6. **History store** (§5.4) → `v6/history_store.py`: compact (64×64 uint8, deduplicated by hash), append-only, lazily loaded views. Replaces his full JSON rewrite and reload per action.
7. **Library** (§5.5) → `v6/library.py`:
   - his `collect_functions` intake, then our validator (compile plus a scratch run with actions blocked), then a per-game store;
   - `programs()`, `read_code()`; seeds `grid.py` and `objects.py`;
   - **reminder:** the 10 most recent programs, re-appended after the library changes, and when the merged eviction ranges intersect the logged span of the carrying reminder (exact rule) or a rebase drops its observation;
   - A4's rules reminder later uses the same scheduler.
8. **Tool output** (§5.6) → `v6/tool_output.py`:
   - his JSON payload;
   - our full-board recogniser → 4× PNG, deduplicated, ≤ 4 per result, caption "Board image: your printed board (RxC)";
   - a 2,048-token text cap, middle-truncated with the §6.4 marker.
9. **Solver and runner:** his `HarnessSolver` (auto-RESET, echo, trace, guards, animation chain) and the taaf `Benchmark` (as the control runner uses), with:
   - our external server (his local-server management off);
   - the contract (`arm25_*`: 25 public games, time budget, expected sequences = the server's);
   - the server preflight tolerant of the other arm;
   - `score.json` via taaf `evaluate_runs`;
   - `prolong/game-NNN/log.jsonl` (`action` records with `level_completed` / `run_complete`) for the token-matched scripts.
10. **Two-arm runs:** the control arm via `run_gameplay_arm_s2.sh` (`client_mtp2`); the v6 arm via `run_v6_arm.sh`; one server, 50 lanes (job 160 shakes the machinery down).

## Tests (before any GPU run)

- **Unit:**
  - echo and verdicts; the stale and terminal latches; the batch no-op stop; the transient exemption;
  - library intake, validation and reminder scheduling (synthetic receipts);
  - board-conversion cap and deduplication (r11l's 68- and 112-grid payloads); the truncation marker;
  - the yield trigger; `length` → continuation; HC1b sizing;
  - prefix identity (≥ 3,104 tokens);
  - the history store against his JSON views on replayed `g8b1` frames.
- **Pod smoke:** 4 games for 240 s against the job-160 configuration. Checks:
  - rc 0; no protocol violations;
  - an opener's measured append size against §4's bounds;
  - the prefix shared (`prefix_share` swaps), resident image tokens logged;
  - no capacity refusals.
- **Then:** send the rendered system prompt, the opener layout and the parameters to Harness Design (spec §11 step 13), and run v6.0 against control as a two-arm run.
