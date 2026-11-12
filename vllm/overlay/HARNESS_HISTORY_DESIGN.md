# Harness wiring for oldest-interaction KV eviction

Implemented by `/root/persistent_harness` (Astra), 2026-09-13, only in this phase-three experiment. Production harness and phase-two files remain unchanged. Server selection/engine mutation are owned separately.

The bounded gameplay entrypoint accepts `--kv-eviction-policy off|oldest_interactions` (default `oldest_interactions`) and `--kv-evict-fraction` (default `0.10`). This is separate from the inherited text-headroom setting, which remains unused by persistent serving. The entrypoint still requires explicit execution, one to four named games, MTP0, no per-game token limit, no artificial response cap, and a gameplay deadline of at most 600 seconds.

The runner passes validated configuration through its private solver/agent factory attributes; it changes no process-global policy or environment. Configuration is normalized and pinned as immutable JSON per session. Direct adapter calls default to policy `off`, preserving ordinary control sessions unless explicitly enabled.

Only the initial `/arc3/persistent/start` body contains the out-of-band control:

```json
{"history_eviction":{"policy":"oldest_interactions","fraction":0.10}}
```

It is not included in `chat`, template arguments, tool results, or later append bodies. The existing managed-suffix sidecar remains independent. The policy contributes to the local immutable staging fingerprint. Caller mutations after configuration cannot change a prepared request. Every response must acknowledge the pinned policy, fraction, and resident token budget before local state commits.

An optional adapter-only `resident_token_budget` permits a smaller diagnostic trigger. Normal gameplay omits it and uses the 40,000-token context limit. The harness's actual `ContextPolicy.context_limit` remains 40,000; no model configuration is silently reduced for a synthetic control.

When policy is enabled, the following fixed reminder is appended once to the existing system prompt before its first prefill:

> Memory reminder: older completed observations and exchanges may no longer be present in your attention context. Recover earlier findings, programs, board states, and actions from the program library and game log when needed. Continue solving from the current board and the retained conversation.

Normal solver construction adds it before the first observation is built or logged. The lazy adapter path also upgrades an already-constructed but still unsubmitted initial system message. Once submitted, the exact system text is protected by both an explicit equality check and immutable external-history staging. Later calls cannot remove/repeat the reminder, rewrite the system, or refresh the current observation's program/snapshot suffix. With policy off, the system text is unchanged.

The server owns rounding the token target upward to oldest complete interaction blocks, protecting the system and active interaction, and releasing KV. The harness keeps its complete local chronological history for dispatch and audit; it sends only new external messages after the initial request. The server's exact assistant-token ledger remains authoritative. An unfinished reasoning continuation still sends no new message and uses the verified cumulative parse, even when the server first frees historical capacity.

There is no text-trimming, retokenizing-old-history, or refill fallback. If the protected active interaction alone exhausts available capacity, the server's explicit capacity refusal stops/closes the game session. History eviction does not promise recovery of malformed partial tools or unlimited active reasoning. Response receipts and exact raw wire logs remain available for audit.

HTTP 413 is a zero-charge capacity stop only when the typed `persistent_session_error` body acknowledges the pinned policy/fraction/budget, a recognized capacity fallback, `triggered=true`, and `eviction_committed=false`. The exact structured error is saved as `persistent-error`. Other 413 responses retain conservative uncertain-request accounting. Both paths close the owned session. A safely handled capacity stop remains a capacity-validation failure for the gameplay runner, rather than evidence that the chosen operating configuration succeeded.

Local validation covers the real PRO-LONG tool loop with an injectable HTTP fixture: start-only policy, on/off reminders, unchanged inspection suffixes, immutable caller config, wrong/missing receipt rejection, preserved 40K context under a diagnostic budget, exact continuation after an eviction receipt, lazy initialization, normal prompt configuration, late-system rewrite rejection, and bounded CLI parsing. These tests establish harness wiring only; the actual server transaction and live model remain separate evidence.
