# Managed-suffix ownership in persistent gameplay

This experiment removes older program/snapshot **KV spans when a new real observation arrives**, while preserving every submitted message in the local audit/dispatch history. The latest observation is rendered once and remains frozen across all assistant/Python inspection exchanges. Phase 1 and production defaults are unchanged.

## Boundary and wire contract

The existing suffix builders explicitly record their original content bases. These private wrappers establish three consecutive regions of the new user message: original observation and current PNG; optional program list; optional historical snapshot. No textual matching or model-visible marker is needed. The helper validates those bases against the actual content prefixes, then sends a separate control field alongside the normal chat/external-message payload:

```json
{
  "managed_suffixes": {
    "version": 1,
    "observations": [{
      "message_index": 1,
      "observation_id": "obs-00000001",
      "content_sha256": "SHA256 of the complete wire content",
      "base_parts_end": 2,
      "program_parts": [2, 3],
      "snapshot_parts": [3, 8]
    }],
    "evict_observation_ids": []
  }
}
```

`message_index` is relative to this operation's `chat.messages` on start or `external_messages` on append. All ranges are half-open **content-part indices**, never client-estimated token spans. Absent suffixes use `null`. Original string content is treated as one text part only when suffix decoration has converted it to a part list. The digest is SHA256 over `json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')`.

Observation IDs are monotonic and scoped to one session; repeated identical board states remain distinct observations. A descriptor with no suffix still marks a real observation, so it can retire an earlier suffix when appropriate. System text, action summaries, the current board and user-message structural framing are never managed suffixes. Sidecar fields are not inserted into messages, tools, chat templates or tokenizer input.

The protocol layer independently validates the descriptor/content digest and resolves content-part boundaries against its exact rendered token stream. Program and snapshot segments retain separate semantic ranges, but eviction uses their combined contiguous suffix owned by the observation ID. This avoids splitting text tokens that span the program/snapshot boundary. The protocol layer must reject any unprovable mixed boundary; the harness never estimates a byte/token boundary or invents delimiters to force one.

## Selection and atomic acknowledgement

Start registers the first observation without eviction. An append containing only tool output, or a no-input reasoning continuation, registers nothing and evicts nothing. On the next real observation, remove the currently live prior suffix owner, then register the new owner. This applies across level changes and unchanged-board actions alike. Snapshot/program contents discovered by inspection are deferred until the new observation.

The helper caches a prepared operation by request ID before transport. Retries of that same operation return the identical suffix content, descriptor and eviction selection even if the workspace library changes. Altered message bases, altered renderer settings, or a different unresolved request ID are refused. A definitive pre-admission rejection can explicitly discard the pending candidate; an uncertain network response cannot. A repeated acknowledgement of the last committed request is idempotent. Later stale requests are refused locally.

The service acknowledgement is `persistent.managed_suffixes = {registered_observation_ids: [...], evicted_observation_ids: [...]}`. Registration includes no-suffix observations; they do not become live eviction targets. The helper commits only when both ID lists match its prepared selection exactly and the authoritative input count is valid. Capacity/validation rejection leaves both historical messages and ownership unchanged. This acknowledgement does not prove physical KV release; the engine audit does.

## Limits

This phase does not truncate historical text, refresh suffixes during an observation, replay old assistant text, change observation/snapshot prompts, or select importance-based tokens. A later chronological policy may remove the oldest configurable fraction of tokens, rounded up to complete interactions, but is outside this helper. Exact token positions, image embeddings, hybrid recurrent state and cache eviction remain the protocol/engine responsibilities. Local checks establish ownership and transport correctness only.
