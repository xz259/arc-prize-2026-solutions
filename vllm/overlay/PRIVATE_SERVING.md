# Mode3 serving candidate, normal compilation

Local derivative of frozen `baseline_server_v1`. Only launch environment/PYTHONPATH and candidate label change; the original command builder, all kernel/runtime/persistent-state sources and normal graph configuration remain unchanged. The reviewed startup adapter is nested as `private_moe`; its settings pin the passed native producer receipt.

Phase: `nvfp4-private-moe-mode3-20260914-v1`. Worker receipt: `private-startup/<worker-pid>.json`, state `warmup-verified`. Use the owned live worker with matching boot/start_ticks and process ancestry. `candidate-server.json` remains the nonblocking launch identity, **not readiness**. Owner must wait for matching `/v1/models`, full startup receipt and actual captured N25/N30 graphs.

Before launch, run the adapter's cheap stdlib preflight and 18 CPU tests plus these four exact-lineage checks under the pinned wrapper. Native top-k admission passed210 direct/50 graph checks and is pinned in the immutable settings. Normal `.95`, N30,40K,MTP0, no prefix caching, unfused deterministic MoE and FULL sizes1,2,4,8,16,24,25,30,32 are retained. No exact compile sizes or eager mode.

Do not interpret startup receipts as device dispatch proof. Root's excluded N25/N30 traces must show candidate symbols in full replay and preserve large-prefill fallback; profile client remains unchanged. Then use the thin private serving ABBA client. No production promotion follows from this candidate's one-layer operator gain alone.

No server, GPU job or upload was launched during local preparation. The inherited `stop-baseline` action is not authorized by this artifact; the owner retains all process scheduling and identity checks.
