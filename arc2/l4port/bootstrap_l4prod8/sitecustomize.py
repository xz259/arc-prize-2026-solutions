"""L4 port production-launch bootstrap (2026-10-31, ARC2 - main; the user: get the harness working with the PP profile on 4x L4).
For the production launcher's L4 copy (k_launch_scaling_l4.py, via ARC3_PY_PREPEND): runs the production chain's head
(L4_NEXT_SITECUSTOMIZE, set by that launcher to the outermost production bootstrap: stage2_v1, which chains MTP0X, glue fusion,
fold, the PLE handoff, the FP8 tier, the dense adapters), then installs the L4 port's hooks in the order bootstrap_ple ... ple12
install them, each gated by its own env as there. bootstrap_ple's PLE copy-handoff hook is NOT installed here: production's combined
bootstrap installs it (ARC3_PLE_HANDOFF_IMPL=copy), and a second Finder (a different module object, so the hook's own
double-install assert cannot see it) would wrap the connector twice.
Without L4_NEXT_SITECUSTOMIZE (v2, 2026-10-31: k_runtime_python.sh also puts ARC3_PY_PREPEND first on the launcher's own path, and
the launcher sets the head only for the server it starts) this is a pass-through: the next sitecustomize on sys.path runs, as it
would without this directory first, and no hook is installed.
v3 (2026-10-31): ARC3_L4PORT_EVPP=1 installs l4prod/l4port_evpp.py last (production's session / eviction stack under TP x PP).
v4 (2026-10-31): with EVPP and the 'draftring' part, l4prod/l4port_draftring_pp.py (draftring's worker extras per pipeline stage).
v5 (2026-10-31): the drafter-head hook from bootstrap_ple13 (v2, even split over the TP ranks).
v6 (2026-10-31): ARC3_L4PORT_TPIDENT=N loads the TP-identity diagnostic (l4prod/l4port_tpident.py).
v7 (2026-10-31, bootstrap_l4prod2; v6 stays in bootstrap_l4prod for the running chains): ARC3_L4PORT_PLE_HPTP=1 installs
ple_hostpoll_tp/install.py right after the head (production's host-poll PLE handoff, with ARC3_PLE_HANDOFF_IMPL=hostpoll,
made TP-aware and CUDA-free on the offload side; it must run before vLLM imports the PLE modules).
v9 (2026-10-31, bootstrap_l4prod4; v7 stays in bootstrap_l4prod2): ARC3_L4PORT_IDX_SHARD=1 loads l4prod/l4port_idxshard.py (lever
4: the QSA indexer's compressed keys split by block parity over the TP ranks; before vLLM imports the model modules).
v10 (2026-10-31, bootstrap_l4prod5; v9 stays in bootstrap_l4prod4): ARC3_L4PORT_IDX_SHARD=1 loads l4prod/l4port_idxshard_v2.py
instead (the sharded compaction and digest; the composite-key merge, ARC3_L4PORT_IDX_SHARD_MERGE=keys|sort).
v11 (2026-11-01, bootstrap_l4prod6; v10 stays in bootstrap_l4prod5, the shipped v2d / v2e): ARC3_L4PORT_QSA_WS_MIB=N (N > 0) loads
l4prod/l4port_qsaws.py (the QSA selection's logits workspace, 128 MiB stock, before vLLM imports the model modules).
v12 (2026-11-01, bootstrap_l4prod7; v11 stays in bootstrap_l4prod6): run-to-run determinism on the TP2 x PP2 path, each opt-in:
ARC3_L4PORT_MARLINDET_EP=1 loads l4prod/l4port_marlindet_ep.py (marlindet's canonical expert-segment order under expert
parallelism, which marlindet passes through); ARC3_L4PORT_QSA_SORT=1 loads l4prod/l4port_qsasort.py (the native QSA top-k's
selection sorted ascending in the stock select).
v13 (2026-11-01, bootstrap_l4prod8; v12 stays in bootstrap_l4prod7): ARC3_L4PORT_PP_DEFER=1 loads l4prod/l4port_ppdefer.py after
l4port_ppdraftbc (the non-last stage's PP token / count / draft receives posted at the consuming step, so no NCCL receive spins
during a forward: the Kaggle decode gap)."""
import importlib.util
import os
import runpy
import sys
import traceback

L = "/mnt/shared/arc2/l4port/"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _passthrough():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in sys.path:
        if not p or os.path.abspath(p) == here:
            continue
        cand = os.path.join(p, "sitecustomize.py")
        if os.path.isfile(cand):
            runpy.run_path(cand, run_name="l4prod_passthrough")
            return


def _install(head):
    assert os.path.isfile(head), ("L4_NEXT_SITECUSTOMIZE", head)
    runpy.run_path(head, run_name="l4prod_next_sitecustomize")
    E = os.environ.get
    if E("ARC3_L4PORT_PLE_HPTP") == "1":
        # v7 (2026-10-31): the host-poll PLE handoff for TP > 1 without the offload process's CUDA contexts
        _load("l4port_plehptp", L + "ple_hostpoll_tp/install.py").install()
    if E("ARC3_L4PORT_MARLINDET_EP") == "1":
        # v12 (2026-11-01): deterministic Marlin MoE under expert parallelism (the PP path's non-bitwise reruns)
        _load("l4port_marlindet_ep", L + "l4prod/l4port_marlindet_ep.py").install()
    if E("ARC3_L4PORT_QSA_SORT") == "1":
        # v12 (2026-11-01): the native QSA top-k's selection in ascending order (sm_89 has no radix selector)
        _load("l4port_qsasort", L + "l4prod/l4port_qsasort.py").install()
    if int(E("ARC3_L4PORT_QSA_WS_MIB", "0") or 0) > 0:
        # v11 (2026-11-01): the QSA selection's logits workspace (lane item: the drafter's 118-MiB select transient on PP 1)
        _load("l4port_qsaws", L + "l4prod/l4port_qsaws.py").install()
    if E("ARC3_L4PORT_IDX_SHARD") == "1":
        # v10 (2026-10-31): lever 4, the indexer dedupe v2 (after the PLE installer: it extends VLLM_BUILD_URL's AOT salt)
        _load("l4port_idxshard_v2", L + "l4prod/l4port_idxshard_v2.py").install()
    if E("ARC3_L4PORT_PP_SKIPMIXER") == "1":
        _load("l4port_pp", L + "bootstrap_ple/l4port_pp.py").install()
    if int(E("ARC3_L4PORT_VPTOPK", "0") or 0) > 0:
        _load("l4port_vptopk", L + "bootstrap_ple2/l4port_vptopk.py").install()
    if E("ARC3_L4PORT_PP_MTP") == "1":
        _load("l4port_ppmtp", L + "bootstrap_ple3/l4port_ppmtp.py").install()
    if int(E("ARC3_L4PORT_PP_WARMCAP", "0") or 0) >= 2:
        _load("l4port_ppwarm", L + "bootstrap_ple4/l4port_ppwarm.py").install()
    if E("ARC3_L4PORT_KVDUMP") == "1":
        _load("l4port_kvdump", L + "bootstrap_ple5/l4port_kvdump.py").install()
    if E("ARC3_L4PORT_PP_BCASTPAD") == "1":
        _load("l4port_ppbcast", L + "bootstrap_ple6/l4port_ppbcast.py").install()
    if E("ARC3_L4PORT_PP_DRAFTBC") == "1":
        _load("l4port_ppdraftbc", L + "bootstrap_ple9/l4port_ppdraftbc.py").install()
    if E("ARC3_L4PORT_PP_DEFER") == "1":
        # v13 (2026-11-01): the PP receives posted when consumed (after l4port_ppdraftbc, which it subsumes on the receive side)
        _load("l4port_ppdefer", L + "l4prod/l4port_ppdefer.py").install()
    if E("ARC3_L4PORT_PP_EMBHOST") == "1":
        _load("l4port_ppembhost", L + "bootstrap_ple10/l4port_ppembhost.py").install()
    if E("ARC3_L4PORT_PP_SPLIT"):
        _load("l4port_ppsplit", L + "bootstrap_ple11/l4port_ppsplit.py").install()
    if E("ARC3_L4PORT_DHEAD_TP") == "1":
        # v5 (2026-10-31): v2 of the drafter-head hook (the 32K draft vocabulary split evenly over the TP ranks; L4 l12e 794.2 tok/s
        # vs 763.1 at 27 lanes); ARC3_L4PORT_DHEAD_EVEN=0 restores v1's split by vocab shard
        _load("l4port_dheadtp", L + "bootstrap_ple13/l4port_dheadtp.py").install()
    if int(E("ARC3_L4PORT_TPIDENT", "0") or 0) > 0:
        # v6 (2026-10-31): the TP-identity diagnostic (l4prod/l4port_tpident.py; off unless asked for)
        _load("l4port_tpident", L + "l4prod/l4port_tpident.py").install()
    if E("ARC3_L4PORT_EVPP") == "1":
        # v3 (2026-10-31): the session / eviction stack under TP x PP; registered by name, because the engine core sends its
        # worker wrappers by reference (cloudpickle) and every process resolves them from here
        mod = _load("l4port_evpp", L + "l4prod/l4port_evpp.py")
        sys.modules["l4port_evpp"] = mod
        mod.install()
        if "draftring" in E("ARC3_STAGE2", "").split(","):
            # v4 (2026-10-31, after l4p35): draftring's worker extras per pipeline stage (PLE on stage 0, the drafter on stage 1);
            # swapped into stage2_v1.draftring before the model runner is imported
            _load("l4port_draftring_pp", L + "l4prod/l4port_draftring_pp.py").install()


try:
    _head = os.environ.get("L4_NEXT_SITECUSTOMIZE", "")
    if _head:
        _install(_head)
    else:
        _passthrough()
except BaseException:
    traceback.print_exc()
    os._exit(94)
