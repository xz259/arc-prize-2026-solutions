"""dryrun_l4.py LABEL [K=V ...] (2026-10-31, ARC2 - main): the server command and environment k_launch_scaling_l4.py would start,
without starting anything. Builds the environment as arc2_server_l4.py does (KDEF_sp20 g60f8inp128 rendered by kaggle_run, base
keys, EXTRA, L4ENV, DROP, the K=V overrides), then runs the launcher with the 'status' action in a child (k_runtime_python.sh) whose
driver prints server_command() and the server environment's PYTHONPATH head, L4_NEXT_SITECUSTOMIZE and the parallel keys."""
import importlib.util
import json
import os
import subprocess
import sys

spec = importlib.util.spec_from_file_location("arc2_server_l4", "/mnt/shared/arc2/l4port/l4prod/arc2_server_l4.py")
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)
kr = srv.load_kr()
D = json.load(open("/mnt/shared/arc2/l4port/sidepod/KDEF_sp20.json"))
prof = D["profiles"]["g60f8inp128"]
label = sys.argv[1]
senv_def = kr.render(D, "server_serve", prof, label, "0,1,2,3", 1399)
drop = set(srv.DROP)
l4 = dict(srv.L4ENV, ARC3_STAGE2=",".join(p for p in senv_def["ARC3_STAGE2"].split(",") if p not in drop))
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    l4[k] = v
base = {k: os.environ[k] for k in kr.BASE_KEYS if k in os.environ}
senv = dict(base, **senv_def)
senv.update(srv.EXTRA)
senv.update(l4)
driver = r'''
import json, os, runpy, sys
sys.argv = ["k_launch_scaling_l4.py", "status"]
path = os.environ["DRY_LAUNCHER"]
g = {"__name__": "__main__", "__file__": path}
try:
    exec(compile(open(path).read(), path, "exec"), g)   # 'status' raises at the end (no identity): the globals stay
except BaseException as exc:
    print("launcher action ended: %r" % (exc,))
if g:
    argv = g["server_command"]()
    env, upd = g["m"].server_environment()
    print("ARGV", json.dumps(argv))
    pp = env.get("PYTHONPATH", "").split(":")
    print("PYTHONPATH_HEAD", json.dumps(pp[:6]))
    print("L4_NEXT", env.get("L4_NEXT_SITECUSTOMIZE"))
    for k in ("CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST", "NCCL_P2P_DISABLE", "ARC3_STAGE2", "ARC3_QSA_MODE", "ARC3_MOE_FINALIZE",
              "VLLM_USE_DEEP_GEMM", "ARC3_DENSE_AOT_ROOT", "ARC3_DENSE_FP8_TIER", "ARC3_L4PORT_FP8TIER_NEXT", "ARC3_PLE_HANDOFF_IMPL"):
        print("ENV", k, "=", env.get(k))
'''
dp = "/tmp/l4_dryrun_driver.py"
open(dp, "w").write(driver)
senv["DRY_LAUNCHER"] = kr.S2 + "/k_launch_scaling_l4.py"
p = subprocess.run(["bash", kr.S2 + "/k_runtime_python.sh", dp], cwd=kr.S2, env=senv, capture_output=True, text=True, timeout=300)
print("rc", p.returncode)
print((p.stdout or "")[-6000:])
print("STDERR", (p.stderr or "")[-3000:])
