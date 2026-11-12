"""Extract the pinned runtime once. Does not patch vLLM or load a model.

Run with the system Python; the extracted packages require Python 3.12 later.
Completed layers are journaled so preparation can resume after interruption.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(os.environ.get("ARC3_ROOT", "/workspace/arc3-inference")).resolve()
LAYERS = ROOT / "reference/runtime-layers"
DEST = ROOT / "runtime/base"
MANIFEST_SHA256 = "e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad"
# The reference setup's FAST_RUNTIME_WHITEOUTS for these exact SHA-pinned layers.
# Avoid traversing every installed package after each layer on a network volume.
WHITEOUTS = {
    8: ["usr/local/cuda-13.0/targets/x86_64-linux/lib/.wh.libcudart.so.13.0.88"],
    17: ["tmp/.wh.requirements-cuda.txt", "tmp/.wh.common.txt"],
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def main():
    manifest_path = LAYERS / "runtime-manifest.json"
    if sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("Runtime manifest does not match the supplied reference")
    manifest = json.loads(manifest_path.read_text())
    layers = sorted(manifest["selected_layers"], key=lambda row: row["index"])
    journal = ROOT / "runtime/extraction.json"
    journal.parent.mkdir(parents=True, exist_ok=True)
    if journal.exists():
        state = json.loads(journal.read_text())
        if state["manifest_sha256"] != MANIFEST_SHA256 or state["destination"] != str(DEST):
            raise RuntimeError("Existing extraction belongs to another runtime")
    else:
        if DEST.exists() and any(DEST.iterdir()):
            raise RuntimeError("Refusing to overlay an untracked runtime directory")
        state = {"manifest_sha256": MANIFEST_SHA256, "destination": str(DEST),
                 "status": "extracting", "layers": [], "model_loaded": False,
                 "source_patches_applied": []}
        save(journal, state)
    if state["status"] == "complete":
        print("Runtime already extracted:", DEST, flush=True)
        return
    # Reference's conservative extracted size is 18.36 GB; leave working space.
    if shutil.disk_usage(ROOT).free < 22 * 1024**3:
        raise RuntimeError("Need at least 22 GiB free for runtime preparation")
    DEST.mkdir(parents=True, exist_ok=True)
    completed = {row["index"] for row in state["layers"]}
    for row in layers:
        if row["index"] in completed:
            continue
        archive = LAYERS / row["file"]
        if not archive.is_file():
            archive = archive.with_name(archive.name + ".blob")
        print(f"Verifying layer {row['index']}: {archive.name}", flush=True)
        if archive.stat().st_size != row["size"] or sha256(archive) != row["sha256"]:
            raise RuntimeError(f"Invalid runtime layer: {archive}")
        started = time.monotonic()
        # Matches the reference extraction. Inputs are exact SHA-pinned OCI
        # blobs, never arbitrary archives; extraction stays in a dedicated root.
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(DEST),
                        "--no-same-owner", "--exclude=*/__pycache__",
                        "--exclude=*/__pycache__/*", "--exclude=*.pyc",
                        "--exclude=*.pyo"], check=True)
        whiteouts = []
        for relative in WHITEOUTS.get(row["index"], []):
            marker = DEST / relative
            if not os.path.lexists(marker):
                raise RuntimeError(f"Expected reference whiteout is missing: {marker}")
            if marker.name == ".wh..wh..opq":
                raise RuntimeError(f"Opaque whiteout needs review: {marker}")
            target = marker.with_name(marker.name[4:])
            if not target.parent.resolve().is_relative_to(DEST.resolve()):
                raise RuntimeError(f"Whiteout outside runtime: {marker}")
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            whiteouts.append(str(marker.relative_to(DEST)))
            marker.unlink()
        state["layers"].append({"index": row["index"], "sha256": row["sha256"],
                                "bytes": row["size"], "whiteouts": whiteouts,
                                "extract_seconds": round(time.monotonic() - started, 2)})
        save(journal, state)
        print(f"Extracted layer {row['index']}", flush=True)
    site = DEST / "usr/local/lib/python3.12/dist-packages"
    required = [site / "vllm/__init__.py", site / "torch/__init__.py",
                site / "vllm/v1/ple_offload/worker.py",
                DEST / "usr/local/cuda-13.0/bin/nvcc"]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Incomplete extracted runtime: {path}")
    # Preserve original Python-source fingerprints for the later patch review.
    fingerprints = {str(p.relative_to(site)): sha256(p)
                    for p in sorted((site / "vllm").rglob("*.py"))}
    save(ROOT / "runtime/original-vllm-source-sha256.json", fingerprints)
    state["status"] = "complete"
    state["vllm_python_files"] = len(fingerprints)
    save(journal, state)
    print(f"Runtime ready: {DEST}; {len(fingerprints)} original vLLM source hashes", flush=True)


if __name__ == "__main__":
    main()
