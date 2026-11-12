"""Stage the exact public OCI blobs from the supplied runtime manifest.

No image execution or extraction. Faster alternative to uploading the 7.9 GB ZIP.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

root = Path(os.environ.get("ARC3_ROOT", "/workspace/arc3-inference"))
raw = (root / "reference/runtime-manifest.json").read_bytes()
if hashlib.sha256(raw).hexdigest() != "e9453f8d0e9c5eb2e14712e0f8563aaa96752ddc1705f245cac327537502baad":
    raise RuntimeError("Unexpected reference manifest")
manifest = json.loads(raw)
destination = root / "reference/runtime-layers"
destination.mkdir(parents=True, exist_ok=True)
(destination / "runtime-manifest.json").write_bytes(raw)


def fetch(row):
    path = destination / (row["file"] + ".blob")
    part = path.with_suffix(path.suffix + ".part")
    if path.exists():
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(chunk)
        if h.hexdigest() == row["sha256"] and path.stat().st_size == row["size"]:
            return {"index": row["index"], "bytes": row["size"], "sha256": h.hexdigest(), "reused": True}
        raise RuntimeError(f"Existing reference blob differs: {path}")
    token_url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:vllm/vllm-openai:pull"
    with urlopen(token_url, timeout=30) as response:
        token = json.load(response)["token"]
    url = "https://registry-1.docker.io/v2/vllm/vllm-openai/blobs/" + row["digest"]
    request = Request(url, headers={"Authorization": "Bearer " + token})
    started = time.monotonic()
    h, count = hashlib.sha256(), 0
    with urlopen(request, timeout=60) as response, part.open("wb") as f:
        while chunk := response.read(8 * 1024**2):
            f.write(chunk)
            h.update(chunk)
            count += len(chunk)
    if count != row["size"] or h.hexdigest() != row["sha256"]:
        raise RuntimeError(f"Runtime blob mismatch: layer {row['index']}")
    part.replace(path)
    record = {"index": row["index"], "bytes": count, "sha256": h.hexdigest(),
              "seconds": round(time.monotonic() - started, 2), "source": "official OCI registry"}
    print(json.dumps(record), flush=True)
    return record


with ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(fetch, manifest["selected_layers"]))
(root / "evidence/runtime-layer-transfer.json").write_text(json.dumps(records, indent=2) + "\n")
print("All six reference layers verified", flush=True)
