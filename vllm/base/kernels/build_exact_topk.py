"""Build the isolated CUDA selector without importing or rebuilding vLLM."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
sources = [ROOT/'exact_topk_extension.cu', ROOT/'exact_topk_radix.cuh']
identity = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
output = ROOT/'build'/digest
output.mkdir(parents=True, exist_ok=True)
cuda = Path(os.environ.get('CUDA_HOME', str(ROOT.parent/'runtime/base/usr/local/cuda-13.0')))
argv = [str(cuda/'bin/nvcc'), '-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC',
        '-gencode', 'arch=compute_120,code=sm_120', '-gencode', 'arch=compute_120,code=compute_120',
        '-I'+str(cuda/'include'), str(sources[0]), '-o', str(output/'libarc3_topk.so')]
with (output/'build.log').open('w') as f:
    subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=600, check=True)
manifest = dict(sources=identity, argv=argv, library=str((output/'libarc3_topk.so').relative_to(ROOT.parent)),
                library_sha256=hashlib.sha256((output/'libarc3_topk.so').read_bytes()).hexdigest())
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(ROOT/'build/latest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
