"""Exact CUDA13 link-only correction; no compiler or native library load."""
import hashlib
from pathlib import Path
import re

CUDA_ROOT=Path('/workspace/arc3-inference/runtime/base/usr/local/cuda-13.0')
LIBRARIES={
 'libcudart.so':dict(target='libcudart.so.13.0.96',bytes=704288,sha256='96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0'),
 'libnvrtc.so.13':dict(target='libnvrtc.so.13.0.88',bytes=109338064,sha256='a49e67e8e74590f1e98de55c39c6287efd3f59e3c3797464d7bbe0fe01349b11'),
 'stubs/libcuda.so':dict(target='stubs/libcuda.so',bytes=70368,sha256='67f959d1ab945005774464fcbf5a10722f97d0781b5800dd8636dc56747c2735'),
}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def libraries(cuda_root):
    root=Path(cuda_root).resolve(strict=True)
    require(root==CUDA_ROOT,'Unexpected CUDA toolchain root')
    require(not (root/'lib64').exists(),'Legacy CUDA lib64 now exists; link correction needs review')
    base=root/'targets/x86_64-linux/lib';rows={}
    for relative,pin in LIBRARIES.items():
        path=base/relative;target=path.resolve(strict=True)
        require(target==base/pin['target'] and target.is_file(),'CUDA link library target differs: '+relative)
        require(target.stat().st_size==pin['bytes'] and sha(target)==pin['sha256'],
                'CUDA link library bytes differ: '+relative)
        rows[relative]=dict(path=str(path),resolved_path=str(target),bytes=pin['bytes'],sha256=pin['sha256'])
    return rows


def replacements(cuda_root):
    base=Path(cuda_root)/'targets/x86_64-linux/lib'
    return {'-L$cuda_home/lib64':'-L'+str(base),
            '-L$cuda_home/lib64/stubs':'-L'+str(base/'stubs'),
            '-lnvrtc':'-l:libnvrtc.so.13'}


def rewrite(text,cuda_root):
    require(text.count('cuda_home = '+str(cuda_root)+'\n')==1,'Ninja CUDA root declaration differs')
    changed=text
    for old,new in replacements(cuda_root).items():
        pattern=r'(?<!\S)'+re.escape(old)+r'(?=\s|$)'
        require(len(re.findall(pattern,changed))==1,'Expected exactly one original link token: '+old)
        changed=re.sub(pattern,lambda _:new,changed)
    inverse=changed
    for old,new in replacements(cuda_root).items():
        pattern=r'(?<!\S)'+re.escape(new)+r'(?=\s|$)'
        require(len(re.findall(pattern,inverse))==1,'Expected exactly one corrected link token: '+new)
        inverse=re.sub(pattern,lambda _:old,inverse)
    require(inverse==text,'Link rewrite changed unrelated bytes')
    return changed


def prepare_link_fix(ninja_path,cuda_root,prepared):
    root=Path(prepared).resolve();ninja=Path(ninja_path).resolve(strict=True)
    require(ninja.is_relative_to(root/'build'),'Ninja outside prepared build')
    inventory=libraries(cuda_root)
    original=ninja.read_text();changed=rewrite(original,cuda_root)
    saved=root/'ORIGINAL_LINK_NINJA.txt'
    with saved.open('x') as f:f.write(original)
    ninja.write_text(changed)
    require(libraries(cuda_root)==inventory,'CUDA libraries changed during link preparation')
    return dict(cuda_root=str(cuda_root),libraries=inventory,original_ninja_path=str(saved),
                original_ninja_sha256=sha(saved),rewritten_ninja_path=str(ninja),
                rewritten_ninja_sha256=sha(ninja),replacements=replacements(cuda_root),
                compile_source_or_flags_changed=False,library_loaded=False)


def verify_link_dependencies(plan,prepared):
    root=Path(prepared).resolve();proof=plan['link_dependency']
    require(proof['cuda_root']==str(CUDA_ROOT),'Link dependency CUDA root differs')
    require(proof['libraries']==libraries(CUDA_ROOT),'Pinned link library receipt differs')
    require(proof['replacements']==replacements(CUDA_ROOT) and
            proof['compile_source_or_flags_changed'] is False and proof['library_loaded'] is False,
            'Link-only correction contract differs')
    old=Path(proof['original_ninja_path']);new=Path(proof['rewritten_ninja_path'])
    require(old==root/'ORIGINAL_LINK_NINJA.txt' and not old.is_symlink(),'Original Ninja receipt path differs')
    require(new==Path(plan['ninja_path']) and new.resolve().is_relative_to(root/'build') and not new.is_symlink(),
            'Corrected Ninja path differs')
    require(sha(old)==proof['original_ninja_sha256'] and
            sha(new)==proof['rewritten_ninja_sha256']==plan['ninja_sha256'],'Link Ninja bytes changed')
    require(new.read_text()==rewrite(old.read_text(),CUDA_ROOT),'Ninja differs beyond exact link correction')
    return dict(proof,exact=True)
