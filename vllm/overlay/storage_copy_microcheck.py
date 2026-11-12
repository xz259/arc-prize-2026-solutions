"""Bounded synthetic physical-copy comparison; no model, harness, or server.

Example: runtime_python.sh storage_copy_microcheck.py --heads 2 --rows 40000 --owner-passes 12
Optional wider-row stress: --heads 32 --rows 8192

The repeated passes deliberately reuse one fixture to bound memory. They are
NOT an independent-owner cache benchmark; cache hotness and the lack of model
interference can differ from serving. CUDA event intervals include CPU launch
gaps, so they are not a measurement of kernel execution time alone.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import eviction_storage as candidate
from eviction_validation import stored_rows_digest


REFERENCE_SHA256="e13dc1c894143effbbfc3e2e093a09f9754c90956c764d85354bfb37407d3d23"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(path):
    if file_hash(path)!=REFERENCE_SHA256:
        raise RuntimeError("Reference source differs from the pinned phase-3 original")
    name="arc3_storage_microcheck_reference"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def kept_rows(case,rows):
    if case=="earliest_ten_percent":
        begin=min(512,rows//16)
        end=begin+math.ceil(rows*0.10)
    elif case=="managed_suffix_near_tail":
        width=min(1024,rows//8)
        begin,end=rows-2*width,rows-width
    else:
        raise ValueError(case)
    return [*range(begin),*range(end,rows)],(begin,end)


def physical_pages_hash(cache,pages):
    checksum=hashlib.sha256()
    for page in pages:
        # At most one physical page is copied to CPU, never the entire fixture.
        checksum.update(cache[page].contiguous().cpu().numpy().tobytes())
    return checksum.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=Path(__file__).resolve().parent.parent/
                        "persistent_history_eviction_20260913"/"eviction_storage.py")
    parser.add_argument("--rows",type=int,default=8192)
    parser.add_argument("--heads",type=int,default=2)
    parser.add_argument("--page-size",type=int,default=2768)
    parser.add_argument("--owner-passes",type=int,default=1)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=1)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if not (2048<=args.rows<=40000 and 1<=args.heads<=32
            and 1<=args.page_size<=4096 and 1<=args.owner_passes<=12
            and 1<=args.repeats<=9 and 0<=args.warmup<=3):
        parser.error("Arguments exceed this bounded diagnostic's supported limits")
    reference=load_reference(args.reference)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This explicitly GPU diagnostic requires CUDA")
    pages=math.ceil(args.rows/args.page_size)+2
    fixture_bytes=pages*args.page_size*args.heads*296
    estimated_bytes=2*fixture_bytes+candidate.DEFAULT_SCRATCH_BYTES+64*1024*1024
    if estimated_bytes>1024**3:
        raise RuntimeError("Estimated fixtures, scratch, and index allowance exceed 1 GiB")
    free,total=torch.cuda.mem_get_info()
    if free<estimated_bytes+256*1024*1024:
        raise RuntimeError("Insufficient free GPU memory for fixtures plus 256 MiB margin")
    torch.manual_seed(95013)
    # Match the token-major backing layout reported by the bound model cache:
    # logical [pages, heads, tokens, 296], strides [page_bytes,296,row_bytes,1].
    source=torch.randint(0,256,(pages,args.page_size,args.heads,296),
                         dtype=torch.uint8,device="cuda").transpose(1,2)
    working=torch.empty_like(source,memory_format=torch.preserve_format)
    physical=list(range(1,pages-1))
    random.Random(95013).shuffle(physical)
    variants=("original_64","candidate_64","candidate_default")
    report={
        "kind":"physical-row-copy-synthetic-v1",
        "config":{**vars(args),"reference":str(args.reference),
                  "output":str(args.output) if args.output else None},
        "source_sha256":{"microcheck":file_hash(__file__),
                         "candidate":file_hash(candidate.__file__),
                         "reference":file_hash(args.reference)},
        "device":{"name":torch.cuda.get_device_name(),"torch":torch.__version__,
                  "free_bytes_before_fixtures":free,"total_bytes":total},
        "fixture":{"shape":list(source.shape),"strides":list(source.stride()),
                   "each_fixture_bytes":fixture_bytes,
                   "estimated_working_bytes_with_allowance":estimated_bytes,
                   "reuses_same_fixture_across_owner_passes":True},
        "timing_note":"Reset, validation, and warmup excluded; synchronized wall and CUDA stream intervals include host launch gaps. Repeated owner passes use one hot fixture, not distinct model owners.",
        "cases":[],
    }

    for case in ("earliest_ten_percent","managed_suffix_near_tail"):
        keep,deleted=kept_rows(case,args.rows)
        plan=candidate.page_compaction(physical,keep,old_rows=args.rows,
                                       storage_block_size=args.page_size)
        old_plan=reference.page_compaction(physical,keep,old_rows=args.rows,
                                           storage_block_size=args.page_size)
        expected=stored_rows_digest(source,plan,token_axis=2,chunk_rows=512)
        untouched=[p for p in range(pages) if p not in plan.kept_block_ids]
        untouched_expected=physical_pages_hash(source,untouched)
        case_report={"name":case,"deleted_rows":deleted,"retained_rows":len(keep),
                     "validation":{},"samples":[],"summary":{}}

        def run_copy(variant):
            if variant=="original_64":
                return reference.compact_cache_rows_(working,old_plan,token_axis=2)
            options={"scratch_rows":64} if variant=="candidate_64" else {}
            return candidate.compact_cache_rows_(working,plan,token_axis=2,**options)

        # Every implementation must independently match pristine source bytes;
        # agreement between two possibly incorrect implementations is not enough.
        for variant in variants:
            working.copy_(source)
            receipt=run_copy(variant)
            torch.cuda.synchronize()
            observed=stored_rows_digest(working,plan,token_axis=2,after=True,chunk_rows=512)
            if observed!=expected or physical_pages_hash(working,untouched)!=untouched_expected:
                raise RuntimeError(f"Byte or untouched-page mismatch for {case}/{variant}")
            case_report["validation"][variant]={"retained_bytes_equal":True,
                "untouched_pages_equal":True,"retained_sha256":expected["sha256"],
                "receipt":receipt}

        for _ in range(args.warmup):
            for variant in variants:
                working.copy_(source)
                for _ in range(args.owner_passes):run_copy(variant)
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        for repetition in range(args.repeats):
            order=variants if repetition%2==0 else tuple(reversed(variants))
            for variant in order:
                working.copy_(source)
                torch.cuda.synchronize()
                wall_start=time.perf_counter()
                start.record()
                for _ in range(args.owner_passes):receipt=run_copy(variant)
                submitted=time.perf_counter()
                end.record()
                end.synchronize()
                wall_end=time.perf_counter()
                case_report["samples"].append({
                    "variant":variant,"repetition":repetition,
                    "passes":args.owner_passes,
                    "wall_ms":1000*(wall_end-wall_start),
                    "host_submission_ms":1000*(submitted-wall_start),
                    "cuda_stream_ms":start.elapsed_time(end),"receipt":receipt,
                })
        case_report["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        case_report["peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        for variant in variants:
            samples=[x for x in case_report["samples"] if x["variant"]==variant]
            case_report["summary"][variant]={key:statistics.median(x[key] for x in samples)
                for key in ("wall_ms","host_submission_ms","cuda_stream_ms")}
            case_report["summary"][variant]["median_wall_ms_per_pass"]=(
                case_report["summary"][variant]["wall_ms"]/args.owner_passes)
        report["cases"].append(case_report)
    report["completed"]=True
    text=json.dumps(report,indent=2,sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n")
    print(text)


if __name__=="__main__":main()
