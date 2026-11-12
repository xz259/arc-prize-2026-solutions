"""glue_fusion_v1 (table v2): lever-1/2 decode-step glue work behind the knob ARC3_GLUE_FUSION (0 | all | comma list of parts).
Served configuration after the N39 profile diff: `gemm,gdn`. `all` (adds hc) was measured once at N39 and N8 and rejected.

hc   : REJECTED. hc_silu + cuBLAS up-GEMM -> one Triton launch (silu fused into the A load) at all 97 hyper-connection
       mix sites. In situ 17.8 us vs 2.1 + 6.8 us for the vendor pair: every program recomputes silu over the whole
       [BLOCK_M, 320] A block (320 programs x 20k sigmoids), which costs more than the launch it saves.
gemm : shape-keyed Triton dispatch inside UnquantizedLinearMethod.apply: in_proj_ba [96,2560] for 2 <= M <= 15 only
       (cuBLAS heuristic pathology: 16.2 -> 4.3 us cold; 36 sites, inactive at M = 1 and M >= 16). v1 also dispatched
       shared_expert down [2560,640] (net zero at N39, dropped).
gdn  : drops the two b/a .contiguous() copies on the packed non-spec decode path (the packed kernel takes row strides)
       and the output-gate copy before the gated RMSNorm (same kernel, same tile, strided z).
Numerics: gdn is bit-identical; hc/gemm keep every bf16 boundary but accumulate the fp32 dot in a different order from
cuBLAS (rounding-level, deterministic). Prefill (M > 64) always keeps the vendor path.
hcf  : stage 14 (2026-09-27). Every HC mix site: down GEMM + split-K reduction + hc_silu in one launch (kernels_hcf
       hcf_down, deterministic fixed-order fixup) and the up projection as a 640-program GEMM (up_gemm, bit-identical
       to cuBLAS on the benchmark); hc_gate_mix unchanged. 17 <= M <= 64 only. Exclusive with hc.
splitk: stage 14. One split-K launch with the same fixup (was cuBLAS GEMM + splitKreduce) for the MoE router [512, 2560]
       (17 <= M <= 64), GDN in_proj_ba [96, 2560] and QSA index_qk_proj [640, 2560] (17 <= M <= 32); below M = 17
       in_proj_ba keeps the gemm part's kernel when gemm is on. Microbenchmark at M = 25 (HBM-cold): HC site 19.71 ->
       15.95 us incl. combine_norm, router 4.81 -> 3.61, in_proj_ba 4.33 -> 2.80, index_qk 5.17 -> 4.02 us; ~0.5 ms
       per 25-lane step before in-situ confirmation. Rejected in stage 14: gate_mix fused into the up GEMM (10.9 us
       whole-row, 7.2 us stream-split, vs 6.9 plain up + vendor gate_mix) and L2 weight prefetch under PDL (+3 us).
"""
