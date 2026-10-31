# Algebraic workload and memory — no timings

Let M=4,N nodes,E unchanged directed COO messages,D input width,H hidden width,T=int(H*selected_multiplier),C classes,L residual SAGE blocks. The original performs approximately

`M*N*(D*H + L*(2*H*H+3*H*T) + H*C)` dense MACs,

plus M*E*H neighbor feature reductions per layer, private scaling/biases, affine LayerNorm,GELU,concatenation and residual operations. Leading-dimension batching retains this dense/message feature arithmetic. One common W can improve cache/device utilization and reduce dispatch; it is not one per-node dense transformation for the bank. Degree metadata can be computed once per batched propagation rather than four serial calls, but lifted message features remain member-specific.

For trajectory chunk q and stem chunk s<=q, main FP32 value sizes include s*N*D input-scaled temporaries,q*N*H private hidden states,q*E*H lifted messages,q*N*2H concatenation and q*N*T FFN intermediates. At float64 double these value bytes. Exact peak includes concurrent original input/features, graph indices, stacked outputs, LayerNorm/BLAS/scatter workspaces, caching/allocator state and any contiguous copies. The default q4,s1 avoids4*N*D stem values at once but retains4*N*H/E*H hidden/message work; q1/q2 may have better memory or throughput.

Original selected parameter storage remains exactly the selected checkpoint's storage; there are no new learned parameters. Untied packing has every private selected W/bias/norm; its true serving parameter count is unchanged. Cold conversion can overlap original+packed storage, but packed-only private serving is permitted after dropping original references. Heads retains one genuine common trunk and four private capable heads; batching only their readouts does not erase those heads. Actual allocated/unique storage and workspace, not persistent forced duplication or a parameter-only ratio, determine serving memory.

Keeping stem1 preserves most initial dense launches. Whether shared-body batching helps depends on the measured fraction of evaluation time in those operations, graph message/memory work and hardware utilization. No seconds,speedup,memory-peak or toy timing is claimed. Full trained-checkpoint measurement and equal control optimization are required by QUALIFICATION.md.
