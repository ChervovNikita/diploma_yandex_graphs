# Typed V3 observed-label mass: information scope

2026-10-10. Pinned provider/context construction only; no arrays, current outcomes, source changes or jobs. This note refines the interpretation of the fixed study without changing its source or roster.

## What the source actually propagates

Native SeHGNN IMDB uses raw five-bit multi-hot labels, not one-hot labels or labels divided by their row sum. Author main lines210–215 sets IMDB label_source[TRAIN]=init_labels.float(); retained native engine.py160–163 does the same. It is **graph adjacency rows** that are normalized (engine.py127–128). Metapath products are then diagonal-removed; typed provider/context explicitly preserves their substochastic row sums without renormalizing.

For a frozen context half, write k_j=1 when movie j's label is observed, Y_jc in{0,1}, and A_p for the native nonnegative diagonal-removed path operator. Typed V3 constructs:

- Existing positive channel: L_pc=(A_p(k*Y))_c, for all12 native paths.
- Observed-label mass: M_p=A_pk, for local MAM/MDM/MKM.
- Signed balance: B_pc=2L_pc/M_p−1 when M_p>0, otherwise0.

Balance adds no information once L_p and M_p are known. The mass uses observed membership and graph weights already available to the context builder; it introduces no new ground-truth labels or VALID/TEST supervision.

Exact source: provider.py39–51 and81–103; context.py110–136. The native model receives all five positive classes at each path, not only the queried class.

## Is mass recoverable from positive channels?

Let q_j=sum_cY_jc be each observed movie's genre cardinality. Then

**sum_c L_pc=A_p(k*q), not A_pk.**

If every observed row has exactly one positive class, M_p=sum_cL_pc. More generally a fixed known cardinality d gives M_p=sum_cL_pc/d. The pinned raw-multi-label provider does not enforce fixed cardinality, but **role_loader.py134–143 requires a nonempty genre list for every development movie**, so q_j is in1…5. V3 also checks binary finite targets and positive/negative support per class across each whole context. Actual cardinality variation was not inspected; source permits it.

There is no universal class-sum reconstruction of M for raw multi-label seeds. For example, one weighted observed seed with label(1,1,0,0,0) and weight.25 produces the same positive vector as two weight.25 observed seeds with labels(1,0,0,0,0) and(0,1,0,0,0), yet masses are.25 versus.5. All rows can have a positive label; nonnegative substochastic operators are sufficient for this counterexample. It refutes a universal local reconstruction rule, not a claim about the actual saved IMDB operators.

Under the admitted provider, max_cL_pc≤M_p≤min(sum_cL_pc,row_sum(A_p)) in real arithmetic. Exact joint/global recovery from all12 operators, fixed graph features, labels and context-ID knowledge could be possible for particular operators or representations. No operator injectivity/rank, data-specific redundancy or native learned embedding rank was inspected here. **Source alone therefore does not certify genuinely additional information on the actual IMDB inputs.**

## The other four label channels matter

If L_pc=0 while any other L_pc'>0, then M_p>0. Because weights/labels are nonnegative, all positively weighted observed contributors are negative for classc. That negative support is already implicit in the complete positive vector; B_pc=−1 for such a case without needing the numeric mass. Do not call every zero queried-class positive “unknown negatives.”

If all five positives are0, the admitted provider's nonempty-row guarantee implies M_p=0: no observed mass contributes. Thus **observed all-five-negative versus unobserved is not an ambiguity of this pinned IMDB construction**. The standalone generic TrainOnly context validator does not impose nonempty rows, but the actual role loader does; its looser standalone interface must not be used as an information claim for this study.

For positive L_pc, the exact negative mass is N_pc=M_p−L_pc. Variable genre overlap/cardinality prevents reconstructing this magnitude from the five class counts by a universal sum rule. Making M explicit can expose observation strength and normalizing denominators, but that is not evidence for independent task information or useful prediction changes.

## Correct claim and consequence

Call the field **explicit observed-label mass and a deterministic balance of existing positives with that mass**. Under raw multi-label propagation, it can be nonrecoverable from local positive sums. On the actual fixed graph it may instead be redundant, globally recoverable or correlated with existing channels. The source cannot select among those cases.

The fixed same-information local-additive, global-context, single and untied/native references remain necessary in either case. A benefit could be explicit statistics, conditioning/optimization, calibration, capacity, different context tasks or an ensemble interaction. No new-information, known-negative novelty, graph-expertise assignment or causal remedy follows from the field name. No change to the current study is requested.
