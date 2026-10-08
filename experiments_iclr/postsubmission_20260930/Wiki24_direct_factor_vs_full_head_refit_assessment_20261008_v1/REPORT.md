# Prefer a direct frozen-head refit pair

8 October2026. Assessment and inactive plan only. No scientific arrays, checkpoints, labels, models, server operations, fits, source preparation or gate edits occurred. The previously sealed78-probe plan remains unchanged and inactive.

## Recommendation

**Prefer the smaller direct factorized/full head pair for the immediate weakness question.** The broad78-probe plan asks what information/fusion a linear processor can exploit. It introduces private biases, several feature geometries and aggregation families; it cannot by itself diagnose the original tied head restriction. The direct pair keeps the same frozen member inputs, common bias, own-label objective and starting effective maps, while changing only the trainable readout family and its optimization coordinates.

This is representative multiclass WikiCS analysis using all580 TRAIN and5274 selected-development nodes, three existing seeds and all requested strong reference families. It acquires no teachers or backbone refits. It is not a novel stacking method, a frozen-state generalization study or a proposed substitute for full from-scratch experiments.

## Shared feature scaling preserves the family

For each feature coordinatej, sett_j to the RMS ofH_m(i,j) over allTRAIN rows andmembers of that bank. If that RMS iszero, sett_j=1 and log the zero-training coordinate; do not drop it or inspect development to choose a scale. UseHhat_m=H_m/t, **without centering**. Scale original effective weightsA_m intoAhat_m=A_m diag(t), retaining the original bias.

For the BE bank, Ahat_m remainsD_s,m W_hat D_r,m withW_hat=W diag(t). Common scaling commutes withr_m. Thus the original common-bias/diagonal family and its initial function are preserved in real arithmetic. Member-specific centering would instead createb_m'=b+A_m mu_m, which is generally a private bias. That is appropriate for a broad information probe but changes this matched head-family question.

Use the same TRAIN-only scaling convention for ordinary/single refits. Preserve their initial native functions, exact selected local/global modes and checkpoint identities. Reconstructed FP64 effective-map logits need not be bitwise equal to sequential FP32 native factor operations; original retained scores remain authority, and meaningless tiny replay differences do not trigger a new scientific study.

## The paired BE interventions

From each unit+contrast selected state:

| Refit | Trainable variables | Initial state/bias |
| --- | --- | --- |
| Factorized bank | CommonW_hat, private r_m/s_m, one commonbias | Original head factors/base/bias, transformed for common feature scaling |
| Full bank | IndependentAhat_m, one commonbias | Ahat_m=D_s,m W_hat D_r,m and the **same** commonbias |

Freeze everyH_m and all upstream state. Each refit minimizes mean TRAIN own softmaxCE acrossmembers plus one fixed effective-map L2 penalty, with no pooled risk or auxiliary target added. Serve the unchanged mean of member probabilities. Full heads contain the entire factorized readout family at these fixed features and common bias. No reinitialization, head architecture, label exposure or selection difference is used to manufacture a gain.

**Penalty convention must be frozen once before fits.** Recommendlambda=0.001 applied to mean squared norm of the class-centered effective matrixJ_C Ahat_m, whereJ_C removes the class-row mean. This is effective-weight L2 on the discriminative margin map. It avoids penalizing a softmax-irrelevant commonclass row. No alternative penalty arm or lambda grid is proposed. Root may choose the original raw-effective-matrix convention instead, but must preserve that choice and its interpretation limits prospectively.

Raw||Ahat_m||² is invariant to changes of factorization that leaveAhat_m fixed. It is not invariant toAhat_m→Ahat_m+1v_m^T, which leaves softmax probabilities unchanged. Therefore a raw-L2 objective gap can partly reflect softmax-gauge penalty geometry. The proposed centered convention removes that specific nuisance; it does not remove nonconvexity, feature-metric dependence or all factor gauges. Keep the native bias class-mean offset fixed, or use a single documented zero-mean bias correction; do not change initial probabilities.

Use one fixed FP64 deterministic full-batchL-BFGS implementation for both families, initialized asabove, maximum1000 iterations, one declared gradient tolerance and no restarts. Freeze any line-search/finiteness/failure convention before fitting. Retain objective, ownCE andpenalty separately, actual calls, convergence, parameter/effective-map diagnostics and costs. The full-head problem is convex in its effective matrices/commonbias; the factorized problem is nonconvex with redundant parameters. Small parameter gradients in that gauge do not certify global optimality or directly comparable effective stationarity.

## Matched regularization references and count

For each seed6101/6203/6307, also refit unrestricted heads on the existing ordinaryindependent4 andsingle states with the same frozen-feature/scaling/ownCE/effective-penalty/solver recipe. Retain ordinary native private bias ownership andsingle bias ownership, initialized from their original values. This gives the ordinary reference **more bias freedom** than the paired BE common-bias bank; disclose it rather than covertly replacing native biases with their mean. It is a strong native-ownership reference, not an exact identical-bias family comparison. No extra offset/bias ablation is automatically added.

The proposed complete study has12optimization bundles: four conditions perseed×three seeds. Their outputs are39head sets:12factorized BE outputs,12full BE outputs,12ordinaryfull outputs and3singlefull outputs. The27unrestricted heads are not27new backbones. A common bias couples the four BE member objectives; count bank optimization and component outputs separately. All12bundles/failures are complete andsealed before development scoring. No development-based best iteration, member, penalty or condition is chosen.

Exact original selected-state custody matters: the ordinary final CLOSED_OWN_BEST_BANK includes four independently selected states/modes, whereas BE uses its coherent joint selected state. TRAIN already fitted each base, and the scored development population selected them. This cannot become independent validation by holding it out only during readout refitting. Existing original scores stay unchanged.

## What a result would mean

1. FullBE beats factorizedBE on whole development accuracy/NLL while retaining or increasing correct alternatives: direct readout freedom/optimization merits further investigation. The causal claim is limited to this frozen-head/refit intervention.
2. FullBE has better TRAIN objective but no development gain: extra flexibility or fit strength did not transfer; do not promote a head-capacity remedy.
3. Factor/full refits improve similarly: ordinary refitting/regularization can explain the gain. An unrestricted-only architecture is unnecessary for this diagnostic.
4. Ordinary/single refits improve similarly or remainstronger: no demonstrated shared-ensemble advantage. Different backbone objectives/selection identities also prevent a causal attribution to sharing.
5. Full coefficient matrices violate saved cross-ratios: this alone is not a prediction-capacity certificate. Class-common weight rows, feature nullspaces and changing bases can preserve probabilities with different matrices.

No observed difference from a nonconvex factorized fit proves a strict capacity bottleneck. Even a converged convex full fit supplies no globally optimal factor-family comparator. Whole-network private upstream states can compensate for head restrictions. Conversely, a useful fullhead change can be scientifically worthwhile through easier optimization without a new expressivity theorem.

The multi-output restriction is node-task-specific. For a scalar output with nonzero sharedW entries, choosing private r_m freely represents any linear weight row, ignoring bias/optimization. Do not claim a corresponding matrix-cross-ratio bottleneck for scalarMolHIV orCollab outputs.

## Readout and later action

For everycondition/seed retain member mean/minimum competence, any-correct coverage, served accuracy/NLL/Brier, and all native-error repairs/introduced errors. Freeze old common-rival cohorts before new predictions; count actual new member rankings and served recovery, not just coefficient distance. Include the full5274-node population and every failed bundle. Seed-node totals remain repeated descriptive instances.

Positive frozen-state evidence could motivate a **known**, zero-initialized private task-head residual trained fromscratch with the target method, shared backbone and qualified local/global head ownership. It would require complete objective-matched GNN comparisons and unused confirmation; it is not a two-stage independent-teacher proposal. No residual implementation or execution is admitted here. Existing target/attention plans and their capacity caveats are preserved.
