# Post-family explanation of the four-bank label-correction screen

**For root, 8 October 2026.** Plan only. Run after all three native seeds and all twelve correction records close under the frozen source/role/selected-state custody. No partial outcomes, new checkpoint selection, mask intervention, tuning, training or modification of the screen. Whole **5,274-node development endpoints** remain primary; this already used role is exploratory.

## 1. Separate correction from selecting a different native epoch

For every seed/arm, reconstruct its own coherent selected native+corrector state. On all development IDs, obtain both corrected predictions and the feature-only native predictions **from that exact native state/mode and forward**. Do not substitute the native-own-best checkpoint, another arm's native checkpoint, an end-of-training state or a newly chosen epoch.

For accuracy, NLL and Brier, retain corrected metric, same-state native metric and their difference. For each co-primary comparison C4 versus R (`S_joint4head` or `U4_sharedB`), report the exact decomposition:

`metric(C4)−metric(R) = [metric(native_C4)−metric(native_R)] + [(metric(C4)−metric(native_C4))−(metric(R)−metric(native_R))]`.

Accuracy gains and risk reductions have opposite preferred signs; retain the original signed values. This decomposition explains the selected endpoints, not a causal effect or an unbiased estimate. All arm selectors already used these development labels.

If a pipeline advantage is predominantly a better selected native epoch while its actual correction has negligible or harmful same-state effect, do not attribute that advantage to useful label-context corrections. The frozen pass/fail result still stands separately; the explanation must be narrowed.

## 2. Fix structural cohorts without outcomes

Using only permitted TRAIN IDs and the unchanged incoming nonself edge records, define:

- **Covered:** a development node has at least one permitted TRAIN-label source neighbor.
- **No visible TRAIN neighbor:** no such source exists.

At serving all permitted TRAIN labels are available, with scale one. Do not use a sampled training Q, class correctness, learned attention threshold or outcome to define coverage. Report supports and the same before/after metrics and repair flows on both cohorts, as well as the whole population. Duplicates affect weights but not the existence test.

For every arm, no-neighbor correction must be zero and its predictions must equal its **own same-state** native predictions. Nontrivial violations indicate a state/operator/diagnostic error. Across arms this cohort can still differ because their selected native epochs differ. Covered nodes need not receive useful label evidence: actual labeled attention mass can be tiny, and class compatibility can be misleading. Report that mass continuously; no favorable mass bin becomes a success gate.

## 3. Predeclared prediction and error diagnostics

For C4/U4, report:

- Top1 disagreement, correctness disagreement, any-member-correct coverage, pooled accuracy minus mean member accuracy, and mean-member minus pooled NLL.
- **Pooled-only rescue:** all members have wrong top1 but probability pooling is correct.
- **Pool harm:** some member has correct top1 but the pool is wrong.
- All-member-wrong nodes with a **strict common wrong rival**: some class c outranks truth in every member. An unchanged convex probability mixture cannot rescue that rival. All-members-wrong alone is insufficient for this conclusion.

S_joint4head has **one predictor**, not four prediction members. Its four hidden attention heads cannot be counted as oracle member coverage; the same applies to S_one_path.

For each corrected arm versus its same-state native baseline, count native-wrong→correct repairs, native-correct→wrong introduced errors, wrong→different-wrong churn and persistent errors. Repairs minus introduced errors must equal the full-population correctcount change. Retain the analogous co-primary across-arm flows, explicitly noting their different selected native states.

Use the saved operator `delta_m=T_m g_m`: inspect row-centered compatibility T_m, actual weighted class histogram/label mass g_m, and changes in target-versus-rival margins. Different attention among the same anchor class can leave g_m unchanged. Different latent factors/embeddings can preserve T_m; T_m or histogram differences can also be decision-invisible. These quantities explain failures, not replace prediction quality.

## 4. Interpretations that reject or narrow utility

- Failure of either frozen co-primary whole-family comparison or its safeguards rejects promotion of shared-factor utility. No covered cohort, seed, latent distance or alternative epoch rescues it.
- Same-state correction gains matched by S_joint4head/U4 support generic label-aware correction, not a unique benefit of factor sharing or four separately served routes.
- Stronger members with unchanged coverage/pool benefit support supervised regularization more than useful complementarity. Extra disagreement with more harms or worse risk is not success.
- An advantage explained by native-epoch selection does not establish corrective evidence learning. Uncovered-node across-arm advantages necessarily come from native state/selection, not this one-hop label correction.
- Absence of label reach is an information-interface limitation. Failed competitor margins with present context can implicate attention/compatibility estimation; endpoint diagnostics alone cannot identify weak TRAIN feedback, gradient interference or a causal mechanism.

A positive first screen still needs a true independently acquired GNN ensemble, capable label-aware references and unused confirmation. Any distinct next method must follow complete findings and receive a new prospective test. This plan prescribes no rescue, coefficient, denominator or basis change.

## 5. Common-mask algebra is training context, not endpoint evidence

Root's saved finite-population calculation is consistent: at fixed pre-update state and a TRAIN target in Q, `Cov(delta_m,delta_n)=N(N−t)/t × S_mn` for `N=579,t=290`, including zero non-neighbor contributions. Cross terms need not be positive or symmetric. For **own-route CE in logits**, Jensen gives `E_Q CE(z_m(Q),y) >= CE(E_Q z_m(Q),y)`; it does not establish a bound for the served probability-mixture CE or useful specialization.

Selected development serving is unmasked. Do not call that training-mask covariance measured endpoint diversity or calibrated uncertainty. No mask resampling or extra TRAIN-query intervention is included in this bounded collector.

## Compact retained outputs

Three summary tables suffice: same-state correction metrics/flows by seed-arm/cohort; co-primary native-epoch/correction decomposition; route complementarity and harms. Preserve individual seed values/supports and work. Any necessary per-node logits/weights remain in the authorized research repository; avoid duplicate large local archives. No new significance or subset-performance claim follows.
