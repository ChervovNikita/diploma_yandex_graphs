# PT additive bias coordinates: bounded control-design conclusion

11 October 2026. Scope: the sealed `private_continuation_PT_architecture_design_20261011_v2/ARCHITECTURE_BINDING.json` and the saved complete `pubmed_factor1_complete6_scalar_comparison_root_20261010_v1/REPORT.md` and `pubmed_normalization_closed_interpretation_20261010_v1/REPORT.md` only. No external reading, source/module imports, arrays, numerical/remote execution, partial outcomes, source edits, fit or manuscript review. PT dataset, optimizer, selectors and numerical pilot criteria remain unbound.

## Conclusion

**Yes: the current mixed-single/I4 comparison retains a material optimizer/reference confound. Prospectively include one fixed same-function bias-coordinate single/I4 control family in the first pilot.** This is a risk to an ensemble-specific explanation, not evidence that redundant biases improve PT or cause an observed result.

The candidate gate uses `z_m=h(c_m)+beta2+e_m`. Writing `e_m=mean(e)+delta_m` gives a common effective bias `beta2+mean(e)` plus route contrasts `delta_m`. Moving a vector from beta2 into every e leaves all gate functions unchanged. Softmax also has its usual common-logit shift invariance. The mixed single retains beta2 but removes e, so it removes both route differences and a redundant common-bias coordinate path.

Function equivalence does not imply optimizer equivalence. Even plain SGD on a simple duplicated bias `beta+q` changes its effective step versus one bias; coordinate-wise Adam moments/epsilon and aggregation of unequal route gradients need not preserve the same effective trajectory. No added continuation regularizer is required for this difference. The actual PT optimizer is not yet fixed, so no quantitative Adam advantage or necessary score effect is asserted.

The saved PubMed lesson is directly relevant in scope: factorized M1 reached 90.816844% versus 89.937426% ordinary native, while shared M4 reached 90.901404%; the reported shared-minus-M1 mean was only 0.084559 pp. A capable same-function reparameterized single reproduced most of the apparent shared accuracy gain. That history motivates this reference; it does not predict PT. The separate failed private-Adam convention control supplies no warrant to tune bias learning rates/epsilon or adopt a normalization grid.

## What the middle-state-mean control resolves

It retains the candidate's beta2/e coordinates, parameter objects, source, initialization law and decoder. Thus candidate versus middle-state mean is not confounded by merely adding/removing redundant bias coordinates. It tests the stated package of continued private histories, differentiated backward coupling and associated block2 computation changes.

It does not equip the ordinary mixed single/I4 with the candidate's redundant bias coordinates, show that private routes beat a capable same-function single, isolate forward history from backward coupling, or establish private-address necessity/novelty. The two controls answer different questions.

## One prospective reference family

Keep the existing mixed-single/I4 anchors. Add a **bias-coordinate mixed single** with the same two blocks, maps, beta2, head, Wo, one continued state and four tail-dropout TRAIN losses. Retain four e rows per block with the candidate's fixed `onehot(m)-0.25` initialization, but use only their arithmetic mean in every gate:

`z=h(c)+beta2+mean_m(e_m)`.

The mean starts at zero, so its initial gate and entire function class equal the existing mixed single; the extra coordinates do not create private histories. Retaining all 32 e coordinates also avoids choosing an arbitrary smaller duplicate-bias inventory. Give its I4 four independently fitted/selected complete copies with separate parameters, optimizers, RNGs and stops; source ancestors may be reused read-only with their acquisition charged once. Preserve each single's four-draw TRAIN law and one eval prediction.

Freeze this one construction, initialization, explicit optimizer groups/decay, selector, raw/member/confidence populations and full cost/closure rules before fitting. Use it as a capable reference, with no coefficient, coordinate-count, LR or epsilon grid and no outcome-dependent replacement. Equal-coordinate presence will still not match all candidate route gradients; it tests whether a single without private continuation can obtain useful quality through that coordinate package. Candidate improvement over both this reference and middle-state mean would support the bounded pilot package, not a pure causal or novelty claim. This report recommends a prospective roster decision only; it authorizes no source or experiment change.
