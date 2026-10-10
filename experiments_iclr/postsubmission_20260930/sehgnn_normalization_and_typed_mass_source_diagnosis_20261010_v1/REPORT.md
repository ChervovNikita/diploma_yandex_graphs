# SeHGNN normalization: bounded source diagnosis

2026-10-10. Source-only author diagnosis for the pinned HGB IMDB SeHGNN commit e92bd37d0b803457339555684f139b4c8f3e160d, native model SHA0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a. No model/provider imports, arrays/checkpoints, current outcomes, jobs, remote mutations or source/roster changes. Earlier synthesis and factorized PubMed plan remain untouched.

## Conclusion

**Normalization removes some scalar degrees of freedom, but source does not establish empirical route collapse or the cause of the shared IMDB coverage deficit.** The final five-logit LayerNorm is a concrete class-calibration constraint in every pinned native predictor. Both genuine independent controls keep it and still acquire more correct alternatives. A normalization explanation specific to shared fast routes is therefore unproven.

Earlier adapter documentation already accounts for cross-channel feature LayerNorm, preserved shared heads/normalization, private buffers and zero-start semantic gamma. The newly explicit facts are output LayerNorm's common-shift/radial restrictions and the precise ownership distinction. They identify a bounded control question, not a causal remedy or new method.

## Actual placement and ownership

Representative native construction is hidden512,25 feature+12 label channels,2 feature-projection layers,4 task layers,5 IMDB Bernoulli outputs and no raw-feature residual. Exact native model: [model.py](../sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/model.py); constructor: engine.py186–194.

| Location | Normalization/parameter | Shared factor bank | Genuine plain/same-factor I4 |
|---|---|---|---|
| feature_projection.1 and.5 | LayerNorm([37,512]), native learned weight gamma and bias beta | Modules copied but both affine parameter objects are native slow objects shared across members | Separate native affine parameters in every independently initialized body |
| task_mlp.1.1,.2.1,.3.1 | BatchNorm1d(512,affine=False), default running statistics | No gamma/beta; running_mean/running_var/counter buffers are already member-owned | No gamma/beta; each body's buffers private |
| task_mlp.4.1 | LayerNorm(5,elementwise_affine=False) immediately after task_mlp.4.0 classifier | No output gamma or beta; stateless per-example operation, not across members | Same no-affine five-logit LayerNorm |
| semantic_fusion.gamma | Native scalar residual-attention gate, initialized0 | Shared native slow parameter | Independently owned/learned per body |
| semantic_fusion beta | Computed attention-softmax weights, not a learned norm beta | Can vary with each route's Q/K factors/input | Computed separately per body |

The adapter wraps **six sites only**: feature_projection.0/.4, semantic_fusion.query/key/value and fc_after_concat. Every native parameter, including task head, PReLU slopes, feature-LayerNorm gamma/beta and semantic gamma, is shared by identity. Only input_factor/output_factor are private. All registered buffers are deep-copied and ownership-checked. The final classifier has **no existing fast r/s factors**.

Grouped sites immediately precede whole-[channel,hidden] LayerNorm; this normalizes all37×512 coordinates jointly, not each metapath separately. Q/K/V sit inside residual attention after those norms. fc_after_concat feeds PReLU/dropout and three residual hidden-BatchNorm task blocks before the shared classifier/output LayerNorm. There is no direct normalization immediately after every fast site.

Exact adapter: grouped_member_factors.py18–21,121–124 and install_member_bank beginning267. Its real-arithmetic factor algebra and native ownership are in the two adapter forward methods and the installer. Ownership/Adam: bank_training.py232–264. Independent constructors and no-shared-storage check: independent4.py119–171.

## Mathematical invariances, with assumptions explicit

For a pre-normalization vector z, let u_e(z)=(z−mean(z)1)/sqrt(var(z)+epsilon), with variance over the actual normalized coordinates.

- **Common shift:** u_e(z+b1)=u_e(z) exactly in real arithmetic, for any epsilon. Subsequent fixed affine gamma/beta cannot recover that removed shift.
- **Positive common scale:** u_e(az+b1)=a(z−mean(z)1)/sqrt(a²var(z)+epsilon). Exact scale invariance holds at epsilon0 for a>0; with the actual nonzero default epsilon it is approximate when variance dominates epsilon. Negative scaling flips the centered direction, rather than disappearing. Zero/degenerate variance requires separate treatment.
- The radial derivative for delta=z−mean(z)1 is epsilon·delta/(var(z)+epsilon)^(3/2): potentially small, **not identically zero** at positive epsilon.
- **Coordinate-specific r/s:** generally change normalized direction. They are not reduced to a common scale/shift and are not automatically erased. Some changes can still lie in normalization or downstream null directions; source supplies no measured fraction of useful versus null updates.

The exact adapter algebra is Y=((X*r)W)*s+b with bias outside s. Uniform scalar output scaling a gives Y'=a(Y−b)+b, which is not az+constant unless the native bias is zero/constant over the normalized coordinates. Native biases start zero but can learn. Thus “all scalar factor changes cancel” requires assumptions even at the two immediately normalized grouped sites. Per-metapath scalar changes survive relative-channel normalization unless all normalized coordinates share the same scale.

Q/K gains can change attention temperature/weights; value scaling changes the gated branch relative to its skip. The native semantic gamma is exactly0 initially, making Q/K/V predictor derivatives initially zero for every native/shared/I4 model. Once gamma learns, that initialization fact supplies no persistent-collapse result. It was already documented in PRIVATE_SITE_JACOBIANS.md.

At the final nonaffine five-logit norm, each member's returned logits have zero class mean and variance at most1 in real arithmetic. Therefore each absolute logit is at most sqrt(5−1)=2; native sigmoid probabilities are bounded approximately.119–.881. A nonconstant member row cannot have all five logits strictly positive or all strictly negative. This is a **structural calibration/sign-coupling constraint**, not an observed error cause; no target-cardinality distribution or active-state variance was inspected. A pool averages sigmoid probabilities and need not retain the member logit constraint.

A hypothetical class-specific gain before the output norm can rotate the five-class direction; a common class gain/offset is removed approximately/exactly as above. A class gain/offset **after** that norm affects sigmoid confidence/thresholds directly. Existing fast factors are hidden/metapath/semantic coordinate factors upstream of the head, so neither hypothetical final-class placement is currently implemented.

## What completed IMDB does and does not support

The completed mask interpretation records shared1059 common-wrong/3 lost-alternative dependent role-events versus plain independent727/268 and same-factor independent746/247. Role3 mean-member F1 is higher than both controls while coverage is lower. Completed confidence bins include both error signs and599/1059 events with every member at least.10 from.5.

These observations do not identify the cause as shared gamma/beta, output LayerNorm, a classifier nullspace or calibration. Plain and same-factor I4 retain the same output normalization; their entire feature/task/norm weights and starting functions are independent, with different own selectors. The comparison does not isolate normalization ownership. Large margins do not prove that missing alternatives were latent in the features or that a bias/scale change would improve net F1/BCE.

Evidence: historical_imdb_mask_evidence_followup_20261010_v1/REPORT.md and the completed confidence COUNTS/TERMINAL pair. No current fixed candidate18 result was read.

## Known ancestry and one inactive control implication

Saved Edward2 DenseBatchEnsemble source has per-member input/output factors and ensemble bias (batchensemble_source.py470–609). Saved MIMO WideResNet source has ordinary shared hidden BatchNormalization and expanded input/output slots (mimo_model_source.py21–43,84–104); MIMO itself is not a private-normalization proof.

Private normalization affine is explicit prior art: saved Tiny Deep Ensemble scoped method says normalization beta/gamma differ by ensemble member while other weights are shared, including Batch/Layer/Instance/GroupNorm. Its pinned repository inspection found README-only implementation scope, so no exact training-loss/SeHGNN equivalence is claimed. Saved NENS scoped interpretation is another shared-backbone/member-normalization antecedent. These are reused bounded saved reads, not a new full-paper or absence audit.

**Only inactive implication:** if later authorized, isolate a unit-initialized class affine *after* the native output LayerNorm, comparing member-private versus shared affine ownership with capable single/untied same-placement controls. Initial gamma1/beta0 preserves the native function; M4 private freedom is40 scalar parameters versus10 shared. This tests output calibration/head freedom and an ownership interaction, not additional graph information. Generic private norm affine is known, not novel. Gains explained by a single/shared affine remain single-model calibration effects; exact competence, coverage, repairs and introduced harms would still be required. No implementation, grid, source/roster amendment or launch is proposed here.

## Related typed-mass information check

[LABEL_MASS_NOTE.md](LABEL_MASS_NOTE.md) derives the actual raw-multihot positive/mass relations. The three masses are not a universal simple sum of the native five positives under the permitted variable-cardinality source. The pinned loader requires at least one positive genre per observed movie, so all five positive channels zero already means zero observed mass in real arithmetic; a zero queried-class channel with another class positive already supplies known-negative support. Source alone cannot prove that the actual graph supplies new information rather than an explicit redundant/recoverable statistic. Current fixed controls remain applicable in either case.

[READ_BINDINGS.json](READ_BINDINGS.json) records exact source/interpretation paths and scope. Exploratory broad searches had truncated displays; no claim is made to have semantically read every literature-index field or source line.
