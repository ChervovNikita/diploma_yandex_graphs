# P: SeHGNN feasibility and prior assessment of the typed tangent proposal

## Decision

**Defer a complete initializer pilot.** The native source eliminates the proposed missing-information rationale for SeHGNN. An attributed covariance/tangent allocation hypothesis remains possible, but no measured SeHGNN channel/tangent deficit currently supports spending a complete pilot on it. Root confirms that the finished attention cohorts have not yet received comparative decoding; completion supplies no scientific finding. No new mechanism is proposed here.

The original proposal packet remains immutable. This assessment neither reopens closed negative candidates nor changes a source, recipe, selector or gate. No outcomes, checkpoints, numerical providers, models or servers were accessed.

## What native SeHGNN actually does

The pinned HGB source is `sehgnn_IMDB_TRAIN_VALID_native_reference_source_20261009_v2/model.py`, SHA256 `0948239c4c4d06dd258cd106fd2fa7b6aaa2a62688fe70dd662413fa1eae532a`, from author commit `e92bd37d0b803457339555684f139b4c8f3e160d`.

- Lines 126–140 define per-metapath projection, joint `[num_channels, hidden]` LayerNorm, PReLU, semantic attention and a `num_channels * hidden -> hidden` concatenation map.
- Lines 57–68 retain a tensor with one output per token: attention contributes `gamma * (beta @ h)` and the original token tensor is added as a residual. Gamma initializes to zero at lines 30 and 49.
- Lines 211–217 stack every feature/TRAIN-label token, project it, apply semantic fusion, transpose and **concatenate** it before `fc_after_concat`.

Consequently, an across-token contrast is already accessible to the learned concatenation map. At gamma zero the native semantic path is an identity on all projected tokens, not their mean. The original proposal's symbolic `ker(L)` argument applies to an actual linear sum bottleneck; it cannot be invoked as information recovery for this SeHGNN source. Subsequent compression to hidden width does not establish that the native map loses the particular label-relevant directions proposed for private allocation.

SeHGNN also precomputes typed/meta-path neighborhood means. Private factors in its semantic processor do not acquire new raw neighbors, recover within-meta-path neighbor distributions discarded during preprocessing, or create learned edge transport. Those would require a different information path and a different proposal.

## What remains of the mechanism

The original hypothesis uses rank four per member: three common-sum activation directions and one member-assigned direction from covariance projected onto across-token contrasts, with outgoing factors zero. It changes which initial outgoing-factor derivatives are available. It does not change the initial predictor. Once incoming factors train, the contrast restriction need not persist.

This is a **type-conditioned starting basis**, with three material limitations:

1. **Equal latent width is not semantic coordinate alignment.** Native tokens pass through independently learned per-metapath maps. The proposed sum/difference compares matching hidden coordinate numbers without demonstrating that they have the same meaning across tokens. In the deterministic gamma-zero map, separate within-token hidden-coordinate permutations can be applied to the last projection and corresponding LayerNorm affine parameters and compensated in the concatenation map. Contrast allocation can therefore depend on a function-equivalent coordinate choice. This statement uses dropout off; it is not a pointwise fixed-dropout replay claim. The full trained attention model is not claimed to admit arbitrary independent rotations. The source-initialization example questions an intrinsic graph interpretation; it does not disqualify every competent warm state with learned attention. Such a state would need its own justification for the proposed token-coordinate alignment.
2. **Variance is not competence.** The paper-style second moments are uncentered; native means, scales, label-channel frequencies and metapath redundancy can dominate them. Three leading common directions provide no label-information floor. Assigning the first through fourth contrast eigenvectors can give members unequal useful signal. A repeated eigenvalue also leaves the chosen member directions dependent on a basis/tie rule.
3. **Target alignment is broader than graph specificity.** Independently permuting target rows per contributor tests paired multi-view alignment. That principle applies to ordinary multi-view inputs. A positive aligned-versus-permuted result alone would not establish graph-specific information acquisition, graph geometry or a new ensemble principle.

The native tokens include propagated TRAIN-label channels where enabled. Covariance estimation can avoid using heldout labels, but the resulting initializer should not be described as label-free when those native inputs already contain TRAIN-label information.

## Nearest prior comparison

| Prior, reused at its saved scope | What it already establishes | What it does not establish here |
|---|---|---|
| HGEN, IJCAI2025, main §§3.1–3.4/Eqs1–9 and saved bounded author source | Different meta-path neighborhoods, separate feature encoders/graph learners, within-path attention and fused predictions; direct graph evidence ensemble ancestry | That four starting covariance directions on one shared native processor supply stronger or complementary members |
| EVA, `2410.07170v1`, saved pp3–5 | Activation-SVD incoming LoRA directions with outgoing zero; variance-based rank allocation | That the proposed common/contrast split or graph-token assignment improves competence |
| CorDA, `2406.05223v1`, prior packet §§3.1–3.4/Eqs1–5 | Context second moment, SVD(WC), inverse-based reconstruction and function-preserving top/bottom low-rank weight splits | A graph ensemble, this trainable shared-core branch, or an error-complementarity guarantee |
| LoRA-GA; How to Train a Shallow Ensemble, saved gradient/covariance scopes | Task-gradient orientation, centered head covariance and orientation-versus-magnitude distinction | Graph-type allocation superiority; short warm states do not license a stationary posterior interpretation |
| Saved outgoing-zero nonlinear message/SVD and private native-block proposals | Graph dictionary initialization, placement before aggregation and partial private capacity are already open attributed hypotheses | Support for a new missing-information claim at native SeHGNN concatenation |

HGEN is a closer collision for the *goal* of useful graph evidence ensembles than the proposed initializer is a demonstrated solution. Its separate learner ownership also explains why it cannot be used to certify competence of a small shared-core adapter. No HGEN result, theorem or implementation guarantee is transferred.

There is no need for another primary retrieval: saved SeHGNN scopes include the complete main projection/fusion method and its complete HGB source; HGEN, EVA, CorDA and the covariance work cover the claims under assessment. This packet adds **zero primary method scopes, zero full-paper reads and zero public requests**. Selected already-scoped native source lines were reopened for the concrete architectural decision.

## Is a competent teacher-free pilot technically possible?

**Possible in principle; not source-ready or resource-qualified.** The initializer requires one common native warm state and FIT token activations. It does not require independent teacher predictions, teacher embeddings, teacher training or distillation. A prospective native single can supply that common warm state at a predetermined step, without selecting it for contrast utility. Genuine independent4 remains a competence comparator, not an initializer teacher.

The native IMDB interface has 37 tokens of width 512 and a flattened width of 18,944. A dense concatenated covariance has dimension `18,944 x 18,944`; it should not be assumed cheap merely because the adapter rank is four. Matrix-free covariance products or an algebraically equivalent sample-space eigensolver could avoid materializing it. Their exact precision, degeneracy rules, convergence, charged cost and role permissions still require source review and qualification. No resource estimate is an observed runtime result here.

There is also a location issue that a concrete source proposal must resolve. A standard additive adapter on `fc_after_concat` consumes **post-attention** flattened tokens. The originally described initializer consumes **pre-attention** projected contributors and adds a width-512 correction later; that is a separate skip branch. Those are different mechanisms after gamma becomes nonzero. The native flattening follows a transpose, so the common/contrast projector must use the actual feature-coordinate order. No silent hook substitution is qualified by the current paper design.

Every fitted comparison would need identical branch location, rank, inputs, row scale, warm state, factual objective, FP32/normalization policy, continuation, selectors and deployed pooling. An unconstrained activation-covariance initializer and a strong same-information single remain necessary. A same-information single should receive the total bank capacity; collapsing linear adapter sums at a fixed shared input remains possible. Adding an ensemble label does not make four incoming dictionaries necessary.

## Observations and the condition for revisiting

The saved IMDB negative readout measures excess label3 false positives in roles1/2 and weaker served quality, with little assignment-specific deployed effect. The molhiv and private-sheaf readouts measure weak members or insufficient useful pooling despite acquired corrections or repaired learning. These observations justify separating competence from diversity. They do **not** identify a deficient SeHGNN contrast tangent, establish cross-token coordinate alignment, or show that an initializer would repair the actual errors.

First decode the already completed attention comparisons under their frozen criteria. If competent shared native members do not exhibit an unresolved deficit, or an existing attention change already addresses it, defer this initializer. If the diagnosis is an implementation/precision/normalization failure, repair that failure before considering a new scientific initializer.

A concrete reason to revisit would be: **at the same predetermined competent native warm state, a single prospectively fixed FIT-only directional assay shows a useful task-loss descent component in the proposed contrast subspace that the matched global residual subspace underrepresents, and the advantage disappears when only targetwise contributor alignment is broken.** Check that conclusion under a predeclared coordinate-equivalent representation, with identical tangent scale and parameter budget. No layer, direction, rank split, seed, horizon or arm is chosen from the assay. This is evidence for a bounded allocation pilot, not evidence of heldout gain or ensemble necessity. The assay itself has not been run or authorized by this assessment.

At present, even that limited observation is absent. **Decision: defer the complete pilot pending actual comparative diagnosis; retain no causal assignment from the aggregate errors.**
