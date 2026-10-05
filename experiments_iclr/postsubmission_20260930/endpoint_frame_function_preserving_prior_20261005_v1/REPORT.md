# Prior check: equal initial functions with different tangent features

5 October 2026. One bounded prior check, requested before a predictive schedule. No experiment, server action, new agent, canonical edit or manuscript verdict.

## Conclusion

**The axis initializer does not establish a new initialization principle.** Function-preserving adapter branches with different Jacobians are already explicit in the retained LoRA-Ensemble implementation. ETHER+ supplies an even closer reflection-based neutral construction. A second primary method, HTA, already uses reflection–diagonal–reflection coefficient matrices.

The candidate's exact local distinction is applying the same reflection to both link endpoints before their coordinatewise product, and choosing different coordinate-axis normals so that initial products coincide while their permitted first-order changes differ. The reviewed sources do not state this complete pre-product ensemble recipe. This bounded search cannot establish its absence elsewhere. At present, it is a concrete combination and placement of established ingredients, with an untested utility hypothesis. It is not a defensible new general principle or an established methodological contribution.

## Closest retained ensemble operation

The latest index v59 already contains **LoRA-Ensemble: Efficient Uncertainty Modelling for Self-Attention Networks**, arXiv:2405.14438v5. Its retained author source, pinned at `812ced225717fdcce3c1e236e5dd17d14c2b1b2b`, is particularly decisive:

- `models/lora.py:69–105` constructs `A` and `B` and serves `W(x) + B(A(x))`.
- Lines119–153 provide Gaussian, Kaiming and Xavier choices for `A`, each with `B` initialized to zero.
- Lines206–233 construct separate adapters for the members and stack their states.

For a shared base map, every such adapted map initially equals `W(x)`. Nevertheless, its derivative with respect to `B_m` in direction `δB_m` is `δB_m A_m x`, which depends on the member's initialized `A_m`. The derivative with respect to `A_m` initially vanishes. Different adapter tangent features while preserving the initial map are therefore already an implemented ensemble mechanism. Private heads or other private operations can still make the complete predictors different initially; this is an adapter-map statement, not a full-network identity claim.

The relevant LoRA-Ensemble paper scope was already retained, including Appendix M initialization prose. This check adds a targeted read of the saved initializer source to answer the named unresolved question; it does not count a new paper or another full read.

## New scoped primary methods

### ETHER, arXiv:2405.20271v1

Read main §§3.1–3.4 and targeted implementation passages. The method defines one-reflection adaptation `H = I − 2uuᵀ` and the relaxed adapter

`H+ = I − uuᵀ + vvᵀ`.

For unit vectors with `u = v = q`, the relaxed transformation is exactly the identity. The paper explicitly describes this cancellation in §3.3. At that neutral point, its first-order variation is

`δH+ = (δv − δu)qᵀ + q(δv − δu)ᵀ`.

Thus neutral transforms can have different tangent directions according to the common normal `q`. This derivative follows directly from the displayed operation; it is not a reported ETHER theorem. The read passages do not certify that the authors actually initialize every ETHER+ pair equally, select different normals for ensemble branches, or apply the transformation before a graph-endpoint product. No such implementation claim is made here. The general neutral reflection-adapter construction is nevertheless prior.

§3.1 also describes OFT's zero skew-parameter initialization, yielding an identity Cayley transform. Orthogonal adaptation that initially preserves the base is therefore not a new idea either. The paper's preservation, learning-rate and efficiency claims are not transferred to our candidate.

### HTA, arXiv:2410.22952v1

Verified title: **Efficient Adaptation of Pre-trained Vision Transformer via Householder Transformation**. Read main §§3.1–3.3 and its short implementation-details passage. The proposed adapter explicitly has

`W_HTA = H_left D_H H_right`,

and adds this matrix alongside a low-rank update to a pretrained linear map. Each side uses a learned vector and `D_H` is a learned diagonal matrix. This is a direct prior for the reflection–diagonal–reflection sandwich, rather than merely an unrelated orthogonality title.

For fixed endpoint states, one linear readout of our reflected product is `aᵀ H D_c H b`. Its coefficient matrix is the tied-left/right special case of that published sandwich family. This local coefficient equivalence does not make the whole nonlinear NCNC ensemble identical to HTA, or establish that HTA uses an endpoint-product ensemble.

The printed HTA definition is `H = I − vvᵀ`, without the standard factor two or an explicit normal-length condition in the read method passages. For this displayed form to be a nontrivial orthogonal reflection, `vᵀv` must equal two. Choosing `v = sqrt(2)q` with unit `q` gives the standard Householder operator. The algebraic sandwich overlap is clear; the paper's claim that unrestricted learned vectors remain orthogonal is not independently qualified here. No implementation or result is adopted.

## What remains specific to our axis construction

At normal `v = e_k`, the reflection flips endpoint coordinate `k` at both operands, and the flips cancel in their product. Hence the initial feature vector is exactly `a ⊙ b`. For a tangent perturbation with `δ_k = 0`, the derivative is

`δp_k = 2 sum_(j≠k) δ_j(a_k b_j + a_j b_k)`,

`δp_j = −2δ_j(a_k b_j + a_j b_k)` for `j ≠ k`.

Distinct axes expose mixed-coordinate interactions involving distinct selected coordinates. This is an exact fixed-state property of this placement. It supplies no semantic alignment between a coordinate and a graph relation, no guarantee of useful learned differences, and no graph-quality theorem. Relabeling hidden coordinates also relabels which interactions the chosen axes emphasize.

The mechanism is available to an ordinary independent ensemble and to a single predictor supplied the same four frame products. Its existence therefore does not identify a benefit of shared-backbone ensembling. Those controls remain essential if utility is tested.

## Decision before predictive scheduling

Do not freeze a predictive cohort under a claim of a new function-preserving initializer or a new orthogonal/bilinear operation. A bounded utility screen can still answer whether this particular placement helps a competent shared model, provided the same-operation single and independent controls are frozen with it. A positive result would first establish adapter utility; a reproducible advantage over those controls would support an incremental empirical method-design result. This prior check alone supplies neither result nor a novelty certificate.

## Same-four-frame single: avoid delaying three frame gradients

The proposed capable single concatenates four frame-product vectors into one linear map. Initializing its blocks as `W0 = Wnative` and `W1 = W2 = W3 = 0` preserves the initial native outer score, but the derivative through frames1–3 is initially zero because their feature vectors are multiplied by zero blocks. Their map blocks can begin learning immediately; their frame vectors cannot influence the loss until those blocks change. This is an unnecessary initial disadvantage when testing whether one predictor can use all four frames.

The parent's deterministic alternative is justified as **one prospectively fixed control initializer**:

`W0 = Wnative`, `W1 = C`, `W2 = C`, `W3 = −2C`, with `C = Wnative/4`.

All four coordinate-axis frame products initially equal the native product, and the blocks sum to `Wnative`. Keeping the recursive branch on `W0` also retains its original algebraic map. The outer bias must occur once, and the concatenated linear map must precede the original normalization, dropout and activation; frame-specific perturbations before summation would break the cancellation. At this neutral point, frame m has derivative `Wm δp_m`, so all four vectors can receive first-order gradients. Nonzero blocks remove the structural zero-gradient barrier; they do not guarantee nonzero gradients on every example or parameter.

No RNG draw, fitted coefficient or outcome-selected choice is needed. This is a control construction, not another methodological contribution. Record its coefficients, native-weight donor and recursive use before any comparative outcomes, and use this one initializer instead of searching its alternatives. If an earlier predictive cohort were already frozen with the zero-block initializer, this choice would require a separately declared successor rather than silently changing that cohort.

The cancellation is exact algebraically, not a promise of identical floating-point summation. It also changes optimization geometry: the extra blocks have different signs and magnitudes, and four independently trainable blocks receiving duplicated initial features do not have the same effective update as one native block. Even the original zero-extra-block initializer has that latter issue, because those blocks still receive nonzero weight gradients. Preserve the declared optimizer/selection protocol and assess the control's competence; an underperforming inadequately trained single would not establish an ensemble advantage. The alternative removes one known handicap, without certifying a perfectly matched optimizer or whole-network capacity.

## Scope and preserved limitations

- First queried index v59; saved exact index hash and topical reuse records in `INDEX_REUSE.json`.
- Exactly three external OpenAlex metadata queries. Queries1 and3 returned largely unrelated work; they are preserved as failed retrieval strategies, not absence evidence. Query2 located ETHER and HTA among its leads.
- Exactly two newly retrieved primary method scopes, zero full-paper certifications. Parsing a source or retrieving a paper is not itself a read.
- Retained one local parser-import failure (`bs4` unavailable), before any network retrieval, then used the standard library without installing anything.
- Keyword locators exposed limited abstract/introduction, results and implementation snippets. A method-section TFLOPs table and an implementation hyperparameter table were also exposed. These are recorded as incidental source exposure; no numeric quality or cost outcome is adopted.
- No author-code retrieval for either new paper, native reproduction, complete proof audit, exhaustive prior search or outcome access.

Exact primary source hashes, passage custody and accounting are saved in `READ_SCOPES.json`, `PAPER_CONCLUSIONS.json` and the `primary` directory.
