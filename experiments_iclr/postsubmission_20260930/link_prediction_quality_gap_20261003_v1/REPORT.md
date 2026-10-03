# Link-prediction quality gap and one conditional extension

3 October 2026. Source-only research assessment. Literature memory **index_v31** was consulted before retrieval; its 130 conclusion records are not 130 full-paper reads. This assessment made no public requests, new primary-paper reads, imports, fits, dataset access or outcome-dependent choices.

## Decision

**A fixed shared BUDDY cache limits which query evidence any of its members can use. It does not imply that four nonlinear predictors cannot improve ranking.** The prepared factorized BUDDY arm already has four private nonlinear trajectories, rather than only four scalar heads on one learned representation. Independent BUDDY predictors can reuse the same deterministic cache. Its current comparison therefore studies a useful architecture and cost question; it supplies no new graph-learning principle or measured quality result here.

**Retain one conditional hypothesis for source qualification: preserve the association between a member's NCNC completion weights and its own nonlinear decoder, instead of pooling those weights before decoding.** The proposed operation is attributed to NCNC neighborhood completion and BatchEnsemble parameter sharing. It changes query-dependent evidence before the final score and is distinguishable from final-head diversity. A representative paired collab validation is scientifically warranted *if* the precise private-versus-pooled implementation is qualified. This is an unadopted composition, not a claim of a new completion primitive, a previously unpublished complete method, superior predictions or an accepted paper. No seeds, arms, freeze, driver, allocation or experiment have been adopted.

The reason to retain this hypothesis is a concrete information–decoder pairing that pooling removes, with a direct matched comparator. The reason is neither current GPU availability nor an assertion that familiar ingredients automatically preclude a contribution. A positive result would first support this explicitly attributed architecture comparison. Methodological novelty and broader LP merit would need separate assessment.

## 1. What the current shared representation permits

Let a query be q=(u,v), the supplied graph and features be (G,X), and the deterministic cached query input be

    phi(q;G,X) = [sketch-derived structural counts,
                   their degree normalization,
                   cached endpoint feature vectors].

Every admitted BUDDY member receives these inputs. The native sign_k=0 preprocessing still performs one fixed normalized feature propagation. The active source has no trainable graph pass, learned node embeddings or RA input. A learned endpoint projection and its endpoint Hadamard product occur *inside each member*, followed by private nonlinear and normalization paths [1,2].

### Information limit

If phi(q)=phi(q'), every deterministic evaluation member h_m(phi) returns the same score for those queries. An arbitrary number of heads, different losses or new initializations cannot distinguish them without another input. Structural counts do not expose the feature vectors of the actual common neighbors as such; fixed propagated endpoint features do not generally identify the joint allocation of features to overlapping neighborhoods. NCN/NCNC, LPFormer and PENCIL explicitly access such query-conditioned information [3–5].

This is an information boundary, **not evidence that useful exact cache collisions occur on collab**. No such collision, missing-feature benefit or hash-error burden was measured. The sketches themselves are already a substantive query-structural remedy for ordinary node-embedding limitations. Improving a weak feature-only GNN is therefore insufficient evidence against BUDDY. Any remedy must also show benefit over a competent single predictor allowed the same extra query evidence.

For a different architecture with one shared *learned* z(q) followed by four affine scalar heads, mean raw scores are exactly one affine head on z. Nonlinear heads can enlarge its function class; separate member losses can alter its training even in the affine case. Neither architecture can restore distinctions already erased in z. These terminal-head statements must not be attributed to the prepared all-map factorized BUDDY port.

### Function-class and optimization limits

The prepared port uses each learned linear map as

    W_m = diag(s_m) W diag(r_m),

with shared bias and private BatchNorm affine/running states. Every native learned map is modulated; endpoint products and nonlinearities remain member-resolved [2]. For a single map with nonzero reference entries, W_m[i,j]/W_n[i,j] factors into a row term and a column term. Thus it cannot realize arbitrary independently chosen member matrices of that map. This is a per-map sharing restriction, not an impossibility theorem for the effective multistage predictor. Independent models are the relevant stronger sharing control.

Identical member factors/states, the same deterministic inputs, the same loss and identically applied randomness preserve member symmetry under a symmetric optimizer. Different sign initializations, dropout and private normalization can break it. Parameter differences alone establish neither complementary correct rankings nor calibrated uncertainty. The active source uses distinct sign factors and private normalization, so exact symmetry is a limiting example, not a diagnosis of its trained behavior.

For binary scores z_m and their mean z_bar, the saved objective analysis gives

    mean_m BCE(z_m,y) = BCE(z_bar,y) + D,
    D = mean_m softplus(z_m) - softplus(z_bar)
      = mean_m KL(Bernoulli(sigmoid(z_bar)) || Bernoulli(sigmoid(z_m))) >= 0.

The training source averages member BCE while serving mean raw logits [2,6]. This includes a score-agreement Jensen term. It can change optimization and regularization; it does not prove harmful collapse or justify removing it. The sampled BCE target is also different from the official pooled Hits@50 negative-tail ranking target. BCE improvements, embedding separation, member disagreement and oracle unions cannot replace the served official metric. Changing the objective would be a separate intervention and is outside the retained hypothesis.

## 2. Competent collab comparisons

No author score, leaderboard number, runtime or superiority claim was adopted. The following roles follow retained method conclusions and the existing static source qualification, not reproduced predictive results.

| Family | Evidence supplied to its query score | Role and remaining limits |
| --- | --- | --- |
| BUDDY/ELPH [1] | Sketch-derived neighborhood counts and endpoint features; BUDDY precomputes fixed inputs | Native-width BUDDY and same-cache independent BUDDY are necessary. The latter receives the same preprocessing reuse. The active width-1024 reference is a disclosed full-TRAIN/TRAIN-only-test adaptation. |
| NCN/NCNC [3] | Embedded common-neighbor sums; NCNC also scores missing links to weight residual neighbors | Direct attribution and strongest local mechanism reference. The pinned collab recipe is identifiable; environment, runtime and reuse permission are unqualified. NCNC completion scores are detached in the inspected native source. |
| LPFormer [4] | GCN features, target-pair context attention, learned PPR-relative encoding and structural counts | Relevant pair-conditioned capacity prior. Saved scoped primary conclusions only; source was not qualified. Strong/SOTA claims remain unverified. |
| PENCIL [5] | Sampled query-subgraph node/adjacency/role tokens, bidirectional attention and reconstructed-adjacency propagation | One recent 2026 competence reference. Pinned native collab source/config is available conditionally. It is not a BUDDY cache drop-in, and its published advantage is not established here. |
| Link-MoE / LP stacking [7,8] | Multiple expert/model scores and supervised combination | Direct prior for structure-conditioned routing or changing the pool. Link-MoE's collab gate consumes part of official validation for training, so that supervision must be matched or explicitly separated. Neither is the selected extension. |

### Protocol differences cannot disappear inside one score table

The existing source report [9] records:

- **Graph access:** active BUDDY uses complete weighted TRAIN topology and TRAIN-only TEST input. Native NCN/NCNC uses complete TRAIN, discards weights, and its README flag adds VAL for TEST. Native PENCIL filters TRAIN records to year >=2007 and unconditionally adds VAL for TEST. Full declared nodes remain. Native community recipes and a complete-TRAIN/TRAIN-only comparison are separately named settings.
- **Target removal:** native NCN/NCNC removes the whole training-positive minibatch before graph encoding and completion. PENCIL samples a query subgraph first and removes that query edge afterward. Target-presence effects on sampling must be disclosed. Active BUDDY's fixed native cache has its own training-edge convention.
- **Features/exposure:** NCN uses provided features. PENCIL defaults to structure-only and has an author-supported feature option; its YAML uses roughly half of filtered positives per epoch. NCN drops the shuffled incomplete last training minibatch. Changing either exposure is an adaptation.
- **Selection/heldout:** native NCN and PENCIL inspect TEST each epoch. A prospective wrapper must preserve validation Hits@50 selection and an explicit tie rule, persist selected weights, and open TEST only after the complete comparison is locked. Official shared negative pools and strict greater-than Hits ties remain unchanged. HeaRT is a distinct query/metric setting.

A broad LP-quality claim would require NCN/NCNC and the recent PENCIL reference in a clearly declared setting, in addition to the BUDDY controls. A narrowly scoped sharing study may begin with the causal twins below and native NCNC, but cannot label that screen state of the art. Independent members must receive competent packing and deterministic cache reuse wherever their own architecture permits it. Every encoder, candidate score, sparse aggregation, backward path, selection, cache, fit and serving pass is paid work.

## 3. The single retained hypothesis

### Member-resolved neighborhood completion

Start from the native depth-1 NCNC operation, rather than adding a new scalar head to the BUDDY cache. One shared learned node encoder produces h on the current minibatch-masked TRAIN graph. Four decoder branches use shared learned decoder matrices with private rank-one factors and normalization states. Each branch retains its native endpoint product, feature-residual transformation, base scorer, clamped completion weights and final nonlinear decoding. This is a prospective architecture, **not a port already prepared or an exact native independent-NCNC reproduction**.

For q=(u,v), let C(q)=N(u) intersect N(v) and R_u(q)=N(u) minus N(v), with R_v defined symmetrically. Write h_tilde_m for the native member-resolved aggregate features (including its xlin residual). The missing-link weight for w in R_u is

    w_m(q,w) = kappa(stop_gradient(a_m(v,w))),

and analogously for R_v. Here a_m is that member's depth-0 native scorer with its prescribed recursive feature transformations. Kappa is the *existing NCNC* clamp:

    p0 = sigmoid(scale * (a - offset)),
    kappa(a) = alpha * pt * p0 / (pt*p0 + 1 - p0).

Use the fixed native pt convention initially; a learned pt would be another change. The native source accumulates actual common-neighbor embeddings and the two weighted residual-neighbor sums, then applies the member's nonlinear decoder [3,9]. All candidate sets and base-scoring queries must use the same masked graph. Stop-gradient applies to the completion scorer as in the native depth-1 path; it must not silently remove the gradient through downstream aggregate features.

**Proposed changed operation:** preserve w_m inside member m's aggregation/decoder. **Direct control:** compute the same four candidate scorers, apply kappa separately, then give each member the same w_bar=mean_m w_m for that candidate. Keep its own h_tilde_m and downstream decoder unchanged. Pool *weights after the clamp*, rather than clamping average scores. This avoids changing two operations at once. Both twins pay for all four candidate-score paths; a cheaper deployable common scorer is a later cost comparison, not the causal twin.

The scientific hypothesis is that a member's score-conditioned completion weights sometimes identify neighborhood evidence its decoder can use better than common weights. Pooling can instead denoise inaccurate completion scores, so the private version can lose. A shared encoder may already erase useful feature distinctions; completion cannot reconstruct feature information absent from h. A differing completion weight is not a verified missing edge, posterior probability or calibrated uncertainty.

### Why this contrast is distinguishable, and what it does not establish

In a simplified limit with common aggregate features h_w and affine member decoder A_m, the private-minus-pooled score contribution is exactly

    mean_m sum_w (w_m(q,w)-w_bar(q,w)) * A_m h_w.

This is the association of each completion weight with its decoder response. It vanishes when all decoder responses are identical. With four members, one scalar residual feature h_w=1, weights (.8,.2,.8,.2), decoder responses (1,0,1,0), and no base term, private pairing gives .4 and pooled weights give .25. This arithmetic witness has no graph labels and makes no claim about which score is better. Native nonlinear maps and private features provide further ways the predictions can differ.

The complete member bank can be written as one deterministic grouped network producing the identical pooled score. A single model allowed the entire same candidate bank can preserve this association too. Therefore **the contrast does not prove a new ensemble function class or information advantage over every capable single predictor**. Its potentially useful contribution is the measured quality/cost effect of preserving that association under the specified parameter sharing and training. An exactly equivalent grouped implementation is an accounting/correspondence check, not a second scientific method to fit.

NCNC already introduced learned neighborhood completion. BatchEnsemble already introduced rank-one sharing. Native independent NCNC members already retain their own completion/decoder association. Preserving it in a shared decoder composition is a precise proposed delta relative to a common-completion design; the sources inspected here do not certify that the complete composition is new. This is stronger than attributing novelty to a final BE head, while remaining an attributed, testable architecture hypothesis.

## 4. Conditional representative validation

First qualify the exact recursive source semantics, member axis, stop-gradient boundary, normalization/dropout states, masked candidate sets, pooled-after-clamp control, selected-checkpoint replay and source reuse status. Verify the equality limit when the four member branches are identical and the affine common-decoder reduction above. These are implementation requirements for a later package; no model tests were executed here.

If qualified, one representative complete-collab development comparison should retain:

| Required comparator | Question resolved |
| --- | --- |
| Private-completion factorized twin versus pooled-weight factorized twin | Does retaining the weight/decoder association improve the *served* ranking, with the same capacity, candidates, objective and paid scoring work? |
| Native capable single NCNC and one prospectively specified capacity/cost-capable single | Is the gain attributable to ordinary structural capacity or a better single model? Exact parameter count alone is insufficient cost matching. |
| Independent ensemble of the same completion architecture, with independent encoders/decoders/fits and competent packing | Does parameter sharing retain useful independent-member quality at the measured total budget? Four independent decoders on one learned encoder are not four independent predictors. |
| Existing native-width BUDDY / same-cache independent BUDDY, followed by recent PENCIL for broader claims | Are the resulting quality/cost values competent beyond the selected completion family? Existing frozen BUDDY settings stay fixed. |

Use the same complete official development graph/query/negative contract within each named adapted setting, and preserve native essential masking/recipe choices unless explicitly declaring another adaptation. Prebind the candidate enumeration, normalization, initialization, positive/negative streams, optimizer, member objective, mean-raw-logit serving pool, checkpoint budget/ties and full-family heldout barrier before opening comparison outcomes. Multiple paired seeds on the fixed temporal split describe conditional optimizer variation; they do not provide independent-graph or chronological-population uncertainty. Report all admitted fits and paired effects, including failures. No seed numbers, numerical continuation gate or new selected arm are frozen by this assessment.

The primary quality measure is official pooled Hits@50. Record its paired variability, every selected epoch, full cost and peak memory. Pooled/member BCE, completion-weight response differences, ranking-error complementarity and post hoc private-to-pooled counterfactuals are descriptive mechanism diagnostics; none substitutes for quality or provides a new selector. A useful gain must survive a capable single and independent control and be large enough to matter under the intended use. A null or adverse result closes this tested composition, not all possible GNNM extensions.

**Resource estimate: unknown.** The shared encoder may amortize one learned graph pass, but four recursive scorer paths, native member feature transforms and weighted sparse aggregations remain. Static source/config sizes are not timing or memory measurements. Qualification should measure the complete representative epoch and serving operation before any resource request. Current availability does not reject the hypothesis; no GPU request or launch recommendation is made by this packet.

## 5. Closed evidence and custody limits

The permitted completed Squirrel audit was read directly [10]: its arithmetic trigger was true, with only tens of micro-nats improvement and all selected states after one/four continuation updates. That supplies no practical/general mechanism confirmation. The filter packet's retained closure conclusions and binding file were reused for the PPI and complete35 relation-CP failures [11]; the original PPI/CP outcome files were not opened. They reject their tested recipes at their original gates, not the different LP completion composition. No successful-subset reinterpretation or cross-task benefit is inferred.

Source-only BUDDY contract/model/training passages and the saved pinned NCN decoder passages were read. Saved method conclusions were reused for BUDDY, LPFormer, PENCIL, Link-MoE and stacking; no paper was newly read in full or in scope. No public search or retrieval was necessary. No active run, cache, checkpoint, label/tensor, test array, raw outcome, remote host or model runtime was opened. Historical outcome-summary exposure occurred and is explicitly recorded. Prior auditor packages, indices, ledgers, freezes and manuscript scores remain unchanged. The active Information Fusion/graph-filter access question was not duplicated.

The packet's bindings distinguish retained conclusions from the narrowly revisited source passages. This bounded evidence supports the information limits and the conditional matched comparison above. It provides no predictive result, global novelty certificate, calibration guarantee, cost advantage or acceptance recommendation.
