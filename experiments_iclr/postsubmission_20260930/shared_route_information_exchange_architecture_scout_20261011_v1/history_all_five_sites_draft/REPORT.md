# One acquisition proposal after fresh primary searches

11 October 2026. **Inactive research proposal; no fit, runtime qualification, server action, model import, numerical study, reviewer or manuscript edit.** The proposed change is activation-sign-dependent coupling of existing dropout masks in original-boundary shared SAGE GNNM. It is an attributed dependent-sampling operation, not a new antithetic primitive or a new prediction architecture. Root owns source admission and any execution.

## Why the question survives the closed failures

Complete Physics E loses 1.462764 accuracy points to genuine equipped I4, with mean members 5.174743 points weaker and 9,369 more pooling losses despite 4,379 more covered alternatives. Its endpoint increment over N is only 0.028158 points; that attribution gate failed. Penn internal sharing loses 0.979907 points to freshly acquired original boundary GNNM. The strong Physics source-derived meanSAGE single scores 93.058986%, versus E 91.326372% and equipped I4 92.789136%. These are whole procedures with different recipes. They do not establish that dropout causes the acquisition gap.

The affine, Context9/alignment, affinity/fixed-key and TRAIN-gradient studies remain closed at their recorded failures. The one sampled TRAIN-gradient diagnostic supplies no uniform cancellation explanation. The Rank-1 source and its 21/39/105 proposal remain parked. This proposal uses neither a learned factor-noise law nor the original-boundary × receiver-action combination being considered separately.

## A useful direction rejected as an architectural contribution

Fresh reads covered Graph-Tuple §4, DeGTA §3–4.3, statistical-signal-processing neighborhood aggregation §2–4, FOG §3, BGNN §3, PNA §2–3 and MM-GNN §2.2/4. Matched-neighbor cross-route covariance preserves correspondence that separate route means lose. But a capable PNA message function can recover it cheaply by

\[
\operatorname{Cov}(a,b)=\tfrac14\{\operatorname{Var}(a+b)-\operatorname{Var}(a-b)\}.
\]

PNA explicitly learns messages before mean/standard-deviation aggregation. FOG instead sums center×neighbor Kronecker products; BGNN represents different-neighbor products; MM-GNN computes coordinatewise multi-order moments with a moment adaptor. Graph-Tuple mixes ordered intra/inter-view convolutions; DeGTA combines structural/positional/attribute attention and local/global interaction. None of these reads licenses calling ordinary covariance pooling a new interaction principle. That candidate is not nominated.

## The refined operation

At each existing dropout site and each node/channel, let the current four scalar route activations be \(a_0,\ldots,a_3\). Detach these values for the mask decision. Keep the native retention probability \(q=0.8\), native inverse scaling \(B_m/q\), own-route CE/4, factors, initialization, optimizer, selector and deterministic raw-logit mean serving.

Enumerate the three perfect matchings: `(01,23)`, `(02,13)`, `(03,12)`. For each pair `(i,j)`:

* If \(a_i a_j\ge0\), draw one fresh uniform \(U\), set \(B_i=1[U<q]\), \(B_j=1[1-U<q]\).
* If \(a_i a_j<0\), set both masks to \(1[U<q]\).

The two pairs use independent uniforms, and different coordinates, sites and updates use independent fresh uniforms. Choose the matching minimizing the sum of its two covariance contributions \(2a_i a_j c_{ij}\); break exact ties by the listed order. No label, donor prediction, learned gate, extra loss, sampler gradient, teacher or fitted coefficient enters that decision.

For antithetic Bernoulli masks,
\(c_- = \max(0,2q-1)-q^2=-\min(q^2,(1-q)^2)\); for common masks, \(c_+=q(1-q)\). At `q=.8`, these raw-mask covariances are `-.04` and `+.16`. Each route's mask tensor is conditionally ordinary independent-coordinate Bernoulli(q), even though the joint route law depends on the current activations. At fixed current weights the own-route marginal stochastic forward and expected own-loss target remain the ordinary-dropout ones. **The shared optimizer trajectory and final model/route distributions are changed.** No equality of trained models is asserted.

For the arithmetic mean \(Y=\frac14\sum_m a_mB_m/q\), conditioned on the four current values,

\[
\operatorname{Var}_{\rm pair}(Y)-\operatorname{Var}_{\rm IID}(Y)
=\frac{2}{16q^2}\sum_{(i,j)\in\mathrm{matching}} a_i a_j c_{ij}\le0.
\]

The sign rule makes each summand nonpositive; choosing the matching gives the best of these three pair laws. This is not an optimum over all four-mask joint distributions.

Uniform balanced dropout is an essential rival, not an assumed successful intermediate. Its rule chooses `K=3+Bernoulli(.2)` and a uniform K-of-4 retained subset. It minimizes retained-count variance, but for unequal signed values it can increase variance of their mean. With fixed values `(1,-1,1,-1)`, IID mean variance is `.0625`, uniform balance gives `.078125`, and the proposed rule can pair opposite values with common masks to give zero. With `(1,1,1,1)`, uniform balance gives `.015625` and the pair rule `.046875`; thus the pair rule does **not** dominate uniform balance. These are analytic witnesses, not dataset measurements.

## The graph consequence and its limits

For a common fixed linear graph aggregation \(P\), independent source-coordinate mask draws imply that each target-coordinate variance of the arithmetic mean of linear messages is the sum of \(P_{vu}^2\) times source-coordinate variances. The local nonincrease therefore passes through that linear segment. This is a concrete protection against signed private-route responses making a negative-correlation rule harmful.

The source check matters. Native `TABMModel` drops hidden values after the boundary projector, before GELU, and at FFN linear1/linear2 in each of two SAGE blocks: five sites, each `34493×128` for the selected Physics architecture. It does **not** drop all 8,415 raw input features at those hooks. LayerNorm, GELU and residual additions intervene before later graph aggregation. The fixed-P lemma is therefore not a variance theorem for the entire unchanged native network. It also gives no reduction in gradient variance, Adam displacement, proper loss or prediction error. No cause is inferred from the Physics failure.

To obtain all four current activations, the candidate needs layerwise execution of the four native routes. Keep each native route operation and four own-loss backward calls in route order, with one optimizer step after them. Detached matching must create no gradient links among route graphs. A matched layerwise IID arm distinguishes mask coupling from forward/RNG scheduling. The exact original sequential/global-RNG recipe remains a quality anchor. Full-graph memory and numerical equivalence with fixed masks remain unqualified.

## Strongest checked ancestry

* **Ren, Zhao and Ermon, ICML 2019, Adaptive Antithetic Sampling**, primary PDF pp.1–5 and the method prefix of p.6: a learned joint Gaussian copula preserves marginals and minimizes estimator variance; it explicitly discusses integrand dependence and wrong-antithetic harms. This occupies adaptive marginal-preserving dependent sampling. The proposed finite sign rule is a restricted analytic instance, with no new primitive claim.
* **Differentiable Antithetic Sampling**, arXiv1810.02555v2 §2.2/4: moment matching under correct marginals is established; its approximate transform loses the exact unbiasedness guarantee. Our discrete pair law uses exact Bernoulli marginals.
* **Multi-Sample Dropout**, arXiv1905.09788v3 §II: multiple stochastic copies share parameters and average losses; deterministic inference needs only one copy. Its late-dropout CNN cost advantage cannot be transferred to whole SAGE copies.
* **Standout**, NeurIPS2013 primary PDF pp.1–3 (§2 and the main §3 algorithm): activation-dependent dropout is established. It changes retention probabilities and learns/ties gate parameters. Here q stays fixed and only dependence among route masks changes.
* **Fraternal Dropout**, arXiv1711.00066v4 §2: shared stochastic copies plus a prediction-consistency penalty. That objective is absent here.
* **Optimal Transport Dropout**, arXiv2609.33377v1 §3.1–3.2: a learned within-mask transport and Energy Score alter the perturbation and prediction objective. That preprint does not establish the present native own-CE procedure.

The primary ACM2006 dependent-rounding paper, the ACM2024 cross-view-confluent paper and AAAI2026 active-state-selection paper remain unread after their recorded failed primary access. GENN remains unresolved in the inherited audit. The search/read scopes are exact bounded custody, not a worldwide absence certificate. Combination and a local variance lemma do not establish novelty; the possible contribution is an acquisition operating point that survives capable controls.

## One complete representative study, inactive

Use full Physics and the same ten fixed class-stratified blocks `8101,8219,8363,8501,8663,8819,9001,9199,9403,9623`: 100 TRAIN,150 SELECT,150 CAL,34093 evaluation nodes per block, common support530417, all graph nodes retained. These populations have already been released. This is explicitly encountered-graph exploration, not unused confirmation. No split, class, seed, threshold or pool is chosen from its new outcomes.

The ten procedures are: exact native original O4; matched layerwise IID O4; uniform-balanced O4; sign-pair O4; original-equipped M1; that M1 with four IID stochastic copies; that M1 with four sign-pair copies; genuine four separately optimized equipped bodies with IID draws; the same four-private-body opportunity with sign-pair draws; fixed source-derived meanSAGE M1. The last is the already closed exact population reference and may be reused only with matching role/source/selector custody. The nine new procedures have15 optimizer bodies per block. This is a mechanism/control roster, with no coefficient or duration grid.

All original-family procedures preserve the native 5000-update ceiling, patience300, native off-cadence checkpoint rule, original initialization, lr.001/AdamW/zero weight decay and native deterministic logit contract. Four-copy M1 owns one parameter set and one deterministic served prediction. Genuine IID I4 owns four complete parameter sets and four own selectors; sign-pair I4 also has four separate optimizers but coupled randomness, and must not be labeled statistically independent acquisition. If an I4 body stops early, its last acquisition state may supply a forward-only partner until the other bodies close; those extra forwards are paid work. No stopped body receives another update.

Primary acquisition-utility gate: sign-pair O4 must gain at least `.25pp` served accuracy against matched layerwise IID and exact original O4, at least `.10pp` against uniform balance, and at least `.25pp` mean actual member accuracy against matched layerwise IID. All four paired one-sided zero-null tests must pass Holm alpha.05; report full vectors, ordinary/Bonferroni simultaneous95% intervals and all1024 sign-flip assignments as symmetry sensitivity. Ten overlapping populations of one graph are not ten independent domains. Runtime failure or an incomplete roster blocks a complete positive verdict; no survivor subset or automatic retry follows.

Report all capable M1/I4/source-port accuracy and proper losses regardless of the primary gate. A benefit reproduced by four-copy M1 or four-private-body I4 limits the sharing-specific interpretation. Beating only a weaker adaptation does not establish a useful shared operating point against the stronger references. If the sign rule fails the uniform-balance increment, close its sign-adaptation claim. If served accuracy or members fail, the acquisition hypothesis fails even with a favorable local mask variance.

Use the actual selected four members for mean/worst accuracy and NLL/Brier; report covered alternatives, pooling losses, aggregation-only correctness, repairs/harms and the full-population count identity. These are diagnostics, not selectors or cohort rescue gates. Report raw and separate-CAL pooled proper losses. Reject an accepted acquisition result if calibrated NLL mean harm exceeds `.02`, or any block harm exceeds `.05`, versus exact original, genuine IID I4 or the source meanSAGE reference; raw NLL must not worsen matched layerwise IID by more than `.02` on average. These are practical protections, not statistical noninferiority.

After all native selections close, one positive scalar per new bank is fitted for500 CPU-float64 Adam updates on CAL only, at lr.01 and fixed final endpoint. It temperatures the **native pooled raw logits**; there is no member probability-pool substitution or temperature-selected checkpoint. Report complete raw/calibrated accuracy, NLL and Brier. Existing source-reference calibration is reused as declared, with no TEST-chosen recalibration.

### Paid work and admission limits

Maximum new work: **90banks/150unique optimizer bodies/90scalar endpoints**, plus the10 existing source-reference banks/states/endpoints. The result panel totals100banks/160bodies/100endpoints; archived costs remain separate from incremental costs. The original-family ceiling gives750000 optimizer updates,1650000 route/copy TRAIN forward-backward opportunities and1350000 deterministic SELECT route opportunities. Calibration adds45000 fixed scalar updates. Failure, qualification, forward-only stopped partners, collection and idle owner time must be recorded.

Five cached uint8 four-route mask bitsets would occupy22075520bytes at this hidden width; five detached four-route float32 activation copies would occupy353208320bytes if materialized. These are tensor-size calculations, not CUDA peak estimates. Joint route tape retention can cost substantially more than native sequential acquisition. No isolated speed, memory or FLOP benefit is predicted from the existing overlapped Physics timings. Root needs representative full-input qualification and positive finite wall/CUDA/RSS bounds before admission. This packet supplies a complete scientific question and bounded update opportunity, not ready-to-launch source.

`READ_SCOPES.json`, `SEARCH_SCOPES.json`, `SOURCE_RECEIPTS.json`, `STUDY_PROPOSAL.json` and `CONCLUSIONS.json` preserve this exact scope. Existing evidence, scores and source packages were not modified.
