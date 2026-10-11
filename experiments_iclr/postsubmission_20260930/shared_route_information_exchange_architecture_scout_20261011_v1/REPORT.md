# One refined acquisition proposal

11 October 2026. **Inactive research proposal; no model import, numerical study, fit, server/GPU action, reviewer or manuscript edit.** Root and the source agent refined this proposal to one controlled dropout slot, two fixed positions and three fresh complete Physics development blocks. The all-five-sites and old-role/reference-reuse drafts are preserved under their history directories and are superseded. `PROSPECTIVE_AMENDMENT_RATIONALE.md` records the fresh-role change before any new array/role access.

## Evidence and question

Complete Physics E loses 1.462764 accuracy points to genuine equipped I4, with mean members 5.174743 points weaker and 9369 more pooling losses despite 4379 more covered alternatives. E−N is only+.028158 points; its endpoint attribution failed. Penn internal sharing loses.979907 points to freshly acquired original-boundary GNNM. The strong Physics source-derived meanSAGE single scores 93.058986%, versus E 91.326372% and equipped I4 92.789136%. These are whole recipes; none identifies dropout as the cause.

The affine, Context 9/alignment, affinity/fixed-key and one-draw TRAIN-gradient studies retain their exact closed results. No decoder, gradient or endpoint rescue follows. Rank-1 acquisition and its 21/39/105 proposal remain parked. The separate original-boundary×receiver-action proposal is not combined here.

**One question:** can activation-dependent coupling of the existing shared routes' dropout, at a graph-input hook where its local variance effect is interpretable, acquire stronger members and retain their predictions? The possible contribution is one useful shared-SAGE acquisition operating point. Adaptive antithetic sampling is known; neither a new sampling primitive nor new prediction architecture is claimed.

## Fresh search outcome and strongest ancestry

The bounded search executed 29OpenAlex metadata queries and the bound arXiv metadata/query requests. Thirteen distinct primary bodies received explicitly scoped method reads; zero full-paper reviews are claimed. Downloads, headings, abstracts and extraction are not method-read counts. `SEARCH_SCOPES.json` and `READ_SCOPES.json` give exact custody.

A proposed matched-neighbor cross-route covariance direction was rejected as a new interaction principle. PNA §2–3 learns messages before mean/std aggregation and can recover covariance from `Var(a+b)−Var(a−b)`, with the factor 1/4. FOG §3 sums center×neighbor products; BGNN §3 represents different-neighbor products; MM-GNN §4 combines coordinatewise moments. Graph-Tuple §4 and DeGTA §3–4.3 establish existing intra/inter-view convolution and attention/fusion constructions. Ordinary covariance pooling is not nominated.

The nearest checked acquisition collision is **Ren, Zhao and Ermon, Adaptive Antithetic Sampling, ICML2019**, primary PDF pp 1–5 and the method prefix ofp 6. Its adaptive Gaussian copula preserves marginals while minimizing estimator covariance, with explicit integrand-dependent harms and amortized learning. The present discrete sign rule is a restricted analytic dependent-sampling design, not a new antithetic principle. **Differentiable Antithetic Sampling** arXiv 1810.02555v 2 §2.2/4 supplies marginal-preserving moment matching and discloses bias for its approximate transform. **Multi-Sample Dropout** arXiv 1905.09788v 3 §II supplies shared stochastic copies/mean own losses/deterministic single-copy deployment. Its late-dropout CNN cost advantage does not transfer to a full SAGE copy.

**Standout**, NeurIPS2013 pp 1–3 (§2/main§3 algorithm), establishes activation-dependent dropout, with changed learned retention probabilities. Here q is fixed and only dependence among routes changes. **Fraternal Dropout** arXiv 1711.00066v 4 §2 adds prediction consistency, which is absent here. **Optimal Transport Dropout** arXiv 2609.33377v 1 §3.1–3.2 learns within-mask transport and Energy Score; that preprint does not establish this native own-CE acquisition recipe.

ACM2006 dependent-rounding primary, ACM2024 cross-view-confluent primary and AAAI2026 active-state-selection primary remain unread after their recorded failed access. GENN remains unresolved in the inherited audit. No absence or novelty certificate is issued.

## Exact controlled operation

At the single selected dropout slot, gather the current four scalar route activations `a 0..a 3` for each node/channel and detach them for the mask decision. Keep native retentionq=.8 and inverse scalingB/q. Enumerate matchings `(01,23)`, `(02,13)`, `(03,12)` in this order. For a same-sign pair, including zero, draw freshU and use masks `Bi=1[U<q]`, `Bj=1[1−U<q]`. For an opposite-sign pair, use `Bi=Bj=1[U<q]`. The two pairs and all source coordinates use independent fresh uniforms. Choose the matching minimizing its sum `2*ai*aj*cij`, with listed-order exact ties. No label, teacher, learned gate, extra loss or gradient through the mask choice enters.

Antithetic raw-mask covariance is

\[
c_-=\max(0,2q-1)-q^2=-\min(q^2,(1-q)^2),
\]

and common-mask covariance is `c+=q(1−q)`. Atq=.8 these are−.04 and+.16. Each route's current mask tensor remains independent-coordinateBernoulli(q), conditioned on all current prefix values. Within a fixed placement, its own-route marginal stochastic forward and expected own-loss target are the ordinary-dropout ones. **The shared optimizer trajectory and trained model/route distributions change. Moving the slot also changes the marginal acquisition objective.** No final-law equality is asserted.

For the arithmetic mean of dropped values,

\[
Y=\frac 14\sum_m a_m B_m/q,\qquad
\operatorname{Var}_{pair}(Y)-\operatorname{Var}_{IID}(Y)
=\frac{2}{16q^2}\sum_{(i,j)} a_i a_j c_{ij}\le 0.
\]

The sign rule makes each pair contribution nonpositive; matching gives the best of three pair laws, not the optimum over all joint mask distributions. Uniform balance is a serious known rival: chooseK=3+Bernoulli(.2), then a uniformK-of-4 subset. For fixed values `(1,-1,1,-1)`, IID mean variance=.0625, uniform balance=.078125, and opposite/common pairing can give zero. For `(1,1,1,1)`, uniform balance=.015625 and the pair rule=.046875. Thus the proposed rule does not dominate balance. These are analytic witnesses, not empirical measurements.

## Two source-correct positions

1. **Original stem:** after native boundaryBEBlock, before GELU.
2. **First normalized SAGE input:** remove only that stem dropout; apply it after the first `ResidualModuleWrapper` LayerNorm, immediately before `SAGEModule.conv`. The dropped normalized tensor is used by both the actualSAGEConv and the explicit self input concatenated with its message. The outer residual identity retains the undropped stem.

All four original FFN dropout sites remain IID native calls. Moving just after GELU would still leave LayerNorm before aggregation, so it is not the selected position. The relevant masks cover 34493×128 hidden coordinates, not 8415 raw input features.

At the second position, conditioned on the current prefix states and shared slow weights, masks are independent across source/channel coordinates. The covariance change of their route mean is diagonal and nonpositive. The common fixed linear SAGE neighbor/root maps, explicit self input and FFN linear 1 preserve this order:

\[
\Delta\operatorname{Cov}(L\bar h)=L\Delta\operatorname{Cov}(\bar h)L^\top\preceq 0.
\]

This is the covariance of the ensemble-mean first SAGE pre-nonlinear linear output. It is a property of that linear segment, not a theorem for downstream FFN dropout/GELU/LayerNorm, gradients, Adam, confidence or accuracy. The undropped residual bypass and relocation are part of the placement procedure; no isolated activation-order cause is claimed.

A detached four-route prefix pass chooses the masks, then each native prefix is recomputed and the suffix/CE backward runs sequentially. Weights stay fixed until four CE/4 backwards finish and one optimizer step occurs. No four full body tapes are required in principle. This adds prefix forwards. A dedicated sampler generator leaves the four FFN native draw streams independent of sampler draw counts across wrapper arms. Removing the original stem global-RNG draw changes their relation to native original RNG scheduling; no capture/replay or burn scheme is claimed. Native original therefore remains a separately acquired quality anchor. Fixed-mask equivalence, source/default linearity and full-input memory/restore behavior still require root qualification.

## Frozen representative pilot: three complete Physics blocks

Blocks are **8101,8663,9623**, fixed by root before any running original-anchor quality is opened. A custodian constructs NEW class-stratified development roles using `RandomState(block+5000011+211000003)`: sorted classIDs are permuted, first 20 TRAIN,next 30 SELECT,next 30 CAL,restDEVELOPMENT_EVAL; exported unions are sorted. Each uses the entire 34493-node graph/support 530417 with 100 TRAIN/150 SELECT/150 CAL/34093 DEVELOPMENT_EVAL nodes. This is one encountered graph after prior TEST release, with three complete mask/optimizer protocols—not three independent graphs or unused confirmation. No newrole/array access occurred before this amendment. No oldTEST archive/truth/score is read or recalculated for the pilot. All references are freshly fitted on these masks; the running original 10 anchor remains separate context after its whole closure, and its partial quality is excluded.

New acquisition roster:

| Procedure | Banks/block | Unique optimizer bodies/block |
| --- | ---: | ---: |
| O4 original-stemIID | 1 | 1 |
| O4 original-stem sign pair | 1 | 1 |
| O4 normalized-graph-inputIID | 1 | 1 |
| **O4 normalized-graph-input sign pair** | 1 | 1 |
| O4 normalized-graph-input uniform balance | 1 | 1 |
| FactorM1 at each of the two positions | 2 | 2 |
| Genuine factor I4 at each position, own weights/optimizers/selectors | 2 | 8 |
| Four-copy factor M1 at graph input: IID and sign pair | 2 | 2 |
| **New total** | **11** | **17** |

Native-original sequential O4 and source-derived meanSAGE M1 add two FRESH reference banks/bodies perblock. The whole panel is **13 freshbanks/19 bodies perblock =39 freshbanks/57 bodies**. The source single's width/normalization/regularization/selector differ; it is strong task context, not a capacity-matched causal contrast. Four-copyM1 owns one parameter set and serves one deterministic prediction. Its deterministic prefixes are identical, so its sign-pair stem is an ordinary antithetic control, not a four-member learned ensemble. GenuineI4 has four complete separately optimized predictors; it receivesIID masks and no cross-body sampling rule.

All new original-family procedures preserve original boundary initialization/privateR/S/B,5000-update ceiling, patience 300, lr.001/AdamW/zero weight decay, own CE and the native off-cadence selector: strict improvements reset stale; only improving divisible-by 10 updates save/update best; earliest eligible ties. Selection and primary serving use deterministic native float 32 raw-logit mean and first-index argmax ties. Own I4 selectors are retained. No receiver action, pool loss, teacher, noise prior, coefficient or duration grid is added.

### Exact pilot utility and attribution gates

A complete positive **acquisition-utility** screen requires all of:

* Candidate served accuracy mean gain≥.25 pp over graph-inputIID and≥.25 pp over fresh native-originalseq, with every paired block gain>0 in both contrasts.
* Candidate actual mean member accuracy gain≥.25 pp over graph-inputIID, with all three block gains>0. Worst member and raw member NLL/Brier remain visible.
* Candidate mean raw served accuracy is at least each capable factor M1/I4/four-copy M1 reference and the fresh source meanSAGE reference. A stronger reference loss closes the proposed shared quality operating-point screen, while any within-backbone gain remains reported.
* Mean calibratedNLL harm≤.02 and each block harm≤.05 versus native-originalseq, same-position genuine I4 and fresh source meanSAGE; rawNLL mean harm versus graph-inputIID≤.02. These are practical screens, not statistical noninferiority.

A positive **graph-position/sign-adaptation interpretation** additionally requires candidate−stem/sign-pair mean accuracy≥.10 pp, mean factorial interaction `(graphSP−graphIID)−(stemSP−stemIID)`≥.10 pp, and candidate−graph/uniform-balance mean accuracy≥.10 pp; all three corresponding block differences must be nonnegative. Failing an attribution gate closes that exact interpretation, even if a known sampling ingredient has within-backbone utility. A four-copy M1 gain of the same size limits sharing-specific interpretation; report the paired difference of coupling increments without treating it as a proved cause.

Three protocols are a development pilot. Report complete vectors, ordinary paired descriptive 95% intervals and all eight sign-flip assignments as symmetry sensitivity; no confirmatory superiority p-value or independent-graph claim follows. A complete roster/failure is required. No favorable class/cohort, metric substitution, survivor subset, automatic retry or pilot-based pool rescue is allowed. Genuinely unused confirmation and a stronger novelty case remain necessary before claiming the user's intended novel accurate method.

Report covered alternatives, pooling losses, aggregation-only correct, repairs/harms and the full-population count identity using actual selected members. These diagnostics do not select a new rule. Report all raw/calibrated/member accuracy,NLL,Brier and all fixed controls regardless of gates.

### Calibration and paid work

After all 57 selected native body states and all 39 bank exports close, fit 39 positive scalars on their fixed native pooled-logit endpoints using CAL only:500 CPU-float 64Adam updates, lr.01, betas(.9,.999), eps 1e-8, weight decay 0, initiallogT0 and fixed final endpoint. All 39 scalars close beforeone completeDEVELOPMENT_EVALrelease. No existing reference state/scalar is reused on differentmasks. No per-member probability-pool replacement, temperature-selected checkpoint, same-CAL proper-loss scoring, evaluation-truth refit, retry or grid is permitted.

**Fresh total work:39 banks/57 uniqueoptimizerbodies/39 scalarendpoints.** The 33 wrapper/control banks 51 bodies and sixfresh referencebanks/bodies are separate cost scopes, without doublecounting. Original-family ceilings give 270000 optimizerupdates and 630000 mainTRAINroute/copyopportunities; the source meanSAGE reference's 100000 maxupdates×3 adds 300000 optimizerupdates/routeopportunities. Total maxima are 570000 optimizerupdates,930000 mainTRAINroute/copyforward-backward opportunities and 840000 deterministicSELECTrouteopportunities. The prefix-recomputation schedule permits 420000 additional detachedprefixforwards for the five shared and two four-copy M1 arms. Calibration adds 19500 fixedCPUupdates. Recordactualprefix/qualification/failure/collection/owner costs. Root's coarse **2–6 hour** planning range for 57 fits is not a measured runtime bound or isolated speedclaim; finiteactualbounds must follow the full-inputcheck.

One uint 8 four-retention-bit cache at the moved/original slot would occupy 4415104 bytes; four detachedfloat 32 prefix activation copies would occupy 70641664 bytes if materialized. These are size calculations, not CUDA peaks or measured savings. The source agent's inactive wrapper avoids four whole-body tapes but adds prefix work. Root requires one full-input update/restore qualification and finite positive owner wall/CUDA/RSS bounds before any representative acquisition is admitted. No isolated speed benefit is certified from prior overlapped timings.

`STUDY_PROPOSAL.json` is the structured scientific contract; `CONCLUSIONS.json`, exact read/search scopes and source receipts preserve limits and unresolved ancestry. No original score, sealed evidence or old source package was changed.
