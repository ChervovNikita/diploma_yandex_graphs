# Assessment of count-conditioned source-pattern supervision

4 October 2026. Literature and algebra only. The existing four-arm screen and sealed five-case CUDA diagnostic are unchanged. No project scores, datasets, checkpoints, fit, numerical experiment, or remote compute dispatch was accessed. The hypothesis remains untested and has no novelty clearance.

## Assessment

The proposal is a useful way to ask whether the auxiliary learns **which residual nodes carry observed TRAIN incidences after their two side counts are fixed**. It removes the direct reward for predicting those counts from its conditional output law. It does not establish that the old objective mainly fitted counts, that count calibration survives training, or that arrangement learning improves link ranking. Those are separate falsifiable questions. The supplied motivation is treated as a hypothesis; no earlier project result was reanalysed.

The density and recurrence are classical. A narrowly scoped new read of Chen and Liu (1997), PDF pp.1-4 / printed pp.875-878, confirms independent Bernoullis conditioned on their sum, the weighted-subset law (Eq.3), its subset-product normalizer (Eq.6), and a positive addition/multiplication recursion (Eq.9 and the following cell description). The paper credits earlier sampling/conditional-likelihood work; this assessment does not claim historical priority for the 1997 paper. Its remaining 14 pages, proofs, applications, code, and numerical experiments were not read or certified.

## Closest prior ingredients reused from index_v43

| Prior | Relevant established operation | Boundary for this proposal |
| --- | --- | --- |
| Chen & Liu, 1997; Tarlow et al., *Fast Exact Inference for Recursive Cardinality Models* | Fixed-size weighted subsets; binary Gibbs laws with count potentials and exact cardinality inference | Conditional Bernoulli, ESP normalization, count supervision, and exact inference are attributed ingredients. |
| GRAN, *Efficient Graph Generation with Graph Recurrent Attention Networks* | Graph-conditioned finite mixtures of factorial Bernoullis, component-wide likelihoods and responsibilities | Closest mixture mechanism. Conditioning components and then mixing uniformly changes the local model, but does not create a new generic mixture principle. |
| MaskGAE; Graph Guided Diffusion | Masked edge/degree supervision; visible-entry zeros can inflate reconstruction competence | Counts and source-observation prevalence are real confounds to control. A TRAIN zero is not a verified latent nonlink. |
| sMCL; *Joint Training of Deep Ensembles Fails Due to Learner Collusion*; Wood et al., *A Unified Theory of Diversity in Ensemble Learning* | Specialist assignment, joint-training failure modes, loss-dependent diversity analysis | Responsibilities can specialize or collapse; neither spread nor conditional NLL guarantees useful pooled predictions. |
| PIFM, *Prior-Informed Flow Matching for Graph Reconstruction* | NCNC informed priors with coupled graph refinement | Direct graph/NCNC structural-refinement comparison ancestry. Its native residual adaptation and harmonized Collab protocol remain unqualified in the stored scope. |
| Graphite, SIG-VAE, SeeGera, DiGress, KREPE | Shared latent/contextual graph laws, masking, repeated refinement or autoregressive/masked completion | A capable structured single is a substantive comparator. Conditional per-edge losses do not make an integrated latent/recurrent model globally factorial. |
| *On the Role of Edge Dependency in Graph Generative Models*; SGDiff; CAM linksets | Identity-aware motifs, graph-dependent joint generation and linkset context | Graph arrangement has established ancestry. A local four-component objective does not imply a coherent full-graph posterior. |

The complete reused conclusions, exact old scope limits, and source hashes are retained in `REUSED_CONCLUSIONS.json` and `INPUT_BINDINGS.json`. They are reused conclusions, not new full-paper reads. Search errors, irrelevant Crossref results, and a publisher soft-404 are preserved as retrieval records and supply no absence evidence.

## Decisive mathematical limits

1. **`C_mu` loses its additional structured head.** For its frozen law `exp(eta·z + g_C(K_L,K_R))/Z`, conditioning on both counts makes `g_C` constant over every remaining pattern, so it cancels exactly. Its conditional law is just the native unary conditional Bernoulli on each side. All count-head gradients are zero, including through the head's context dependence. With the existing detached completion and zero weight decay, merely replacing its auxiliary by conditional NLL would also remove its count-head learning signal. It is not a capable identity-sensitive structured-single control. Conditioning on total count alone would be a different model and would not generally cancel a two-sided potential.

2. **The proposed memberwise control is misnamed.** Mean component conditional NLL, called `W_K` here, already has within-side dependence from the fixed-count constraint. It differs from frozen `F_P`, which is the product of pooled Bernoulli marginals. A conditional analogue of `F_P`, called `F_K`, first pools slot probabilities, converts them to odds, and then conditions each side. `J_K` versus `W_K` tests mixture responsibility weighting versus requiring each member to fit; it does not isolate dependence cleanly.

3. **Both extremes provide no arrangement supervision.** When a side's count is zero or its support size, its conditional NLL and slot gradient are exactly zero. If both sides are extreme, the full auxiliary is zero. Count conditioning can therefore remove much of the auxiliary signal in a sparse reconstruction problem. Report informative-query coverage and signal magnitude; any benefit might come from reducing conflicting supervision rather than better arrangements.

4. **Component interactions remain restricted.** Inside each conditional component, replacing selected node `i` by `j` has odds `exp(eta_j-eta_i)` independent of the other selected identities. Cross-side association arises only through the shared member identity; its pattern-by-pattern probability matrix has nonnegative rank at most four. Neither learned graph-motif interactions within a component nor arbitrary graph coherence follows from conditioning.

5. **Training and serving differ.** Teacher counts are auxiliary labels only. Conditional laws do not identify count probabilities or member logit offsets. Shared nonlinear updates can still alter unconditioned native completion weights. The served pooled raw target logits are not the conditional pattern-mixture density; no likelihood guarantee transfers to ranking.

## Graph-specific distinction worth testing

The useful possible distinction is a restricted supervision interface: exact native residual support from the supplied TRAIN record mask, identity-sensitive assignment within fixed side counts, one shared member index for both sides, and the unchanged pooled completion/target decoder at serving. Graph meaning comes from endpoints, actual residual nodes, visible topology, and their contribution to native completion; fixed-size set prediction by itself is not graph-specific.

The two-sided assignment can represent correlated choices with identical counts: with two nodes per side, one positive per side and two opposing members, a shared member mixture can prefer matching left/right identities while each pooled side remains uniform. This proves a representational distinction from `F_K`, not predictive usefulness or new methodology. Repeated/overlapping masks, endpoint exchange, duplicate-edge handling and dropout must retain their actual semantics; they do not automatically define one consistent graph law.

## Representative falsifiable comparison

Freeze one common-replay Collab screen before any fitting: `P0`, unchanged unconditioned `J_P`, candidate `J_K`, independent-side conditional mixture `J_K_sep`, memberwise conditional `W_K`, and a capable count-constrained identity-sensitive single `S_K`. Keep the existing width64 native bank, target loss/serving, admitted TRAIN teacher and record masks, seed610041, replay seed2026100401, 100 epochs ×17 complete65536-record updates, coefficient1, original per-query support-size normalization, and first maximum of complete official VALID Hits@50. Each arm starts fresh; no engineering state is a donor. This is a proposed comparison, not implementation or an execution release. `S_K` needs its own frozen source and feasibility qualification before a bank-specific claim.

Primary contrasts are `J_K−J_P` (conditioning), `J_K−J_K_sep` (shared member identity across sides, with whole per-side laws and unary marginals exactly matched at fixed parameters), `J_K−W_K` (responsibility learning), and `J_K−P0` (served benefit). Comparison to `S_K` limits a bank-necessity claim. `F_K` is the algebraic conditional analogue of F_P, but pooling before conditioning also changes conditional unary marginals; it is not as clean a cross-side control as `J_K_sep`. If used as another scientific condition, freeze it before fitting and describe its changed operations. These are named mechanisms, not a lambda/seed/route search. All failures and costs remain reported.

Measure conditional identity NLL against the uniform fixed-count baseline, separately for informative count strata, alongside original count NLL, observed-positive/zero competence and served ranking. Decompose the old mixture using its **count-reweighted** conditional law, not `J_K`. Use only prospectively admitted TRAIN mask contexts for arrangement probes, and treat overlapping query contexts as dependent. No TEST or model selection on those diagnostic probes is proposed.

Stop the useful-supervision claim if `J_K` does not improve served ranking over `P0` and the matched comparisons, even if its conditional NLL improves. Stop the bank-specific claim if a competent `S_K` matches it. A favorable one-seed screen warrants a separately frozen replication; it does not establish superiority, causal graph coherence, or novelty. Current native/full-batch admission and the missing `S_K` source must be resolved first.

See `ALGEBRA.md` for derivations and `PAPER_CONCLUSIONS.json` for scoped attribution. No model score or prospective outcome is assigned here.
