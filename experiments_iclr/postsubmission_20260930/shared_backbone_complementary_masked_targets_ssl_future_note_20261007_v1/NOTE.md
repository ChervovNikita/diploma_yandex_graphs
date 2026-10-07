# Shared frozen graph backbone with complementary reconstruction targets

Masked feature supervision on unlabeled nodes is a plausible way to train useful private graph representations when a competent backbone already fits the TRAIN labels well. That explanation is a hypothesis: low TRAIN loss does not establish that useful private gradients have vanished. The scientifically useful question is whether assigning complementary reconstruction targets improves the served four-member predictor beyond ordinary graph self-supervision and a capable single predictor with the same active capacity. This is a prospective experiment; it does not alter the fixed or running studies.

## Established building blocks and the remaining question

| Prior | Relevant operation | Consequence for this proposal |
| --- | --- | --- |
| [GraphMAE, KDD 2022](https://doi.org/10.1145/3534678.3539321), inspected arXiv2205.10803v3 Section3 | Mask raw node features before graph encoding, re-mask the selected latent rows before a GNN decoder, reconstruct raw features with scaled cosine error, and discard the decoder for inference. | The reconstruction objective, leakage precaution, graph decoder and loss have direct ancestry. GraphMAE trains its encoder; freezing a classification backbone is a separate adaptation that may reduce usefulness. |
| [GraphMAE2, WWW 2023](https://doi.org/10.1145/3543507.3583379), inspected arXiv2304.04779v1 Sections2.1–2.2 | One masked online encoding feeds several random decoder re-masks. All decoder views reconstruct the same input-masked target nodes. A second loss predicts latent targets from an EMA network applied to the unmasked graph. | Multiple reconstruction views from a common encoding are already established. Its EMA target generator also requires an unmasked network forward. Complementary private classification paths are a narrower prospective question; raw feature targets require no teacher network. |
| [MVGRL, ICML 2020](https://proceedings.mlr.press/v119/hassani20a.html), inspected arXiv2006.05582v1 Section3 | Dedicated encoders process adjacency and diffusion views; node representations in one view are contrasted with graph summaries in the other, using corrupted negatives in the transductive setting. | Graph self-supervision on unlabeled nodes has established contrastive alternatives. Two structural encoders and negative processing do not supply the proposed single shared corrupted forward. |
| [Deep Sub-Ensembles](https://arxiv.org/abs/1910.08168) and [TreeNets](https://arxiv.org/abs/1511.06314), reused saved method conclusions | Frozen learned trunks with private task networks and probability averaging, or shared early layers with private later networks and ensemble objectives. | Shared/private ensembles have direct preprint ancestry. Adding an established SSL objective does not establish architectural originality. |

Saved MaskGAE and SeeGera conclusions already cover masked structure/feature reconstruction and richer shared graph latents. The saved Adapt, Agree, Aggregate preprint method covers independently augmented GCNs, pseudo-label selection and a consensus GCN; its linked 2026 publication is credited with the saved version-equivalence limitation. These precedents motivate attribution, rather than a new claim about unlabeled graph supervision. None of the inspected operations establishes the exact efficacy of the target assignment below. That is a bounded empirical gap, not novelty clearance.

## One prospective representative experiment

Use WikiCS split 0 and seeds 17, 29, 43 after separately adopting a complete protocol. For each seed, all five conditions use the same competent frozen backbone and initial private draws. Let H0 and z0 be its clean hidden representations and logits. Four active private graph paths produce 512-dimensional representations Rm from raw features and the shared backbone representation; each member has a residual classifier anchored at z0. Use the adopted classification stage lengths and the same classifier initialization across conditions. No extra teacher acquisition is required.

Define U as nodes outside TRAIN, using only their public features and graph. Exclude zero-norm rows from reconstruction. The SSL prephase has 25 blocks of four AdamW updates, learning rate 0.001, weight decay 0, no SSL dropout, and scaled cosine exponent 2. These are fixed prospective choices, not selected results. Within each block:

1. Sample about half of eligible U, rounded down to a multiple of four, as S. Replace each complete feature vector in S with zero. Partition S uniformly into equal quarters Q0,Q1,Q2,Q3.
2. Compute Hmask = frozen_backbone(Xmask,A) once in evaluation mode without gradients. Keep this common masked representation for the four updates of the block.
3. On update t in {0,1,2,3}, the repeated condition gives every path target Qt. The complementary condition gives path m target Q((m+t) mod4). All paths see the same Xmask,Hmask,A in both conditions.
4. Each path recomputes Rm on that input. Re-mask every row in S to zero before a one-layer GCN decoder: reconstructed_Xm = normalized_adjacency × Remask(Rm,S) × Wm + bm. For its assigned quarter, minimize the mean of (1 − cosine(original_x, reconstructed_x))². Average the four branch losses; gradients update the private paths and decoders, while the backbone remains frozen.

Both target schedules give every branch every masked target exactly once per block. Each update uses |S|/4 targets per branch and |S| total target appearances. Aggregate coverage and multiplicity match over a block. Concurrent coverage differs by construction: complementary targets spread across S at one update, while repeated targets concentrate on one quarter. This is a test of target synchronization and finite-budget gradient order, not persistent semantic specialization. The branches have the same population objective; useful predictive diversity is not guaranteed.

After SSL, discard the decoders and fit the classification predictors on clean X,H0,z0 with TRAIN labels only. For the bank, retain the adopted staged own-CE plus pool-Brier procedure and fixed mean probabilities. For the single, train all four private feature paths jointly through one unrestricted residual classifier on their concatenation, using CE plus the same Brier coefficient on its served probability. Give each private path the same number K of clean classification updates: four bank stages of K updates versus K joint single updates.

| Condition | SSL assignment | Served predictor |
| --- | --- | --- |
| B0 | None | Mean of four private member probabilities |
| BR | Repeated targets | Same four-member mean |
| BC | Complementary targets | Same four-member mean |
| S0 | None | One jointly trained classifier on all four private representations |
| SC | Complementary targets | Same capable single, with exactly the SSL signal and four useful reconstruction heads of BC |

This is five conditions × three seeds, with fixed endpoints and no hyperparameter grid. The single has the same four active graph feature paths and SSL decoder capacity; it can learn a joint class decision instead of obeying a fixed probability mean. Its 2048-to-10 classifier has 20,490 parameters, versus 20,520 for four 512-to-10 member classifiers, a disclosed difference of 30. This is a capacity control within the frozen-backbone family. A broader superiority claim would still require a fully trainable native single supplied with the same SSL opportunity.

## Leakage and actual work

The SSL paths and decoder may receive only Xmask,Hmask,A. Clean cached H0, clean z0, original features of target nodes, and clean cross-view representations must never enter the reconstruction forward. Original x appears only as a loss target. Masking private raw input while supplying clean H0 would leave a shortcut through the backbone and would not test masked prediction. Decoder re-masking all of S is common to BR,BC,SC. Neighbor features and topology remain legitimate reconstruction context. No VALID or TEST class labels, pseudo-labels or class-conditioned masks enter SSL. Feature reconstruction is the sole initial pretext task; an edge target would require removing that edge from every message path.

Each 512-to-300 GCN decoder has 512×300+300 = 153,900 trainable parameters; four have 615,600, or 2.46 MB of FP32 weights. The fixed adjacency normalization adds no trainable matrix, but each decoder still pays a graph propagation, an affine projection, and backward work. The four reconstruction heads are active in SC, not dummy parameter padding, and all are removed at serving.

Per SSL condition and seed, the prephase adds **25 frozen backbone forwards, 400 private graph path forward/backward passes, and 400 graph decoder forward/backward passes**. The common frozen Hmask is cached only within a block. With the proposed WikiCS dimensions 11701×512, that one FP32 tensor occupies 23,963,648 bytes, about 24 MB; this is not a total memory estimate. Retaining 100 views or paying four masked backbone forwards per update is unnecessary for this joint prephase. These counts depend on evaluation-mode frozen encoding and the fixed four-update common input.

The no-SSL controls omit that additional work. Clean classification pays 4K active private graph forward/backward passes in both schedules. For the proposed bank, compute and cache the three deterministic frozen peer predictions once at the start of each of four stages, charging another 12 private graph forward passes. Both schedules require one clean backbone forward to form H0,z0. At serving, both families require one clean backbone forward plus four private graph paths; the single is not a cheap one-path reference. Report backbone acquisition, SSL, classification and serving costs separately, charging the added decoder and optimizer work. No reduction in end-to-end time or memory is inferred from forward counts.

## Interpretation and rejection rules

BC versus BR tests the complementary assignment schedule. BC versus B0 tests the added SSL prephase; SC versus S0 identifies a generic SSL benefit available to a capable single. BC versus SC tests whether keeping four probability members adds value after matching the active feature capacity and SSL signal.

Use complete served accuracy as the primary endpoint, with held-out loss and calibration as secondary outcomes. Future donor TRAIN loss and private-path gradient measurements can test the proposed weak-supervision premise. Member competence, common wrong-node overlap, disagreement and reconstruction loss are mechanism diagnostics; they cannot substitute for served accuracy.

If BR matches BC, complementary assignment is unnecessary. If SC matches or exceeds BC, ordinary SSL with joint single prediction is an adequate explanation. If reconstruction or disagreement improves without served accuracy, reject the proposed accuracy mechanism. Feature reconstruction can emphasize nuisance information, and masking can be poorly suited to a frozen classification backbone; GraphMAE2's discussion of noisy or weakly discriminative feature targets makes this a substantive risk. No rescue objective or additional grid follows from a failed representative comparison.

## Source scope

The three new method reads are bounded primary arXiv HTML scopes, not full papers or accepted-version equivalence audits. KDD2022 and WWW2023 publication metadata were verified through publisher-deposited DOI records; ICML2020 through PMLR. No reported benchmark improvement is adopted. Exact URLs, passage ranges, retained source hashes, reused conclusion hashes and the corrected unrelated GraphMAE2 locator are recorded in SOURCE_SCOPE.json and METADATA_RETRIEVAL.json. This work accessed public literature and saved research conclusions only.
