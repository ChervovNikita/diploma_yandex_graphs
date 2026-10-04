# Recent method gap and public DDI recipe scout

Date: 2026-10-04. Source-only research packet; no execution adoption.

## Result

Four new, **scoped** primary reads add useful boundaries: OCN, CPD-PIFM, CurvPrompt, and EO-GNN. Their inspected formulations do not establish the same two-side, fixed-count identity likelihood with a shared posterior member responsibility. This is a comparison of the read equations, not a novelty certificate or an exhaustive absence claim.

The defensible experimental question remains narrow: does sharing responsibility across the two count-conditioned identity subsets improve an unchanged, count-free native ranker beyond independently mixed sides? Conditional Bernoulli distributions, normalized mixtures, topology-conditioned experts, degree auxiliaries, and shared graph-pattern mixture components are established prior operations. The reused index conclusions for Chen–Liu, MaskGAE, GRAN, Link-MoE, PALP, MoSE and PIFM preserve those limits.

Three author repositories contain explicit public DDI recipes beyond NCNC/PENCIL. **HL-GNN is the clearest additional backbone to qualify first**, because its served computation is a graph encoder and a pair MLP. OCN is the most closely related recent structural comparator, but its public implementation has material candidate-batch dependence. Neo-GNN provides another structural baseline but its DDI entrypoint loads pretrained files. No performance superiority or runtime feasibility is certified here.

## Comparison contract retained from prior work

For each side, the component law is classical conditional Bernoulli: `q_ms(z_s|k_s,C) = exp(eta_ms · z_s)/e_ks(exp eta_ms)` on `sum(z_s)=k_s`. The shared construction is `mean_m[q_mL q_mR]`; the independent-side construction is `mean_m[q_mL] * mean_m[q_mR]`. At **fixed logits**, these have identical complete side laws and unary marginals. Their difference is association of member identity across sides. Separately fitted models need not retain matched learned marginals. An empty/extreme side, or a side whose component laws are identical, removes that association difference.

Uniformly mixing already conditioned components is generally different from conditioning an unconditioned mixture, which reweights components by count evidence. A count-only potential cancels from the conditional identity law. The labels describe observed TRAIN incidence: a zero is unobserved TRAIN incidence, not a verified latent nonedge. Counts and these labels belong to the auxiliary likelihood and are absent from the served ranker.

## Read accounting and provenance

- New primary method reads: **4 scoped; 0 full papers**. Exact block ranges are in `READ_SCOPES.json`.
- Reused serialized conclusions: 11 inspected records from `literature_memory/index_v45/LITERATURE_INDEX.json`; additional selected records are retained for context in `INDEX_REUSE.json` without being counted as new reads.
- Three public author source recipes inspected; 16 pinned README/code files independently match the repositories' Git blob identities (`GIT_BLOB_VERIFICATION.json`). These are text reads, not author-code execution.
- HL-GNN's primary paper was inspected only for repository-link provenance. UTMP arXiv:2406.16687v2 was inspected only for metadata and its replication-code link. Neither is added to the new primary-method count. The exploratory filenames beginning `mplp_2406_16687` retain their acquisition names; that paper is **UTMP, not MPLP**.
- The search response titles and a small abstract shortlist were screened. Search result counts do not count as paper reads, and the date-filtered search is not treated as a complete 2025 survey.
- Every requested URL, returned URL/status, byte count, UTC retrieval time, SHA256, and failure is saved under `retrieval/`. Initial guessed repository URLs returned 404 and remain in the ledger. No failed lookup is treated as proof that a method lacks public code.
- Excluded GraphLP arXiv:2301.00169v1 and GAD-NR arXiv:2306.01951 were not retrieved or read.
- No numerical imports, training, scoring, support census, dataset/archive/checkpoint bytes, saved scores, SSH, server actions, extra agents, manuscript/index edits, compilation, or GENLINK operations occurred. Only the new scout directory was written.

## New method reads

### 1. OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction

Juntong Wang, Xiyuan Wang, Muhan Zhang. [arXiv:2505.19719v2](https://arxiv.org/html/2505.19719v2). Published 2025-05-26; this version updated 2026-03-09. HTML SHA256 `b2a8dfb1cd01ad3fc3949d5305c4f77eff3af2ffa81dfc7093e16ab7f6e27666`.

Read: §§2, 4, the opening operational part of §5, §6's model construction; Appendices G/H and the scoped NCN comparison in N. Proofs and numeric result tables were not adopted.

OCN retains the MPNN-then-structural-feature arrangement, combines endpoint Hadamard features with first- and second-order common-neighbor aggregates, and preprocesses common-neighbor features by orthogonalization and walk-participation normalization. §4.1 maintains a running inner product across mini-batches; §4.2's polynomial alternative compromises strict orthogonality. At order one, the paper relates its normalization to Resource Allocation. The served predictor remains sigmoid(MLP(link features)).

This is a strong structural-representation comparator to consider, not an identical probability law. Its inspected target objective and features do not define fixed-cardinality subset identities, an ESP denominator, or a posterior responsibility shared by two auxiliary response subsets. The official DDI source qualifications below matter before adopting it as a controlled comparator.

### 2. Feasible Flow Matching for Graph Reconstruction via Within-Sampling Primal-Dual Guidance

Haoming Chen, Nicolas Zilberstein, Santiago Paternain, Santiago Segarra. [arXiv:2609.32980v1](https://arxiv.org/html/2609.32980v1), 2026-09-26. HTML SHA256 `45ab4b6a4aad3640612e70ed00fc95dd33fc227fe51a0d575b73b650ea9308be`.

Read: §2 problem formulation and §3 construction/Eqs.2–6, including the sampler's operational algorithm math. No theoretical-bound proof, experiment table or implementation was audited.

CPD-PIFM reuses a trained graph-conditioned flow and prior. During sampling it predicts an endpoint, evaluates degree/density/triangle budget violations, updates nonnegative dual multipliers, and adds normalized projected surrogate gradients to later Euler steps. It restores observed entries after each step. Exact binarized statistics drive the multiplier updates, while differentiable surrogates supply guidance. At one step, the initial zero multipliers give no adaptation.

The paper's optimization places constraints on **expected statistics**, and its clipped sampler maintains observation/symmetry support. Neither operation is the exact conditional identity law on `sum(z)=k`. Degree or density budgets used to alter served graph samples also differ from label-only TRAIN counts in an auxiliary branch removed at serving. This read further blocks broad claims that graph completion with structural side information or graph-level coupling is itself new.

### 3. Dynamic Graph Prompting via Topology-Routed Mixed-Curvature Experts

Quanxin Wang et al. [arXiv:2608.06031v1](https://arxiv.org/html/2608.06031v1), 2026-08-06. HTML SHA256 `b92f82364b8badc148e62dc32081b429618c9da3ce61b7d278860dca257f415c`.

Read: complete extracted Methodology block range, covering the dynamic setting, expert updates, topology gate, aggregation, and training/downstream objectives. Experiments, proofs, code and benchmark quality were not audited.

CurvPrompt uses curvature-diverse experts with private prompts and a shared node–time condition bias. The topology encoder mean-pools several sizes of the most recent historical-neighbor list: these sizes are not hop radii. Training uses soft Top-K weights; downstream adaptation freezes the experts/router and uses uniform weights over the selected Top-K experts. The routed objects are embeddings, and training uses temporal edge BCE plus distortion, gate-balance and prompt-orthogonality terms. The expert bank remains in the served computation.

A shared gate context and uniform selected-expert weights are not posterior responsibilities for an observed two-side pattern. The inspected objectives contain no fixed-count subset normalizer. The relevant prior is topology-conditioned expert specialization and shared context for computation/selection, already broader than the proposed adaptation.

### 4. Edge-Level Automorphism in GNNs: A Quantitative Framework and Effective Designs for Link Prediction

Chen Shao, Donald Loveland, Tobias Käfer, Danai Koutra. [arXiv:2609.34729v1](https://arxiv.org/html/2609.34729v1), 2026-09-28. HTML SHA256 `4f67203c3226ecbf446dced0579ce4e15fa2ec4bcb3fedc0868a61a8ce41109a`.

Read: §4's dropout/orbit-feature construction and link readout; Appendix D.1's stated edge-orbit proposition and proof; Appendix H's loss statement only. The larger expressiveness claims, other proofs, experimental results and code were not audited.

EO-GNN introduces orbit-size-dependent stochastic dropout and noisy WL-role embeddings, then combines common-neighbor aggregation with endpoint products in the pair readout. It is another structure-conditioned stochastic training prior rather than the proposed conditional subset law.

Its Appendix D.1 proposition preserves an orbit assignment under `Aut(G)` while discussing why equality need not hold for other permutations on that same fixed graph. That is narrower than a test that **relabels the whole input graph, supports, features and labels together**. Its stochastic symmetry-breaking and that proposition do not certify the current ordered Single's full relabeling behavior. No official DDI recipe was qualified for EO-GNN.

## Public DDI recipes

These are source-backed contenders, not a numeric ordering. Official commands are retained verbatim in the README source copies; no command was run.

### HL-GNN — first additional backbone to qualify

Primary paper [arXiv:2406.07979v2](https://arxiv.org/html/2406.07979v2) points to [LARS-research/HL-GNN](https://github.com/LARS-research/HL-GNN). Pin: `0855b0de74a8f0586b8cc203e9ba4dbbb57243f4`.

README lines75–80 explicitly specify DDI with embeddings/encoder/predictor width512, three training negatives, dropout0.3, and the label `WeightedHingeAUC`. The remaining source defaults include HLGNN encoder, 15 propagation steps, KI coefficient initialization with alpha0.5, a two-layer Hadamard MLP, Adam at0.001, batch65536, epochs500, evaluation every5 epochs and runs1. DDI's command does not request node features, pretrained embeddings, validation-edge insertion or random-walk augmentation.

The encoder's actual source is a learned linear input map followed by a weighted sum of normalized propagation powers; it is a useful different backbone from NCNC. Serving uses encoder embeddings and the pair MLP, without auxiliary counts.

Qualifications:

- `BaseModel.calculate_loss` uses WeightedHingeAUC only when TRAIN supplies a weight margin. Otherwise it falls back to `auc_loss = sum((1-(positive-negative))²)`. This packet has not inspected DDI data bytes, so the existence of a TRAIN weight field is left to the owning dataset contract. A loss name alone cannot establish the executed objective.
- Training uses graph-global nonedge sampling with selfloops excluded. The inspected training function uses the supplied adjacency without target-edge removal. The source `test` routine reads both VALID and TEST and runs at each evaluation interval; a held-TEST adaptation must be explicit.
- Extending the command to multiple runs without changing source is not sufficient for independent restarts: the model/optimizer are constructed before the run loop, and HLGNN.reset_parameters resets the propagation coefficients without resetting `lin1`. Qualify a fresh model/optimizer per independent run.
- No dependency/runtime or cost result was established.

### OCN/OCNP — recent, closely related structural comparator with source issues

Primary paper points to [qingpingmo/OCN](https://github.com/qingpingmo/OCN). Pin: `530c142779f3039fd978e97a2e775426d9d3d49f`.

README lines57–60 and96–99 specify DDI for cn5/cn7 respectively: 100 epochs, 10 runs, train/test batches32768, width64, three pure-GCN message-passing layers, target masking, block sparse adjacency-square construction and extensive dropout. The model uses learned node-ID embeddings for featureless DDI. TRAIN negatives come from the source's PyG negative sampler; target TRAIN edges are removed from the adjacency for each batch. The runner evaluates VALID and TEST every epoch and retains validation-best entries.

Qualifications:

- Both DDI commands have `--alpha 7.18--probscale 4.31`, a missing space that does not supply a valid float argument. A corrected invocation is an explicit source repair, not a byte-identical author command.
- cn5 is `CNLinkPredictorOringin`; cn7 is `CNLinkPredictorbaselearn`. Several README predictor flags are not wired into these constructor routes, so the full command's appearance does not establish all intended predictor settings.
- cn5 computes `cn1.sum(dim=0)` across the **current queried-edge batch**, normalizes by these sums, and zeros columns whose sum equals1 (model lines2261–2272). Its higher-order normalization also uses the current query rows. cn7 likewise begins with per-query-batch column sums and handles sum1 columns using `args.sum` (lines3114–3126). Thus scoring has candidate-batch dependence in source; no numerical magnitude has been measured. This must be addressed or declared before using it in a native ranker/candidate protocol.
- cn5's running inner-product state updates while training and is reused during evaluation. Reproduction of paper claims and equivalence of public preprocessing to the paper's global description remain unverified.

### Neo-GNN — available structural baseline with pretraining ownership

[seongjunyun/Neo-GNNs](https://github.com/seongjunyun/Neo-GNNs) identifies itself as the official NeurIPS2021 implementation and links the primary OpenReview paper. Pin: `a27fe3023bf0a0f65793041ab12dcf91d89439b4`.

README explicitly gives `python main_ddi.py`. Its DDI source uses learned node embeddings, feature and structural probability branches mixed by a learned two-weight softmax, and separate BCE terms for the structural, feature and mixed branches. Negatives use PyG dense nonedge sampling. Default structural/GNN training batches are2048/65536, width256, fine-tuning epochs40 and runs10. The zipped loaders stop at the shorter, large-batch loader: a full small-batch pass must not be assumed.

Before each run, the entrypoint loads `ddi_GCN_model.pt`, `ddi_GCN_predictor.pt`, and `ddi_GCN_emb.pt` from common pretrained filenames. Tree metadata confirms these filenames exist, but no checkpoint bytes were retrieved. The provided `main_ddi_gcn.py` has a validation-selected pretraining stage that writes those shared filenames and also scores TEST during periodic evaluation. Pretraining seed/selection/custody and its resource budget must be included in a comparison. The README's old dependency versions and GPU statement are author requirements, not a current runtime certification.

### Additional discovery limit

UTMP arXiv:2406.16687v2 links replication code at `https://doi.org/10.5281/zenodo.15019863`. It was not acquired or inspected, and no DDI recipe was qualified. The unsuccessful exploratory MPLP/GitHub search is not a code-availability verdict. Existing BUDDY/LPFormer paper conclusions were reused; their source recipes were not newly audited here.

## Two next ideas

### 1. A relabeling-safe, genuinely structured Single control

Use one shared auxiliary network to define a directional law

`q_LR(z_L,z_R|k_L,k_R,C) = CB(z_L;k_L,eta_L(C)) * CB(z_R;k_R,eta_R(C,SetEnc(selected_L)))`,

and the swapped direction `q_RL` with tied, side-swapped parameters. Use `q_sym = (q_LR+q_RL)/2`. SetEnc must be permutation invariant over selected identities and equivariant inputs; no visitation order or node ID is used as an auxiliary positional cue. Each conditional CB has its exact ESP denominator, so each directional product is normalized by iterated summation, and the two-direction average is normalized. The second side can depend on the observed first-side selected set, providing association without a learned member bank. The orientation average enforces endpoint exchange when the contexts/parameters transform accordingly.

This is a proposed control construction, not an attributed new method or a demonstrated feasible implementation. It has an orientation mixture but one parameter bank. It needs the same TRAIN/mask/support contract, adequate fitting and stated parameter/work matching. It would replace the order-confounded Single as evidence about bank necessity. The auxiliary remains absent at serving. A competent Single matching the bank refutes necessity; failure without these qualifications does not support it.

### 2. One controlled DDI transfer check on the qualified HL-GNN recipe

After the root's dataset and runtime gates, qualify the pinned HL-GNN baseline under the declared TRAIN-only fit/VALID-selection/held-TEST protocol, including the actual loss branch and fresh run initialization. Then compare that baseline with the same shared-responsibility auxiliary and its independent-side counterpart while preserving the served HL-GNN encoder/MLP and candidate protocol. This checks transfer across a materially different native backbone without a tuning grid. Auxiliary likelihood improvement alone is insufficient: the served ranking must improve, and a J-versus-separate comparison remains necessary to attribute benefit to shared association.

No fit, scoring or server action is authorized or initiated by this packet. Root owns dataset acquisition, execution decisions, source integration and literature-index updates.
