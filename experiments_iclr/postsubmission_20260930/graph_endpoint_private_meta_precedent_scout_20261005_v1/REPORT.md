# Graph endpoint and private-update precedent check

The complete active combination is not established in the checked saved scopes or the two new bounded primary method scopes. This is a scoped finding, not historical novelty clearance. Broad claims about shared meta ensembles, head adaptation, persistence, fixed deployment, or recomputation are already unavailable from the saved priors. The new AUX-TS read strengthens the graph recomputation ancestry.

The check began with `literature_memory/index_v65/LITERATURE_INDEX.json` (243 records; SHA256 `4b8cd1ed8a6b1dd1b8be705ad35a662d4f346af7de760053bd0ab6e68d5c97e7`) and saved SELAR, Meta-Graph, ANIL, BMAML, MLDG, MetaReg and persistence scopes. H-GRAM, Meta-iKG and Episodic DG were located in the saved `endpoint_private_transfer_novelty_stress_test_20261005_v1/SOURCE_LEDGER.json`; their primary papers were not reread. H-GRAM explicitly supplies one-graph task ancestry. Consequently, Meta-Graph's multiple-graph setting cannot support a claim that one-graph meta-training is new.

## Active configuration bound by this check

The persistent core consists of one NCN encoder and dense predictor basis. Four private factor/LayerNorm/beta rows carry separate Adam histories. The current outer endpoint union includes endpoints from **both positive and negative queries**; every current inner positive and negative pair must avoid that union. Other endpoint context remains in the support graph, with the paired target-positive masking rule.

A virtual private update starts from the old private state. Its outer loss differentiates into the shared blocks. After the actual shared update, the private inner gradients are recomputed at the new shared blocks from the unchanged old private parameters and moments; this recomputed private step is committed and virtual moments discarded. Serving uses mean raw logits of the committed rows, without adaptation. The loss is half pooled BCE plus half mean member BCE. Source bytes are pinned in `CANDIDATE_BINDINGS.json`; no target source was imported or executed.

## Two new primary method scopes

### AUX-TS: another graph recomputation precedent

**Xueting Han, Zhenhuan Huang, Bang An and Jing Bai. _Adaptive Transfer Learning on Graph Neural Networks_. [arXiv:2107.08765v2](https://arxiv.org/html/2107.08765v2), revised 20 July 2021.** Read: complete §3.1–§3.3, Eqs. (1)–(9), Algorithm 1 lines 0–20. HTML subsection IDs are `S3.SS1`–`S3.SS3`; algorithm ID is `alg1`. Published-version equivalence was not checked.

§3.1 shares GNN parameters among a target task and auxiliary task-specific layers. §3.2 uses target/auxiliary gradient cosine similarity, loss and task-type signals to generate auxiliary loss weights. Eqs. (7)–(8) and Algorithm 1 lines 11–12 evaluate a target meta fold after a virtual joint GNN update and update the weighting model. Lines 14–19 recompute the similarity and joint loss at the original GNN parameters using the updated weighting model, then commit the main-model update. This directly establishes **virtual update → slow meta update → recomputed persistent graph-model update**, in addition to the saved SELAR precedent.

The state partition and data rule differ materially. AUX-TS's slow variable is the weighting model; its virtual fast update changes the joint GNN model. Algorithm 1 line 5 splits a supervised batch, without specifying endpoint eligibility. Line 19 commits on `Dsup ∪ Daux`, which includes the meta fold, rather than replaying only private inner learners. The selected scope does not establish persistent same-task private factor rows, separate carried optimizer histories or fixed raw-logit ensemble serving. Deployment implementation was not inspected.

### TMAG: interaction separation does not establish endpoint exclusion

**Yuxiang Shi, Yue Ding, Bo Chen, Yuyang Huang, Yule Wang, Ruiming Tang and Dong Wang. _Task Aligned Meta-learning based Augmented Graph for Cold-Start Recommendation_. [arXiv:2208.05716v2](https://arxiv.org/html/2208.05716v2), first posted 2022, revised 4 May 2024.** Read: complete §3.1, §3.2, §4.1, §4.2.2, §4.3 (including §4.3.1–§4.3.3), §4.4 and §4.5; Eq. (1), Eqs. (6)–(18), Algorithm 1 lines 1–14. Exact HTML IDs and paragraph/equation locators are saved. §4.2.1 and Eqs. (2)–(5) were not read.

§4.2.2 clusters users using attributes and constructs support/query interactions within clusters. It explicitly states `S_k ∩ Q_k = ∅`. §4.3.3 Eq. (14) uses observed and unobserved interactions in a BPR loss. These are relevant graph-task and positive/negative-supervision precedents; disjoint interaction sets do not imply exclusion of both endpoints from all positive and negative outer queries. That exact endpoint rule is not specified in the read scope.

§4.5 Eqs. (17)–(18) and Algorithm 1 describe support adaptation followed by a query meta update to initial parameters, then graph augmentation. They do not specify a recomputed inner-only private commit. §3.2 explicitly performs meta-test support updates, and §4.5 describes rapid adaptation for new users/items. This differs from adaptation-free deployment of committed private rows. Algorithm 1 uses `theta_k` without explicitly defining its reset/carry convention; row persistence cannot be settled from that notation alone. No author code was inspected.

## What the evidence permits

[COMPARISON.md](COMPARISON.md) compares the five active requirements. AUX-TS/SELAR cover generic recomputation order; BMAML/EMAML cover meta ensembles and shared features/private classifiers; ANIL and OML cover restricted adaptation and representation learning; MLDG/MetaReg/Episodic DG cover relevant train-only transfer, persistent partners and ordinary fixed deployment. None of these ingredient credits establishes the exact conjunction in the active source.

The specific endpoint eligibility and persistent shared/private update/serving configuration can be stated as the configuration studied. Their exact assembly, optimizer replay conventions and half/half objective do not by themselves establish a new learning principle. Current endpoint exclusion does not erase earlier supervision or retained correlated graph context, and supplies no independence or unseen-node generalization guarantee. This assessment does not reject the configuration on single-model representability grounds and makes no utility or acceptance verdict.

## Search and reading limits

The retained discovery logs cover graph link-prediction meta-learning, endpoint/node disjointness, support/query overlap, graph meta-regularization, cold-start methods and meta ensembles. Candidate IDs were checked against all 243 v65 records and local saved method/conclusion memories before retrieval; the repeated structured check in `DEDUPLICATION.json` found no selected-identity method record among 1,125 checked local memory files. A generic Crossref metadata collection mentioned the Adaptive title, which earned no prior method-reading credit.

ML2E (`10.1109/ACCESS.2020.3022796`), dual-customization recommendation (`10.1145/3597458`) and positive/negative-sampling KG recommendation (`10.1145/3654804`) remain metadata-only leads. Their names or abstracts do not resolve the active rule; no favorable or unfavorable method conclusion is assigned to them. This bounded search is not an absence proof.

Reading credit: **two new bounded primary method identities; zero new full-paper reads; zero retained primary rereads; zero author-code or result-section reads.** Heading navigation and abstracts exposed author performance language, which was not adopted. Selected source text and exact retrieval hashes are saved; temporary full HTML was removed. No allocation/18.77/MacLink contact, model/data/outcome/checkpoint access, numerical work, extra agent, canonical index/status change or frozen-study edit occurred.
