# Persistent private graph learners: focused sharing follow-up

**No supported successor is identified.** Two new bounded method scopes show how persistent graph clients share representations through averaging, personalized mixtures and local regularization. They do not establish endpoint-conditioned private-learning credit as an already published exact graph operator, and they provide no evidence that the current common-core ensemble has the sharing failure those interventions address. The active proposal remains graph meta-regularization with current endpoint-conditioned label eligibility.

This packet uses five public arXiv metadata searches (59 distinct result identities), one metadata-only follow-up, and **two new primary method scopes; zero full-paper reads and zero retained primary rereads**. The new scopes contain eight complete subsections and 34 HTML paragraph/list-item containers plus the indicated equations and algorithm text. Retrieval of a whole HTML page is separate from semantic reading. No experiment results or proofs were assessed. The v63 index and the saved TMetaNet, SELAR, Episodic DG and counterfactual-credit conclusions are reused. Their primary papers were not reopened.

## 1. What the new graph sources establish

| Primary source and exact inspected scope | Sharing and persistence | Credit boundary |
|---|---|---|
| Sim and Park, [LPSFed, arXiv:2603.20338v1](https://arxiv.org/html/2603.20338v1), §2.1 and §§3.1–3.3; Eqs.4–15 and Algorithms1–3 | Persistent clients train pooling and predictive MLPs on separate user-item subgraphs. The server averages both MLP parameter sets and a scalar margin each global epoch (Eq.13). A spectral similarity score to a generated reference graph controls local/global interpolation (Eq.14); bias encoders remain local. | The selected equations and algorithms specify local learning, arithmetic averaging and interpolation. They contain no query-loss differentiation through a private learning step, no outer-query endpoint exclusion and no fixed mean-logit client ensemble. |
| Zhang et al., [PFedEG, arXiv:2406.11943v1](https://arxiv.org/html/2406.11943v1), §§3.1–3.4; Eqs.1–12 and Algorithm1 | Persistent clients train entity and relation embeddings on local KG triples. The server builds personalized supplementary entity embeddings using entity overlap or embedding cosine affinities, ownership normalization and a residual local mixture (Eqs.4–12). Each client initializes entity embeddings from its mixture and trains with a proximity penalty (Eqs.2–3); relation embeddings persist. | The server's affinity recomputation follows previously trained embeddings. Algorithm1 supplies no outer loss or meta-gradient through the local updates. The output is each client's learned embeddings, rather than a served fixed ensemble. A recomputed affinity is not itself differentiation through private learning. |

Both are relevant to dependent graph supervision and repeated private learners. Both learn shared complementary representations through those learners. Neither has one simultaneous common representation block differentiated through multiple private update maps. Their exact sharing operations deserve attribution if borrowed; combining either with meta-learning names would not establish novelty.

The motivating failures differ from the current pilot. LPSFed addresses structural imbalance among different client graphs. A similarity descriptor computed from the current *common support graph* would be identical for all four members; its min–max normalization would have a zero denominator when all scores coincide. Distinct client graphs or a new descriptor would be an additional modeling choice. PFedEG addresses semantic disparities and partially overlapping entity sets across client KGs. Its personalized embedding exchange is not evidence that one common-core encoder suppresses an otherwise useful private correction in the frozen task.

Neither paper's empirical superiority or theoretical guarantees are imported: experiment sections and proofs were outside the selected scopes.

## 2. Established ingredients and the exact endpoint rule

The previously retained scopes already establish the ingredients:

- **ANIL/BMAML:** shared representation gradients through private head adaptation, including shared-feature/private-classifier ensembles.
- **OML/La-MAML/FTML and Episodic DG:** repeated learning and persistent task/domain partners; the precise commitment and derivative partitions differ. In Episodic DG, the private classifier is constant in the cross-domain shared-feature gradient.
- **MLDG/MetaReg:** training-only transfer objectives and deployment without adaptation.
- **SELAR:** graph training with a virtual fast update, a slow meta update, then a recomputed persistent fast update. Its slow variable is loss weighting; its committed fast update includes the complete primary minibatch/meta fold and auxiliaries rather than the candidate's inner-only private commitment.
- **G-Meta, H-GRAM and Meta-iKG:** graph-local support/query learning, including an explicit one-graph task precedent in H-GRAM. Their inspected link protocols do not establish the candidate's current positive-and-negative outer endpoint-union label exclusion.
- **TMetaNet:** recent topology-conditioned graph update rates on adjacent snapshots, updating full GNN weights rather than a persistent private ensemble.

Thus, private-learning credit and virtual/meta/recomputed graph training are established principles. **The exact simultaneous configuration—current endpoint-union eligibility, persistent private state, inner-only commitment at the updated shared representation, and a fixed served raw-logit ensemble—was not found in these bounded scopes.** This is a scope-limited non-match, not a global absence proof or a supported methodological novelty claim.

Endpoint conditioning changes current eligible supervision. Historical parameters, optimizer moments and incident graph context still carry endpoint information. The dependent episodes are not independent tasks, unbiased cross-fitting folds or evidence of unseen-node generalization. These limits remain after adding the two federated citations.

## 3. What the unchanged cohort can test

The frozen ten-cell, three-block study remains at **30 fits**, four members and selector `first_maximum_complete_VALID_MRR_rounded4`. No cells, gates, coefficients, logging or training operators are added.

The existing within-F4 factorial contrast is

`Delta_b = (E_end_live - E_end_detached) - (E_random_live - E_random_detached)`.

It can describe whether live private-update sensitivity is more useful under endpoint eligibility than matched random eligibility in this cohort. It cannot alone establish a sharing-specific cause. The live rule also exists for a single learner and an untied bank.

The capable-single comparison remains necessary for quality, but F4 and that single assign their dense head bases to different outer/private roles. Their contrast mixes member count with label access, adaptation capacity, mixed derivatives and optimizer history. The role-matched F1 comparison is absent. `U_end_live` adds an untied quality anchor, but the cohort lacks the corresponding untied geometry-by-live/detached factorial. Therefore it cannot identify an endpoint-credit interaction unique to parameter sharing. Ordinary and paid controls bound utility and compute; they do not resolve this missing mechanism comparison.

One conditional diagnostic remains possible **only if the existing trace already stores the required components**: for a prospectively fixed outer query at the same episode state, let `c_q = g_live,q - g_direct,q` and inspect `-delta_theta^T c_q`. This measures a local alignment of the shared displacement with private-learning sensitivity. It is not realized prediction improvement, an independent-task certificate or a sharing-specific causal contrast. Aggregate logs are insufficient. This packet requests no trace replay, new fits or new logging and does not access the traces.

Consequently, no additional mechanism beyond the existing endpoint-by-live attribution is identifiable from these new papers without changing the intervention or controls. Any complete positive cohort result could motivate later sharing-specific research under the saved decision boundary; it would not certify that mechanism.

## 4. Successor decision

**No new operator is proposed.** The new papers' cross-client structural and semantic failures are not established for the current common support/core. Their interpolations and proximity terms can also be applied to an untied bank and encoded by a capable branched single with the same blocks, streams and paid paths. Naming clients or members cannot supply the missing causal distinction. This does not claim that a smaller ordinary NCN single must reproduce those models.

The saved counterfactual-credit screen also remains decisive: own-update routed marginal credit recovers the served-loss adaptation meta-gradient already present in the live rule; symmetric credit combines ordinary adaptation benefit with counterfactual ambiguity; the pure interaction cancels the explicit outer label at fixed outputs and can increase while the served prediction stays wrong and unchanged. The two new graph sharing recipes do not overturn those identities.

The next defensible statement is narrower: assess the complete frozen quality/compute comparisons and endpoint-by-live contrast under their declared limits. No supported ensemble-specific successor follows from the literature inspected here.

## Audit

`INPUT_BINDINGS.json` binds eight existing inputs; `SOURCE_LEDGER.json` records the two new identities, exact scopes, findings and metadata exclusions; `PRIMARY_RETRIEVAL.json` records versioned URL receipts and original HTML hashes; `SCOPE_EXCERPTS.json` retains only selected method text. `VERIFICATION.json` verifies unchanged bound inputs and frozen spec; `MANIFEST.json` inventories this packet. Raw full HTML is removed after scoped extraction. This packet does not modify the canonical index or any frozen pilot file and contains no model execution, data/checkpoint/outcome access, server/MacLink action or acceptance verdict.
