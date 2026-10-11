# Bayesian Sheaf Neural Networks version and prior boundary update

A newer author revision exists: **arXiv:2410.09590v2**, posted **12 September 2025, 16:40:19 UTC**, following retained v1 of 12 October 2024, 16:46:48 UTC. The current primary record is [the arXiv abstract page](https://arxiv.org/abs/2410.09590); the inspected method is [v2 HTML](https://arxiv.org/html/2410.09590v2). This is one version revisit and **zero new paper identities**.

The revision does not change the closest-prior conclusion on the inspected ownership, geometry sampling or predictive acquisition axes. BSNN already learns one network and an endpoint-conditioned sheaf-distribution learner jointly, then ensembles predictions over sampled sheaves. That broad shared-network/private-geometry motif remains occupied.

## Scoped version comparison

| Axis | Retained v1 | Verified v2 scope | Consequence for E |
| --- | --- | --- | --- |
| Learned ownership | Network parameters theta and conditional-distribution parameters phi, trained jointly (§3.1) | Same roles (§3.1, paragraphs S3.SS1.p1–p8; ELBO Eqs18–20) | No revision to the shared learned network plus private sampled-geometry overlap. |
| Geometry distribution | Product over incident node-edge pairs; endpoint MLP predicts its factors (§3.2) | Product assumption and tractability rationale remain explicit; ordered endpoint features predict the factor parameters (§3.2, S3.SS2.p1–p4/Eq21) | No global stochastic graph-mode latent or persistent member parameter bank is introduced in this inspected scope. Shared phi still couples learned distribution parameters; the draws remain conditionally factorized. |
| State and layer sampling | Projected input features generate the distribution; iid sheaves, one per network layer (§3.3) | The same construction remains (§3.4, S3.SS4.p1–p5) | This retains the distinction from E's maps generated from each route's live source/receiver states. |
| Acquisition | One-sample MC likelihood/ELBO estimator, weighted/annealed prior KL (§3.1/§3.3) | Same estimator/annealing and backpropagated network update (§3.1 and §3.4, S3.SS4.p6) | E's persistent member own-CE acquisition without posterior KL remains a declared procedure difference; it establishes neither better learning nor a new principle. |
| Prediction ensemble | Testing ensembles predictions (§3.3) | V2 explicitly says multiple sheaf samples are drawn from the learned posterior and corresponding forward passes ensembled (§3.4, S3.SS4.p6) | The sampled predictive-ensemble reading is clarified, not removed. E uses a fixed bank of point-estimated factors and probability serving. |

V2 reorganizes the method: the Cayley-distribution treatment is now §3.3 and architecture is §3.4, replacing the old architecture locator §3.3. The method block also contains separation/distribution theorem statements; their proofs and validity were not audited or transferred to E. No author code, published scores, full-paper comparison or stochastic baseline execution was adopted.

## Publication status

The canonical arXiv page lists v2, no journal reference and no publisher DOI; its DOI is the arXiv-issued `10.48550/arXiv.2410.09590`. Saved OpenAlex metadata exposed only the repository record. An author-constrained Crossref query returned unrelated works; one title-only follow-up also returned no exact BSNN match. No official proceedings or journal publication is verified by this bounded check. This remains **unresolved**, rather than a claim that none exists; metadata searches cannot prove absence. No further search or blocked-endpoint retry is nominated.

## Exact primary custody

| Primary carrier | Bytes | SHA256 |
| --- | --- | --- |
| Canonical arXiv abstract page, retrieved 11 October 2026 | 41,706 | `97d5b8249ca00c383515d00c7aab942ff1b9947ee55d116518b65f03d2579fdf` |
| `https://arxiv.org/html/2410.09590v2`, retrieved 11 October 2026 | 669,925 | `3e08236aa19bf9ef6a8af156f0ff53b8d40ff48592a0f56433cdd5aeff97288c` |
| Retained `https://arxiv.org/html/2410.09590v1` | 549,707 | `3335bf627ed7d30c6bc40af86d4f4b49f4a2ed443b57dd3d58198ea6fbb320bf` |

`VERSION_AND_PUBLICATION.json` records exact submission dates, retrieval timestamps and URLs; `READ_SCOPES.json` distinguishes the bounded method revisit from metadata and unaudited exposed statements. Raw primary bytes and request receipts are retained in `sources/`.

The useful update is citation freshness and repaired section locators. It does not manufacture a methodological gap, private-builder necessity, posterior interpretation of deterministic factors or superiority over BSNN. Existing confirmation priorities and failed recipes stay unchanged; V4 is untouched.
