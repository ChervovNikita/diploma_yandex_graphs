# MORGAN full-primary resolution attempt

## Result

**Full primary remains unresolved; the central source distinction is resolved for the inspected pinned implementation.** The new routes did not produce a MORGAN paper PDF, full HTML, or full XML. Verified author code learns spectral operators from eigenvectors, gates them from eigenvalue summaries, and applies them to node features during ordinary supervised training. The inspected paths do not construct graph-filtered training residual/cotangent VJPs to initialize expert-private parameters. This is a bounded source finding, not a full-paper or exhaustive absence claim.

Target: **MORGAN: To Bridge Mixture of Experts and Spectral Graph Neural Network**, Lihui Liu and Yuchen Yan, DOI [`10.1609/aaai.v40i28.39553`](https://doi.org/10.1609/aaai.v40i28.39553). New Crossref metadata identifies the version-of-record PDF and publication date **14 March 2026**, Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, issue 28, pages 23783–23791. The registered PDF URL is the same previously failed URL; it was not fetched again. No preprint ID/version was established.

## What changed

A GitHub author search led to `lihuiliullh`, whose profile links a [homepage](https://lihuiliullh.github.io). The homepage identifies Lihui Liu as a Wayne State assistant professor, matching the cached paper affiliation. Its [publications page](https://lihuiliullh.github.io/publications/) lists the exact MORGAN title, AAAI 2026, and both authors. The author's repository listing supplies **[lihuiliullh/Morgan](https://github.com/lihuiliullh/Morgan)**.

The repository is pinned to commit **`a31af5d9ac4c838cb1dfbeb4131852bafe86e6c6`**, **27 November 2025, 14:16:38 UTC**. Its README declares itself the official repository for the exact title, names Liu and Yan in the citation, gives matching Wayne/Illinois contact addresses, and cites the **40th Annual AAAI Conference**. Its introductory prose says “accepted at AAAI 2025,” conflicting with the publisher and author's AAAI 2026 record. Preserve that discrepancy. This lane did not qualify a native source implementation or silently correct the README.

The repository tree contains README, model/training/configuration text paths and dataset paths; it contains no paper PDF/HTML/XML. README plus six targeted model/training/configuration text files were retrieved; all seven match the pinned Git blobs. No author code was imported or executed, and no data files were downloaded. The author's publications entry has no MORGAN paper link. The complete, non-truncated author-site source tree has three generically named small PDF assets and no MORGAN/spectral-named paper asset. Those unlinked, unidentified PDFs were not treated as MORGAN or fetched.

This verified author-source lead was not found by the earlier broad/exact repository queries. The narrow source findings below resolve the most important implementation question; full-paper theory and source-to-paper equivalence remain open.

## New routes and access limits

The prior failure log was consulted first. The prior article-view URL and article-download URL were **not repeated**.

| New route | Outcome | Supported conclusion |
|---|---|---|
| Crossref exact DOI record | Success | Canonical title/authors/date and registered version-of-record PDF target; only the previously failed PDF URL is registered. |
| OpenAlex content PDF and GROBID XML endpoints named by the saved work record | Both HTTP 401 | Full-text service requires authentication in this session; no paper bytes obtained. |
| Official OJS OAI `GetRecord`, exact article identifier | `RemoteDisconnected` | New official metadata route failed; no record/full text obtained. |
| Official OJS separate article/galley-view path `39553/43514` | `RemoteDisconnected` | New PDF-view route failed; no viewer source/paper bytes obtained. |
| arXiv exact title phrase | Search returned no results | No matching preprint located by this query; no absence inference. |
| arXiv title “MORGAN,” first 50 returned results | Returned unrelated titles | No matching item located in the inspected results; not an exhaustive preprint search. |
| Wayne site search | Generic search page only | No additional paper link extracted. |
| GitHub user profiles and profile-linked author homepage/publications/repository list | Success | Verified author and official MORGAN repository lead. |
| Pinned MORGAN README/tree/head and author-site tree | Success | Official author-source identity, revision, README year discrepancy, and no identified linked full paper in these inspected routes. |
| Six pinned model/training/configuration text files | Success | Narrow static source distinction and native optimization-path discrepancies; no execution or full-paper claim. |

There were **25 new retrieval requests**, of which 21 returned bytes and 4 failed. All request URLs, UTC times, response types/errors, saved byte counts and hashes are in `RETRIEVAL_LOG.json`. Searches were restricted to this paper and its authors. No unrelated primary papers were read.

## Exact pinned source findings

`model.py:61–181` defines the inspected `EigenExpert` and `Morgan` path:

1. Construct five separate `EigenExpert` MLPs, a gate, a feature map and a shared classifier. The constructors instantiate ordinary `nn.Linear` modules; no checkpoint copy or label-residual seeding operation appears in this path.
2. Divide the eigendecomposition into five contiguous **index groups**. The ordering/normalization convention of the uninspected eigensolver helper is not inferred here.
3. Apply each expert MLP to scalar entries of each eigenvector and average over nodes to obtain learned per-eigenvector weights `V`. Construct `A_b = U_b diag(V_b) U_b^T`. The source does not multiply `V_b` by eigenvalues in this expression.
4. Feed the five mean eigenvalues to the softmax gate. This creates **one band-weight vector per supplied graph/subgraph**, not a node-specific or label-residual-specific gate. Expert and gate inputs do not include labels, prediction errors or CE cotangents.
5. Sum the gated operators and apply the result to node feature matrix `X`, followed by shared feature map, optional batch normalization, activation, dropout and classifier. The classification output is one fused log-probability prediction, not four separately continued/poolable member predictors.

`train.py:257–324` constructs `Morgan` for the admitted single-layer/non-residual setting, creates Adam over its parameters, passes sampled **node features**, eigenvalues and eigenvectors to the forward path, then uses key-node labels for NLL, ordinary `backward`, and `optimizer.step`. The source class-count dimension depends on labels; that is distinct from a label-dependent private-parameter seed. The same file's `preprocess_eig_file` accepts topology/sample information and calls the eigensolver before model training; it does not receive prediction residuals/cotangents (`train.py:15–37`).

A targeted symbol search across the six retrieved text files found the ordinary loss backward calls and no special warm-checkpoint load, graph-filtered supervised cotangent construction, VJP/JVP initialization call, gradient-orthogonal centering, or Armijo routine. Ordinary backpropagation is itself a derivative operation; the supported difference is the absence of the **specific explicit initializer construction in these inspected paths**, not an assertion that MORGAN uses no VJPs mathematically.

Native optimization paths require caution. `config.py:1–54` defaults to `optimization='am'`. `am.py:50–91` toggles only top-level `nn.Linear` and `Velocity` modules; `Morgan` has no `Velocity` child. Nested expert/gate parameters are not toggled by those branches, while shared top-level linear maps are frozen in mode `d`. The AM outer path constructs a fresh `Morgan` and alternates ten-epoch ordinary NLL phases (`am.py:113–139`). The GP path instead constructs `SLOG_B_gp` and tunes two scalar values with a Gaussian process while continuing the same model/optimizer (`gp.py:110–176`). These are source observations, not a verified paper recipe or behavioral test.

The exact source scopes and excerpts are saved in `READ_SCOPES.json` and `evidence/PASSAGES.json`; the comparison is summarized separately in `SOURCE_FINDINGS.json`.

## Consequence for the round15 cotangent initializer

The **cached abstract**, not a new full-method read, supports graph Laplacian eigendecomposition, frequency-band partition, dedicated per-band experts, learned input-dependent spectral fusion, and a localized sampling variant. Thus frequency-band expert assignment remains a relevant published ancestry that the candidate must acknowledge.

The round15 candidate instead defines graph Bernstein operators acting on a masked training CE logit cotangent, then parameter VJPs into active route-private boundary factors of an identical warm predictor, with gradient-orthogonal centered directions, common descent, functional/null checks, and finite Armijo checks. Its active slices and continuation/output contract remain those of the immutable round15 packet. Against the inspected source, the concrete delta is **one-time supervised cotangent-to-private-parameter expansion** versus **continuously learned eigenvector-based filtering of node features with fused prediction**. General spectral expert ancestry is shared; the exact initializer is not present in the inspected MORGAN paths.

The full paper's projector/filter formulation, expert initialization description, any paper-level warm/copied pretraining or gradient-space diversification, losses/regularizers, spectral decomposition convention, theoretical guarantees, native cost, and equivalence to the inspected revision **remain unresolved from a full primary**. Static source narrows the implementation comparison; it does not establish an exact full-paper novelty boundary.

Do not reject the candidate because it has spectral/expert ancestry. Do not claim a supported original extension or utility until the precise near-prior boundary and the prospective utility controls are established. The unchanged warm-copy null from the separate StarSSE/FAGEL review remains a useful required control for initializer utility; this retrieval packet changes none of that protocol.

## Preservation and next step

This is a new source/literature-only packet. Prior conclusions, scores, proposal/source packets and the closed coordinate/factor NO_GO branch were not modified. No SSH, checkpoint/tensor/dataset load, author-code import, scientific execution, fitting/scoring, testing or compilation occurred.

`PAPER_CONCLUSIONS.json` retains full-primary-unresolved status while adding the verified author-source revision. `READ_SCOPES.json`, `evidence/PASSAGES.json`, `SOURCE_INVENTORY.json`, `PRIOR_BINDINGS.json`, `RETRIEVAL_LOG.json`, and `MANIFEST.sha256` make the access result reviewable.

The next useful step is access to an author-linked/publisher full paper or a separate engineering qualification if a native MORGAN control is intended. The current static source finding is sufficient to preserve the candidate's specific initializer hypothesis as unresolved rather than treating spectral expert ancestry as exact duplication. No further broad search or repeated failed URL is warranted by this bounded attempt. **No exhaustive absence claim is made.**
