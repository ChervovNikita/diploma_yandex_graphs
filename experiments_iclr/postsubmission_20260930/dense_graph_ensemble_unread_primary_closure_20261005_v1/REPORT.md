# Closest unread graph ensemble leads: primary scope closure

## Result

Two relevant method scopes are now read: **GETS v2** and **A3-GCN v1**. They establish adjacent graph-conditioned calibration and ensemble pseudo-label training operations. The inspected operators do not transport observed member-error Gram matrices. **Graph ensemble neural network** remains a metadata-only unresolved lead; it supplies no equivalence or absence verdict. The preceding scout is preserved unchanged.

| Lead | Primary scope obtained | Closest operation |
|---|---|---|
| GETS, arXiv2410.09570v2 | Complete relevant §§3–4, selected method/setup appendices, pinned author code | Graph-context supervised calibration of one frozen predictor; node-weighted calibration experts. Source is class-specific vector scaling and can alter argmax. |
| Adapt, Agree, Aggregate / A3-GCN, arXiv2503.17842v1 | Complete §4, Algorithm1, notation and selected setup/parameter prose | Augmented independent GCNs, confidence-set agreement controls pseudo-label training, one final consensus GCN. |
| Graph ensemble neural network, DOI10.1016/j.inffus.2024.102461 | Public publisher coredata only | Method unknown. |

## GETS: exact operation and critical paper/source difference

The paper learns node-specific mixtures of graph calibration experts using logits, raw features and degree inputs, with noisy top-k softmax weights. It describes NLL fitting on the validation/calibration set. Its scalar-temperature motivation and accuracy-preserving language suggest a positive common scale across classes.

Pinned author commit **4403410fbf730eae9c5d2018d8554a6b645ba50f**, dated2025-02-26, implements a materially broader map:

- Each GCN/GAT/GIN expert emits **C values per node**, rather than one scalar node temperature (GETS.py39,109,163).
- The gate sees projected raw features concatenated with the frozen base logits (386–389).
- The expert vectors are weighted first: t_ic=Σ_m g_im expert_m(i)_c.
- The final logits are z'_ic=z_ic·softplus(t_ic) (403–405).

Positive class-specific scales can change the predicted class. For example, logits(2,1) multiplied by positive factors(1,3) become(2,3). Therefore native source accuracy preservation is **unqualified**. A scalar scale common to all classes recovers the order-preserving temperature special case. This static source finding also makes graph-conditioned vector scaling a relevant accuracy comparator; discarding GETS as calibration-only would miss its inspected source operation.

All experts execute in the source (394–398), despite zeroed top-k output weights. No conditional-execution or efficiency benefit is adopted. Its coefficient-of-variation penalties concern expert importance/load; they are not member-error covariance. The source supports expert_select=M, so dense versus sparse weights does not create a novelty boundary.

The fitter caches base logits without gradients, fits calibration CE on masks[1]=VALID, and selects calibration CE on masks[0]=TRAIN. The base classifier itself was selected on VALID. This fitting opportunity must stay explicit in any comparison; it is not a TRAIN-only fusion recipe. No source code was executed or native reproduction qualified.

GETS is stated as **ICLR2025 Spotlight** by arXivv2 metadata and the pinned author README, which links OpenReview qgsXsqahMq. Official forum access returned browser-verification HTML, and public API routes returned403. We have not audited the accepted conference PDF. README uses an alternate “Nodewise Ensemble Calibration” title and a2024 BibTeX year; retain the exact arXiv title/version and these bibliographic limits.

## A3-GCN: ensemble agreement trains a served consensus model

The complete preprint method independently initializes GCNs on independently edge-dropped graph views. For each model i, high-confidence candidates satisfy max_c p_i(u,c)≥θ. A **global set-overlap ratio**

s=|intersection_i H_i|/|union_i H_i|

controls the random training-subset ratio and updates θ_j=θ_(j−1)+α(s_(j−1)−s_j). This is overlap of confident-node sets. Models can be confident on the same nodes while predicting different classes, so this ratio is not label agreement or an error-covariance statistic.

A separate GCN on the original graph is trained using majority-vote targets on nodes satisfying a class-agreement threshold. Native β=1 requires unanimity and can include low-confidence unanimous nodes. Algorithm1 returns the **single consensus GCN**. The member outputs supervise that model through targets; the method does not serve dense learned member weights or concatenate their hidden states into its decoder.

The source claims about reliable agreement and reduced confirmation bias are not guarantees: unanimous errors remain possible. The preprint leaves ratio-to-count conversion, empty-union behavior, bootstrap, threshold clipping and tied votes unresolved. Algorithm1 random-subsets L∪Lpseudoi, while prose emphasizes random subsets of confident nodes; protection of original labels is ambiguous. The linked author repository returns HEAD404, so there is no source resolution or native qualification.

A primary Springer preview now confirms the published successor **10.1007/s13042-026-03047-y**, Int.J.Mach.Learn.&Cyber.17,262(2026): received2025-03-22, accepted2026-02-22, published/version of record2026-04-24. Only its title, abstract, dates and metadata were read; the full published method is subscription content. The preprint provides the method evidence. Exact equality of published equations/Algorithm1 is unverified.

## Graph ensemble neural network: unresolved, with bounded routes recorded

Authors Rui Duan, Chungang Yan, Junli Wang and Changjun Jiang; DOI10.1016/j.inffus.2024.102461; Information Fusion, October2024 issue. The public Elsevier API returned1775 bytes of coredata and no abstract or method. A public FULL request returned401 and the public publisher abstract route403. No credentials or restricted access were used.

Exact-title arXiv search returned0. OpenAlex and SemanticScholar provided closed-access metadata and no OA copy. Author-copy search met a DuckDuckGo human challenge; two Bing queries produced unrelated results. These are retrieval limits, not a whole-literature absence finding. The full method remains unknown, including whether it uses dense prediction weighting, hidden fusion or residual statistics. A specific public manuscript or an authorized supplied copy is needed to close that scope.

## Consequence for the earlier accuracy hypothesis

The earlier graph-local uncentered-error-moment proposal remains an attributed composition, with no quality qualification. GETS establishes graph-context supervision and postprediction vector scaling; A3-GCN establishes ensemble-controlled graph self-training and a final consensus predictor. Neither inspected relevant operator explicitly transports labeled residual Gram fields. That scoped difference supplies no global novelty certificate, and generic graph stacking remains a strong capable comparator.

The source-informed nearest comparison is a small graph-conditioned vector scaler trained with the same permitted fusion-fit labels. This is a comparator role only: no native port, grid, data read or experimental admission is added. Preserve the earlier fixed disjoint fit/development masks through every moment, residual diffusion and corrector; a bank already selected using fullVALID still yields retrospective development.

READ_SCOPES.json records exact blocks, equations, Algorithm1 and author-source line ranges. QUERY_RETRIEVAL_LEDGER.json preserves exact URLs, UTC times, response hashes and failures. Two new method work identities, zero full-paper reads and zero old indexed primary revisits. Reported paper/README results were textually exposed where adjacent to method context; no numerical results, target data, histories, models/checkpoints or training were adopted or accessed.
