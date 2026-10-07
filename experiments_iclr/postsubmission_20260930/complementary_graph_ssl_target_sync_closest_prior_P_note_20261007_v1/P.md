# Complementary graph reconstruction and branch learning

The balanced quarter schedule in the earlier SSL note changes the order of reconstruction targets seen by each private branch. It does not create different branch objectives. Three distinct primary method reads sharpen that conclusion and show ways to adapt shared representations without acquiring independent teachers first. A separate persistent graph-target alternative is recorded in ALTERNATIVE.md; neither proposal is launched or added to the existing studies.

## Closest newly scoped methods

| Method and source scope | Actual objective and schedule | Relevant overlap and boundary |
| --- | --- | --- |
| [DIBS](https://arxiv.org/html/2003.04514v3), notation and main method, blocks 25–50; [AAAI2021 DOI](https://doi.org/10.1609/aaai.v35i11.17163) | A trainable shared encoder f(x) feeds stochastic private latent generators and task decoders. Member likelihood is combined with a variational information-bottleneck interpretation. A discriminator distinguishes pairs from the prior, the same latent head and different heads; generator/discriminator objectives promote inter-head separation and prior matching. Training alternates likelihood, generator and discriminator objectives. The main classification readout is the modal member class. | Direct shared-encoder/private-head diversity ancestry with a continuing diversity objective. The shared encoder is updated; no pre-acquired independent teacher bank is specified. This is latent-distribution diversity with likelihood anchoring, not complementary feature reconstruction, balanced target order, or a graph-specific repair mechanism. Printed adversarial explanations are not an implementation audit. |
| [CAE-Ensemble](https://arxiv.org/html/2111.11108v1), Algorithm1 and Sections 3.2.1–3.2.4, blocks 35–39,72–106; [PVLDB DOI](https://doi.org/10.14778/3494124.3494142) | Grow convolutional autoencoders sequentially. Transfer a random parameter fraction from the previous partially trained model. A new model minimizes squared reconstruction error minus λ times squared distance from the current ensemble reconstruction. The final anomaly score is the median of member reconstruction errors. | Explicit reconstruction competence versus continuing output diversity, with warm parameter reuse and no independently acquired teacher bank. It reconstructs the same underlying time series, not different persistent graph components. Transfer is not established as one permanently shared frozen encoder; transferred-parameter freezing and exact ensemble-detachment semantics remain unresolved in this paper scope. Anomaly scoring does not transfer to full-label graph classification. |
| [ParetoGNN](https://arxiv.org/html/2210.02016v3), Section 2 and AppendixB, blocks 15–43,95–124; ICLR2023 acceptance stated by primary arXiv metadata | One trainable graph encoder and task-specific heads learn masked features, topology, representation decorrelation, node–graph mutual information and node–subgraph mutual information. Each task has its own sampled/augmented graph. MGDA minimizes the norm of a convex combination of task gradients on shared encoder parameters, then supplies the resulting weights to joint task optimization. Learned node embeddings are used for downstream tasks. | Actual teacher-free graph multi-task adaptation of the shared representation. Complementary objectives are different graph pretexts, rather than different row orders of one pretext. Gradient reconciliation seeks a common descent direction, not prediction diversity among served members. Multiple task views require their own encoder work; it is not the one common masked forward design. Pareto stationarity of SSL losses is not a classification/common-error guarantee. |

DIBS and CAE publication metadata were verified through publisher-deposited DOI records. CAE's inspected source is an extended preprint linked to PVLDB, not a version-of-record equality audit. ParetoGNN's official OpenReview metadata API returned403; retain the primary arXiv acceptance statement without claiming accepted-body verification. No benchmark outcome is adopted.

## What BC versus BR actually tests

For one fixed corrupted input and four target quarters, write a branch's losses as ℓ0,ℓ1,ℓ2,ℓ3. BR visits the order0,1,2,3 in every branch. BC gives branch m the cyclic order m,m+1,m+2,m+3 modulo4. Every branch has the same target counts and aggregate loss:

Lbranch = (ℓ0 + ℓ1 + ℓ2 + ℓ3) /4.

The common encoder is frozen, and SSL private-path/decoder parameter sets are disjoint. There is no learned shared-parameter update or inter-branch term in this prephase. Assigning different targets concurrently has no extra coupling effect; the actual intervention is different per-branch target minibatch order before the later supervised pool fitting.

For stateless SGD and smooth losses, four updates in order π have the expansion

θend = θstart − η Σq gq + η² Σa<b Hπb gπa + O(η³),

with gradients and Hessians evaluated at θstart. The leading summed-gradient term is order invariant. This is a local mathematical comparison, not a theorem about the proposed AdamW runs: moment accumulation and adaptive normalization add their own order dependence. Accumulating all four quarter gradients at fixed parameters and taking one update would remove the BR/BC difference entirely.

The exact balanced rotation was not specified in the three new method scopes or the reused GraphMAE2 scope. Its predictive usefulness remains an empirical question, but it is a narrow order/optimizer control. It cannot establish a new complementary semantic-task objective or a distinct graph specialization principle. The prior 15-cell proposal should retain that interpretation.

## What freezing does and does not restrict

A decoder that sees only frozen H0 cannot recover information that H0 discards. The current private graph paths also receive raw node features and perform new message passing, so frozen H0 is not an absolute information bottleneck. The anchored logit z0 can be offset by learned residual logits. Shared wrong predictions therefore do not establish an irreducible common-error floor. Finite private capacity, available labels, optimization, representation mixing and fixed graph receptive fields can still limit repair.

DIBS, the saved ONE method, ParetoGNN and saved ADaMoRE update their shared representations during training. ONE constructs an online ensemble target while training shared/private branches; it does not first acquire a bank of independent teachers. ParetoGNN obtains shared-encoder supervision from public graph pretexts. These methods remove a freezing restriction, but none of the inspected scopes proves repair of this candidate's particular common errors. A frozen-versus-live encoder explanation requires a matched adaptation comparison and paid forward/backward accounting.

The saved FoRDE method already uses continuing repulsion between normalized true-label input gradients, including an explicit saturation motivation; it is not a one-time initialization or reconstruction-target method. Saved PCGrad reconciles conflicting shared task gradients, saved Self-Error Adjustment creates error-directed regression targets, and saved stochastic Multiple Choice Learning assigns examples to the member with minimum loss. These are reused ancestry, not new paper reads. None licenses treating arbitrary latent orthogonality as class-error repair.

## Persistent graph target alternative and prior limit

ALTERNATIVE.md proposes a separate, persistent objective using four fixed graph-frequency components of raw features. BernNet supplies the partition-of-identity filters. The closest saved graph combination is [ADaMoRE](https://arxiv.org/html/2510.21207v1): cohesive/dispersive structural channels, multi-hop low/high experts, dense residual experts, fused masked-feature reconstruction, CKA diversity and cross-filter reconstruction of learned expert representations. Its entire model is adapted, with alternating structural-view and main-model updates. Its reconstruction uses fused embeddings and a common decoder; the proposed alternative assigns fixed raw-feature components to separate private paths whose classifiers still predict all labels.

That distinction supports a bounded mechanism comparison. Low/high graph views, spectral experts, graph reconstruction and residual experts are established components. ADaMoRE is a preprint in the saved scope; its masking/cross-filter prose contains ambiguities, and no implementation or efficacy equivalence is assumed. Saved BernNet, MORGAN and TFE-GNN further close general claims about graph-band construction, frequency experts and filter combinations. Neither the new target formulation nor the search establishes originality.

## Scope

Three genuinely distinct primary method documents were newly scoped, with zero full-paper, proof, author-code or empirical-result credit. Saved conclusions and one retained ADaMoRE method excerpt were reused to answer the specific new target question; their rereads add no paper identity. Exact URLs, block ranges and source/reuse hashes are in SOURCE_SCOPE.json. Discovery metadata includes unsuccessful arXiv queries and other uninspected leads. Work used public literature and saved research notes only.
