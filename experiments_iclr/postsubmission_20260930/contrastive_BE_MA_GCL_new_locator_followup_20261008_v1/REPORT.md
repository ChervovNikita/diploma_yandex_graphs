# MA-GCL: recovered author locator and scoped method boundary

8 October2026. The unresolved MA-GCL method was recovered through one genuinely new locator. The accepted DOI metadata names the authors'GitHub repository; its README links the paper'sarXiv version. No failed publisher request was repeated. This is one new primary method scope, zero full-paper/code-audit credits and no scientific execution.

## Locator and identity

- Accepted-paper identity: [MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning](https://doi.org/10.1609/aaai.v37i4.25547), Xumeng Gong, Cheng Yang and Chuan Shi, AAAI2023. OpenAlex metadata associates the exact title/DOI/authors with the accepted publication.
- New author locator: [GXM1141/MA-GCL](https://github.com/GXM1141/MA-GCL), explicitly listed in that metadata. Only its README was used to locate the paper; implementation files were not read.
- Inspected body: [arXiv2212.07035v1](https://arxiv.org/abs/2212.07035v1), PDF SHA1c2e03d80535e885218ff9e7693d68fa8963db7a870f229edc9a6eb4d3b832c3. The preprint title/authors match. Exact accepted-version/body equivalence remains unverified.

## What the method already contains

The proposed Methodology section, PDF3–5, and the immediately referenced operator/positive-pair definitions were inspected. Pages3/5 were rendered to check the loss and Algorithm1. No benchmark results, appendix proof or author implementation was audited.

| Operation | Inspected specification |
| --- | --- |
| Shared encoder weights with different views | Keep the same learnable nonlinear transformationsh_i, but assign different numbers of linear graph propagationg operators to the two view encoders. Different graph receptive fields under tied weights are explicit prior. |
| Asymmetric strategy | Different total propagation countsL/L' with the same transformation count. This intentionally separates model views without untying the transformation parameters. |
| Random strategy | Draw propagation counts every epoch rather than retain one fixed depth. |
| Shuffling strategy | Vary where propagation operators occur among nonlinear transformations. The isolated strategy describes matched total propagation count; the combined Algorithm1 additionally requires different total counts and different per-position counts. These are not silently equated. |
| Combined training | Draw two edge/feature data augmentations; form their graph filters; use the sharedh_i and projector in two encoder compositions; minimize their contrastive loss. The operator-order argument also discusses identical graph inputs/parameters, so unchanged input topology by itself would not distinguish GNNM. |
| Positive relation | Printed Eq3 pairs the same observationi across encoder views. Other indicesj≠i supply its own-view denominator negatives. No TRAIN-class-compatible positive-neighbor selection or different such relation per persistent route is prescribed. |
| Evaluation | Fix an encoder architecture with equal per-position propagation countsK and remove the embedding projector. The inspected operation does not retain a jointly supervised four-member predictive BE bank. |

The graph filter, different propagation lengths, shared encoders, model-view diversity and operator-order augmentation therefore cannot be advertised as GNNM inventions. The printed denominator should not be silently replaced by a different modern InfoNCE convention when reproducing this source.

## Relationship to current context steering

The complete operation differs from the frozen context9 intervention: context9 keeps the native predictor graph and architecture, assigns persistent different TRAIN-label-compatible neighbor-positive targetsQ_m to internal BE routes, retains own classification supervision and serves all four predictions. MA-GCL instead varies shared encoder compositions across training views and keeps the same-observation positive correspondence in the inspected loss.

This scoped difference is not novelty clearance. It neither establishes that every part of the GNNM recipe is unpublished nor excludes related supervised/mined-positive variants. SupCon, BotSCL, AMCL/CGCL and the retained positive-mining leads remain necessary ancestry. PMGCL's complete method is still unresolved; its saved abstract was not reread and no new locator for it was attempted.

The spectral discussion removes nonlinear transformations, uses one-hot inputs and orthonormal collapsed weights, and substitutes a positive-alignment surrogate for the full contrastive loss. It does not supply a theorem for native nonlinear supervised BE performance, member competence or useful pooling. Numeric spectrum examples were visible in the method page but were not adopted as evidence. The proof appendix and experiments remain unread.

## Disposition

Update MA-GCL's knowledge state from inaccessible method to **author-linked preprint method scoped; accepted-body equivalence unverified**. Retain the already running context9/Wiki12/Mol18 families, sources, targets, gates and original scores unchanged. No new algorithm or experiment is admitted from this reading.

For eventual interpretation, keep the attention caveat too: private score ownership is not automatically a strict capacity gain. Existing BE scales admit restricted attention/value compensation; native normalization/mixed paths block the simple general construction, not all possible whole-network representations. A later private-attention gain could be an optimization/parameterization effect. Nothing in MA-GCL resolves that separate question.

## Evidence and scope

The isolated folder retains exact locator requests, accepted DOI metadata, author README/body hashes, the pinned PDF, automatic page extraction, selected method/definition text and rendered method pages. Automatic extraction of9 PDF pages is not semantic full-paper reading. The README's run command and rendered experiment-opening prose were incidentally visible; no command was run or experimental claim adopted. Retained AMCL/CGCL/ASPECT/filter bodies were not reread. No dataset, model, target, checkpoint, current outcome, server or GPU was accessed.
