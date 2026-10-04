# Prospective decision after the native gradient diagnostic

Recorded before seeing any native J_K/J_K_sep gradient output. Original paper scores remain unchanged.

## Scientific question

Each of four members scores residual candidate neighbours around both ends of an edge. The auxiliary observes TRAIN neighbour membership while conditioning on the observed count. J_K assigns one latent member responsibility to the two ends together; J_K_sep assigns responsibilities independently. The practical question is whether the joint supervision improves the existing count-free edge ranking after training. Different gradients alone do not answer it.

Conditional Bernoulli likelihoods, mixtures, cardinality inference, topology/degree supervision (MaskGAE), and shared mixture components over graph blocks (GRAN) are prior ideas. Any new contribution must be the supported graph-specific construction and predictive transfer beyond relevant controls. It cannot be generic mixture novelty or a theorem that shared mixtures are novel.

## First native diagnostic

Use the unchanged first complete 65,536-positive/65,536-negative TRAIN batch, seed0 and the sealed initialization. Compare target, J_K and J_K_sep derivatives from the same retained graph and parameters. No optimizer steps or heldout scoring. Preserve zero-signal, resource-limit and failure outcomes. No different batch may replace an unfavourable result.

The failed first execution was a CUDA startup failure before loading data or constructing a model. The separately reviewed v4 startup correction has the same scientific workload in a fresh execution directory. It is an engineering correction, not additional predictive evidence.

Proceed to a small paired predictive study only if the actual diagnostic yields finite nonzero auxiliary derivatives connected to the model, a genuine joint-versus-separated distinction on native informative supports, and plausible full-training cost. Report the magnitudes relative to the target gradient and the resource costs. Do not treat the cosine with the target gradient as a predictor of generalization. If the gradients agree within the fixed rule, report that result and examine its cause before proposing training; do not invent a useful mechanism.

The complete TRAIN census estimates both sides variable on5.8242% and both genuine on1.3599% of positive queries. This sparsity is a real threat to useful transfer. No whole-dataset predictive claim follows from informative examples alone.

## Small representative development comparison

Use three paired seed blocks, the same complete TRAIN graph, native masking/negative sampling, official author recipe, full epoch budget and VALID-only checkpoint selection. Arms are target-only F4, F4+J_K, and F4+J_K_sep. Preserve the existing target loss, count normalization and auxiliary coefficient1 rather than select coefficients from this result. Equalize update counts and data streams where construction permits; record any stream differences. Report all three seed blocks and paired uncertainty, not the best seed or epoch. Compare against the already completed native single and independent ensemble as context; these are not newly matched training runs.

The already consumed Collab TEST cannot be called unseen. Use VALID for this development decision, disclose the history, and freeze the successor before evaluating an independent benchmark. A later Collab TEST score would be a disclosed development-associated observation. A promising VALID result is insufficient to establish generalization or superiority over the field.

If joint supervision beats target-only but not J_K_sep, the evidence supports auxiliary supervision, not cross-side member association. If the separated control matches it, simplify the claim. If neither arm improves, preserve both and close this candidate rather than broaden a tuning grid. Ordering-sensitive structured-single comparisons cannot demonstrate an equivariant ensemble advantage.

## Independent confirmation and stronger methods

Require a second graph/link-prediction benchmark with official splits and a competitive native baseline before a methodological claim. Pubmed-HeaRT is a candidate only after its native recipe and preprocessing competitiveness are resolved. Existing low validation MRR and repeat controls are not predictive success. Include a relevant recent strong link predictor using an identifiable public recipe (competitiveness audit pending), plus the native independent ensemble. Training and test release need a separately frozen representative plan; this memo authorizes no execution.

## Implementation cost

Exact count-conditioned dispatch may be the practical bottleneck. The support-bucket implementation is a separate disabled hypothesis. It may replace the direct implementation only after equivalence and native resource evidence, with source identity preserved. A faster exact implementation is useful engineering, but cannot by itself establish the requested predictive contribution.
