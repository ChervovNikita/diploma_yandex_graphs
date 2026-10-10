# Teacher-free conditional predictive families

## Decision

**No new methodological gap is established in this bounded scout.** A learned
stochastic predictive family is substantially different from another deterministic
repulsion or auxiliary-task loss, but the central constructions already exist.
The strongest useful next reference is an attributed graph-coherent **Rank-1
BNN**, not a claimed new GNNM principle. No fits, implementation or GPU request
are admitted by this report.

This conclusion follows the saved failure evidence: near-identical decisions
survive hidden separation, useful independent alternatives exist, and updates to
common parameters couple the learned paths. It does not infer that stochastic
inference will repair those decisions. The consulted reports are development
evidence on an already consumed WikiCS population, not untouched confirmation.

## The distinct mechanism considered

Replace a small bank of point-estimated private factors with a learned
distribution over compact factors. Keep one shared deterministic weight bank.
A complete graph-level draw generates one full predictor; serving averages its
class probabilities. The learning target is expected supervised likelihood plus
a posterior/prior regularizer, rather than merely increasing pairwise hidden
distances. There is no need to train four independent teachers first.

The plausible failure distinction is **point-mode collapse versus learned
within-mode uncertainty**. Four factor means can sit in nearly the same functional
basin; a fitted distribution may characterize that basin instead of mislabeling
four nearby means as four distinct competent hypotheses. This is an existing
Rank-1 BNN interpretation. It is a testable alternative explanation for the
project failure, not a demonstrated explanation of it or a new theorem.

One can additionally infer the factor distribution from a set of observed TRAIN
labels and their graph context. That changes available evidence and the
conditional prediction problem; it cannot be presented as a pure diversity
improvement. Message Passing Neural Processes already supply this core
teacher-free graph-label context-to-latent-to-predictor construction.

## Direct priors, checked before a nomination

| Prior and bounded scope | Verified mechanism | Consequence |
|---|---|---|
| **Rank-1 BNN**, arXiv2005.07186, §§3.1–3.5 | One deterministic W; stochastic rank-1 r/s factors; variational inference and priors; multimodal posterior components; comparison of component-average log likelihood with log pool likelihood | This is already the proposed compact teacher-free stochastic shared-backbone family. The paper explicitly discusses nearby deterministic components collapsing in function space and representing uncertainty around modes. A GNN port does not establish novelty. |
| **Message Passing Neural Processes**, arXiv2009.13895, main method/ELBO and generative description | Partially labeled node contexts; graph encoder; global Gaussian latent; graph decoder conditioned on a common sample; variational context/target objective and class-aware summaries | A shared model that constructs conditional node-label hypotheses from TRAIN evidence already exists. Replacing latent concatenation with BE factor generation would be an attributed parameterization change, not a new principle by itself. The native inspected likelihood is Gaussian over softmax/one-hot outputs; categorical CE would be a declared port, not its unchanged native recipe. |
| **Graph Neural Processes**, arXiv1902.10042, §4 | Context aggregation, local spectral edge features, a shared conditional decoder and categorical edge-value CE; training over graph families | Graph-conditioned prediction construction already has edge-imputation ancestry. Its deterministic context summary is not evidence of multimodal global label inference. |
| **Conditional Graph Neural Processes**, arXiv1812.05212, §2 | Bipartite graph-convolution functional autoencoder; graph context representation; target mean/variance prediction | Local graph-conditioned functional prediction and uncertainty are not an unoccupied design space. Its described conditional mean/variance does not itself establish joint multimodal label samples. |
| **Graph Neural Processes for Spatio-Temporal Extrapolation**, arXiv2305.18719v1, §§3.3–3.5 | Graph-conditioned Gaussian evidence aggregation, local hierarchical latent variables, context/target variational inference | Moving from one global latent to local graph-conditioned uncertainty is also established ancestry. Targets are explicitly independent in the inspected model, and the task is temporal extrapolation; a coupled categorical node-label process would need its own distinct mechanism and comparisons. Neither difference alone supplies novelty. |
| **Bayesian Neighborhood Adaptation**, arXiv2602.05358v1, §3 | Shared GNN likelihood; beta-process/concrete-Bernoulli latent hop-feature masks; Monte Carlo variational learning | Teacher-free stochastic structural hypotheses inside one GNN are a current direct prior. It adds no ready novelty clearance for another inferred graph mask or hop mixture. Proofs, native source and empirical results were not audited. |

The following previously saved boundaries remain active:

- BGCN already associates graph-conditioned weights with sampled topology;
  Bayesian node copying already offers task-conditioned neighborhood alternatives.
- SIG-VAE already propagates stochastic graph-latent noise and neighboring-node
  distribution information; Graphite already refines latent-induced topology.
- GMNN already learns graph label conditionals with approximate variational EM.
- ABMLL already has shared/global and task-local low-rank variational factors.
- GAMLP already uses reliable labels and propagation, so a label-context model
  requires a strong same-information deterministic comparator.
- Learned fusion and graph-conditioned aggregation have FFL, E2GNN/MGL/MoE and
  related ancestry. A changed decoder or pooling layer is not a clearance.

These are scoped component and operation facts. They do not assert an exact
collision with every possible graph latent-factor design, nor a global
literature absence certificate.

## A falsifier worth preserving as an attributed reference

The specific empirical hypothesis is: **a fitted, non-degenerate compact
factor posterior can recover useful full-class alternatives that the matched
point-factor bank fails to retain, without losing member competence.** The
conceptual screen below can falsify it. It does not turn an existing inference
framework into a novel method.

Use a complete task with all classes represented in every evaluation split,
several prospectively fixed matched splits/optimizer seeds, a capable native
backbone and a source-qualified inference recipe. No class or node subset is
allowed. Preserve every outcome and runtime failure. Do not recalculate or
replace the submitted paper's scores.

| Required condition | What it decides |
|---|---|
| Capable native single | Whether extra inference has useful prediction value |
| Matched deterministic shared point-factor bank | The exact point-mode anchor |
| Shared K4 Rank-1 posterior with one coherent factor draw per mode over the whole graph | The attributed stochastic-family candidate |
| The same fitted posterior evaluated at its component means, with no additional training | Whether integrating fitted uncertainty contributes beyond its learned means |
| Genuine capable independent4 | Whether the compact shared family actually recovers the intended alternatives |
| Matched untied stochastic factor paths | Whether any benefit depends on shared learning rather than merely stochastic regularization |

All constructors, optimizer groups, priors, KL normalization, posterior scale
parameterization, initialization, sampling schedule and validation selectors
must be source-qualified and frozen before execution. No prior variance or
temperature grid is specified here. The inspected papers do not provide a
qualified PolyFormer/node-classification recipe; a native weak stochastic port
would not be an adequate test.

For a graph forward pass, a weight-posterior draw must remain the same for every
node and all occurrences of its layer. Independently sampling each node's
factors changes the model into a node-latent stochastic field. Likewise, mixture
identity across layers must be explicit. These are model-definition requirements,
not new graph-coherence principles or claims about an uninspected implementation.
Training/public graph edges do not grant access to held-out labels.

Serve the predeclared mean probabilities; count actual posterior draws and
complete model forwards. Freeze the serving sample budget, repeat inference with
fixed independent sampling seeds, and report Monte Carlo variation alongside
split/optimizer variation; a favorable sampled bank cannot be selected. Use a
matched forward budget when making an inference-efficiency comparison.
Accuracy/NLL against both capable single and genuine
independent4, mean/worst component quality, any-correct coverage, common wrong
rivals, pooled-only rescues and introduced errors decide utility. Mixture
component means, stochastic draw predictors and an averaged component predictor
are different objects and must not be swapped in a reported member table.

**Falsify the hypothesis** if no competence-preserving whole-population coverage
and pool-quality gain appears relative to the point-factor bank. If the mean
evaluation matches the sampled predictive average, predictive integration is
unsupported; stochastic training could still have changed the fitted means.
Extra variance, entropy,
representation spread or an isolated good seed do not rescue that outcome.
If untied paths alone improve, shared learning remains unsupported. If stochastic
regularization improves a single equally well, ensemble-specific value is
unsupported. Any positive result would first support an attributed efficient
Bayesian graph reference, not a novel inference principle or clear accept.

Label-conditioned latent generation is deliberately **not** added to this first
falsifier. It would need a separate matched deterministic label-context model,
MPNP/GAMLP ancestry and a context/target split wholly within TRAIN. Mixing those
changes would prevent attribution of a posterior-mode benefit.

## Remaining research boundary

A genuinely distinct successor would need a precise reason why the new
conditional shared/private law represents useful alternatives that the direct
Rank-1 BNN/MPNP family cannot preserve, plus a matched same-information falsifier.
The following alone are insufficient: a hypernetwork instead of latent
concatenation, Gaussian-to-mixture replacement, local latent variables, graph
correlation, graph-aware kernels, a new KL weight, or an alternate optimizer.
These have direct ancestry or unresolved close priors and currently add no
demonstrated task mechanism. This scout therefore returns **no promoted new
method**, rather than spending fits on a renamed composition.

Six bounded primary paper scopes were inspected, zero full-paper reads. Complete
retrieval/extraction is mechanical and not additional reading credit. Some
method paragraphs contain public qualitative results; none are adopted as
project evidence. Rank-1 BNN's printed derivative expressions in §3.4 are not
imported as a certified derivation. No proofs, figure pixels, native author model
code, numerical results, runtime or predictive competence were qualified. No
model/data/checkpoint, pending Q/K or initializer outcome, server, canonical
memory, paper source, Desktop artifact or original score was touched.
