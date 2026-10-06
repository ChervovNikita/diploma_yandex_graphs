# Shared native encoder with four full NCN heads

This supplemental capacity control separates the restriction of diagonal factorized heads from sharing the encoder. The active model has one shared trainable native NCN encoder and four independently stored, fully trainable nonlinear native CNLinkPredictor heads. All five blocks are copied from the same successful TRAIN-only warm state. No initialization perturbation, r/s factors, extra warm fit, or method-novelty claim is introduced.

Shared lower computation with private upper networks is established: TreeNets, arXiv1511.06314v1 §5, in the saved scoped assessment; and CAMERO, arXiv2204.06625v1 §3.1, in the saved primary conclusion. This control uses the original full graph-aware NCN heads rather than only linear output heads. No published quality or exact training-recipe equivalence is inherited.

## Parameters and native computation

| Block | Ownership | Parameters |
|---|---|---:|
| Original width256 pureGCN encoder, input projection and JK | One shared trainable copy | 948,225 |
| Full nonlinear native NCN head | One independent copy per member | 463,362 |
| Four full heads together | Private | 1,853,448 |
| Complete control | Shared plus private | 2,801,673 |
| Original shared F4 reference | Shared plus factorized private | 1,434,634 |

Each native head retains its full xlin, xcnlin, xijlin and lin sequences, all native LayerNorm affine parameters, biases and beta. The head uses the current native support/common-neighbor operation. At a dropout-free serving call the encoder runs once, and all four heads score its common hidden state. The served output is the arithmetic mean of raw logits, exactly as in the initializer pilot. In an own-head training call only the requested head executes; the shared encoder uses that route's existing independent dropout stream.

The donor warm model is released after the initial audit. The model constructor does not retain the donor through a closure. Runtime assertions require disjoint parameter objects and storage among the encoder, all heads, and the donor, exact copied native states, native classes, the counts above, and trainability of every parameter. The driver has no factor attributes and records perturbation and first-R statistics as null.

## Matched training and selection

The driver is derived from sealed initializer run.py SHA256 b319ab6b2d523aeee60858c16fe1b5d97a71861fd135fc5cc013526d630d071b. Its own_loss, ordinary_step, Adam, metric, state/checkpoint and complete-TRAIN audit functions are unchanged at the AST level.

Reuse the same successful same-seed/same-provider 20-cycle TRAIN-only common warm checkpoint for seeds0/1/2. Fit60 complete postwarm cycles. Every head gets its own existing inner route list, as F4 does. One joint Adam covers the shared encoder and all four heads. The three committed passes remain mean own inner BCE, 0.5 pooled raw-logit BCE plus 0.5 mean member BCE on the outer queries, and repeated mean own inner BCE. This differs from the independent-warm4 control's four optimizers and union inner queries.

Complete227-positive by500-negative VALID evaluation occurs every five cycles. The first maximum MRR rounded to four decimals selects one pooled checkpoint; member metrics accompany it. Optimizer/dropout reset, actual draws and masked support, initial/final states, selected state and provenance logging retain the parent contract. TEST remains closed. Preserve all original initializer conditions, seeds, comparisons and scores.

## Native qualification before fitting

Root execution first checks exact full-TRAIN deterministic equality of every copied head to the warm native single. One existing masked TRAIN episode then checks each own route's gradients reach the shared encoder and its complete native head, while other heads remain unused. It checks that the mean own loss and pooled-plus-own loss connect every active model parameter with finite gradients. The qualification performs no optimizer step, advances no dropout stream, and uses separate sampling streams. It cannot change the first training draw or select a condition.

These runtime checks are implemented, not executed by the author. Only AST, source-delta, JSON, hash and integer parameter-count checks ran here. No numerical framework, model, dataset, label, score, checkpoint or server was accessed.

## Root activation

COMMANDS_DISABLED.json contains three fit previews for each existing provider. Only shared_full_heads is admitted; the supplemental driver rejects warm phases and all other conditions. Root copies the complete sealed supplemental packet, activates a separate external PLAN/job, binds the packet manifest and same-seed warm FREEZE/checkpoint, supplies new source-review evidence, approves the native gate, and attaches the existing finite hard/resource supervision. Existing provider custody and native dependencies remain authoritative. The allocation environment includes the already accepted dependency overlay correction;77 follows the root's BPR/capacity release. No server7 route is added.

The templates are disabled. The existing F4 prior caps15000s soft/18000s hard, GPU10GiB, RSS64GiB and output/log limits are reused as proposed bounds; actual cost remains unmeasured. The first native gate and all training/evaluation work are charged. No new owner process, launch, retry, horizon reduction or automatic deadline expansion occurs in this packet.
