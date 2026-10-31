# Shared graph weights with separate Adam moments: diagnostic study

Prospective design, recorded while the separate 432-cell validation-grid
training was running and before its outcomes were opened. This is a mechanism
test of the existing boundary-projector GNN. Separate per-member optimizer
state is related to prior multitask optimization and local-update methods,
including Yang et al.'s AdaTask (arXiv:2211.15055v2) and one-step FedAvg.
AdaTask already separates first and second moments by task. Our averaged
member updates adapt that idea to these graph ensemble paths. We do not
claim a new optimizer, a guaranteed descent step for the mean loss, or a
general benefit. The result may be positive, negative, or null.

## Fixed cells

Use the same frozen four graph tensors and split-0 masks as the validation
grid: Cora, WikiCS, Actor, and official filtered Chameleon. For every graph,
rerun TIED and UNTIED at the fixed default `(learning rate 0.001, decay 0)`
for seeds 0, 1, 2. Give SYNC the prospectively fixed six-candidate grid:
learning rate `{0.0003, 0.001, 0.003}` crossed with AdamW weight decay
`{0, 0.01}`, again with seeds 0, 1, 2. Each graph has six control cells and
18 SYNC cells, totaling **96 fresh cells**. Every cell uses one runner,
width-128 two-block SAGE, four boundary-projector paths, LayerNorm, dropout
0.2, full-batch message passing, and exactly 1,000 epochs if finite.
All three arms minimize the mean of four member training cross-entropies.
TIED has one graph stack with accumulated mean-loss gradient. UNTIED has four
independent graph stacks. SYNC has four graph stacks with **identical weights
at every forward**, but separate AdamW first and second moments.

For SYNC, the mean-loss backward gives each private graph copy `g_m / 4`,
where `g_m` is that member's loss gradient with respect to its graph copy.
Immediately before AdamW, multiply **only** each private graph gradient by
four. AdamW then sees `g_m` for each copy, using the candidate's learning
rate, betas, epsilon, and decay. At the default candidate these values match
TIED exactly. Shared boundary projector and final
normalization gradients remain gradients of the mean loss. After AdamW,
arithmetic-average every corresponding graph parameter over the four copies
and copy that value back into all four. Keep their optimizer moment tensors
separate. This compares AdamW on the mean graph gradient with the mean of
member-specific AdamW updates while forcing equal graph weights. Decoupled
weight decay is zero in the default intervention and 0.01 in half the SYNC
tuning candidates. The four copies can be collapsed
to one stored graph stack for inference, but training retains four graph
parameter tensors and four moment pairs.

## Required implementation gates

Before any mechanism training, freeze source, model, original data/tensor
manifest, all 96 cells, numeric tolerances, and this protocol. Independently
review source and test firewall. On CPU and then one free A100, run an SGD
identity smoke with zero momentum and zero decay for five steps from matched
initialization. TIED's update on the mean gradient must match the averaged
per-member updates of SYNC within calibrated numerical tolerance. Match the
dropout RNG states and assert the two arms finish each step with the same RNG
states. Run an AdamW one-step smoke that checks graph weights equal across
SYNC copies after synchronization, four distinct moment storages, and a
nonzero difference between member moments. For every training epoch, assert
the four graph copies are exactly equal before forward and after averaging.
At each selected validation checkpoint, collapse SYNC into the ordinary TIED
module and check all member validation logits and class decisions numerically.
The four projector arm constructions used here must have matched canonical
initial graph/boundary parameters, RNG states, and member logits within the
original 1e-5 absolute tolerance for each seed.

Select the whole-model checkpoint each epoch by highest pooled raw-logit
validation accuracy, then lowest pooled validation CE, then earliest epoch.
For each graph, choose one SYNC learning-rate/decay pair for all three seeds
by highest mean selected validation accuracy, then lowest mean selected
validation CE, then lower learning rate and lower decay. Record all six
candidate validation means before test scoring. A candidate with any
nonfinite seed is invalid. If all SYNC candidates fail, report the graph
incomplete rather than substitute a setting. No test labels enter training,
checkpoint selection, or candidate choice. An independent audit of all 96
traces, checkpoints, source/data hashes, and collapsed SYNC states writes a
validation-only mechanism lock. Test inference is allowed only when that
lock **and** the separate complete 432-cell validation-grid lock both exist
and match their frozen source/data manifests. Score the default TIED/UNTIED
controls and both the selected and predeclared default SYNC candidates after
those gates. No other SYNC candidate may be scored on test. Report every
candidate's validation mean, all scored seeds, and paired default-arm test
differences. Tuned SYNC versus default controls is descriptive because only
SYNC received a search. The three default arms supply the matched mechanism
intervention. Three
optimizer seeds on one split are optimizer replicates, not independent graph
draws. A selected checkpoint at the 1,000-epoch ceiling limits convergence
claims. The complete mechanism audit cutoff is 2026-09-26 06:00 UTC.

This study isolates a difference in optimizer aggregation under identical
forward graph weights. It does not establish a general rule for choosing
TIED, UNTIED, or SYNC on an unseen graph, and it does not remove the capacity
difference between the fully untied and tied arms.
