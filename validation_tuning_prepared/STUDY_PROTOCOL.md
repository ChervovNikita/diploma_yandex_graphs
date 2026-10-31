# Frozen validation-tuning sensitivity study

This file is part of the prospective source hash. Freeze it with the runner,
independent verifier, copied model definition, four raw graph datasets, and
graph-tensor fingerprints **before any study training**. The operational
cutoff for a complete, independently audited study is 2026-09-26 06:00 UTC.
Partial or unaudited outcomes do not support a complete-study claim.

## Matrix and architecture

Four published settings: Cora Planetoid public, WikiCS official split 0,
Actor Geom-GCN split 0, Platonov filtered Chameleon split 0. All use their
source features and masks, full-batch message passing, coalesced symmetrized
edges, and no explicitly added self-loops. Symmetrized filtered-Chameleon
scores use a different edge protocol from its source paper's directed graph.

Six arms: BASE ordinary SAGE; ENS four independently initialized ordinary
SAGE paths; TIED four boundary-projector paths with one stored SAGE stack;
PRIVATE-FIRST or PRIVATE-LAST with one private residual block per member;
UNTIED with both residual blocks private. The four projector arms have copied
canonical initial parameter values and post-construction RNG state, with
initial member logits checked numerically at 1e-5 absolute tolerance. BASE
and ENS use their ordinary initializations, so their contrasts are practical
comparators rather than initially matched interventions. Each graph uses two
residual SAGE blocks, width 128, LayerNorm, dropout 0.2, four members where
applicable. All runs use AdamW, mean member training CE for four-path arms,
and exactly 1,000 full epochs if finite.

Every arm and graph receives the same six-candidate grid: LR 0.0003, 0.001,
0.003 crossed with weight decay 0, 0.01. Each candidate uses optimizer seeds
0, 1, 2. This is 432 planned cells. Every epoch computes pooled raw-logit
validation accuracy and CE. The selected whole-model checkpoint maximizes
validation accuracy, then minimizes CE, then uses the earliest epoch. An
observed nonfinite seed invalidates its entire three-seed candidate; all
remaining planned cells still run. If all six candidates fail for a group,
the group and complete comparison are reported incomplete, with no invented
substitute setting.

## Validation selection and test firewall

The independent verifier checks all 432 cell identities, source freeze,
1,000-epoch traces (or documented nonfinite failure), trace hashes, checkpoint
hashes, selected epochs, and validation scores. Per graph and arm, a single
candidate is chosen by greatest arithmetic mean selected validation accuracy
across the three seeds, then lowest mean selected CE, then lower LR and lower
weight decay. The verifier writes one immutable, hashed global validation
lock with all 24 group choices before any test inference. Each arm's chosen
candidate and the prespecified (0.001, 0) default may then be tested, and no
other grid candidate may be scored on test labels. Fresh independent replay
checks those selected/default checkpoint predictions and decisions.

For an operational partial-family recommendation, PRIVATE-FIRST and
PRIVATE-LAST are each first tuned by the above rule. Choose between their
validation-selected configurations by higher mean validation accuracy, then
lower mean validation CE, then PRIVATE-LAST on an exact remaining tie. This
chooses one partial architecture using **12** configurations per graph,
versus **6** for each individual comparator arm; its selected-family outcome
does not have an equal total search cost. Both partial arms and all outcomes
remain reported. The selected candidates may use different optimizer
settings, so tuned-arm differences describe an accuracy/storage frontier and
do not isolate a causal effect of tying. Three seeds on one split measure
optimizer variation, not independent graph draws. Parameter count describes
stored parameters, not inference latency. A checkpoint at the epoch cap limits
convergence conclusions.

## Reproduction and provenance

Run `python tuning.py freeze` once, followed by `check-freeze`, `preflight`,
and `run --dataset ...` on the assigned GPU. The freeze contains exact source,
raw-file, graph-tensor, configuration, seed, and cutoff information. The
training bundle excludes test indices and labels. Only after collecting all
432 results in one study directory run `python verify_tuning.py
audit-and-lock`. Transfer the identical lock to each host, run `python
tuning.py score --dataset ...` for its assigned graphs, collect only the
allowed score artifacts, and run `python verify_tuning.py audit-scores` for
final independent replay. Preserve every cell, including failed candidates,
the frozen sources, full traces, checkpoints, lock, and score audit.
