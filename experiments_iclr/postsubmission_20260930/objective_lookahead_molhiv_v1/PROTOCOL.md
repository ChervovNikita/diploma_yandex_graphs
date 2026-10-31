# Actual AdamW lookahead for molecular sharing decisions

## Status and purpose

This packet is **source-only**. It provides an engineering implementation and a
prospective mechanism screen. No MolHIV dataset, checkpoint or real screening
result has been admitted. No CPU execution is claimed by the source delivery.
The runner refuses real inputs before opening the supplied binding path.

The correct originally authorized one-GPU allocation must be verified before
any remote preparation or execution. The unrelated seven-GPU allocation is
outside the user's authorization. Runtime, data, checkpoints and results
acquired there are out-of-scope evidence and cannot qualify real bindings for
this packet. Local source preparation does not grant remote authorization.

The screen tests whether copying one GINE block changes the ensemble's pooled
training loss after a small, actual optimizer continuation. A gradient-based
criterion can disagree with AdamW because AdamW uses coordinate-wise moments.
The relevant comparison therefore advances both the tied and private models.
The unchanged source checkpoint is not the comparison baseline.

Positive results would justify a separately designed predictive study. They
would not establish predictive accuracy, a new method, superiority, or a
confirmed mechanism. Previous exploratory evidence motivated this design.
Screening placements on these graphs does not make the selected placement
independently confirmed.

## Unchanged model and future source requirements

The exact source files are:

| Source | SHA-256 |
| --- | --- |
| `new_tasks/gine_adapters.py` | `7416ddddd5a4057ec80624c648f32a5c8369c8c2628e8b58c450c82fe22e3c86` |
| `source_snapshot/models.py` | `07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303` |

The model has four boundary-projector members, width 128, three residual GINE
blocks, hidden multiplier 2, dropout 0.1 and one graph-level binary output.
Original OGB atom and bond categories are preserved. A private block includes
its bond encoder, LayerNorm, GINE MLP and epsilon. AtomEncoder, boundary
projectors and output LayerNorm remain shared. Every arm trains all its
parameters. First, middle and last placements each add three copies of one
equal-sized block. These controls match placement capacity; they do not match
the tied model's parameter count.

Real-input admission requires a **separately versioned, coordinator-approved**
packet with a successful grouping exporter, a qualified frozen CPU runtime,
passed CPU execution gates and all six complete P-only epoch-25 cells:
seeds 0, 1 and 2 at label fractions 10% and 100%. Each cell must include its
checkpoint, probe contract/protocol, fit trace, artifact manifest and successful
deterministic replay. Its source-training loss and label mask must be recorded.
Every continuation uses that cell's exact common source. No source is selected
using validation or test performance.

CPUv3 and the failed v5 grouping exporter are not valid parents. A qualified v6
exporter alone is insufficient: its matching CPU sibling and all six P25
artifacts from an authorized allocation must first pass admission. Previous
qualification on the wrong allocation cannot satisfy that prerequisite.
Nothing in this packet modifies previous
probe, selector, ledger or model files.

## Training groups and label visibility

P supplies source fitting. Q supplies this continuation and its mechanism
screen. P, Q-update and Q-screen must be disjoint by the accepted molecular
group identifier, including the accepted scaffold grouping policy. A successful
export and mapping audit must establish these identities before admission.

The future fold lock must record the exact IDs and its construction. Within Q,
sort the group IDs, shuffle them using Python's local random generator with
seed 704, allocate the first floor(number of groups / 2) to Q-update and the
rest to Q-screen. This split uses no labels. If an accepted parent already
supplies this exact division, reuse its lock. Otherwise a new compatible fold
lock must be approved before real inputs enter. Never silently replace an
accepted parent's groups or masks.

Within each label budget, include only officially training graphs whose labels
are observed under the accepted mask. No validation or test graph is loaded.
No wholly hidden graph contributes to a loss. No class balance repair,
resampling, group movement or seed retry follows screening outcomes. Both
Q-update and Q-screen need at least eight observed groups per cell. Record
graph and group counts, positive and negative counts, group-size distribution,
largest group's weight, and the number of graphs actually seen by updates.
Large counts alone do not establish independence or representativeness.

## Eight paired continuation arms per source

For each common source, deep-copy the tied model into eight arms:

| Training objective | Placements |
| --- | --- |
| Mean of the four member BCE losses (`own`) | tied, first, middle, last |
| BCE of the mean of four raw logits (`pooled`) | tied, first, middle, last |

Copies must have exactly equal evaluation logits before updating. Private
blocks are copied before constructing their optimizers. Every arm receives a
fresh AdamW optimizer: learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8,
weight decay 0. Warmup optimizer moments are not transferred. The continuation
is therefore conditional on resetting optimizer state; it does not estimate
what an uninterrupted optimizer would have done.

The direct pooled objective is an essential explanation control. It checks
whether a result depends on training separate member losses while evaluating
pooled logits. Both objective families begin from the same accepted P25 source;
this is not a comparison of independently trained, from-scratch objectives.

All arms take K = 8 minibatch steps with batch size 128, native CPU float32 and
one intra-op and inter-op thread. Sort update graphs by unit ID. For each
complete pass, shuffle their order with seed `21109 + 1000*source_seed + cycle`,
where cycle starts at zero. Preserve a short last batch before starting the
next pass. Thus the screen can see at most 1024 graph presentations, often fewer
unique graphs. Report exact coverage and all update IDs.

Before each step set dropout seed `27431 + 1000*source_seed + step`, where step
starts at zero. The model layout retains the same forward order and dropout
calls. All arms therefore share update batches and random streams. The two
training objectives are distinct families and their results stay separate.

## Scoring and fixed abstention

After K updates, score **every observed graph in Q-screen**, in sorted ID order,
with dropout disabled. Use BCE of the mean of the four raw logits. Each graph
receives equal weight. In the general loss helper, each graph's observed target
entries are averaged first; MolHIV has one observed binary target per included
graph. Accumulate native per-graph losses using float64 summation. Retain the
per-graph values, mean loss, update trace and state hashes for every arm.

For each objective family, the gain of a private placement is the tied model's
pooled BCE **after its own K updates** minus the private model's pooled BCE
after K updates. The tied gain must be exactly zero. Positive gain means lower
screening BCE. Initial losses are retained for audit, but do not replace this
paired baseline.

The rule below is fixed before any MolHIV result from this packet. It is a
practical continuation rule, not a confidence interval or significance test:

- Primary budget: 10% labels, all three source seeds.
- Numerical margin: 0.000001 BCE.
- A candidate requires at least 0.001001 BCE gain at every primary seed and
  in their mean. The practical component is 0.001 BCE, chosen as a small
  absolute loss improvement worth a later study, without calibration to these
  outcomes. Its relevance remains to be evaluated.
- At 100% labels, every seed's gain must be at least -0.000999 BCE. This retains
  a numerical margin inside the fixed 0.001 maximum-harm guard.
- Within each objective family, select the eligible placement with greatest
  mean primary gain. Exact ties choose first, then middle, then last.
- If no placement qualifies, abstain. Retain all 48 gains and all failures.
  Never merge the two objective families into a new, outcome-selected score.

These absolute thresholds are not uncertainty estimates. Three source seeds
on the same group division cannot support a broad generalization claim. A
selected candidate still needs a new protocol with full-budget training,
validation-based selection, appropriate efficient-ensemble and capacity
controls, independent seeds/splits, and uncertainty and cost reporting.
No validation or test result is released through this mechanism screen.

## Synthetic engineering gate

`verify_cpu.py` uses only constructed categorical graphs and analytic logit
fixtures. It does not instantiate a dataset provider or load a checkpoint.
The gate must run in a fresh CPU process through the repository's approved
logged command wrapper and supervisor. It checks:

- Exact own/pooled logit gradients and graph-equal losses with unequal numbers
  of observed targets; hidden-target gradient zero; malformed inputs refused.
- Exact initial function equality, equal private-placement capacity, all
  actual gradients finite, fresh AdamW and eight completed optimizer steps.
- Duplicate tied copies with exactly equal final state, loss traces and
  screening scores for both objective families.
- The advanced tied state and screening loss differ from the unchanged source,
  so the baseline check cannot accidentally pass by using the source alone.
- Full-screen scoring, paired update/dropout streams, input immutability,
  source gradient reference/value preservation, source modes and external CPU
  random-state preservation.
- Group, unit, split and fixture-scope rejection; partial minibatch coverage;
  complete 48-arm arithmetic; fixed abstention; and real refusal before any
  binding file is opened.

Passing this gate establishes engineering checks on a synthetic fixture.
It does not establish a working predictive method or a qualified MolHIV run.

## Delivery and later audit

Preparation is standard-library-only and writes an exclusive new output in
this sibling folder. CPU outputs must also use a new filename here. The six
source/document files and the two unchanged model sources enter the source
manifest. No output is overwritten.

A later admitted real runner must additionally hash the accepted group/mask
lock, exact P25 source state, visible graph tensors, runtime freeze and source
snapshot. It must preserve all eight arms per cell, both families, coverage and
class/group imbalance, complete screening values and state/optimizer artifacts
for replay. This source packet does not save future real checkpoints or claim
that those replay artifacts already exist.
