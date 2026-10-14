# Literature-to-experiment decisions, 15 October 2026

These are scoped readings and reused project conclusions, not a full-paper
reading count. Retrieval receipts distinguish downloaded text from sections
actually inspected. Primary passages below are the scope of the current read;
previous project exposure is not counted again as a new unique paper.

## TabM — arXiv:2410.24210v3

Read the abstract, Sections 3.2–3.5 and 5.1. The earlier attempted identifier
2410.24258 was wrong; its failed receipt remains. The correct primary source
and text extract are saved here. The model is existing prior work.

The source directly supports three decisions. First, initialization is part of
the method: keep the first multiplicative input adapter random and initialize
later multiplicative adapters to one. Second, pooling can improve prediction
even when individual shared members are weaker than a single network. Member
competence must therefore be measured alongside error complementarity, rather
than used as an absolute requirement that every route beat a native single.
Third, its independent-model and packed-model comparisons have different
checkpoint selection contracts. Our independent reference retains separate
member selectors; this distinction must be visible in the analysis.

The paper's tabular result does not establish a graph result. The present pilot
keeps the native categorical AtomEncoder and all graph/bond/message-passing
operations. Its nonlinear private correction is after the complete native
encoding. That insertion is an attributed adapter experiment, not a claim to
have invented input specialization. The pilot compares this ingredient with
base GNNM, a graph objective and their combination under the same stopping rule.

## PCGrad — arXiv:2001.06782v4

Read the abstract and method/Algorithm 1 passages (Sections 2–3) and the stated
scope of Theorems 1–2. The algorithm projects each task gradient sequentially
against original other-task gradients in random order, then aggregates the
projected gradients. It can feed a standard optimizer including Adam.

In shared GNNM, member losses can be treated as competing objectives for the
shared parameters. This is a candidate control for optimization interference;
it is not a novel method or an Adam no-harm theorem. Negative gradient cosine
alone does not establish harmful interference. A study must record the
unmodified losses and member competence, retain ordinary private gradients,
count reverse collections, and compare to ordinary mean-gradient GNNM. An
improvement would support this mechanism on the tested settings only.

## LoRA-Ensemble — arXiv:2405.14438v5

Read current abstract/bibliographic record only; retained project record126
already covered this prior. The record lists TMLR 2026. It describes private
low-rank attention matrices around a shared pretrained network. Neither its
claimed wins nor the pretrained-attention setting transfer to from-scratch
message passing. This is required ancestry for a private low-rank GNN wrapper,
not evidence that the present nonlinear input adapter is new or beneficial.

## Flow Matching for Generative Modeling — arXiv:2210.02747v2

Read abstract/bibliographic record only. It describes learning vector fields
for specified conditional probability paths, including optimal-transport paths.
The saved recent ensemble-flow search contains abstract-level discovery hits
from other applications; no full-method transfer or performance conclusion has
been established from them.

Flow-based discriminative steering remains a research lead. A defensible
version needs a TRAIN-only source/target distribution and coupling, an
inference-time vector field that does not receive the unknown label, and an
explicit solver/work budget. A CE-trained residual block is not thereby flow
matching. One possible target is a class-conditional distribution in the native
latent space, but arbitrary prototype targets could erase useful variation.
Before native-data training, specify the coupling and compare the exact same
residual/solver with CE-only and prototype-loss controls. This is deferred for
method definition, not rejected because of current GPU availability.

## Prediction quality and uncertainty

A wider distribution of embeddings is not evidence of better ensembles.
The pilot will compare whole-VALID ROC-AUC/BCE, each route's prediction,
common errors and recoverable errors. Validation is used for selection and
exploration; selected validation gains are optimistic. Three paired seeds on
one fixed scaffold split are a screen, not a significance or transfer claim.
Further claims require a frozen independent confirmation and heldout evaluation.
Standard deviations describe variation; they do not by themselves establish
that two methods differ. The full benchmark must report paired effects and
uncertainty across the declared trials, alongside every preset and failure.
