# Prospective MolHIV and MNIST development freezes

These are finalized source and method plans, created before predictive fitting.
They are not runner-generated runtime protocols: exact training-label subset
hashes, the actual package/provider environment, matching capacities and GPU
identity still require the declared CPU gates and runtime freezes. No server,
GPU job, raw tensor load, validation score or test score was used in preparing
this package. Existing paper scores and prior artifacts retain their meaning.

MolHIV uses official scaffold train/validation partitions (32,901/4,113 graphs),
bond-aware residual GINE and a prespecified middle-private block candidate.
Eight arms, three paired optimizer seeds and two nested training-label fractions
make 48 fits. MNIST uses released 55,000/5,000 development partitions, SAGE and
mean node-logit graph readout. Eleven arms make 66 fits. The four common/narrow
cells distinguish centering from feedback; packed source GNNM and parameter
matched independent/single controls test whether the new architecture adds
value beyond execution layout or stored capacity. Every fit gets 100 epochs,
the complete validation partition and one fixed recipe.

Use `molhiv_source_freeze.json` and `mnist_superpixels_source_freeze.json` for
the bound recipes, hypotheses, contrasts, uncertainty and stop/go rules.
`launch_schedule.json` contains the exact114-fit serial order and command
templates as data. It launches nothing. Run all 10% comparisons then full-label
comparisons; neither budget nor seed is selected for a favorable result.
No GPU-hour estimate is claimed from the CPU checks. The workload contains
2,242,500 training minibatches before validation or evidence writing.

The latest source amendment adds `base_parameters` and `ens_parameters` to
MolHIV schema-v2. They use complete bond-aware GINE, not packed nonlinear GINE.
Width matching enumerates actual CPU parameter counts over integers 8..512,
targets `middle_private` at width 128, breaks ties toward smaller widths and
requires no more than 1% difference. The new synthetic gate must confirm the
actual chosen capacities, RNG preservation and finite gradients. The old
MolHIV usage document describes the earlier eight-arm version; this package's
explicit arm list and schema-v2 recipe supersede it for these new fits.

Both runners now bind `new_tasks/shareable_guard.py`, which requires explicit
unscored/unaccessed-confirmation status, unchanged shareable-v2 provenance,
absence of test targets and class summaries, consistent removal counts and
the canonical test input-fingerprint hash. The original freezer did fingerprint
test targets, and public provider files contain test labels. The boundary is
procedural and source enforced, not a cryptographic seal of public labels.

Before any fit, the coordinator must rerun:

1. `python new_tasks/verify_molhiv_pilot.py` and
   `python new_tasks/verify_mnist_pilot.py` in the existing graph environment;
   CUDA visibility must be empty. These are synthetic checks.
2. CPU preflights with 128 complete training graphs for MolHIV `middle_private`,
   `base_parameters` and `ens_parameters`, and MNIST `centered_feedback`,
   with new output paths and the original host's exact shareable data manifest.
3. The two runner freeze commands in `launch_schedule.json`, including their
   explicit reduced arm lists. Inspect actual matching counts and bind the
   same source/data/environment as these plans. Frozen server paths must not
   be rewritten or followed from this local preparation host.
4. Record an explicit authorized idle physical GPU, its UUID/model, serialized
   ownership and resource ceiling. None was idle when this plan was prepared.

The runner freeze creates the exact subset/index hashes using training labels
only. Runtime hardware belongs in the launch contract and fitted evidence;
this local package fabricates none of it. A source/data/environment change
requires a new amendment and output directory before fitting. Prior runners
and verifiers are preserved in sibling `before_runner_amendment/` with hashes.

Report every arm/seed/fraction and failed attempt. The primary validation metric
is ROC-AUC for MolHIV and accuracy for MNIST; margins are 0.005 and 0.002.
Paired seed summaries and descriptive 95% t intervals describe optimizer
variation on each corpus. They do not establish independent-task generality.
Efficiency promotion requires measured cost and the separately frozen
inference profiler; stored counts or adjacency widths cannot establish speed.
Failure of the prespecified quality or mechanism rules stops expansion rather
than initiating a width/rank/depth search. The same source MNIST images in
GOOD-CMNIST cannot count as independent image-corpus confirmation.

Test scoring and independent confirmation remain closed until a separate
candidate/confirmation protocol is locked. This package makes no manuscript,
acceptance, state-of-the-art or predictive performance claim.

To prepare another source-only amendment, run the sibling preparation script
with a new output path inside this phase. It rejects existing paths and writes
every JSON/README/artifact-manifest file exclusively. The manifest covers all
its siblings except itself. No prior files are overwritten.
