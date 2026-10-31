# Source-only integration design

## Scope

This is a prospective design for integrating the short actual-AdamW lookahead
with freshly qualified molecular P25 probes and a focused predictive comparison.
It admits no real input, binds no outcome, changes no existing model or probe,
and launches no job. The original six-file lookahead packet remains unchanged.
The accompanying `INTEGRATION_PLAN_v1.json` enumerates every proposed fit.
The coordinator must approve and freeze a separately versioned execution packet.

The verified route is the originally supplied `anogena-2` account, reaching one
A100 80GB. Each future job must use `protocols/run_authorized_v2.py`. Its
`authorization.json` must bind the exact supplied `GNNM_SSH_DESTINATION`, account,
authorized project root and one-device resource identity. The route record at
`protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json` is necessary route
metadata; it does not replace per-job authorization or qualify data by itself.
Previous wrong-account binaries, bundles, checkpoints, runtime freezes and
results remain out-of-scope. Matching hashes alone do not restore authorization.

## Admission chain

The execution packet must first verify all of the following as correctly
authorized, immutable references. Any missing or inconsistent reference stops
admission before checkpoint deserialization.

1. Fresh correct-route exporter command, supervisor completion, authorization,
   source snapshot, runtime freeze, molecule/scaffold mapping gates and exact
   official-train bundle hash. The bundle must retain every official training
   graph, original categorical features and group mapping.
2. Freshly qualified GPU class-source and package identities. The new GPU
   lookahead paired-null gate and low-rank gate must pass in that environment.
   Earlier gate outputs do not satisfy this requirement. The private-copy
   and low-rank actual counts must be emitted there. An unlaunched CPU molecular
   sibling is not a dependency; optional CPU analytic checks are engineering
   checks, not replacement GPU qualifications.
3. All six P-only probe runs: seeds 0/1/2 at fractions 0.1/1.0, exact epoch25,
   P-only own-member BCE fitting, accepted mask/fold locks, contract, trace,
   checkpoint, record, replay, complete artifact manifest, source and runtime
   references, supervisor completion and per-job authorization.
4. Exact train ID and tensor identity across the exporter, probe and lookahead.
   Verify each checkpoint's source state against its saved record; verify
   complete P/Q group disjointness and all graph coverage. Source recipe,
   architecture, label fraction, seed and epoch must agree across references.

All six source cells and the unchanged recurrence locks must be finalized
before the first predictive validation comparison. The lookahead additionally
requires all six cells and both objective families before its global eligibility
and ranking rule is evaluated. This forbids partial-cell selection.

## Bridge from a P25 probe to the lookahead core

Use a new admission/bridge module; do not change the preserved probe or its
recurrence policy. The bridge reads only the authorized train-only bundle and
the admitted P25 checkpoint. It must not construct the general OGB dataset
provider, validation loader or test loader.

- Instantiate the tied model through `lookahead_core.source_module()`, then
  load the freshly qualified GPU checkpoint state strictly and move the model
  to the single authorized GPU exposed as `cuda:0`. The probe's class may have been
  imported under a different module name; do not pass that class instance to
  the core's exact-type check. The source bytes, state keys, tensor values and
  architecture must match.
- Use `gpu_lookahead_core_v2.py` for actual GPU updates. Do not transfer the checkpoint's optimizer moments. The eight lookahead
  arms each receive a fresh AdamW optimizer as specified in `PROTOCOL.md`.
- Reconstruct units from the accepted observed-label mask and official train
  IDs. Erase hidden labels before forming units. Use exactly the admitted
  `folds.unit_to_group` mapping; there is no replacement chemistry pass.
- P IDs must equal the admitted `probe_fit_unit_ids`. Q IDs must equal
  `score_unit_ids`. Reuse the fixed seed704 split of Q specified by the
  lookahead protocol. Record the explicit update and screen ID/group lists,
  reject duplicate IDs, overlap, nontrain data or fewer than eight observed
  groups in either portion. Changing to a graph-count-balanced split requires
  a prospective new version, before outcomes.
- Keep exported graph tensors on CPU and move only each collated minibatch to
  the GPU. Convert units into `lookahead_core.Unit` with `synthetic=False`.
  Both current source-only cores refuse these. The separately admitted packet must implement
  its own explicit opening of real execution; editing the existing v1 flag
  is not an admission procedure.
- Keep native GPU float32, no AMP or TF32, deterministic algorithms, no cuDNN
  benchmark, one intra-op/inter-op thread and `CUBLAS_WORKSPACE_CONFIG=:4096:8`
  set before CUDA initialization. Keep the exact source, recipe, masks,
  grouped split, update streams and practical rule. Report
  every trajectory, all screening graphs and class/group imbalance.

The integration should expose the GPU-v2 trajectory function through a
source-frozen admitted copy or a versioned public admission API, retaining
the mathematical implementation and tests. It must not monkeypatch the v1
`SOURCE_ONLY` constant or change placement logic at runtime. Deterministic
kernel failures abort; there is no fallback to CPU, lower precision or a
post-outcome numerical-margin change. `verify_gpu_v2.py` supplies the prospective
synthetic paired-null/source-preservation gate and refuses real inputs. Run it
only through the authorized wrapper with a fresh exclusive output such as
`GPU_GATE_authorized1gpu_run01.json`. An additional
real-batch initial-function/capacity gate runs before the first continuation;
it cannot be counted as predictive evidence.

Save final model states and AdamW states in the later admitted packet, together
with per-graph screening logits/targets, eight-step traces, initial/final hashes
and replay contracts. The GPU-v2 source returns the final model/optimizer and
screen logits as objects for serialization by the future admitted runner;
it does not save real artifacts. The integration must add an independent
forward/trace replay without claiming that such evidence already exists.

## Complete private-block state and storage accounting

A full block includes LayerNorm weights/biases, every BondEncoder table, both
GINE MLP weights/biases and the learned epsilon, plus any registered buffers
in the qualified implementation. Copy the complete block via the preserved
`privatize` function. Replacing only its linear layers is a different method.

For placement index 0, 1 or 2, instantiate the tied common model and deep-copy
the selected block four times. Remove the unused shared block by the preserved
`nn.Identity` replacement. The extra parameter count is exactly `3*p_block`.
All three placements must have equal actual counts. The actual optimizer's
parameter identity set must equal the complete trainable parameter set, with
no orphan shared block or duplicate registration. Its state starts empty.

The gate must compare the complete source block state to each private copy,
including buffers, and check distinct storage among the four copies. Within
an arm, the atom encoder, boundary parameters/factors and output LayerNorm stay
shared across members. Across arms, all parameters have independent storage.
Initial evaluation logits must agree exactly for tied and copied placements.
Record each parameter/buffer name, shape, dtype, bytes, trainability and role,
plus total model, checkpoint and optimizer-state bytes. Count the full block
consistently in the budget, source contract, construction and reports.

## Declared low-rank comparison

Use the separately qualified source at
`closest_controls_v1/trainable_low_rank_gine_v1`. It is a declared
TabLoRA-inspired GINE adaptation, not an exact author reproduction and not a
new-method claim. It retains shared original parameters and adds trainable
per-member factors to the six hidden dense maps. Atom/bond encoders, epsilon,
biases and normalization remain shared. Both factors train; random A / zero B
starts with zero correction and a live B gradient. No frozen basis is used.

The rank is selected by **actual registered counts without data or metrics**:
enumerate ranks 0 through128, minimize absolute difference from one full
copied-private middle-block model, and break ties toward lower rank. Emit all
129 count candidates and the chosen mismatch on the authorized runtime.
Do not import previous wrong-allocation count reports as evidence. If mismatch
exceeds1%, stop the proposed size-comparison admission; do not retune rank on
performance. Report an approximate size match, never an exact match unless
the dynamic count is identical.

Use initialization seed `29001 + 1000*optimizer_seed`, with the adapter's map
offsets and private CPU generator. Initial pooled/member logits must match the
common source within a declared 1e-6 absolute numerical tolerance. Rankzero
must be exactly tied. Both objective families use the same constructed model
and rank. The pooled family is a prospective additional objective control;
the retained adapter's `loss=own` descriptive field must not be silently used
as a claim that pooled training reproduces the prior method. Record the actual
training loss separately in the new run contract.

All five compared layouts retain four complete graph paths and twelve GINE
propagations per forward. Parameter savings do not imply propagation,
activation, latency or peak-memory savings. Measure those quantities.

## Focused predictive matrix

The smallest complete primary block is **30 fits**:

| Factor | Values |
| --- | --- |
| Label fraction | 0.1 |
| Seeds | 0,1,2 |
| Training objective | own, pooled |
| Layout | tied, first-private, middle-private, last-private, declared low-rank |

Register the matching **30 full-label fits** at fraction1.0 prospectively.
The placement/objective matrix is60 fits. Run the complete primary block first;
the full-label block is needed before a predictive continuation claim passes
the adverse-budget guard. A resource-limited or failed primary block cannot
be presented as the complete comparison. Do not drop losing arms or retry
seeds based on results.

Every predictive arm restarts from the same canonical seed initialization;
none receives P25 weights. Construct private copies and zero-delta low-rank
factors before the fresh optimizer. Fit on the entire visible official
training subset, P plus Q, under its existing frozen mask. Keep source/probe
and mechanism costs separate in raw logs, but include them in the adaptive
method's total cost. A fixed-placement baseline does not incur adaptive search.

Freeze one recipe: width128/L3/M4/multiplier2/dropout0.1, AdamW lr0.001,
betas(0.9,0.999), epsilon1e-8, decay0, batch128, 100epochs, float32, noAMP,
num_workers0. The exact ordering implementation and dropout seed rule belong
in the new runner freeze, with shared batch and dropout streams across arms.
Prefer retaining the inherited pilot's DataLoader-generator ordering rule
`19709 + 1000*seed + epoch`; do not silently substitute the probe's Python
shuffle while calling it identical. Snapshot or hash the actual batch-ID stream.

Use full official validation graphs after each epoch. The predeclared inherited
checkpoint rule is maximum pooled validation ROC-AUC, then minimum pooled BCE,
then earliest epoch. Retain all per-epoch member logits and full traces.
Additionally report epoch100 metrics for every arm as a fixed-budget diagnostic;
neither endpoint can replace the other after seeing results. Validation
comparison is development evidence and is affected by checkpoint selection.
No test score is used in this stage. A test-release/confirmation stage requires
a separately frozen protocol after the method definition is fixed.

## Adaptive identities and exact candidate accounting

There are six independent probe source cells, eight mechanisms per cell and
48 mechanism trajectories. These are not predictive fits. Lookahead evaluates
three private placements per objective family using all six cells and can
abstain; its global candidate is at most one placement per family. The unchanged
recurrence policy has its own per-cell locks and can abstain. It remains an
own-loss-gradient policy even when its locked layout later trains with pooled
loss. That distinction must be reported.

The low-rank rank search has129 structural count candidates and zero scored
rank candidates. Its rank is a capacity rule, not HPO. The predictive matrix
has five fixed architecture layouts, two objective families and one optimizer
recipe:30 primary fits, plus30 registered adverse fits. Own and pooled are
reported separately; selecting the best family using validation creates an
additional development choice and cannot be called a preselected method.

A locked adaptive method may refer to an already fitted fixed placement if
and only if source seed, mask, model definition, initialization, loss, optimizer,
training streams, budget and selection rule are exactly identical. Map
abstention to tied. Map a single selected index to its corresponding fixed arm.
Reject multiple indices or any missing arm; never fall back to middle.
Locks must predate the first validation comparison. Record the alias explicitly
with source lock and target run hashes. This is reuse of an identical run, not
an additional independent replication or validation-best placement choice.
Probe/search overhead still counts for the adaptive method.

## Mandatory expanded comparisons, frozen before primary outcomes

If either training-only locked candidate passes the complete primary signal
rule below, finish all30 full-label placement/objective fits and all of these
comparisons at **both** fractions and all three seeds before promotion:

| New layout/reference | Objective fits | Unique fits |
| --- | --- | --- |
| Closest all-layer BE-GINE adaptation | own and pooled |12 |
| Parameter-matched BASE | one member; own equals pooled exactly |6 |
| Parameter-matched independent ensemble | own and pooled |12 |
| Published OGB bond-aware GIN-Virtualnode reference | one member; one published recipe |6 |

There are36 additional baseline fits, hence96 unique fits after mandatory
expansion. Single-member BASE/reference runs are explicitly aliased across
families because their own and pooled losses coincide; this does not provide
extra replications. All-layer readout endpoints add no fits. The entire expanded
block is required once launched. One successful family does not justify omitting
the other family's declared controls or adverse results.

The all-layer model source is the retained declared GINE adaptation in
`closest_controls_v1/gine_port_v1/source/all_layer_gine_adapter.py` and
`member_factor_linear.py`. Freshly qualify it on the authorized GPU; no old
gate or freeze is reused. It wraps both hidden maps in each block, preserving
member states and shared encoders/norm/epsilon. Its published-prior adaptation
and probability pooling must be described honestly. Retain both raw-logit-mean
and mean-member-probability readouts, with separately predeclared validation
checkpoint selections. Evaluate each candidate under the matching readout;
require the contrast under both endpoints before superiority to this comparator.
Do not select a favorable readout after results.

For size matching, construct all widths8..512 without data or metrics for the
existing `MolecularSingle` and four independent complete singles. Minimize
actual registered parameter-count difference from a **complete one-private-block
model**, breaking ties toward lower width; require mismatch no greater than1%.
Emit all505 width candidates, actual parameter/buffer bytes and chosen width.
Single and independent members have their own encoders, including atom and bond
tables. A mere change in projector width is not the matching rule. All parameters
train under the fixed100-epoch recipe. A single member has one graph path;
the independent ensemble has four full graph paths. Approximate count matching
does not match optimizer-state bytes, memory, runtime or representations.

If recurrence-policy promotion is also attempted and any locked cell abstains,
add controls matched to **tied capacity** for the affected scope. Freeze the
complete second target registry now: matched BASE6 and matched ENS12 at both
fractions/all seeds, conditional solely on a training-only abstention lock,
not validation results. This can add18 fits (maximum114). A private-capacity
control must never be described as matched to an abstaining tied model.

### Published reference recipe and declared wrapper differences

The primary source is Hu et al., *Open Graph Benchmark: Datasets for Machine
Learning on Graphs*, arXiv:2005.00687, with official molecular code pinned at
snap-stanford/ogb commit `61e9784ca76edeaa6e259ba0f836099608ff0586`.
The exact README, `conv.py`, `gnn.py`, `main_pyg.py`, commit response and paper
abstract are retained with URLs/hashes in `published_reference_sources_v1`.

The reference uses OGB's **bond-aware GIN-Virtualnode**: original atom/bond
fields, learned bond messages `ReLU(x_j + bond_embedding)`, five layers,
width300, BatchNorm/ReLU, virtual-node updates, dropout0.5, last-layer node
representation, mean graph pooling and one graph head. The published default
recipe uses batch32, Adam lr0.001 with its standard default betas/epsilon and
decay0,100epochs and full features. It is relevant to molecular GINE-style
message passing but is not the same residual LayerNorm GINE architecture.
It is a published reference recipe, not an assertion of current SOTA or an
exact paper-score reproduction.

Retain its model source and optimizer defaults. A new wrapper supplies the
accepted10%/100% masks, deterministic seed/batch streams, provenance and retained
logits/checkpoints. The original main creates a test loader and evaluates test
every epoch: **do not run that main directly**. The wrapper loads only the
admitted train/validation exports; heldout test loading remains absent. Record
any inherited skip of a one-graph/one-node training batch rather than silently
changing BatchNorm handling. The common maximum-AUC/BCE/earliest checkpoint
tie rule and fixed epoch100 diagnostic are declared wrapper differences.
Do not tune its width, virtual-node choice, dropout or optimizer using our results.

## Proposed prospective continuation and promotion rules

This is a proposed developmental rule for the new execution freeze; no outcome
is attached. It must be approved before validation comparisons.

- For each objective family, evaluate the training-only locked lookahead
  candidate against both same-family tied and low-rank controls. Abstention
  produces no private candidate and cannot satisfy a private-method success rule.
- Complete all three primary seed pairs. Require positive ROC-AUC differences
  at every seed and a mean gain of at least0.005 (0.5percentage points) against
  each control. The absolute margin is a practical continuation threshold,
  not a significance threshold or an expected result.
- If a family meets that primary rule, complete **the whole registered full-label
  and mandatory baseline expansion**, including both families and all registered
  layouts. For that candidate, no full-label seed may lose more than0.005 ROC-AUC
  against tied or low-rank controls. This is a budget guard, not promotion.
- Promotion to a separate independent-confirmation study additionally requires
  mean ROC-AUC gain at least0.005 and three positive paired differences against
  all-layer BE-GINE (both predeclared readout endpoints), correctly size-matched
  BASE/independent ENS, and the published GIN-Virtualnode reference at **both**
  label fractions. Preserve exact capacity matching when evaluating abstention.
  These are practical developmental gates, not statistical significance claims.
- If no family meets the primary rule, retain the completed30-fit primary
  matrix and abstain from expanding this particular mechanism direction.
  A fixed-layout observation can motivate a distinct prospective hypothesis;
  it cannot replace the locked candidate's failed result.
- Report every paired difference, mean, sample SD and range. Add descriptive
  paired uncertainty from retained logits and the accepted validation groups,
  if those group identities are independently qualified. Any group-bootstrap
  interval is conditional on the fixed training and split; three seeds and one
  reused official split do not establish broad or independent confirmation.

The primary30-fit screen alone resolves none of the broader baseline concerns.
The mandatory expansion addresses closest efficient ensembles, size controls
and a published molecular reference. Previously declared fully-private,
same-width BASE/multihead/independent-ensemble comparisons and independent
task/split confirmation remain required before broad superiority or efficiency
claims. A go decision after the96/114-fit development matrix still does not
establish novelty or independent confirmation.

## Execution and evidence discipline

The new packet must run serially on the one authorized device unless a separate
resource assignment is explicitly authorized. Do not assume the old seven-GPU
allocation exists for this task. Use observed correct-route engineering timings
to estimate the30-fit primary and96/114-fit expanded budgets. No ETA is derivable from source alone. If a
promising primary result makes the full matrix infeasible on the allowed GPU,
report the exact remaining workload and ask the coordinator to obtain more
authorized resources before expansion.

Each command needs the approved wrapper's authorization, run_logged command,
environment/source snapshot, supervisor completion, output manifest and replay
report. Keep all project manipulations inside the authorized repository. Record
every failure, correction and protocol amendment. Preserve all result cells;
none enters the manuscript until its full admission/audit chain passes.
Fresh reviewer passes must inspect the resulting paper without campaign context.
Neither this design nor an eventual passed engineering gate warrants an accept
recommendation or a novelty claim.
