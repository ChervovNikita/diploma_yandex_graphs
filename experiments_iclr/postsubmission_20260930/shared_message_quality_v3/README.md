# Shared-message model and qualification source v3

## Narrow generated-code amendment

Both sealed predecessor packets are preserved. V2's authorized synthetic run
exited unsuccessfully after approximately 9.75 supervised seconds when its
final import audit rejected PyG's legitimate generated
`torch_geometric.nn.conv.sage_conv_SAGEConv_propagate` module in project tmp.
There is no completed passed synthetic gate from that failed run.

V3 changes the source/import evidence guards for this runtime behavior.
`GENERATED_MODULE_POLICY_v1.json` binds root's actual stdlib-only provenance
capture receipt, installed PyG version, exact generator/template source hashes
including the nested `collect.jinja`, and the exact captured full generated
module names/source hashes. Installed source pins are checked through package
metadata before Torch/PyG imports and checked again at final module audit.
Each allowed generated module must have its exact full name, exact emitted
source hash, full-name filename prefix, `.py` suffix, and a non-symlink path
confined to phase/tmp. Unknown generated names, arbitrary tmp modules,
changed source and symlink/outside paths are refused. RESULT records each
actual generated module's path and source SHA plus the policy and installed
source provenance; downstream synthetic admission audits those exact fields.

Models, counts, acquired-data interfaces, recipe, tolerances and schedule
weighting are unchanged from v2. This source repair is not numerical passage
or scientific admission. A new completed authorized synthetic run is required.

## V3 admission amendments

This is a separate source generation. The sealed v1 and v2 packets are preserved.
All public model, count and acquired-data interfaces and recipe remain the
same. This generation repairs the independent source audit's admission and
runtime evidence issues; source sealing does not establish numerical passage.

Before the first phase-package import, the qualification entrypoint removes
script/cwd/PYTHONPATH source paths, binds the manifest to the exact request,
hashes the import guard, then checks all sealed own/inherited dependencies.
`RUNTIME_IMPORT_POLICY_v1.json` lists the exact recursive `.py` importable
tree of this packet and both inherited packages, including the legitimate
`engineering_entry_bridge_v2.py`. Unsealed Python, bytecode, extensions,
module-shadow directories, cache directories and symlinks are refused. A
restricted phase finder admits only listed modules and compiles their hashed
source without reading/writing bytecode. Other phase directories cannot
shadow installed Torch/NumPy/PyG or stdlib. Imported phase modules and the
verified runtime tree are recorded in RESULT. Installed third-party packages
are frozen by versions; their entire installation files are not source-sealed.

Every operation consumes the external `bounded_run_v1.py` START named by
`GNNM_BOUND_START_JSON`, binding exact request, CLI, inner directory, route,
source and cap. This wrapper starts a fresh process group and reserves
TERM/KILL grace inside its cap. Cooperative wall checks remain supplemental.
Real admission requires the completed outer TERMINAL as well as completed
inner authorization/logging. The terminal includes evidence-hash costs in
`whole_supervised_seconds`; final terminal-write/print micro-overhead is
caller-accounted. This is an external process-group timeout mechanism, not a
literal OS absolute deadline guarantee.

Synthetic CUDA checks reset peak memory and check peak reserved memory after
each named check and training-step boundary with CUDA synchronization. RESULT
records the final peaks, cap and boundary measurements. These checks detect
peak-memory cap breaches after operations; no external hard memory limit is
claimed. Failures preserve metadata and observed peaks when CUDA initialized.

Real admission links the exact synthetic request, source, prescribed output,
START/RESULT/artifact manifest, completed CLI/environment, authorized GPU,
outer START/TERMINAL and import receipts. Runtime policy, equivalence
tolerances and GPU must match the real request exactly. A result copied from
another request, output, supervisor or runtime is refused.

Engineering path selection uses `median(train + validation / 5)` and then
variant ID. The weights reflect the prospective 300 updates and 60 validation
passes. Scientific schedule and budget admission remain separate.

This is a new, source-only packet for the complete transductive Reddit2 and
ogbn-products graphs. It has no scientific runner, fitted weights, provider
data, numerical result, quality score or runtime admission. Its author has not
imported Torch/NumPy/PyG, accessed data or connected to a remote host. Existing
sealed packets are unchanged. Independent source audit and actual root-owned
runtime/resource qualification are required before execution.

## Five models

Let `Z_m = LayerNorm(H_m)` and let `P` be the exact fixed incoming-neighbor mean
from the audited provider graph. The member root states remain separate.

| Packet arm | Neighbor message | Parameters |
|---|---|---|
| `exact_member` | `P Z_m` | Pinned source TABM/GNNM, D64/M4/L2 |
| `shared_mean` | `P mean_m(Z_m)`, broadcast | Identical to exact |
| `detached_shared_mean` | `P stopgrad(mean_m(Z_m))`, broadcast | Identical to exact |
| `single_D64` | Ordinary source single `P Z` | Ordinary D64 source Model |
| `single_matched` | Ordinary source single `P Z` | Deterministic nearest parameter width |

Shared and detached have the same forward function at equal parameters and
noise masks. Both change the forward function relative to exact member
messages: they delete `P(Z_m - mean Z)`. No sampling or color partition is used.
Mean is taken **after each member's LayerNorm**. All input/output factors,
member hidden states, shared affine normalizations, learned neighbor/root
maps, concatenation, FFN, GELU, dropout, residual updates and output heads
remain in the pinned source order. This packet does not compose learned maps
or move dropout to reduce dense work.

For upstream message derivatives `U_m`, the differentiable common channel
contributes `P.T sum_m(U_m)/M` to every `Z_j`. Whole detachment makes this
channel zero, including the member's own contribution. Learned neighbor/FFN
parameters still receive gradients, and the root path still differentiates
through `Z_j`. This comparison does not identify off-diagonal coupling alone.
Linearity of the shared mean does not establish unbiased nonlinear states,
losses, gradients, optimizer updates, useful diversity or predictive quality.

## Initialization, ordinary single width and interfaces

New constructors use the pinned `Model`/`TABMModel`, scoped RNG and source
parameter transfer for fresh seeds **17, 29, 43**. Shared, detached and exact
have exactly paired parameters per task/seed. Ordinary singles are paired by
seed; they have a different parameterization and are not equal-weight pairs.
The original seed guard (0/1/2) and D<=64 fixture guards remain unchanged.

The matched single has a new explicit bounded constructor based on the pinned
`PlainSAGE` blocks. Its integer width minimizes `(absolute count mismatch,
width)` over 1..256 using the exact source count formula. Smaller width breaks
ties; a target exceeding that explicit bound is refused. Width selection uses
input dimension and class count, with no predictions or measured performance.

| Task | Ensemble parameters | Ordinary D64 | Matched width | Matched parameters | Signed mismatch |
|---|---:|---:|---:|---:|---:|
| Reddit2 | 86,384 | 82,985 | 66 | 86,897 | +513 |
| ogbn-products | 52,680 | 51,247 | 65 | 52,697 | +17 |

These are executed stdlib formula counts. Constructors also check actual
Torch parameter counts at runtime. Parameter matching does not match
activations, graph channels, wall time or memory. `PARAMETER_PLAN_v1.json`
reports exact mismatch and symbolic storage/channel counts, without measured
feasibility.

- `counts.plan(task)` returns dimensions, counts, matching rule/mismatch,
  dropout policy and symbolic channels/storage.
- `models.construct(task, arm, seed=17, checkpoint_blocks=False)` returns a
  fresh CPU model; `task` is a named complete task, or a dimensions dict for
  synthetic checks. No existing constructor is patched.
- `models.forward_logits(..., execution="packed", dropout_key=key)` returns
  `[4,N,C]` for ensembles or `[1,N,C]` for singles. Shared/detached forbid
  serial trajectories. Exact supports packed/serial and block checkpointing.
- `models.make_optimizer(model)` creates fresh AdamW with lr=.001,
  decay=.0001, betas=(.9,.999), eps=1e-8.
- `models.train_step(..., execution="packed", dropout_key=key,
  return_loss=False)` uses mean own-member CE and one AdamW update. Serial
  exact accumulates four CE/4 backward passes before that one update.
  Optional returned loss is a detached training-only tensor for a separately
  admitted scientific caller; qualification never requests or records it.
- `models.pooled_inference(...)` returns raw float32 mean logits; exact serial
  streams pooling without retaining four complete-node outputs.
- `models.split_logits(..., indices, execution="packed")` performs one
  complete-node eval forward and returns CPU member slices and exact CPU raw
  pooled slice. Serial slices are copied before pooling mutates the first
  output. This helper has no label argument.
- `data.load_inputs(phase_ref)` reuses sealed acquired training-only export and
  prepared-array audits, returning `(export, lineage, metadata, directory)`.
  Export keys and official ID order remain unchanged.
- `data.load_family(metadata, directory, arm, device="cuda:0")` loads **full P
  only**. Differentiable shared, exact and singles retain forward and adjoint;
  detached alone loads forward without an unused adjoint. No color CSR is
  supplied to any model. The inherited full prepared-manifest audit still
  reads all stored array evidence and its cost must be charged.

Dropout reuses pinned `DropoutKey(task_index, optimizer_seed, epoch)` and its
exact `gnnm-propagation-dropout-v1` domain/private generator semantics. Task
indices are Reddit2=0 and Products=1, member indices 0..3 (single=0), input
layer=-1, residual layers=0/1, sites input/ffn_first/ffn_second. Corresponding
ensemble masks are paired. The unused old `training_colors` sampler is never
called. Qualification's key field named `epoch` holds short engineering update
IDs; these are explicitly not scientific epochs. A separately adopted
scientific protocol controls its own update IDs and complete training budget.

## Synthetic code and real engineering qualification

`synthetic_checks.run` is **unexecuted Torch test code**. It covers runtime
parameter counts and paired initialization, same shared/detached forward with
dropout off/on, keyed mask replay, the full double-precision common-channel
adjoint, whole stopped neighbor derivative, non-equivalence to exact messages,
surviving learned root/neighbor gradients, checkpoint gradient/AdamW equivalence,
packed/serial exact accumulation, serial slice-before-pool semantics, ordinary
single block equivalence and preservation of original fixture/seed guards.
There is no provider import, real target or saved model in this test code.

`qualify.py synthetic` requires explicit root engineering authorization,
source seal, frozen actual runtime and corrected one-A100 supervision before
Torch imports. A completed source-bound synthetic CUDA result is required by
`qualify.py real`. The latter admits both exact complete training-only exports
and profiles 12 fixed variants: shared/detached packed with checkpoint off/on,
exact packed/serial with checkpoint off/on, and each single with checkpoint
off/on. Every variant begins from seed17 and a fresh optimizer. No scientific
schedule, epoch count, fit family, quality margin or promotion threshold is
adopted by this packet.

Real qualification times **actual training-mode own-member CE, backward and
AdamW**, followed by dropout-disabled complete-node inference and official
validation-ID reductions with **no validation labels**. Cold/warm/raw samples
record synchronized wall time, CUDA events, process CPU seconds, operation
counts, setup/H2D, peak GPU allocated/reserved memory, process RSS and potential
checkpoint/logit bytes. It compares first full-input gradients across execution
variants of each arm using frozen tolerances. Internal loss/reduction values
are never reported, saved, ranked or used to choose a path. Updated weights
and optimizer tensors exist transiently and are discarded. Only JSON metadata
is saved; no weights, logits, predictions or held-out target values are saved.

Path order is shuffled by an outcome-independent stdlib seed. Memory-infeasible
variants are retained as failed engineering alternatives; they cannot be
selected. The feasible lowest median train-plus-label-free-validation/5 wall
path, then variant ID, is selected for each arm. This is the fastest among
these measured variants, without a global optimality claim. Missing feasible
arms leave complete qualification infeasible. Nonfinite/numerical failures
stop the qualification. Actual wall and metadata disk caps still apply, and
all failed work remains charged. These short profiles are initialized-model
engineering evidence, not representative predictive observations.

Prior mean-only headroom used `eval()`/`no_grad()` inference and label-free
reductions. It is **not train/backward/optimizer timing**. No prior headroom
number is imported into the new qualification or adopted as training evidence.
This packet has not produced new timing evidence. TF32/AMP remain off; actual
package versions, CPU threads, CUBLAS configuration, device UUID and numerical
tolerances are root-frozen. A false deterministic-algorithm setting is not a
claim of deterministic CUDA execution. APIs reuse the already qualified
Torch2.1.2-compatible source checkpoint/CSR paths.

## Independent launch and scientific protocol boundaries

`REQUEST_SCHEMA.md` gives complete engineering fields. All commands use the
repository interpreter/cwd/wrapper and corrected authorization supervisor on
the sole permitted `anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru`
allocation. Source references are canonical phase-relative paths and SHA256;
outputs are new exclusive directories inside this packet. The seven-GPU route
is forbidden. Actual acquired tensors/providers are never read by the author.

The prospective quality design and RNG/order policy live separately in
`shared_message_study_design_v1`; that protocol is not admitted by this source
packet. Scientific schedules, compact initial/selected/final artifacts,
selection, final replay, target admission, full-family cost forecasts and
quality gates belong to a separate runner/root adoption. Neither the rejected
original 30-fit study nor its thresholds are cloned or silently reopened.
There is no inductive Reddit2 claim, new novelty certification or paper
acceptance review in this packet.
