# Molecular endpoints and future matrix compiler

This prospective MolHIV/MolBACE GINE packet creates no benchmark outcome or GO
receipt. Scientific execution requires root runtime qualification and explicit
human GO. Existing running studies and archived scores are unchanged.

## Data and selected-checkpoint custody

Preparation retains the original joined four-role packs and also saves separate
graph-only and target-only heldout artifacts. `MANIFEST.json.endpoint_inputs`
binds exact `{path,bytes,sha256}` receipts:

| Key | File / contents | Reader |
| --- | --- | --- |
| `cal_graphs` | `CAL_GRAPHS.pt`, graph keys except `y` | Bank export |
| `evaluate_graphs` | `EVALUATE_GRAPHS.pt`, graph keys except `y` | Bank export |
| `cal_targets` | `CAL_TARGETS.pt`, exactly `{ids,y}` | Scalar CAL |
| `evaluate_targets` | `EVALUATE_TARGETS.pt`, exactly `{ids,y}` | Evaluation |

Graph-only prediction has no dummy target. Structural admission preserves all
nine atom fields, three bond fields, partitions, local edges and unique ordered
graph IDs; a `y` field in a graph-only file fails. `_batch` returns no labels for
prediction packs. TRAIN/SELECT still require the original joined labelled schema.

`adapters/molecular_endpoints.py:export_bank` restores each paid parent's own
engine-selected checkpoint with its native model/optimizer/selector/RNG
provenance. It checks heldout disjointness and exports CPU raw
`[members,graphs,tasks]` logits and IDs with byte/hash receipts. It never opens
CAL/evaluation truth. I4 requires the complete typed four-body group, distinct
initialization/dropout/shuffle seeds, one route per body and identical resolved
recipes. Each body has its own paid full trajectory and selector.

## Serving, fixed CAL and missing labels

Serving is `sigmoid(mean(raw_member_logits))`; scalar CAL is
`sigmoid(mean(raw_member_logits)/T)`. CAL fits one float64 log-temperature
initialized at zero using 500 Adam updates at LR .01, clamping logT to [-4,4]
after each update. BCE includes only finite observed CAL targets. It binds the
exported CAL bytes and target receipt and reads no evaluation truth.

Evaluation binds completed export/CAL parents and their serving/target laws,
then reports raw T=1 and calibrated binary NLL/Brier using explicit observed-
target denominators. Missing labels must be declared NaNs; infinity/nonbinary
observed labels fail. Per-task observed/missing and positive/negative counts
are retained. Macro ROC-AUC includes exactly tasks with both observed classes,
matching OGB eligibility. No eligible task gives null AUC with explicit counts;
no observed targets gives null proper losses and zero denominator. CAL requires
at least one observed target. Graph rows are not independent statistical
replicates; paired seed inference remains a root whole-cohort operation.

Successful endpoints separately count completed predictions/forwards and scalar
updates. Failures retain `failure_costs.json` and expose
`exception.endpoint_costs` to the root runner. Counts inside interrupted
prediction calls are completed lower bounds; root wall/resource receipts remain
authoritative. No endpoint counts itself as another backbone fit.

## Compiler, domains and reuse

`compile_molecular_matrix.py` validates actual closed preparation manifests,
split/provider metadata and all declared hashes without decoding graph/target
tensors. It binds source commit, authoritative file fingerprints, archived
source pins, numerical recipe, resources and exposure decisions. Its CLI checks
the authorized allocation hostname/GPU. It performs no numerical provider
import, download, model construction, training, scoring or GO creation.

The GINE catalogue has 40 effective configurations: five learning rates × two
widths × two depths × two dropouts, with decay zero. The native point remains
present. The 100- and 300-epoch domains have separate complete search/comparative
trajectories. No prefix reuse, successive halving or extension is implied.

Eight single/shared procedures receive separate two-seed TRAIN/SELECT searches:
`single`, `single_contrastive`, `factorized_single`,
`factorized_single_contrastive`, `be_unit`, `be_init`,
`be_unit_contrastive`, `be_init_contrastive`. Four I4 procedures inherit their
objective-equivalent complete member's selected configuration. The factorized
M1 contrastive procedure uses source two-view alignment and has zero cross-
member residual, as does each contrastive independent body. No cross-body
coupled TRAIN objective is introduced.

`neighbors` expands alignment weight [.025,.05,.1] and temperature [.1,.2,.4],
and shared-BE residual weight [.025,.05,.1]. M1 residual remains .05 because it
is zero and cannot be an effective HPO axis. Noncontrastive domains have 40
configurations, M1 contrastive 360 and shared contrastive 1080. `native_fixed`
holds all source auxiliary values and has 40 configurations in every cell.

Every comparative seed fits four typed I4 groups with exactly four separately
paid own-selected bodies each, plus four shared BE trajectories. The singleton
reference is prospectively the corresponding group's **fixed body0**, using the
same run/config/checkpoint identity. It is never selected as the best I4 member
after outcomes. Thus each group serves M1 and I4 references with four paid fits
and no checkpoint relabelling.

Per dataset and epoch domain:

| Count | Native fixed | Neighbor coefficients |
| --- | ---: | ---: |
| Search optimizer starts | 640 | 6080 |
| Comparative optimizer starts / physical bodies | 60 | 60 |
| Independent comparative body fits | 48 | 48 |
| Shared comparative BE optimizer fits | 12 | 12 |
| Search-selection endpoints | 8 | 8 |
| Served banks / exports / scalar CAL / evaluations | 36 each | 36 each |
| Genuine I4 banks / reused fixed body0 artifacts | 12 each | 12 each |
| Comparative served prediction views | 108 | 108 |
| All run records | 816 | 6256 |

Complete TRAIN passes are optimizer starts times the fixed epoch domain. Actual
Adam updates depend on graph count and complete tail minibatches and are
measured by the engine. Physical bodies, shared fits, independent fits, served
prediction views and cheap endpoints are counted separately.

The disabled plan uses root registry/scheduler/engine and phase receipts:
whole search before comparative; whole comparative before exports/CAL/evaluation;
whole CAL before evaluation. Assignment is allocation-only. Missing original
GNNM binary graph reference leaves that scientific comparison unresolved. This
packet does not claim whole-inventory coverage or published author replication.

## Explicit exposure/resource inputs

Exposure JSON has `datasets` keyed by official dataset name and declares
`prior_exposure`, `evaluate_purpose`, `protected_confirmation`. Explicit unknown
whole-scope exposure is admissible for development without claiming untouched
evidence. Development requires `protected_confirmation=false` and
`evaluate_purpose=development`. Protected requests are retained as blocked
zero-fit packets pending root's frozen activation/eligibility decision. MolBACE
preparation alone does not certify unused evidence.

Resource `profiles` are keyed `dataset/GINE/epochs`, `dataset/GINE` or `*`, with
positive `reservation_mib`, `wall_seconds`, `estimated_seconds_per_epoch`,
`estimated_export_seconds`, `estimated_CAL_seconds`,
`estimated_evaluation_seconds` and assumption/source `evidence`. Estimates are
not measured runtime qualification.

```text
python training_pipeline_v2_20261012/compile_molecular_matrix.py \
  --project-root PROJECT_ROOT \
  --custody data/molecular_molhiv_roles_v2/MANIFEST.json data/molecular_molbace_roles_v2/MANIFEST.json \
  --resources molecular_resources.json --exposure-decisions molecular_exposure.json \
  --host-bindings allocation_host_bindings.json --output molecular_future_packet_v1 \
  --epochs 100 300 --scalar-coefficients neighbors
```

Outputs are fresh compressed studies, `PLAN.json` and `SUMMARY.json`, never an
overwrite. Tests cover real tiny archived-model four-body own selection/export
with target files absent, fixed-body0 identity, post-pool CAL, changed-target
rejection, failure costs, missing masks/AUC eligibility, domain counts, disabled
gates and synthetic provider custody. These are engineering fixtures, not
scientific MolHIV/MolBACE accuracy evidence.
