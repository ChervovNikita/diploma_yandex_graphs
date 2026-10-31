# Molecular GINE/internal-BE packet

## Implementation and scope

`adapters/molecular.py:make_adapter` implements the v2 engine interface. It loads the exact archived `MoleculeBackbone`, internal-factor functions and molecular objectives; the model forward is reused directly. The archived architecture is bond-aware residual GINE with a trainable virtual node, node/output LayerNorm, and all original OGB categorical atom/bond fields. Shared message factors act before GINE nonlinear message aggregation. This is the project's declared GINE adaptation, not an official OGB author-protocol replication.

The stock trainer has fixed MolHIV counts and logs its last minibatch. This packet wraps the model with a configurable full-epoch controller. It supports single-target binary graph classification and an explicitly declared multitask binary output-head adaptation. Regression and arbitrary molecular feature formats are not supported.

Imports of the adapter itself use only stdlib and the stdlib engine. Numerical dependencies load in the approved headless worker: Torch, NumPy, PyG, OGB molecular encoders and scikit-learn ROC-AUC. There is no dataset downloader, OGB dataset constructor or remote operation.

## Factory and prospective run contract

The adapter entrypoint, relative to the research project root, is:

```text
training_pipeline_v2_20261012/adapters/molecular.py:make_adapter
```

The factory uses the existing worker signature `make_adapter(torch, project_root, study, run, device)`. The study task must be `graph_binary_classification` or `graph_multilabel_classification`. `run.budget` must declare `unit=epochs`, `policy=full_budget` and a positive maximum. A fixed maximum is completed even when an earlier checkpoint wins; the adapter never requests an early stop. The engine retains and restores the selected model, optimizer, module modes, gradient flags, selector and adapter RNG state.

`run.selection` chooses `rocauc` with `direction=max`, or `ce`/`brier` with `direction=min`; first or last ties are explicit. The archived MolHIV recipe uses first tied maximum ROC-AUC, evaluation each epoch, no checkpoint0, and 100 complete epochs. Alternatives are declared standardized recipe changes. Saving is eligible at every validation (`save_every=1`). Validation uses only SELECT labels.

The executable schema is `validate_config`; `MOLECULAR_CONFIG.schema.json` provides JSON Schema tooling support. Additional semantic checks cover full budgets, role receipts, I4 cohorts and contrastive target compatibility. Unknown configuration fields and extra fit-role inputs are rejected. Explicit root orchestration metadata (`stage`, `reservation_mib`, `estimated_seconds`, `selected_config_from`, `resolved_selection`, `maximum_cuda_bytes`) is validated and retained in actual config/checkpoint provenance; it adds no new scientific axis or heldout fit input.

## Input packs and role custody

TRAIN and SELECT each have a separate `torch.save` plain-tensor dictionary with exactly these keys:

| Key | Meaning |
| --- | --- |
| `x` | Concatenated int64 `[total_nodes,9]` original categorical atom fields. |
| `edge_index` | Int64 `[2,total_edges]`; each edge block uses graph-local node IDs. |
| `edge_attr` | Int64 `[total_edges,3]` original categorical bond fields. |
| `node_ptr` | Int64 `[graphs+1]`; starts0, ends total_nodes, strictly increasing. |
| `edge_ptr` | Int64 `[graphs+1]`; starts0, ends total_edges, nondecreasing. Edgeless graphs are supported. |
| `ids` | Unique nonnegative int64 original graph IDs, preserved across roles. |
| `y` | Floating `[graphs,tasks]` binary targets; `[graphs]` is also accepted for one task. |

All tensors load on CPU with `weights_only=True`. The adapter validates graph partitions, all graph-local edge endpoints, categorical vocabulary bounds, binary targets and disjoint TRAIN/SELECT graph identities. It performs no graph repair or data filtering. Every graph must have at least one observed target. `complete_binary` rejects all missing labels. `observed_binary_mask` permits only explicitly declared NaNs, with observed-target BCE; infinity or nonbinary observed targets fails.

Each source and pack requires an exact `{path,bytes,sha256}` receipt. Paths stay project-relative and files are hashed before loading. The study's `data.roles.train` and `data.roles.select` must be the same receipt dictionaries as `config.molecular_inputs.train/select`. CAL and evaluation authorities remain separate study declarations; this factory never opens them. Input preparation is a separate admitted custodial action and must preserve the prospective scaffold/split IDs.

## Arms and independent selection

| Config arm | Predictions / live bodies / optimizers per fit run |
| --- | --- |
| `single`, `single_contrastive` | 1 / 1 / 1. |
| `factorized_single`, `factorized_single_contrastive` | 1 / 1 / 1; source M1 contrastive alignment has zero cross-member residual; one unit row at every dense factor site plus one message-factor row. |
| `be_unit`, `be_init`, and their `_contrastive` variants | 4 / 1 / 1; all internal maps/message factors share the complete slow body. `be_init` initializes only first-stem r with seed+900001 Rademacher signs. |
| `independent4_body`, `independent4_body_contrastive` | 1 / 1 / 1 per paid run; four runs are assembled after their own selected checkpoints close. |
| `factorized_independent4_body` and its `_contrastive` variant | 1 / 1 / 1 per paid equipped body; four separate own-selected runs. |

An I4 config requires `independence_group`. Exactly four model-fit runs with body IDs0–3 must belong to that group. They share the declared role/architecture/optimizer/budget/selector recipe and have four distinct constructor, dropout and shuffle seed bindings. Every run acquires, fits and selects its own body. Their ensemble is a later nontraining endpoint with all four completed parents; it cannot choose another pooled checkpoint. Root's registry/scheduler owns the group closure and ensemble assembly.

`independent4_body_contrastive` is a genuine independent single-body auxiliary fit. Its source residual cross-member term is zero at M=1. It is intentionally distinct from the older portable four-body contrastive fit, whose cross-member auxiliary couples TRAIN representations. The older coupled objective is not silently presented as independent training.

## Objective, epoch logging and RNG

Every arm gets exactly two own-loss stochastic TRAIN views per minibatch. The single/shared objective is the mean of the two native BCE views and mean across actual prediction members. `_contrastive` arms add the archived same-label symmetric two-view alignment and differentiable class-centered same-object cross-member residual contrast, using the explicit panel cap, temperature and weights. Contrastive labels require one complete binary target; no multitask similarity rule is invented. All gradients belong to one live body's optimizer for that fit. The full tail minibatch is included.

The archived default auxiliary weights are .05/.05, temperature .2 and panel cap512. The default architecture is width256, five layers and dropout.5; Adam uses LR .001, weight decay0, epsilon1e-8 and betas (.9,.999), batch128, evaluation batch128. These values are explicit config fields so prospective HPO can declare recipe changes.

TRAIN traces contain the preupdate minibatch scalar objective and auxiliary terms aggregated by graph count over the full epoch. BCE also has an observed-target-weighted mean, numerator and denominator. With missing multitask targets, `objective` and `ce_graph_weighted` are graph-weighted, while `ce` uses observed-target counts; the difference is labelled. No last minibatch is represented as an epoch average. Optimizer weight decay belongs to the declared optimizer recipe and is not folded into the reported data/auxiliary scalar objective.

Operation counts record attempted/completed per-member forwards, backwards and Adam updates, unique supervised graphs/targets, route predictions and two-view label opportunities. An exception retains completed lower-bound operations through `AdapterStepError`; a failed optimizer/backward attempt is not counted as a completed update. Midcall native internal work is not a FLOP or elapsed-time estimate; the external engine/owner retains interruption and cost limits.

Persistent member dropout streams use source-compatible CPU/CUDA forked scopes. Their states, global Torch/Python/NumPy RNG, epoch, own selector and source/data/recipe provenance are captured and restored without random draws. The shuffle order is deterministic at `shuffle_seed+19709+1000*epoch`, matching the archived law. Constructor seeding uses the archived private `seeded` scope; its NumPy seed must fit uint32.

SELECT serving uses the mean raw logits, then sigmoid when probabilities are needed. BCE/Brier use every observed SELECT target. Macro ROC-AUC uses each task with both binary classes, matching OGB's eligibility rule; a ROC-AUC selector fails when none is eligible. `predict_pack` accepts an already admitted pack and opens no role file. Graph-only prediction has exactly the original graph keys except `y`, admitted by `validate_prediction_pack`; `_batch` returns no labels. Fit packs still require labels. See `MOLECULAR_ENDPOINTS_MATRIX.md` for selected-bank export, masked scalar CAL/evaluation and the future 100/300-epoch compiler. An approved endpoint can use it after strict checkpoint restoration; CAL/DEV/TEST ingestion belongs to that endpoint, not this fit packet.

## Source pins and example config

These are the saved archived bytes used for this port:

| Source | Bytes | SHA256 |
| --- | ---: | --- |
| `portable_internal_be_public_interface_20261007_v2/core/models.py` |8758|`2be6d872e962d9fdd341883b0d1060170fe0a176f7a8d6829d988fc6f8e7f9c3`|
| `portable_internal_be_public_interface_20261007_v2/core/factors.py` |3286|`9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3`|
| `portable_internal_be_public_interface_20261007_v2/core/objectives.py` |4824|`9208e677ab5cfb286862216991453f8592a1d373120019a6e2ff14c38fad5bcf`|

Populate actual pack receipts and bind these files plus the adapter/engine in the study source manifest. The minimal config shape is:

```json
{
  "molecular_inputs": {"train": {"path": "roles/TRAIN.pt", "bytes": 0, "sha256": "REPLACE_WITH_REAL_RECEIPT"}, "select": {"path": "roles/SELECT.pt", "bytes": 0, "sha256": "REPLACE_WITH_REAL_RECEIPT"}},
  "molecular_source": {"models": {"path": "portable_internal_be_public_interface_20261007_v2/core/models.py", "bytes": 8758, "sha256": "2be6d872e962d9fdd341883b0d1060170fe0a176f7a8d6829d988fc6f8e7f9c3"}, "factors": {"path": "portable_internal_be_public_interface_20261007_v2/core/factors.py", "bytes": 3286, "sha256": "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3"}, "objectives": {"path": "portable_internal_be_public_interface_20261007_v2/core/objectives.py", "bytes": 4824, "sha256": "9208e677ab5cfb286862216991453f8592a1d373120019a6e2ff14c38fad5bcf"}},
  "model": {"hidden": 256, "layers": 5, "dropout": 0.5, "outputs": 1},
  "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0, "eps": 1e-8, "betas": [0.9, 0.999]},
  "training": {"batch_size": 128, "eval_batch_size": 128, "views_per_update": 2},
  "arm": "be_init_contrastive", "fusion": "mean_logits",
  "initialization_seed": 7101, "dropout_seed": 307102, "shuffle_seed": 7101,
  "head_policy": "archived_scalar", "target_policy": "complete_binary",
  "contrastive": {"max_objects": 512, "alignment_weight": 0.05, "residual_weight": 0.05, "temperature": 0.2}
}
```

This example's pack receipts are placeholders and deliberately fail admission. A multitask head replaces only the source scalar output head before factor installation, is labelled `declared_multitask_linear`, and requires a fresh scientific method/config binding. It is not claimed identical to the source scalar model.

## Verification and release status

`tests/test_molecular.py` checks config/role gates, receipt changes, weighted tail/missing-label logging, engine metric/count acceptance, full-budget own selection and independent-cohort membership. Optional Torch-only tests validate literal tensor packs. A dependency-gated CPU engineering fixture copies the three exact archived modules, constructs their real `MoleculeBackbone`, performs one complete synthetic epoch on methane/ethane graph fixtures, evaluates ethanol/methanol SELECT fixtures, and restores/replays the engine-selected checkpoint after deliberate parameter perturbation. The fixture uses real OGB categorical feature indices, four synthetic graph IDs and arbitrary binary labels; it is no scientific result. `GNNM_REFERENCE_PROJECT` identifies the pinned research source root for staged QA (default: the local package's parent). Missing source files fail the fixture when providers are installed; they are not a skip condition.

Local stdlib checks passed; Torch tensor-pack checks are skipped when Torch is unavailable. Full runtime construction/forward/backward, strict model/optimizer/RNG checkpoint replay, real provider pack custody and resource qualification remain a separate engineering gate. This source packet is implemented; it is not yet a qualified READY migration packet and creates no scientific-launch authorization. Scientific fitting uses the existing `run_study.py` approval gate only after explicit human GO. Existing active families finish unchanged.

## Official MolHIV/MolBACE data preparation

`prepare_molecular_data.py` is a separate no-fit/no-score custodian. It admits only `ogbg-molhiv` and `ogbg-molbace` through the official `PygGraphPropPredDataset` provider, with original categorical fields and `transform/pre_transform=None`. It does not construct a model or optimizer, invoke an evaluator, inspect model predictions, or start training.

Its standardized four-role protocol preserves official scaffold TRAIN exactly, divides official VALID into SELECT and CAL by fixed public-ID SHA256 ranking within each binary class, and preserves official TEST as the protected evaluate role. Per-class SELECT count is the floor of the declared rational fraction; the default is1/2. The split seed is mandatory. Each class must appear in both SELECT and CAL. All four roles must be disjoint and cover the complete official provider universe. Labels are used for prospective role stratification, with no scores or graph structure used for choosing the split.

This protocol divides published official VALID and therefore **does not reproduce the published author split/evaluation protocol**. Author replication requires a separate recipe and study track. Dataset prior exposure and confirmation eligibility remain root-owned declarations; preparing a new pack does not make an encountered graph untouched.

The custodian requires cache/output directories inside the project and a fresh output directory separate from the cache. Downloads default to forbidden; `--allow-download` is explicit consent for the official OGB downloader. Any interactive cache/version upgrade prompt fails rather than deleting or upgrading an existing cache. The caller supplies external resource supervision and the authorized allocation route.

```text
python training_pipeline_v2_20261012/prepare_molecular_data.py \
  --project-root PROJECT_ROOT --dataset ogbg-molbace \
  --dataset-root PROJECT_ROOT/data/ogb \
  --output PROJECT_ROOT/data/molecular_molbace_roles_v1 \
  --split-seed 190131 --select-fraction 1/2
```

The command creates `TRAIN.pt`, `SELECT.pt`, `CAL.pt`, `EVALUATE.pt`, `CAL_GRAPHS.pt`, `EVALUATE_GRAPHS.pt`, `CAL_TARGETS.pt`, `EVALUATE_TARGETS.pt`, `SPLITS.json`, `PROVIDER_CUSTODY.json`, `MANIFEST.json` and `STATUS.json`. The manifest binds pack bytes/hashes, original graph IDs, complete role checks, categorical/local-edge preservation and the preparation program. Provider custody records library versions, actual provider program bytes/hashes, provider metadata, every actual dataset-cache file receipt and any explicitly admitted download event. Interrupted/failing preparation keeps its status and partial artifacts; it does not become a completed dataset or overwrite prior output.

Use `MANIFEST.json.roles.train/select` verbatim as the fit study role authorities and adapter input receipts. `MANIFEST.json.endpoint_inputs` binds exact graph-only/target-only heldout receipts; bank export opens only graph-only files. CAL/evaluate files stay with their separately released nontraining endpoints. `tests/test_molecular_data.py` checks the split/custody logic without importing OGB, downloading data or scoring anything.
