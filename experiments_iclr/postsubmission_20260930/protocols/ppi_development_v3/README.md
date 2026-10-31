# Prospective PPI development v3

This packet prepares four runtime freezes: sharing and architecture at 10% and 100% observed training labels. It binds six freshly supervised training-only probes from `repoenv_repro_v1`. It contains no new predictive result and does not modify historical scores, protocols or probe locks.

The current amended runner is preserved with all six source files under `source_snapshot/`. The previous runner is preserved separately under `protocols/versions/pre_repoenv_selector_amendment_v3/`. Every prepared file is covered by `artifact_manifest.json`.

## What is ready

- `JOINT_DESIGN.json`: exact operations, families, source/data/evidence identities, paired analysis, continue/stop rules and cost measurements.
- `SELECTOR_LOCKS.json`: clean training-run references and their checkpoint, source, numerical-replay and supervisor hashes.
- `QUEUE.csv`: a deterministic maximum of 96 unique fits; five controls are reused across analyses.
- `prepare.py`: a stdlib evidence check and command printer. It never launches a fit.

The six report rows were independently recomputed from the comparison script without invoking its exclusive report writer. The exact updated runner lock function passed without importing torch. Local manifest copies are byte-identical copies of the coordinator-fetched remote manifests; no provider tensors were fetched or loaded locally.

## Stage order

| Stage | Budget | Maximum fits | Decision |
| --- | --- | ---: | --- |
| 1 | 10% | 33 | Three seeds for tied/three private placements, four architecture cells and three matched controls. |
| 2 | 10% | 15 | Finish supported comparator families: untied for sharing and four common complete controls for either family. |
| 3 | 100% | 33 | Run each mechanism core only after its 10% gate and full comparator/cost review support continuation. |
| 4 | 100% | 15 | Finish supported 100% families. |

The 96-fit ceiling is 16 arms × two nested budgets × three seeds. Each fit has 300 epochs and 6,000 graph optimizer steps: at most 28,800 research epochs and 576,000 optimizer steps. The six completed policy probes contributed 1,500 graph steps separately. Saved initialization/profiling work is engineering evidence, not training reproduction or a selected-model performance frontier.

Apply gates only after all three required seed checkpoints replay. Missing or failed fits are incomplete evidence. No one-seed screen can promote a candidate. If both 10% mechanisms stop, stages 2–4 are skipped and the maximum becomes 33 fits. PPI is one development task; finish the broader independent-task program before a transfer or general-method claim.

### Fixed gates

For sharing, the locked selected placement must have lower graph-mean member BCE than tied in at least two seeds and lower BCE than both rejected equal-capacity placements in at least two seeds. Mean pooled BCE deterioration must be at most 0.001 and mean micro-F1 deterioration at most 0.5 points. The selections are middle/middle/first at 10%, first/first/first at 100%. The probe checkpoint is not used to initialize the development fit.

For architecture, centered feedback must improve graph-mean member BCE over both uncentered feedback and centered no-feedback in at least two seeds each. Stop if a single simple narrow/packed/BASE control is within 0.001 pooled BCE and 0.5 F1 points of centered in at least two seeds at the registered parameter budget. This is a quality/parameter screen; deployment latency requires measurements of the actual selected checkpoints.

Full criteria and exact contrasts are in `JOINT_DESIGN.json`. A stopped partial family cannot support a complete-method comparison. A validation-selected replacement layout remains a descriptive oracle with extra selection cost.

## Coordinator commands

Use the existing repository interpreter, the already-authorized physical GPU and a saved reference to that authorization. These variables are inputs from the coordinator's existing authorization; this packet does not authorize a new GPU or request duplicate permission.

```bash
source protocols/repo_env.sh
"$PPI_PYTHON" protocols/ppi_development_v3/prepare.py check
"$PPI_PYTHON" protocols/ppi_development_v3/prepare.py print-freezes \
  --python "$PPI_PYTHON" --physical-gpu "$PPI_PHYSICAL_GPU" \
  --authorization-file "$PPI_GPU_AUTHORIZATION_FILE"
```

The printer returns commands to capture real hardware, run four **supervised CPU freezes** with the installed provider/runtime, then write new joint protocol files under `protocols/ppi_development_v3_runtime_v1/`. The raw runner freezes remain unchanged. `FROZEN_BUNDLE.json` is created only after all four runtime freezes exist. Printing commands does not execute them.

Actual runner freezes remain pending here. They validate frozen provider bytes, installed provider class, current source/environment, masks, confinement and the changed selector path. Previous archived engineering evidence refers to its archived source version. It must not be relabeled as a gate for the new source identity.

Once the coordinator completes those freezes, print one sequential fit/replay pair:

```bash
"$PPI_PYTHON" protocols/ppi_development_v3/prepare.py print-fit \
  --python "$PPI_PYTHON" --physical-gpu "$PPI_PHYSICAL_GPU" --ordinal 1 \
  --runtime-bundle protocols/ppi_development_v3_runtime_v1/FROZEN_BUNDLE.json
```

Run only one fit on one idle GPU. After successful completion, invoke the printed CPU replay from that fit's archived source; it checks all 300 retained epoch metrics, the checkpoint rule and fresh selected predictions. Record the supervisor result and replay hash before advancing. The live runner checks GPU idleness again before moving a model to CUDA. Preserve every failure and log; a documented execution retry uses `--attempt 2` with unchanged settings and a new directory.

For conditional rows, add `--gate-receipt PATH`. The coordinator's receipt must bind `joint_design_sha256`, list `decisions` (`sharing10`, `architecture10`, `sharing100`, `architecture100`), assert `all_three_seeds_replayed: true`, and `test_scored: false`. Each decision must cite its three seed inputs, fit/protocol/result/replay hashes, prescribed comparisons and any full-family completion review. A stage-3 positive decision includes the completed 10% comparator and selected-checkpoint cost review. The printer checks the receipt's identity and allowed condition; it does not substitute for the numerical review.

## Interpretation limits

All arms have one recipe and the same 300 checkpoint opportunities. Endpoints use the same pooled-F1-selected checkpoint. Shared controls are dependent evidence reused across families, not independent studies. The two budgets are nested on the same PPI graphs. Report each paired seed delta, mean, SD and range; a three-seed interval is conditional on this one task.

The available test labels/cache are public, so the restriction is **no predictive test scoring**, not cryptographic label blindness. This runner has train and validation loaders only. Independent confirmation tasks must remain unscored until a candidate and confirmation protocol are locked. Novelty remains provisional given BatchEnsemble, TreeNets, Packed-Ensembles and narrow adapter precedents.
