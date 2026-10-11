# Native matrix packet

12 October 2026. `compile_native_matrix.py` is an allocation-only preparation compiler. It imports no Torch/provider arrays, fits no model, computes no predictive score and creates no GO receipt. The root supplies real prepared data custody, scheduling estimates, host verification and execution transport.

## Inputs and output

Required inputs are a phase `--project-root`, one complete `CUSTODY.json` per dataset, an explicit `--resources` JSON and a fresh `--output` path. All input/output paths stay inside that phase root and retain identical project-relative names when synchronized between hosts. The compiler verifies the allocation hostname/sole GPU before opening project metadata, verifies every custody file fingerprint, and refuses output replacement.

Final generation also requires `--frozen-runtime`, `--runtime-package`, `--metadata-dir`, `--native-model-source` and one or more `--fixture-receipts`. The root snapshots the named native runtime files, Model/TABMModel, numeric/catalogue metadata and qualification receipts outside the live engine first. Factory paths, schema validation and `PLAN.runtime_launcher` use that frozen snapshot. Source binding never scans unrelated or changing adapters/other packets; future migration work cannot invalidate running native jobs. Pure compiler helper defaults use the live engine only for metadata unit checks.

Resource JSON has `profiles` keyed by `dataset_id/backbone_name`, with an optional `*` fallback. Each entry declares positive integer `reservation_mib`/`wall_seconds`, positive `estimated_seconds_per_update`, and saved `evidence` describing the measured source or explicit planning assumption. Estimates are scheduling inputs. Packet status is **PREPARED_ESTIMATED_RESOURCE_PROFILE_REQUIRES_RUNTIME_QUALIFICATION**, not scientific READY. Native wide/full-graph cells still need the root's actual provider/task/resource qualification.

Resource JSON also supplies positive `worker_capacities_cuda_bytes` for `allocation/gpu0`, `server77/gpu0` and `server77/gpu1`. Assignment rejects a group that fits no declared worker; saved capacity bounds do not substitute for live hostname/UUID verification.

The output contains `PLAN.json`, compressed `studies/*.json.gz` (or `--plain-json`), `SUMMARY.json` and `INVENTORY_COVERAGE.csv`. Spec/plan hashes bind the frozen contracts. The exact frozen native runtime, fixture receipts, source metadata, numeric recipes and prepared graph/role files have individual fingerprints. An existing cache's preprocessing version remains a custody limitation; the current import does not retroactively certify it.

## Scope and exact numeric search

The eight source interfaces are **ResNet, GCN, SAGE, GAT, GAT-sep, GT, GT-sep and TAG**, using the actual source `Model`/`TABMModel`. LayerNorm, FFN multiplier 1 and heads 8 are locked. Every interface has exactly **100 deduplicated base configurations**:

| Block | Numeric domain | Count |
| --- | --- | --- |
| Broad standardized grid | LR .00025/.0005/.001/.002/.004 × width 128/256 × depth 2/3 × dropout .2/.4 × AdamW decay 0/.0001 | 80 |
| Source default anchors | LR .00002/.00003 × width 256/512 × depth 1/2/3/4/5 × dropout .2 × decay 0 | 20 |

Every search trajectory uses **5,000 full-graph optimizer updates**, accuracy-first strict earliest selection, checkpoint0 disabled, and fixed full-budget execution. There is no halving, early configuration pruning or runtime extension. Search uses seeds **70101/70103 on registered split0 once per dataset/backbone/method**. Complete within-cell ranking uses mean SELECT accuracy, then mean SELECT proper loss and stable configuration ID. It creates one frozen selection endpoint per method. HPO is not repeated across WikiCS's 20 masks.

All prepared official/custom splits are then compared using **70201/70203/70207**. Comparative configurations resolve from the immutable selection endpoint and cannot add values outside its declared domain. Protected confirmation, binary/Penn94, large-graph sampler interfaces, third-party fixed-author replication and other mechanism packets are outside this compiler. Original GNNM's original five datasets are rejected regardless of aliases. The original five scores are preserved.

## Explicit methods and references

Each native interface searches native single, **original boundary GNNM**, all-map shared4, all-map factorized single and coupled untied4 separately. Original GNNM retains its source boundary architecture, mean-own multiclass CE and **mean raw logits** serving. All-map sharing is a separate method and uses mean probabilities. These are new-dataset standardized comparisons; no archived paper experiment is repeated.

Genuine ordinary I4 inherits native-single HPO; genuine factorized I4 inherits factorized-single HPO. Each bank has four complete independently fitted/selected members, distinct optimizer/model seeds and own unscaled member objective. Its member0 is the exact corresponding comparative single fit; three additional member fits are materialized. The existing member0 counts once as a fresh optimizer trajectory and is explicitly reused by the I4 bank. A coupled untied4 fit is one optimizer trajectory/four physical bodies; it is not called independent I4.

**SAGE only** additionally searches eight scalar-loss transfers: GNCL, within-route SupCon, GNCL+SupCon, ordinary/factorized SupCon singles, paired class-full alignment, paired SupCon Eq2, and paired alignment+residual contrast. A factorized SupCon I4 inherits its matching member search and uses four independent member fits. These are explicitly **new standardized SAGE transfers of audited tensor formulas**, not source-exact PolyFormer/Polynormer/GINE or historical training recipes. No context-target/private-gradient/FoRDE source is silently replaced by a scalar loss.

Default scalar coefficient domains are pool weight .25/.5/.75; temperature .1/.2/.4; and applicable SupCon/alignment/residual weights .025/.05/.1. Every applicable Cartesian tuple multiplies the same 100-point base domain. `--scalar-coefficients native_fixed` is a distinct prospectively frozen alternative using source native coefficients, not an outcome-driven reduction. Paired objectives retain two complete supervised TRAIN views and a fixed first-512 ordered TRAIN auxiliary panel; within-route SupCon uses complete TRAIN. These additional configuration counts appear in the compiled summary.

## Serving, calibration and global embargo

Each selected comparative bank has its own label-free export, one fixed scalar CAL fit and one evaluation endpoint. Export opens CAL/evaluate IDs only; targets remain separate sealed `{ids,y}` files. Original GNNM exports four raw-logit routes with its mean-logit pool. I4 exports four independently selected complete learners with a mean-probability pool.

This packet predeclares **temperature after the native serving pool**, on mean logits or the log of mean probabilities: 500 float64 Adam steps, LR .01, initial log T=0, log T bounded to [-4,4]. This is the packet's standardized calibration policy; it differs from the earlier member-temperature-before-probability-mean design default and is not asserted to reproduce historical native calibration. CAL never selects a backbone/configuration.

Every dataset/backbone dependency group stays on one GPU through `partition.assign`; all three logical workers have two supervised slots. Assignment can be prepared on the allocation without accessing 18.77. Its host binding remains pending until the root's future live verification. No placement constitutes permission to launch.

Global gates have fixed project-relative receipt paths and require `status=complete`, the current `plan_hash`, matching `closed_phase`, and exactly the declared required-host receipts:

| Gate | Source stages | Jobs held behind it |
| --- | --- | --- |
| whole_search | search, search_selection | comparative |
| whole_comparative | comparative | bank_export, calibration, evaluation |
| whole_calibration | calibration | evaluation |

Per-study selection closure does not satisfy whole-search closure. The root collects verified small host receipts and releases stages through the approved launcher/runner gate contract. Missing/failed required references keep comparisons unresolved. No partial CAL/evaluation outcome selects another configuration or restores a failed arm.

## Typed counts and remaining coverage

The compiler enumerates unique fresh optimizer starts by stage, coupled fits, physical trained-body instances, initial segments, update ceilings, selected search banks, comparative banks, route views, I4 banks/body0 reuse, bank exports, scalar CAL fits and evaluation jobs. It reports exact registered fixed maxima, not executed work. Source acquisition fits, continuations and extensions are zero in this packet; failed-attempt actual costs are recorded during execution.

For **one dataset with J prepared splits**, the five base methods across all eight backbones need 8,000 search starts and 264J comparative starts; comparative banks/CAL/export/evaluation counts are 168J each. The default SAGE coefficient-neighbor transfers add 20,400 search starts, 33J comparative starts and 27J banks of each endpoint type: totals **28,400 search + 297J comparative optimizer starts**, **33,200 search + 369J comparative physical trained bodies**, and **195J comparative banks/CAL/export/evaluation jobs**. These formulas assume all eight interfaces and all declared scalar transfers, and describe a single dataset roster; aggregate counts come from actual materialized studies. With fixed native scalar coefficients, additional search is 1,600 rather than 20,400. Comparative counts are unchanged.

`INVENTORY_COVERAGE.csv` retains every canonical inventory entry and distinguishes the native operator packet, SAGE formula-transfer support and uncovered source/migration blockers. **This packet does not cover all 249 predictive candidates.** Grouped panels, source-specific tasks, unsupported samplers and all other variants still need their exact adapter/provider/protocol packets. Author replication and unused confirmation add no ready runs here. The root presents the concrete aggregate plan and waits for explicit human GO before any benchmark training.
