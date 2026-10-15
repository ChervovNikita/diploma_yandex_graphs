# Reporting rebind for the frozen normalization migration

Source-only successor of the existing reporting integration. It consumes the
actual completed normalization migration, reuses its `reporting_delta`
function to verify every changed spec, and keeps the accepted reporter.
No producer or adapter change is needed: the genuine producer already exposes
the full runtime identity, scalar source metadata and original export hook.

## Finite root command

After review/publication, from the allocation repository, set
`REPORTING_SOURCE_SHA256` to this package's reviewed manifest SHA and run:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/benchmark_reporting_normalization_rebind_v1/rebind_registry.py \
  --source-manifest-sha256 "$REPORTING_SOURCE_SHA256" \
  --migration-preparation /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/normalization_campaign_migration_v1/generated_v1/PREPARATION.json \
  --migration-preparation-sha256 9d4a9c494b8b6e83d2aeaf4480c30e9628e5ad23c2adb7695bf9bc0c7134b6c7 \
  --output /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/benchmark_reporting_normalization_rebind_v1/generated_v1
```

Hostname/sole-GPU identity checks precede project reads. The generator reads
JSON metadata and source bytes only. The fresh output contains 27 profiles,
`INDEX.json` and final `PREPARATION.json`; a partial failure retains
`FAILED.json`. It grants no launch, resource, qualification or GO authority.
The old campaign output root remains a recorded proposal awaiting release and
claim-custody review. Changed fit output paths are exactly those already
declared by the migration's unadmitted dispatch delta.

## Preserved identities and explicit attribution

The rebind retains 6,460 physical fits, 270 I4 exports, 6,190 predictor rows,
270 source-reference rows and 270 same-bank-mean rows. The 3,570 unchanged fit
specs and all I4 specs remain identical, including original GNNM, native source
references, M1/body0 and independent native bodies. The 2,890 changed specs use
the exact new descriptor, readout and genuine composed runtime identities.
Their common native/data/controller/evaluator comparison bindings are retained.

Every contrast is reconstructed from the actual successor `CONTROL_LINKS` and
exact fit/cohort pairing, using the existing reporter's format. Existing
contrast definitions remain. Added private-BN comparisons carry
`primary_ingredient_attribution`; retained direct original-GNNM comparisons
carry `total_extension_vs_original_GNNM`. Exact paired fit links are retained
in each profile and in collector output. Same-bank means remain diagnostics
from their own existing selected export and never add a fit or a forward.

Each profile and registry explicitly binds
`composable_normalization_final_scoring_v2/MANIFEST.json`, SHA256
`af692038bec60bbb54e9ed45d659289af98a0cb844ab0637c8923ee6ce70b67b`.
This is a source binding, not a TEST grant or a new scoring roster/seal.

## Reporting source seam

`runtime_entry.py` and `backend_bridge.py` are byte-identical to integration V2.
The observer's existing original writer call and export payload are retained.
For composed jobs only, the successor verifies producer/normalization source
receipts, exact public runtime hashes and the unchanged native helper view.
It checks callback/extension/policy identity and actual source-dependency bytes,
and records the full witnessed runtime identity in scalar invocation and
selected-export sidecars. Failed or missing runtime authority withholds
reporting bindings while preserving invocation costs. The collector requires
that same witness to match the actual RESULT and selected scalar sidecar.
No predictions, checkpoint payloads, exports, scores or qualifications are
created by the metadata generator or source checks.

The dispatcher reporting hook keeps its four fields: `bridge`, `entrypoint`,
`registry`, `manifest`. Its fixed source folder must explicitly change to this
package. Manifest rows contain both package-relative `path` and the existing
`path_relative_to_phase`, so existing closure iteration can remain intact.
The caller must require the successor registry's
`normalization_rebind.schema`, exact `source_manifest`, actual
`migration_preparation`, and repaired `selected_serving_source`; old profiles
must not be accepted. All manifest members, the generated registry/profiles
and the pinned dependency sources belong in the reviewed release closure.
The bridge remains a pure argv transform under the original resource owner.

## Checks and remaining blockers

`PYTHONDONTWRITEBYTECODE=1 python3 -B check_source.py` uses one ten-cohort
metadata profile, actual generator/contract functions, the unchanged reporter,
and the actual observer on failed scalar invocations without selected exports.
It checks stale-parent, missing/duplicate/cross-cohort control and malformed
runtime refusals, preserved costs and withheld all-missing intervals. No
Torch/NumPy, model, array, tensor, native replay or TEST payload executes.
One separately authorized guarded read-only allocation inspection confirmed
actual MolBACE/BUDDY field schemas and no duplicate targets in 5,110 control
rows. It performed no remote generation or mutation.

Remaining work belongs to root:

1. Run this finite metadata generation and review its actual output receipts.
2. Retain the required genuine normalized producer qualification and attach
   an observer witness through existing authorized work; no duplicate native
   replay or new smoke grid is authorized here.
3. Bind exact reporting profiles, source closure, candidates, global ownership
   and claim/checkpoint custody in the separately reviewed dispatcher release.
4. Attach measured resources/costs and reciprocal pair admission, then the
   user's GO for that exact updated campaign. Server77 remains disabled.

Original paper scores and all frozen predecessors are unchanged.
