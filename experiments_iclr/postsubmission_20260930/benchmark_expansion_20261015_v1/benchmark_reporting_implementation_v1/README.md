# Exploratory paired reporting — runnable metadata implementation

This implements the accepted `benchmark_reporting_protocol_v1/PROTOCOL.json`
and `ROOT_REVIEW.json`. It reads JSON metadata only and changes no stored
scores, scientific status, training plan or access permission. No model,
tensor, prediction reconstruction, TEST scoring or reviewer is invoked.

## Commands

Use Python with NumPy1.22or newer. The checks used the existing bundled Python
with NumPy2.3.5; no environment was installed. The allocation's existing NumPy
environment can run the reporter after source deployment.

From this folder:

```sh
python3 -B check_metadata.py
python3 -B reporter.py report \
  --registry metadata_fixtures/REGISTRY.json \
  --observations metadata_fixtures/OBSERVATIONS_COMPLETE.json \
  --contrast-plan metadata_fixtures/CONTRAST_PLAN.json \
  --output my_new_fixture_report.json
```

The included fixture has invented, explicitly labeled **fixture** graph names
and scalar numbers. It is an engineering example, not an experiment. Outputs
must be fresh paths. `CHECKS.json` records nine check groups covering missing
cohorts, unresolved caps, wrong pairing/endpoint identities, duplicate records,
omitted prescribed contrasts, signs/ties, I4 ownership and actual metadata
extraction. No extra native tests are implied.

Other commands:

```sh
python3 -B reporter.py compile-contrasts --result FACTS.json --output plan.json
python3 -B reporter.py extract-runtime --descriptor JOB.json --result RESULT.json \
  --identity frozen_identity.json --binding-evidence binding_evidence.json \
  --output new_normalized_row.json
python3 -B reporter.py extract-I4 --result I4_RESULT.json --identity frozen_I4_identity.json \
  --components four_normalized_native_bodies.json --output new_I4_row.json
python3 -B reporter.py extract-compact-molecular --result compact_MolBACE_REPORT.json \
  --output new_pilot_extraction.json
```

`PRESCRIBED_CONTRASTS.json` is generated from the accepted pinned FACTS:
**1,170 contrasts across27presets**, including27nativeM1/I4 context contrasts.
No scores were read to compile it. Same-selected-bank fixed mean is a separate
readout of the same method. The two task aliases resolve graph geometry to
`task_geometry_alignment_variance` or `link_same_query_endpoint`, and structured
responsibility to `structural_class_quartile_private_v1` or
`link_endpoint_responsibility_v2`. An inapplicable/unresolved control raises an
error. A report rejects omitted/replaced presets, prescribed contrasts or
frozen seed/split identities.

## Input contract

`SCHEMA.json` and `metadata_fixtures/REGISTRY.json` define the JSON interface.
The production registry is a source-generated inventory of expected predictors,
not an inventory inferred from successful results. Each expected predictor
identifies dataset, preset, source/matched track, exact cohort ID/seed, actual
initialization seed, split index/literal split seed, scientific method/runtime
procedure, actual/diagnostic readout, exact fit ID, endpoint name/units/direction
and source/data/controller/evaluator/readout hashes. I4 additionally declares
all four component fit IDs in body order. Expected missing rows remain expected.

`extract-runtime` verifies the existing `native-full-epoch-fit-result-v1`
against its exact descriptor/job hash and identity. It retains raw RESULT,
selected checkpoint/export, selected/trained epochs, stop reason, native proper
loss, runtime binding and actual costs. It does not turn an engineering job or
`matched_pilot` into an expanded benchmark fit. Those scopes remain unresolved.

Expected hashes are **not** copied into observed evidence. Optional binding
evidence must anchor the exact job hash and selected export SHA and identify
its producer receipt. Without it, missing source/data/controller/evaluator/
readout authority remains unresolved. The reporter validates metadata anchors
and the frozen registry; it does not establish those source/data facts by
reopening tensors or independently executing an evaluator.

The I4 extractor consumes the existing
`independent-own-selected-VALID-bank-result-v1` JSON and four normalized native
bodies. It checks all body IDs, seed offsets, selected epochs/checkpoints,
job hashes and exact M1/body0. Four bodies remain four physical fits; the export
adds zero fits. Missing aggregate readout authority is not supplied from the
expected registry. All component epochs and costs remain attached.

A `same_selected_bank_mean` observation must use exactly the actual predictor's
fit ID, selected checkpoint and epoch. It is a labeled diagnostic readout,
not another fit or another selection. Its raw metadata row must come from an
authenticated producer sidecar. The current runtime RESULT does not expose
all mean-sidecar scalars in JSON, so production sidecar extraction still needs
integration; this reporter never opens `selected_VALID.pt` to manufacture it.

## Paired summaries and missingness

Every prescribed contrast emits ten frozen cohort rows, both exact predictor/
fit identities, every available native lhs-minus-rhs difference, failure/
missingness and unresolved flags. Higher endpoints treat positive differences
as descriptive wins; lower endpoints reverse only the win/loss interpretation.
Ties use exact saved values before rounding. Missing pairs are not imputed.

Mean, sampleSD(ddof1) and median retain their available-pair denominator. Fewer
than two scored pairs have no SD. A capped matched trajectory remains unresolved
even when it has a numeric selected score. Its score, sign, epochs and costs
are retained. Source references remain separate; a source fixed horizon is not
treated as matched-controller convergence.

A descriptive marginal95% paired percentile interval is emitted only when all
ten own-selected pairs are present and resolved. Missing/unresolved cohorts
withhold the interval, rather than reducing its sample to survivors. NumPy
Generator(PCG64(130102026)) generates a10000×10 index array in frozen seed/split
order, reused for complete contrasts within the same preset. Quantiles use
linear interpolation. NumPy version is recorded. There are no p-values,
significance/superiority labels, winner replacement or confirmatory claims.

Raw normalized rows and the explicit attempt cost ledger are preserved.
Ledger attempt IDs must be unique. Known worker-wall costs are summed with
unknown counts shown; elapsed wall is left unknown. Row costs are not summed
as complete lifetime cost, and M1/body0, diagnostic readouts and I4 components
are not silently charged twice. Preprocessing, inference/export, retries,
resume/failure, process memory, storage and historical reuse stay explicit or
unknown; no absent cost becomes zero by inference.

## Existing result checks and remaining integration

`COMPLETED_MOLECULAR_EXTRACTION.json` preserves the completed MolBACE24 report's
24 raw fit rows,18 method/cohort rows, exact selected metrics/IDs/epochs and
costs. Its input hash is unchanged. Seeds101/202/303 remain a **three-seed
validation-only pilot**; no ten-cohort bootstrap or new interval is attached,
and prior intervals are neither recomputed nor promoted. The checks also
exercise the actual current v4 descriptor/RESULT schema from the completed
CPU lifecycle, retaining engineering scope and missing binding evidence.

Remaining production work is concrete:

1. Generate the full expected predictor/fit/component registry from the frozen
   compiler descriptors and cohort contracts; freeze it before outcome ingestion.
   The complete prescribed contrast/cohort plan is already generated here.
2. Wire the audited source/data/controller/evaluator/readout binding producer
   to the normalizer. Missing evidence must continue to block complete intervals.
3. Export authenticated selected same-bank mean/support/member diagnostics into
   compact JSON where currently available only inside tensor payloads.
4. Connect the complete attempt/cost ledger, including resumed/failed and
   engineering work. The reporter preserves supplied entries but cannot infer
   absent history, lifetime costs or timing from a final RESULT alone.
5. Freeze paper table/figure serialization on top of these rows; the current
   output is a structured JSON report, not a manuscript or benchmark result.

These gaps mean the implementation is runnable and tested, not that the
6,460-fit campaign/reporting integration is finished or authorized. Root owns
source review, publication and subsequent integration. No source scoring,
original paper experiment or past score was changed.
