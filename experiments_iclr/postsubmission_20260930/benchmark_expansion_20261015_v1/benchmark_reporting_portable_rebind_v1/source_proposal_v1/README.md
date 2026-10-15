# Allocation reporting registry execution rebind

This source prepares a successor registry for the actual
`portable_dispatcher_v2/generated_v2/allocation` projection. The parent
registry, observer, scalar sidecars, collector and reporting entrypoint remain
unchanged. The source is not a launcher or an execution admission.

The root operator must first generate and pin the real portable projection.
Then, on the verified allocation, invoke:

```sh
python3 -B rebind_registry.py \
  --parent-index-sha256 EXACT_PARENT_INDEX_SHA256 \
  --preparation-sha256 EXACT_PREPARATION_SHA256 \
  --candidates-sha256 EXACT_ALLOCATION_CANDIDATES_SHA256 \
  --release-sha256 EXACT_ALLOCATION_RELEASE_UNADMITTED_SHA256 \
  --portable-manifest-sha256 EXACT_PORTABLE_SOURCE_MANIFEST_SHA256 \
  --output /absolute/project/path/to/benchmark_reporting_portable_rebind_v1/generated_v1
```

The input paths are fixed in source. The output must be fresh and inside this
package. An incomplete output without its final `INDEX.json` cannot be adopted.
Full profiles stay on the allocation; only compact generated receipts need to
be transferred or published.

## Fixed verification and changes

The script checks the literal allocation hostname and sole GPU UUID before
project metadata reads. It requires all 6,730 actual tasks to remain assigned
to `allocation/gpu0`, with no GO or task qualification. It verifies the pinned
portable source closure and its reviewed runtime delta, the canonical compiler4
plan, all original reporting profiles and both versions of every descriptor.

On the allocation the source and destination roots coincide. Consequently,
every projected fit descriptor must equal the canonical descriptor completely,
including the policy, seed, native recipe, scientific contract and data/source
receipt values. Changed scientific fields are refused. The entire projected
task must also equal its expected canonical projection. Cohorts, dependencies,
output names, contrasts and all counts are preserved.

Only the exact descriptor and runtime entrypoint receipts change for fits.
The original receipts and canonical job hash are retained in
`canonical_execution`. Common native-source, data, controller and evaluator
fingerprints, wrapper callback authority and the expected input-authority
arrays remain identical. The output root comes from the pinned portable
release.

For I4, the exact aggregator receipt changes. Its readout fingerprint is
recomputed with that actual source using the original reporting definition;
the original aggregator and readout binding are retained. The same four
component fits remain in the same order. Common comparison fingerprints stay
unchanged. Expected I4 reporting rows receive this exact successor readout
binding; other expected rows stay unchanged.

These are expected metadata authorities. This script does not open datasets,
checkpoints, member logits or model objects, and it produces no scalar scores.
Production observation must still verify the actual input bytes before future
execution. No training, TEST evaluation, hardware/resource qualification,
77 access or Git operation is performed. Root owns review, publication,
source/runtime admission and any later campaign GO.

## Current status

Source proposal only. No projection rebind has been executed. The metadata
build waits for the root operator's actual generated projection receipt and
exact pins; it must not substitute an invented or earlier projection.
