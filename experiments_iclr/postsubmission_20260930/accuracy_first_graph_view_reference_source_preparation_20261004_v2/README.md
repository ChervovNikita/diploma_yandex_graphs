# Full-TRAIN native quality references — source preparation

**Source only; exact v2 numerical checks unrun; no scientific execution or release.** This separate package implements the missing references from the bank v2 plan. The sealed four-bank source and its 10% semantic views are unchanged. Accuracy of native-graph predictions is the objective; graph-view diversification is an attributed baseline, not a new mechanism.

## Implemented references

| Family | Physical fit and objective | Training forward/backward passes per update |
|---|---|---:|
| Modern view-augmented single | Ordinary unwrapped native Polynormer; native CE + 0.5 equal-view CE + 0.5 different-view CE | 3 |
| Conventional native single | Explicit alias of predeclared native member0; native CE | 1, charged through member0 |
| Conventional independent4 | Four separately initialized native fits; each has its own native CE, Adam, RNG, clocks and selector | 4 total |

All losses use every official TRAIN label. Native, equal and different pass order is fixed; each weighted CE backpropagates immediately, followed by one Adam step. Native dropout remains enabled. The single coefficients match bank-level native/equal/different coefficients 1/.5/.5, while compute/capacity remain different and disclosed. Native-only references intentionally have no probe supervision. The independent4 quality reference has four independent selection opportunities and is distinct from the synchronized untied comparator.

The ordinary per-member neural code is loaded from the exact immutable v6 native file. Byte-bound v2 helpers provide construction seeds, selected-device normalization/provenance, RNG and CPU-state copying; byte-bound v2 views verify live mask custody. No neural body, wrapper, current study or sealed predecessor is edited. Portable paths resolve against `ROOT.parent`; compilation records resolved provenance and avoids writing bytecode caches into immutable sources.

Use `(split,block seed)=(0,17),(1,29),(2,43)`. Augmented singles use block seeds. Native members use `block_seed+1009*m`, fixed m0..3; the single aliases m0 and never the best member. Each physical fit follows unchanged width512/10-local/1-global, dropout .2/.3/.3, Adam .001/zero decay/native defaults, 200 local + 2500 global. It selects strict native VALIDATION correct-count improvements with earliest ties, one selector across both stages, and update1 as first candidate. At actual200 it restores its own selected-local model+Adam while retaining live RNG; final restoration preserves the overall selected stage, including local winners. Each checkpoint stores full model/Adam/primitives/RNG/modes/gradients/device state. Every selection event is traced; selected-local/final images are saved.

`pool_conventional` requires all four predeclared completed native references with equal source/input context, reopens their selected tensors, and returns a fixed arithmetic class-probability mean. It preserves their separately selected stages/updates and the exact member0 physical alias. There are five physical fits per block, not six: augmented single plus four native fits. TRAIN metrics are fitted-function diagnostics; selected VALIDATION accuracy/NLL/macro-F1 and member competence are exploratory development evidence.

## Caller interface and checks

`python3 -B driver.py` only prints source-only metadata. `load_runtime(execute=True, manifest_sha256=..., protocol_sha256=...)` requires a deliberate root-admitted numerical call. `run_reference(..., execute=True)` takes caller-supplied raw features, full-TRAIN roles/labels, exact fixed-view bundle and VALIDATION labels, plus a bound context. No acquisition, TEST labels, automatic fit queue, HPO, approval ladder or host/sandbox modification is supplied.

The context must bind this source/protocol, actual official TRAIN mask coverage, feature bytes, validation targets and a separate paired protocol binding both bank and reference source manifests. That paired protocol is **pending** and is not manufactured by this source package. Existing server-only masks remain with the authorized caller. Input-projection provenance must be established there; declaring a role is not evidence of its external lineage.

The retained stdlib checks cover loss coefficients/pass counts, fixed seed policy, exact200+2500 stages, strict/local tie selection, four independent selectors/member0 alias, context/duplicate rejection, default import exclusion and portable descriptors. Their deliberate fixtures stay inside this project.

`test_numerical.py` is saved but **unrun**. It checks both kinds/stages: weighted accumulation versus summed loss and one Adam step, active/inactive heads and forward buffers, full state/function restoration and exact next-update replay, selected-local transition preserving live RNG and valid local final restore, fixed probability pooling and member0 alias. Fixtures use small artificial native dimensions; the pooling fixture uses constant artificial logits, never predictions from data. Deliberate temporary fixtures stay inside this project. Numerical verification of bank v2 does not qualify this new driver or objective.

## Remaining limits

Exact reference numerical checks, paired scientific protocol/freezing, full-shape/resource measurements and confirmation/TEST-history rules remain pending. Charge augmented8100 and each native2700 training forwards/backwards, each fit's2700 native selection forwards, checkpoint/restore/reporting and complete host costs. No memory increase is authorized. No scientific mid-run restart orchestration, paired outcome synthesis or competence/gain verdict is implemented. The pinned modern architecture does not establish competence or advantage without the planned comparisons. VALIDATION is selection-associated development evidence; no independent heldout or unseen TEST claim follows.

## V2 admission repairs

This preserved v2 successor repairs the fresh source review and applies the same paired-input requirement to both quality-reference packages. V1 sources and their review/results remain preserved. The mathematical objectives, model operations, seeds, optimizers, clocks, selectors, restoration operations and numerical test code are unchanged.

Created output directories and checkpoint paths are resolved to absolute paths at creation. A checkpoint descriptor therefore reopens the same file after the caller changes working directory. Bound source descriptors retain their portable phase-relative encoding.

Fitting admits CPU and selected-device CUDA only. Unsupported device types, including MPS, are rejected before feature movement, model construction, RNG seeding or fitting. The ordinary native builder has the same guard. The unchanged bank helper's broader parser is not a claim of reference support for other backends.

The future caller-owned paired master has the exact source fields `bank_manifest_sha256`, `reference_manifest_sha256`, and `native_gnnm_manifest_sha256`. Each package verifies its own exact manifest and the bank manifest. It also requires `paired["input_bindings_by_split"][str(split)]` to be an object containing these four fields:

| Field | Required value |
| --- | --- |
| `role` | The live full-TRAIN `role.identity()` |
| `native_edges_sha256` | The live verified bundle's native-edge digest |
| `feature_identity` | The caller context's feature identity, already checked against the live features |
| `validation_labels_sha256` | The caller context's label digest, already checked against the live VALIDATION labels |

A missing split mapping/entry/field or any unequal field is rejected. The ordinary reference also retains exact eligible official-TRAIN coverage-file verification. No actual input bindings or future paired master are produced here; test identities are explicitly synthetic.

Exact v2 numerical checks are **unrun**. Root reported the unchanged v1 small CPU objective/state suites passed (four ordinary checks and two native-GNNM checks). Root also reported that default full-shape CUDA kernels failed exact function/next-update replay despite restored state, with evaluation differences around 1e-5; deterministic CUDA runtime qualification is pending. Those reports do not qualify this v2 source or prove exact GPU replay. No deterministic policy is introduced by this source repair; the prospective all-arm policy remains root-owned.

V2 stdlib verification passed 9 checks, including integration checks for all three split entries, absent/conflicting input/source bindings, relative-path round trips from another working directory, duplicate/missing/conflicting paths and unsupported backends. Deliberate temporary fixtures remain inside this package and are removed. No numerical imports, datasets, servers or predictive results were accessed.
