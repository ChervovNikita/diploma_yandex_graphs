# Independent source review of the disabled native utility qualifier

6 October 2026. **NEEDS_SHARED_SUPPORT_REPAIR; prospective external resource closure also required.** Reviewed qualifier SHA25672812e4ade4d81b0076d3567b3787b466613d882be7dadb3aefaf144d0c67b81. No prepared source/Torch/native import or execution, numerical/data/checkpoint/result payload read, SSH, GPU query, fitting, source edit, pilot change or new agent occurred. AST parsing and SHA/JSON metadata checks used the standard library only. This review does not require or assert a numerical PASS.

## Findings

### 1. Required nontrivial shared mixed branches are not separately checked

`qualify.py:304–309` obtains `h_path` and `g_path` as `(shared_tree, private_tree)`, then tests `maximum(h_path)` and `maximum(g_path)`. The helper recursively takes the maximum over BOTH trees. A nonzero private Hessian contribution can satisfy either test while that branch's entire shared mixed derivative is zero. The diagnostic likewise reports only combined maxima.

The candidate utility code is algebraically correct: `utility_control.py:159–174` computes both own-CE g and weighted-margin h with `create_graph=True`, forms the live signed dot product, and differentiates it with respect to both theta and phi. The isolated support's three expressions correctly expose the two product-rule branches. Its sum comparison and candidate comparison include every shared/private coordinate. The gap is in the promised **nontrivial native shared support** criterion: equality of zero shared branches does not certify that both native shared mixed paths were exercised. The earlier complete-episode nonzero shared-update check can be satisfied by direct credit or one other contribution and does not close this gap.

Before using this caller to certify that stronger support claim, separately require and record `maximum(h_path[0])` and `maximum(g_path[0])` above the prospectively fixed threshold, with all shared coordinates finite. Keep private maxima separate and preserve any zero-path failure. Do not tune the cotangent, threshold or state to obtain PASS. This correction changes checks/diagnostic labels, not the54F/76-native/87-API construction schedule. Root will prepare any successor; no source was changed here.

### 2. Numerical-worker measurement and complete child lifetime need separate resource closure

`run` checks limits after its restoration/save (`qualify.py:565–567`), but `main` subsequently sets PASS and performs its final JSON write/rename/chmod and output (`609–617`). `save` samples resources before serializing/writing (`591–595`). Thus the current worker does not measure or enforce an exactly self-inclusive final-publication lifetime. PLAN/REPORT prose describing final serialization as included in a whole-child enforced cap is stronger than the implementation.

This is a closure requirement for the separately prospective supervisor, not a request for an impossible recursively self-inclusive last write. Keep the worker's final numerical/restoration measurement labeled by that boundary. A root-frozen wrapper must independently record child lifetime through termination/result publication, reconcile child status/exit/source/restoration, and enforce a total elapsed limit (and declared external resource checks) before the wrapper admits PASS. The longer external watchdog is termination protection; merely requiring its duration to exceed the worker cap does not enforce the complete child lifetime against that worker cap. Charge wrapper startup/monitoring/receipt work separately and prospectively. Runtime, GPU availability and all caps remain unset here.

## Requested early-failure output inspection

For ordinary failures reached inside `main`'s try—such as a rejected scope in `read_scope` before Torch—the qualifier uses the same fresh `output` directory in initial save, terminal save and chmod (`584–615`). It writes `output/RESULT.json`; no separate ordinary-worker result path is invoked. The reused base module contributes constructor/snapshot/check helpers; its ordinary `main` is not called. I found no mismatched ordinary-worker output/chmod path in this caller.

There are legitimate pre-receipt failure boundaries: argument errors, `output.mkdir` and the initial worker hash occur before the protected try, and an unwritable/full output can also make receipt saving fail. The prospective wrapper must preserve exit/stderr and classify missing/malformed child results as failure, including hard termination. A saved child PASS or absent receipt alone cannot be wrapper admission. This is part of finding2's external closure, not a third claimed numerical defect.

## Verified source properties

* **Source/gates:** all25 entries in the preparation SOURCE_BINDINGS and all6 manifest files match bytes/hashes; qualifier parses. Exact base/operator/port/native/boundary/sequential/utility/accessor pins are enforced on load and rechecked on exit. Original/candidate gates remain disabled; accessor stays its existing released gate. Root reviews/synthetic prerequisite/scope and public+B manifest are bound before Torch/data access. No default marker is flipped.
* **Cold scope:** exact base.build_family uses seed17, constructor followed by author reset, M4 boundary family, global stage and complete eval. No warm/trained checkpoint path exists. Port verifies complete native architecture,300 features, private shapes, eval/no buffers and parameter/context device/dtype; callback substitutes a complete supplied private row into one forward_member route with strict tied functional_call. The planned full24492-node FP32 context and S2449/R2450/W4898/B9797 sizes remain unchanged. A is IDs only; load_public_b supplies projected B labels and public context, with no A/VALID/TEST evaluator. This remains a cold-state support test, not warmed/H16 support.
* **Original-phi recommit:** utility episode computes theta+ then calls inherited response on its ORIGINAL self.phis (`utility_control.py:202–212`). Qualifier independently constructs a new response at `nt` with original caller `phis` (`qualify.py:433–448`), compares every committed private coordinate plus complete utility/Q/finite/CE/pair diagnostics, and checks inactive partial/state zeros. Neither virtual adapted nor probe rows are recommitted.
* **Diagnostics:** utility allocation costs retain their own kind; paid finite softplus-margin response and CE diagnostics remain distinct. Positivity, original row3e-6/column2e-3 feasibility and FP32 atol2e-6/rtol2e-5 remain. All returned states/query and fresh private partials must be finite. All inactive local-head state/credit coordinates checked by the caller remain exactly unchanged/zero.
* **Restoration:** success/failure finally verifies native parameter values/identities, modes/attributes/global stage, inputs and RNG through unchanged base snapshots; caller theta/phi/role/pair tensor values/object/storage pointers/requires-grad flags are checked separately. Port admits no buffers. Reversible backend/intra-op/env/NumPy/Torch CPU+visible-CUDA/Python RNG/path/argv/prepared bindings/signal state is restored, with errors retained separately. Ordinary-grad observer has its own finally. One-time interop initialization is explicitly not restored and ends with the fresh child. Out-of-place constructed candidates are discarded; no model/checkpoint commit exists. Snapshot assertions certify their declared fields, not every possible unregistered Python alias.

## Independent work-count derivation

| Construction | Native forwards | Native reverse constructions | Ordinary grad APIs incl. small maps/query |
|---|---:|---:|---:|
| Complete utility episode |40|24own/main+8nativeVJP+8margin-phi+8dummy-reverse+4weighted-margin=52|63|
| Fresh theta+ / original-phi response |12|8own/main+4margin-phi+4dummy-reverse=16|16|
| Candidate fixed-member0 utility credit |1|1own+1weighted-margin+1nativeVJP=3|3|
| Isolated product-rule support |1|2private partials+3VJPs=5|5|
| **Total** |**54**|**76**|**87**|

The episode's11 additional APIs are10 cost-map VJPs and1 small query VJP. The full schedule has40 pair primal maps/320 solver iterations and10 map VJPs. Independent forward/API observers increment before attempts, compare exact internal count dictionaries, and restore ordinary autograd.grad. Support manually records each attempted API before invocation. Failed/setup/snapshot/restoration/serialization/supervisor work remains additional paid work. These are construction counts, not FLOP/time/memory equivalence, cheaper-cost evidence or the original pilot5740 bill.

## Disposition

The source-supported formula, original-phi commit and proposed work accounting are correct at this review scope. Shared-branch nontriviality is a concrete blocker to the required native support certificate. Prospective wrapper resource/terminal closure must distinguish numerical-worker measurements from whole child lifetime. Preserve this V1 packet and findings; audit a separately pinned successor before any distinct engineering invocation. Current allocation fitting and the six scientific arms remain unchanged. No native numerical, resource, predictive or novelty PASS is claimed.
