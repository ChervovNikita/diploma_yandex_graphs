# Source-only recheck of native preparation v3

## Decision and immutable target

The sealed target is `graph_conditional_response_native_source_preparation_20261003_v3`, manifest SHA256 **ee06207b22ff3c073f70bcea60d21be88bb9dcf7c63e040d969fb74a4e8d1bd2**. I independently hashed that manifest and checked all 36 payload hashes using standard `jq` and `shasum`; all matched. I did not run the packet verifier, import project/numerical code, construct tensors or execute a prepared check.

**No definite computation/algebra source bug was found in the four new check modules, dispatcher or reset rejection.** The source now prepares the important coverage that the earlier v2 review requested. Numerical passage remains unverified. There is one concrete limitation in how later check evidence would be retained, and a narrower runtime-certificate scope that should remain explicit.

V3 was sealed during this recheck. The selected source bytes I read and subsequently hashed match its seal. No v3 bytes, v2 bytes or the original v1 review were changed. The original review's six findings remain its historical assessment of v2; this report is a separate targeted successor recheck.

## Source coverage now prepared

- **Materialized member oracle:** `prepared_native_reference_checks.py:26–88` constructs independent unwrapped native member modules and uses differentiable effective weights `S[:,None]*W*R[None,:]` and scaled shared bias `S*b`. It substitutes all native parameters, compares complete outputs and gradients through inputs/common/factor parameters, coherently permutes all factor rows, tests other-member isolation at every site and checks total/private counts. Native plain attention order-bias values remain independently initialized from the same spec; wrapped registration does not create a trainable oracle parameter.
- **Reset/replay:** `native_source/adapter.py:111–114` raises before the first wrapped map reset changes stored state. The native top-level, attention and FFN reset paths each reach such a wrapper first. The new reconstruction check compares exact snapshots after rejected reset calls, replays deliberately non-one factors and changed biases, and explicitly reestablishes Stage B permissions/mode/receipts. This covers those reset paths; it does not claim all nested PyTorch `reset_parameters` methods are blocked.
- **Irregular complete views:** `prepared_native_view_checks.py:83–146` uses reciprocal irregular nonloop records, source loops and an isolated node, removes declared paired units, independently reconstructs unit-loop/destination-degree normalized operators, compares every K+1 token and produces three CompleteViewBank entries. It checks view identities, cross-view feature mismatch and separately snapshots/restores mutable input tensors. These are synthetic producer fixtures; they do not validate actual-data cache hashes.
- **Native transactions:** `prepared_native_transaction_checks.py:33–252` differentiates the actual three-view objective, independently checks CE terms and gradient flow, initializes actual AdamW moments, uses actual competence/energy guards, witnesses full/forced fractional acceptance, exhaustive rejection and zero displacement, and tests deliberate primitive/buffer/input/optimizer/RNG impurity and a raised native forward. Expected fractional states use the full ordinary proposed optimizer image once. The failed callback's M4 ledger count is correctly described as declared attempt units, not four completed trajectories.
- **Adjoints:** `prepared_native_adjoint_checks.py:9–98` tests unequal leading member/target axes, a nonsymmetric helper operator with an explicit wrong-transpose comparison, K=0/K=2, complete original feature axes, coherent permutations and direct-X versus adjoint higher-order gradients at every intended private R/S. The nonsymmetric operator does not admit directed scientific graphs. The synthetic raw-logit score remains distinct from a source-qualified FoRDE objective.
- **Deferred execution:** The dispatcher checks explicit native authorization and the actual runtime receipt before calling numerical families. Its imported helper modules retain deferred numerical imports. This source inspection supplies no authorization.

The copied core response, guard, transaction and token interfaces were read as the dependencies of these families. No assertion that the predetermined synthetic factors, activity screen or guard-acceptance fixtures will pass numerically is made. A failure of an activity/feasibility screen must be recorded at its fixed conditions.

## Remaining conditions

### RC-01 — Preserve transaction events and failed-family receipts before relying on a complete check cost record

Affected source: `prepared_native_transaction_checks.py:252`; `core/guards.py:50–60`; `prepared_native_checks.py:168–178`.

The transaction family returns `ledger.snapshot()`, which contains aggregate counts and `recorded_work_events`, but not `ledger.events`. Measured proposal/objective/guard/transaction spans and per-event stage/status details are therefore lost when this local ledger leaves scope. The dispatcher retains a whole-family elapsed time for completed checks, which is useful and should not be described as a breakdown of those discarded spans.

If a family raises, line 171 propagates before appending its receipt. The local completed-family receipt list is also not returned. A raw traceback identifies the failing assertion, but the prepared entry point provides no structured failed-family duration or partial-run receipts. Unexpected failures therefore do not produce the complete retained accounting that the earlier review requested.

Before separately authorized execution that requires such accounting, use a source-bound runner or a later successor that persists completed and failed family records and exposes transaction events even on exception. Preserve exception propagation and stop-on-failure semantics. Alternatively, qualify the receipt explicitly as success-path aggregate counts and whole-family timing only. Do not edit the sealed v3 to address this limitation.

### RC-02 — Keep the runtime certificate at its actual declared-file scope

Affected source: `prepared_native_view_checks.py:45–58,68–70`.

The receipt validates every declared file before importing numerical packages, then requires four package entry files, five imported PyG function files, Torch `_C`, NumPy multiarray and SciPy sparsetools binaries, exact versions, deterministic flags and thread counts. This is stronger than checking only PyG's version string.

It is not a dependency-closure certificate. The mandatory list does not include the actual Torch AdamW/functional optimizer source, the PyG `scatter` helper imported by gcn_norm, or Torch's linked numerical libraries. Those files may be supplied additionally by root, but the gate does not require them. It also checks default dtype and CUDA initialization without explicitly binding the default device; the native attention forward constructs an implicit-device scalar.

The v3 report accurately says **declared compiled dependency files** and disclaims PyG2.3/runtime equivalence. Preserve that scope. If later admission requires exact executed optimizer/helper implementations or a guaranteed CPU-default process, bind those implementation files and selected backend libraries/settings in the actual runtime receipt and audit the executed objects. This is a certificate-scope condition, not evidence that the prepared normalization oracle is wrong or that a compromised runtime exists.

## Historical accounting clarification

The initial targeted search exposed `READ_SCOPES.json.source_only_verification_followup.AST_scope` saying “22 current packet Python files and 26 bound external Python files.” The full file explicitly labels this block as inherited v2 history. It is therefore **not a contradictory v3 count or a blocker**. Its embedded word “current” can be clarified in a later successor. The sealed v3 report and stored source-verification receipt describe 26 packet and 48 external ASTs/92 external bindings. I did not independently rerun those AST/import checks or verify all 92 external inputs.

## Actual read and execution scope

Complete source reads: the five `prepared_native_*checks.py` files, `native_source/adapter.py`, `native_source/native_polyformer.py`, `native_source/native_polyformer_outer.py`, `native_source/tokens.py`, and `core/{response,guards,transaction}.py`. Complete metadata reads: v3 `MANIFEST.json` and `READ_SCOPES.json`. Report content was searched and deliberately read at lines 213–315; PyG gcn_conv imports were read at lines 1–22. The previous v1 review was consulted as existing context; this recheck does not claim a fresh complete reread of that report or other inherited source.

The work used file reads, line locators, standard JSON extraction and hashing only. No project/verifier/prepared body, numerical import, training, fit, dataset/label/checkpoint/outcome file, remote service or mutation was used. All outputs are in this fresh sibling review directory. Neither literature conclusions nor engineering source preparation qualifies a pilot.

**Recommendation:** retain the v3 seal, adopt this targeted source recheck, and keep numerical/runtime/actual-data/recipe/control/resource gates pending. Address evidence retention through a separately bound runner or successor before making full failed-work cost claims.

