# Independent targeted v4 source recheck

Target: `graph_conditional_response_native_source_preparation_20261003_v4`.
Manifest SHA256: `cdece1f2a5e317de54d509d6b961ca41235cab880c3becb959b792aa75e03d72`.

## Decision

RC-01 is repaired in the requested runtime-gate/family and transaction paths at source level, assuming receipt storage works. RC-02 remains an explicit runtime-certificate scope limit. No numerical admission is provided by this review, and no computation/algebra blocker was identified in the changed source.

The new external receipt uses exclusive initial creation and atomic replacement. Runtime-gate failures and family failures record elapsed time and the error, retain completed family records, propagate the failure, and stop without retries. The transaction wrapper returns or forwards both counts and copied `ledger.events`, including on failure; objective/backward and ordinary AdamW spans are recorded in `finally` blocks. Import/setup failures before receipt creation and storage failures remain outside this assurance. A killed process may retain only the last persisted running checkpoint, as the helper documents.

The runtime gate now requires selected AdamW class/unwrapped-step, optimizer-functional, `torch.func.functional_call`, and PyG scatter source files, and binds/requires the CPU default device. Its returned certificate and report still explicitly decline full linked/transitive dependency closure. Preserve that qualification in any later runtime admission.

## Minor receipt consistency issue

At `prepared_native_checks.py:189–191`, the persisted family `elapsed_CPU_seconds` is evaluated before `saved.family_end` persists the receipt, while the returned field with the same name is evaluated afterward. The returned interval therefore includes receipt I/O. This is not a numerical blocker. A future unsealed successor can compute one elapsed value and reuse it, or label the intervals separately. No sealed payload was edited.

## Read and verification scope

All 37 target payload hashes matched the independently hashed manifest. Complete new receipt helper and complete v3-to-v4 diffs for dispatcher, transaction checks and view/runtime checks were inspected; the report's runtime qualification was checked. Manifest comparisons confirmed the core/native implementations, masks, groups, roles, explicit scientific rules and mask alternatives are unchanged from v3. This recheck builds on the immutable v3 source review and does not recertify unrelated source families.

Only file reads, diffs and stdlib/hash metadata work were performed. No prepared check, old verifier, project import, numerical module, tensor, dataset, label, checkpoint, training, outcome or remote execution was performed. External runtime bindings were not independently recertified. Previous source packets and reviews were preserved.
