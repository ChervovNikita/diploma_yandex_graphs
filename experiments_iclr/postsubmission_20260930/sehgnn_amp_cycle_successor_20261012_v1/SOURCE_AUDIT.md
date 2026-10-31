# Scope of the AMP successor

The successor is composed against root controller v13. Its three controller
files have exact predecessor pins in `AMP_CONTROLLER_SCOPE.json` and a complete
unified diff in `AMP_CONTROLLER.patch`.

Existing engine functions changed only `_validate_spec` and `run_trajectory`;
new `_training_counts` performs the narrow native AMP accounting check. The
runner changes only `execute`, adding the typed policy when the SeHGNN config
explicitly opts in. Compiler functions changed only `sehgnn_semantics` and
`typed_counts`. Alias-preserving `_clone_tree`, NSD terminal-state lifecycle,
curvature trial laws, inherited-source custody and endpoint rules are retained
byte-for-byte outside these functions. Root can compose its later Single8 or
native-stopping amendments on its own controller copy.

SeHGNN adapter changes are request/commit/skip observation wrappers that always
delegate and restore the original callable. BASIC moves the finite-gradient
check inside GradScaler's observed optimizer dispatch. Its prior pre-dispatch
check wrongly rejected legal native overflow skips; the restored native path is
an explicit correction. JOINT remains FP32 with exact direct owner calls.

The packet compiler adds explicit native AMP law fields and truthful bounded
map counts. Complete cycles and the scientific roster are unchanged. All source,
provider and new-data cache files are streamed and rehashed before compilation.
The current controller actually loads the resulting disabled packet and confirms
references, dependency coverage and typed totals. That controller metadata
ingestion does not reopen arrays or perform training.

All old packets and engineering proofs remain preserved. Old proofs are cited
only for their actual full-width CPU and source algebra/state scope. The new
root-owned synthetic CUDA fixture establishes the legal AMP skip path and has
zero scientific updates. This is not scientific performance evidence, an ETA,
an original-paper replication, or resource admission.
