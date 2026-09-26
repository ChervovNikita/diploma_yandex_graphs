# All-layer scoring-only field-name amendment (26 September 2026)

The 72-cell all-layer training queue completed with 72 result files, 72 validation prediction files, no in-progress cells, and launcher exit status 0. The independent validation-only audit then replayed all 72 cells and wrote lock SHA-256 `f9095c01dc56f3e0e1fa9d37cdee9ff96d9821ca5c7d7e2dfa1f7a6235ddf399`. The separately frozen 36-cell same-runtime TIED validation lock has SHA-256 `fb860e2eb9c5c107c11c6c7d71547b79c4e690a95b74da460838dfa93d27be26`. Both locks and all original training/validation source files remain unchanged.

The first all-layer scoring invocation failed at the lock precondition, before `load_graph(..., include_test=True)` and before any test score was written. The scorer and its score auditor expected `control_lock["primary_selection_lock_sha256"]`; the frozen TIED36 lock records the same original 432-cell lock SHA under `control_lock["original_validation_lock_sha256"]`. At failure, zero all-layer `score.json` files existed and `FINAL_SCORE_AUDIT.json` did not exist.

The two V2 files in this directory are byte-for-byte copies of their frozen predecessors except for that single field-name substitution in each file:

- `all_layer_factor_study_score_v2.py` from `all_layer_factor_study.py`: SHA-256 `bfaa23a2e69037e08a8838a1089194d050d3d2091ea3a28f8137ce3ff8ab128c` versus original `872a94141972ce2707fa23338ffdd0567741bbc5a50309e4d5668932f7c31a9a`.
- `verify_all_layer_factor_score_v2.py` from `verify_all_layer_factor.py`: SHA-256 `2e5d6fd473eed8bdedce8dc6044bf6c58bf9552cbc9f2b2b61dc6e3206c4d4a4` versus original `71de99c84e66a53af2ff88305e812c436ca32ff681008fd97c26e61b4e681e01`.

The change makes the completeness gate compare the actual frozen TIED36 field with the exact original 432-cell lock SHA. It does not alter any model, training, checkpoint selection, candidate selection, test allowlist, metric, or independent replay operation. The V2 scorer still calls `check_freeze()` against the original frozen source list. The V2 auditor imports the original frozen model source. All test scores and their audit must be produced by these V2 files and retained with this note and the failed invocation record.
