# Parent handoff

Prepared in this new directory only. The source-preparation packet is reviewable
and runnable after admission; it is **not execution-admitted or runtime-qualified**.
The scripts import scientific packages only after argument parsing, and no such
imports, data accesses or fits were performed in preparation.

## Parent-owned next steps

1. Review/freeze `PROFILE.json`, the explicit adaptations in `README.md`, and
   `history_features.sql`. The prior feasibility packet's extension cohort is
   unadopted and is not carried into this packet.
2. Admit the already-present native database/task files and native local GloVe
   asset. Run `custodian_export.py` outside the worker filesystem. Check the
   exported file inventory; retain source/private custody receipts separately.
3. Resolve and admit the full candidate dependency lock and matching CUDA
   pyg-lib wheel. `requirements-candidate.txt` is not a tested installation.
   Critical installed source drift causes an explicit failure.
4. Provide real filesystem/network confinement exposing only public inputs,
   the admitted local asset, source packet and empty output path. Offline loading
   flags and relative input checks are not a filesystem sandbox.
5. Execute Stage 1 and inspect data, sampler, forward/backward, chronological-fit,
   row-coverage, restored-checkpoint, SQL-cardinality, dependency and asset QA.
   Record failures and measured cost. No rescue configurations or extra HPT.
6. Freeze the validation-only competence result. Heldout admission and subsequent
   test inference require a separate stage; no heldout runner is included here.

## Scope/accounting

- Three native seeds: 42/43/44; ten epochs each; 150 updates per fit, 450 total.
- Training median and fixed prior-year driver mean control.
- Raw and historical native GBDT controls: ten trials and one native refit each.
- Native trial domains and booster default seed zero retained; Optuna seeded at
  42 and all trial parameters/outcomes/durations captured in receipts and SQLite.
- Chronological Frame fitting through 2005-01-01; full stable-ID public graph
  through 2010-01-01; untimestamped metadata retained as benchmark assumption.
- Test file exported by key projection only; worker produces zero test predictions
  and zero test metrics. Source cohort is preserved exactly.
- Default members remains four; initializer and ensemble decisions unadopted.

Preparation QA is in `evidence/PREPARATION_CHECKS.json`; packet hashes are in
`MANIFEST.json`. Numeric and SQL execution are unverified. No paper claim or
measured timing/memory is established.
