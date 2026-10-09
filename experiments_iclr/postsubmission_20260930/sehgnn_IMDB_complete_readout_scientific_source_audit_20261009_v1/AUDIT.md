# Independent complete-readout scientific source audit

Status: **not admitted pending minimal source amendments**. This is a source review; no empirical verdict was made.

Exact readout entry: `d63b4c5ec64cf8948eb49b5495a059229252074ad353de4165ebd99136efc279`; thin analysis: `07be5b87f903a2a702c10a57ffe9865a0887865280ef4454326750f9c5e8d61b`; seal: `c9edbd9d10dfa7ddd456b8a0f9cdd5ed40828d77c8ed22ef9a47069cf658d35d`; root freeze: `0e0583e75b92bba8f1e7355c4188febe6d03f8bca4daca4f56671bd4a16183f1`. All reviewed identities are in INPUT_BINDINGS.json.

## Concrete blockers

**S1 — Native aggregate schema does not match thin analysis.** A valid complete source cohort reaches the native contrast then raises KeyError; no complete readout or frozen decision can be produced.

Evidence: sealed native engine.py:245-246 writes BCE/native_logit_micro_F1/native_logit_macro_F1/served_micro_F1/served_macro_F1; new RUN.py:136 stores fresh_selected.fresh_scores.VALID unchanged; new analysis.py:81-82,114,155-156 accesses micro_F1/macro_F1.

Minimal resolution: Consume served_micro_F1/served_macro_F1 or explicitly alias those saved values in the thin reader while retaining all original native fields/five results. The prospective strict probability>.5 rule chooses served values. No logits, new metric formula, gate or source fit change is needed.

**S2 — Entry status and inclusive costs are admitted too late and not validated.** Complete cell/family files can be reported with unrelated or failed entry reports and their costs, losing a declared failure and misattributing inclusive cost scope. The source does not meet the requested entry/cost binding before numerical aggregation.

Evidence: new RUN.py:159-172 computes numerical comparisons before binding/read of shared_entry_report/reference_entry_report at173-175; new RUN.py does not require those reports status=complete, complete=true or family_output_directory=the selected family; original shared entry RUN.py:213-218 and reference entry RUN.py:190-195 can flip complete=false after complete family output when inclusive root resource caps are exceeded.

Minimal resolution: Bind and require the original entry status/identity/counts and selected family paths before comparison calls; tie each entry report to its owning original release and actual custody row. Preserve failed inventories and costs and reject quality aggregation for declared failed/incomplete owners. No new scientific gate is needed.

**S3 — The origin admission invents a transfer for a resident-files readout.** The requested resident original-file execution cannot truthfully attest a transfer that never happens. Setting the current flag true would misstate custody; downloading raw artifacts is outside authorized scope.

Evidence: new RUN.py:98 requires selected_checkpoint_digests_verified_before_transfer=true; ORIGIN_INVENTORY_TEMPLATE_DISABLED.json uses that flag and local file mapping; README.md says original checkpoints are verified before transfer.

Minimal resolution: Explicitly amend the thin entry/template/README contract to exact original-location checkpoint digest verification before readout, resident original saved-file inventory and output-only transfer. Keep checkpoint/logit/data files resident; transfer only small analysis outputs. The reader paths are already portable.

## Scientific source result

The joint frozen quality criteria, separate native micro/member rules, all five controls, both genuine references, all members/labels/BCE/U-D and repair/harm coverage, and honest three-pair uncertainty follow the root freeze. Existing analysis formulas are reused. No retrospective native per-label criterion or extra repetition is introduced. Default release/custody/origin templates remain inactive.

## Acceptance limits

- This source audit establishes no empirical quality, actual fit completion, custody closure or efficiency verdict; no current score, dataset/label/logit/checkpoint/array/actual receipt was opened.
- custody() requires root-authorized bound launch/terminal/actual-observation evidence, exact saved boot/PID/birth/group/session handles, clean exit/reaping, saved handle/group absence and expected CUDA UUID residency absence. It trusts root evidence/booleans; it does not inspect evidence contents or cross-link cohort launch/terminal evidence to chosen family/entry identities. Admission requires root to verify those exact ownership links, and S2 calls for explicit linking in source.
- Source-restoration all-absent/mixed BCE values and per-query masks remain in the exact original diagnostic files; the CSV exports full-input pool/member BCE and U/D. Those retained original files are required if inspecting the detailed source-risk basis.
- Native costs copied to the readout retain wall/CPU/RSS but omit native per-fit/cohort CUDA peaks; original native RESULT/COHORT_REPORT identities remain bound. No cost/efficiency claim should be made from the reduced native cost fields alone.
- Positive U, own-only gains and nonsignificance cannot identify source specificity or equivalence. Matched untied same-J and independently frozen unused-outcome confirmation remain required; this pilot can only guide practical advancement.

Only local source/protocol/template/hash metadata was read. No server, current outcomes, actual receipts, datasets, labels, checkpoints, logits or arrays were opened; no compute, live probe, numerical/parity test, sudo or PDF compilation was run. No source or score was modified.
