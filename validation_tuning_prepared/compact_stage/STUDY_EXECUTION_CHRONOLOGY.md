# Validation-grid execution chronology

This record was prepared during the run, before the complete 432-cell audit
or any tuning test scores. Times are UTC on 26 September 2026.

- Approximately 00:42: the first generated source/data freeze had SHA-256
  `fe7e142d19a5ca45283b326b978e8c7a1eb435abcf51c20873fe55d220622caa`.
  The preflight rejected it because its JSON arrays were compared directly
  with Python tuples. No study training had begun. The old freeze was
  preserved as `FROZEN_STUDY_ABORTED_PRETRAIN.json`; the one-line comparison
  repair and its reason are documented in `PRETRAIN_REPAIR.md`.
- Approximately 00:44: the repaired prospective freeze passed a fresh
  source/data/tensor check and A100 CPU/CUDA six-arm preflight. Final frozen
  study SHA-256:
  `6bbb0b7068141e7adb814413d9ce952bcacbc44ef013147f1e577bbff6fcda73`.
- Approximately 00:46:07: A100 GPU4 queue launched Cora then WikiCS. Launcher
  PID was 2277784. This launch preceded the second host's independent source
  review and GPU preflight. No tuning outcome was opened before that check.
- Approximately 00:49: GPU77 independent static source/test-firewall review
  and CUDA first-step preflight passed with the exact same frozen SHA-256.
  The A100 queue was already running at this point; this is an independent
  audit during training, not a pretraining independent sign-off on A100.
- 00:51:36: GPU77 GPU1 queue launched Actor then filtered Chameleon. Its
  launcher PID was 2131162. All four graphs use the same frozen source/data
  fingerprints, with each graph's six arms on one host.
- Around 01:01: a GPU77 Torch 2.7.1 checkpoint serialization canary was
  transferred to the A100 repository and loaded with Torch 2.1.2
  `weights_only=True`. Only archive compatibility and state tensor count were
  checked; no validation or test metrics were opened.
- Approximately 01:05: post-freeze transport, reporting, strict decision
  replay, and optional inference-profile tools were staged separately. Their
  hashes are in `POSTFREEZE_TOOLS_MANIFEST.json`. They do not change frozen
  training or validation-selection source.

The prospective design fixes a complete 432-cell validation-only audit and
global selection lock before *any* tuning test inference. This chronology is
part of the audit evidence; later completion, lock, score, and packaging times
must be appended as observed rather than inferred.

## Observed completion and post-lock transport

- 03:55:48 UTC: all 432 planned cell results were present, with no active
  `.inprogress` directory; the A100 training launcher had exited. The last
  logged completion was WikiCS/UNTIED candidate `(0.003, 0.01)`, seed 2.
- 03:56:18 UTC: independent validation-only `audit-and-lock` verified all 432
  traces, checkpoints, frozen identities, and selected epochs before any
  tuning test inference. The immutable global lock SHA-256 is
  `176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99`
  (288,631 bytes). The exact bytes were sent to the second host and checked
  there before its scoring began.
- After the lock, each host scored only its two assigned graphs at the
  validation-selected and predeclared default candidates. The 72 second-host
  score cells were imported at 04:00:34 UTC from archive SHA-256
  `ebb529f6b8ef2e86e9932ad358f89257bc8e79d6cd9d7fecb0219b5ef8c783be`.
  The importer checked the archive's lock and freeze bytes and the 216-cell
  validation subset audit against the global lock. Cora and WikiCS contributed
  36 and 33 locally scored cells; the difference reflects overlap between
  selected and default candidates for one WikiCS arm.
- 04:01:43 UTC: independent fresh checkpoint, logit, metric, and allowlist
  replay passed for all 141 selected/default test scores. Final audit SHA-256:
  `63da49d3d797fa7856717d4b21a530c74419bfbdfa92614a03b036aa9950bdde`.
  At 04:02:36 UTC, a supplemental exact-decision replay passed all 141 cells
  with zero validation or test class mismatches; audit SHA-256:
  `a3f0d70f592b13b8ef70881611652ed05362ad4a0afc8e6c6af49c8488d65596`.
- 04:03:31 UTC: compact hard decisions were exported from the audited float
  logits. At 04:04:19 UTC the selected/default report was written. At
  04:05:31 UTC the 24 selected seed-0 checkpoints were packaged separately
  for optional inference profiling.

Two **post-freeze transport/reporting** compatibility repairs were necessary
after scoring; neither changed the frozen training, candidate selection,
checkpoints, score records, or independent audit code. The original
`import_gpu77_scores.py` (SHA-256 `a9c37610f47fa0b14e699e6556450023015096740722c7e66037b07881f7b35d`)
rejected the second host's lock and freeze metadata before extraction. The
revised importer (SHA-256 `4b4b5777c7108f6b633b52e8746fc25aa1ad9da190dc1f480f13f7936bd7d44c`)
allows exactly those files plus the 216-cell subset audit and requires their
hash links before copying score artifacts. The original
`summarize_tuning.py` (SHA-256 `a6a379ba547297c4183131d38a7aeb096b55b0af73a2cd64495e3439522b424c`)
rejected float32 score/replay differences above `1e-8`; 48 of 141 scores
exceeded that reporting-only threshold, with maximum absolute difference
`5.960464477539063e-08`. The revised reporter (SHA-256
`aaf1a77c39257ba40ba7540886859a26bee767877a8275ed619234168d866abb`)
uses the independent score verifier's `1e-7` tolerance. The original
post-freeze tool manifest is retained as a record of pre-outcome source hashes.
