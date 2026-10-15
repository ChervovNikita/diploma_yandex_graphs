# Accepted exploratory reporting protocol v1

This successor transcribes the root-accepted choices for the **6,460-fit exploratory benchmark**. The single protocol is `PROTOCOL.json`; the initial seven-dataset exposure record is `EXPOSURE_LEDGER.json`. Final root artifact review is still pending. This package grants no campaign launch or TEST access.

Read the [finite design and method review](../benchmark_review_protocol_proposal_v1/REVIEW.md), followed by the [adopted exploratory reporting protocol](PROTOCOL.json). The review's earlier options are historical; this successor records the selected reporting rules.

## Reporting contract

- Keep all 27 fixed presets and all ten matched cohorts per preset. Preserve every declared outcome, missing/failure/cap status, selected epoch, trained epoch and actual cost.
- Report every prescribed paired signed difference, mean, sample SD, median and win/tie/loss/missing counts, with explicit scored/resolved/planned denominators. Available-pair summaries remain visibly incomplete.
- For the mean signed paired difference, use 10,000 paired cohort bootstrap resamples, seed **130102026**, with percentile **descriptive marginal 95% intervals**. Preserve whole paired cohort rows. No significance, superiority, simultaneous-coverage or unseen-dataset claim follows.
- **Withhold the complete-preset interval if any required pair is missing or unresolved**, including a scored matched trajectory with unresolved convergence at its cap. No survivor-only interval, imputation or automatic scientific retry.
- Keep source replication separate from matched extended training. Each predictor uses its own selected state and actual readout. I4 has four own selected checkpoints; its member 0 is exactly M1. A same-selected-bank fixed mean remains a diagnostic and cannot reselect a bank.
- Retain complete preset reporting; there is no cross-preset winner replacement. The ancestor 226-contrast sign-flip/Holm policy is not the expanded reporting family.

The counts remain **270 source fits + 1,080 native independent body fits + 5,110 shared fits = 6,460 fits**, with 270 I4 export tasks and 6,730 total queue tasks. The 540 source/matched scheduling groups are not 540 paired observations. Pilots are outside this count.

## Initial exposure record

| Dataset | Historical TEST-specific status | Current campaign |
| --- | --- | --- |
| CiteSeer | Unknown; historical development documented | Unlaunched |
| PubMed | Unknown; historical development documented | Unlaunched |
| Coauthor-CS | Unknown; scoped no-TEST branch evidence retained | Unlaunched |
| Coauthor-Physics | Verified historical SAGE TEST truth release | Unlaunched |
| ogbg-molbace | Unknown globally; scoped selected-VALID pilots/diagnostics documented | Unlaunched |
| ogbg-molhiv | Unknown; historical development documented | Unlaunched |
| ogbl-collab | Unknown; historical development documented | Unlaunched |

The Physics release does not establish exact overlap with the current PolyFormer split. Coauthor-CS custodian label decoding and branch no-scoring flags have distinct scopes. None of the seven datasets is certified pristine or unused. Missing historical records are not negative access evidence.

The MolBACE ledger pins the completed 24-fit ingredient and 3-fit BN reports, their exact nested result-authority collections, and three selected-VALID diagnostic records; each has scoped TEST=false evidence. The nine-fit BN/geometry/correction entry pins its frozen spec, root admission and launch record only. Its completion, current progress and outcomes were not inspected or inferred. It permits selected VALID only and does not grant full-campaign GO.

## Separate confirmation and remaining gates

The later proposed **120-fit confirmation is outside the 6,460 campaign**. Its exact candidate/scope, seeds, node split law when applicable, endpoint rules, budget and authorization must be frozen later. Its unresolved node split law **does not block exploratory TRAIN/VALID execution**. No automatic promotion or cross-preset nomination rule is adopted here.

Current runtime/dispatcher, state/readout qualification, measured resource/cost admission, applicable source-fidelity dispositions and explicit human full-campaign GO still apply. A future TEST release requires its own exact selected-state request, official role/topology receipts, complete VALID parity and human authorization.

## Integrity

`MANIFEST.json` pins this package and every cited local source. `VALIDATION.json` records the bounded metadata checks. The prior `benchmark_review_protocol_proposal_v1` is preserved unchanged. This work used local compact metadata only; it performed no training, monitoring, tensor access, remote operations or Git.
