# Independent branch-result interpretation

5 October 2026. Engineering interpretation only. Complete branch RESULT/TERMINAL metadata and the prior mismatch REPORT were read. No mask payload, scientific data, source edits, imports/runs, SSH or fits. **The new evidence supports local directional derivative consistency at the existing synthetic state; it does not turn the original coarse qualification failure into a pass.**

## Observations

The branch run exited0 with `COMPLETE_BRANCH_DIAGNOSIS_NOT_QUALIFICATION`. Its source identity matches the independently reviewed packet. The local RESULT is exactly the terminal's recorded 3,732,732-byte artifact, SHA256 `661ccc411307d485c39807472416c8e89416f9eae9b345d46e63648fe318fb9c`. Base objective and analytic projection match the prior diagnosis exactly; neither the state nor direction changed.

| epsilon | right FD | left FD | central discrepancy from AD | ordinary sign changes (+ / −) |
| --- | ---: | ---: | ---: | ---: |
| 1e-5 | .00387108861 | .18379846252 | .0899636569 | 2 / 7 |
| 1e-6 | .00387111498 | .00387112253 | 1.00364e-10 | 1 / 0 |
| 1e-7 | .00387111232 | .00387113452 | 4.76330e-9 | 0 / 0 |

AD projection is .00387111866. Both sides now agree at the two fine scales, unlike the prior larger-scale left differences. The modest increase in central error at1e-7 is compatible with finite-arithmetic cancellation; this run does not isolate that contribution.

Every point has20 correctly mapped callbacks,372 captured ordinary sites (124 each before/probe-adapted/query) and248 explicit transformed skips (124 each probe/main private-gradient). All captured counts and sign-hash schemas were audited across the complete JSON; there are no missing/extra site identities or captured zero/nonfinite entries. All ten changed-coordinate comparisons concern ReLU sites. The +1e-6 point changes one probe-adapted sign despite close FD agreement. At−1e-5, a before-update branch also changes, alongside probe/query branches.

Thus actual branch changes occur over the larger perturbations, and the ±1e-7 endpoints preserve every observed ordinary sign. Branch crossings are a supported explanation for scale sensitivity, but the metadata does not establish a specific causal site. A crossing is not sufficient for a material discrepancy, as +1e-6 demonstrates. Endpoint sign agreement also does not prove every intermediate point or skipped transformed operation stays in one region.

The earlier ordinary-autograd construction matched all shared AD coordinates to1.11e-16. Together, these receipts now give stronger evidence for the local chain at this state and direction. They do not prove global smoothness, all-direction correctness, an asymptotic derivative theorem or full-graph higher-order qualification. The displayed Q-chain norm .22318048 is reused from the earlier receipt, not newly measured here.

## Next engineering gate

Root's planned separate qualifier successor fixes float64 FD scales(1e-6,1e-7), retains the original tolerance, exact fixture/direction/seeds/architecture and all later checks. Review its exact source delta before execution. Preserve the original coarse FAIL receipt and its verdict; the successor should describe a local derivative check, with the branch limitations above, rather than a global smoothness claim.

Complete the previously unreached stopped-Q versus independently fixed-Q derivative, unused local-head tangent and forward-response checks. Then run the complete public native episode, independently recompute private state at theta+ from original phi, compare returned commit values, verify native/input/RNG and entry-gate restoration, and retain actual resources/failures. That public episode costs9M=36 full-member callbacks; its independent response adds4M=16, all charged as engineering work. None of these paths was qualified by the branch observation itself.

The branch run's16.90 seconds and4,935,950,336-byte peak RSS include its instrumentation on15 artificial nodes. They are not full-Amazon feasibility estimates. Real24492-node higher-order/sparse/context and resource qualification still requires the separately reviewed enabled V3 accessor/projection, exact graph/runtime identity and a bounded engineering caller. An accessor-only engineering release does not release the six-arm worker, fitting or A scoring.

Native/input/RNG and both hooks were reported restored. The14,504,734-byte mask artifact remains on the allocation with its recorded descriptor; it was not opened or copied by this reviewer. No predictive, novelty or paper-acceptance verdict follows.
