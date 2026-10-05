# Independent bounded source review: V3 projection repair and affected qualifier

5 October 2026. **PASS_SOURCE_REVIEW** for sealed source V3 and its sealed affected-only qualifier. No concrete source defect was found in this bounded delta. This pass certifies the inspected source correspondence and scope only; root has not supplied a numerical qualification result to this auditor. It grants no development execution authority.

## Byte custody and delta

V3 manifest `14d2a075bf48ef54e045f580aaf05822e2f88aceb49248c6d57d483f0e46e4ac`, seal `c4b45cf05e3ef2b22de003cc0df57695d56564badc1b0f2ab5a9872451984298` and numerical source `8bbf79221b7a397a6a49c899d0d67bb07c7a13c0e91c9f75a7f47f120677ec4d` match. All12 source payload hashes/lengths match. Qualifier manifest `ca85dbdf56ee4a248f1ceb906f3c02343f145409041ecf1ee044a9e4d5a30cb8`, seal `eb1781c16e65b2b2ccee7e54a3adfd4fb4a668dd5432c88130797eff13556780` and script `6a700afb54782316e933b2997dc2d54ce14de201d3a8b4f8a30d9a4868dff2d2` match, with all four payload entries checked. Relevant source ASTs parse without import.

The actual four-module unified diff equals the bound `SOURCE_V2_TO_V3.patch`, SHA256 `97a309ecb630c7d868e084bf3542c78566f16d132e489bfd3314545b70846e52`. Deleting only the added `probability_hull_qp` and restoring the prior `projection` exactly reconstructs every V2 numerical source byte, including imports/constants/counters/nonfunction text. All24 other numerical functions/classes are unchanged. `custody.py`, `study.py` and `run_development.py` are byte-identical. The operative protocol and correction remain unchanged; the new numerical amendment explicitly records the arithmetic deviation under the existing repair allowance. Thus the completed V1/V2 audit carries forward for labels/folds/custody, native scoring, graphs/H/C&S, ridge moments, heads/optimizer selection, gates,126 fits and sparse accounting. Unrelated full-source checks and numerical tests were not repeated.

## Solver correspondence

| Requirement | Source correspondence |
|---|---|
| Original convex problem and all faces | `numerical.py:233–343` validates finite probability rows/posterior, enumerates all15 nonempty free faces in size/lex order, fixes inactive weights at0.0125 and starts free weights at their sum-one uniform center. |
| Helmert direct least squares | `273–289` uses orthonormal zero-sum Helmert columns; first-free-member centering removes the common mode. Design has shapeN×5×(k−1); SVD U^T-target multiplication and V multiplication have correct indices. k=1 is a direct vertex. The exact-arithmetic equivalence and minimum-weight-norm proof are preserved from the sealed arithmetic review. |
| Explicit numerical rank | `284–285` retains sigma strictly greater than64eps64 max(1,sigma_max); reported cutoff/rank counts describe every candidate face. This is effective numerical rank, not exact-rank recovery or a ridge. Amendment includes the conservative local risk/tie caveat. |
| Stable objective and frozen ties | `260–265,296–309` reproduces the original node scale from PP^T/Pposterior without solving those Grams, compares direct residual risk/scale, then squared weight norm, using unchanged1e−14 tolerances and first face for remaining ties. The stable risk differs from the former quadratic only by a row-constant in exact arithmetic. |
| Feasibility and KKT | Candidate floor/sum/free/dual thresholds are retained. Original gradient is computed as P(P^T w−posterior). `311–335` applies the same floor cleanup/mass renormalization, then independently checks raw and scaled free stationarity/inactive dual and primal errors at1e−10. Numerical rank checks do not substitute for these original conditions. |
| Uniform ID9 wiring | `371–374` passes the same outer-context posterior to the new solver and leaves probability/log pooling unchanged. Every existing bank ID9 call uses this common function. The old generic simplex/moment solver remains exact, including ridge moments. New QP bookkeeping preserves one call and one solution per row. |

The actual witness supports this repair's focus: source-trace node709 is a real failed first-batch row, while the single-row[1,2] recomputation rejects an otherwise permitted candidate for sum error1.9311e−12. That face retains full KKT rank; full-face rank loss is separate. The amendment preserves this distinction and the recomputation limit. No exact-cutoff equivalence or predictive-effect claim is made.

## Qualifier source and reference audit

The373-line qualifier is authenticated source plus the root-bound19026-byte witness only. It validates exact V2/V3 custody and reconstructs all unchanged source before importing the qualified numerical module, sets one-thread CPU/CUDA-disabled state, and never calls custody.prepare, a head/calibrator, optimizer, metric, sparse propagation, model/checkpoint loader, original scientific payload or label reader.

Its five affected checks are source-correct and scoped to the change:

1. The actual witness is compared with an independent80-digit Decimal analytic free[1,2] constrained line optimum using the represented FP64 inputs; the reference verifies positive floor and global inactive KKT in Decimal. It also checks the actual `projection`/stable pool wiring.
2. Known identical/duplicate minimum-norm, active boundary, interior and near-agreement/cutoff cases check the declared numerical rank behavior. Dyadic perturbations straddle the fixed cutoff; no exact-zero-rank or universal sub-tie perturbation claim follows.
3. A nonzero-residual weak direction has an independent analytic boundary reference and checks the disclosed local raw/scaled risk bounds.
4. Eight fixed-seed generic cases use independent direct-risk SLSQP and original constraints, with weight/risk tolerances recorded.
5. A2051-row batch crosses the native2048-row boundary and compares witness/known-case solutions.

Failure handling preserves each affected test and final report; no study retry occurs. A pass requires all five checks, no H/sparse/CUDA work, reauthenticated source/predecessor and unchanged witness. Prior passed V2 qualification must be bound separately by root; source identity supports reusing it and no150-update head/sparse test is repeated. The references and source assertions are not claimed to have passed numerically in this audit.

## Scope and handoff

Read the exact V3 source delta, numerical amendment/proof/bindings, qualifier in full, relevant safe metadata and the previously root-authorized tiny witness assessment. Performed stdlib JSON/AST/text/hash comparison only. No inspected-module import, numerical solve/test, fit, score, original scientific payload access, SSH or source/protocol/manuscript edit occurred. Root can bind this review before its already planned affected-only qualification; development admission remains separate. The error-analyzer V1 review is a different sealed result, and its V2 repair is assigned to another independent auditor.
