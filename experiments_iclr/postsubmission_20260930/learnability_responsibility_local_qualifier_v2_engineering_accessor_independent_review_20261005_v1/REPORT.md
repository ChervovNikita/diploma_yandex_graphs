# Independent local qualifier and engineering accessor delta review

5 October 2026. **Static delta review PASS; no concrete execution blocker found.** Neither numerical qualification nor scientific release follows from this review. No source edits/imports/runs, SSH, data/mask access or fits by this reviewer.

## Qualifier V2

Worker SHA256 `9247f4d3e405c58bd128ecc5d91ae95d661fc98d2fe0d5a4d409240708409562`; manifest `84737ced1f6056744f70900af39fb466a4e92418d30cf876fa8c663145c06be0`. The complete source bytes equal reviewed V1 with exactly one replacement: float64 FD scales(1e-3,3e-4,1e-4) become(1e-6,1e-7). The actual diff matches the sealed patch. Every other source byte, including both tolerance formulas, float32 scales, fixture/state/direction/seeds/architecture, guards, runtime pins/deadlines, resource/failure receipts and later checks, is identical. All manifest entries and external source/engineering-receipt binding hashes match.

The local rationale is bounded by the already observed two-sided agreement at the fixed fine scales. V2 retains the stopped-Q/fixed-Q, unused-head, complete public episode, original-phi recompute/commit comparison and state restoration checks that the coarse failure prevented from completing. It requires a fresh actual run and preserves the original failed source/result. It claims no full-graph FD result; full float32 FD remains optional and unchanged, and full mode still requires a passed synthetic receipt plus an explicitly reviewed enabled accessor/projection.

**Documentation caveat:** PLAN's phrase “failed at a nonsmooth native state” is stronger than the evidence. Native activations are nonsmooth, and finite perturbations crossed observed branches, but a kink at the base state and the causal contribution of particular switches were not established. Use root's clarified interpretation: the fine FD evidence concerns only the tested local direction. This wording warrants a qualified report, not a numerical or method change. No tolerance relaxation is present or recommended.

## Engineering accessor

Accessor SHA256 `9360e69753b362eb66ac89f133f00addaf15ab0e8c56314c75d7dd5e96fecb85`; root manifest `69262fafa1a9d0f439e696f0e94d4c40b5489461417e5facfd573c03f7677f33`. Its entire source equals independently reviewed V3 with exactly `SOURCE_RELEASED = False` changed to `True`. All root manifest files match their recorded sizes/hashes; this root packet is bound by its supplied manifest pin.

The release enables only custody projection and public W/B readers. Exact native CPU graph preprocessing/helper pins/logical identities, hash-only W/S/R/A roles and label separation are unchanged. The compact TRAIN custodian necessarily decodes A before separating it, with no predictive metrics; readers supply no A labels. The enabled source must create and read the same projection because its manifest checks the custodian's own source hash.

Six-arm warming/fitting, the warm response helper and held-A evaluator remain disabled in their unchanged packets. The native operator/port files remain unchanged; the reviewed qualifier's temporary process-local port gate is confined to its engineering episode. This accessor release grants no warmup, arm execution, fitting, A scoring, selection or acceptance claim.

Proceed only under root's existing explicit engineering authorization and exact source pins. A later numerical result must retain the original coarse failure and report the tested local/synthetic limits; full-native context/resources remain a separate gate. Exact identities and closure evidence are in `PROVENANCE.json`.
