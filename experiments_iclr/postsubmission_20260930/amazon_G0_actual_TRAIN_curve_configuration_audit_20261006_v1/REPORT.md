# Amazon reference TRAIN curve and configuration audit

All ten scalar TRAIN traces passed source hash, schema, finite loss, contiguous update, and stage checks. The references strongly minimize their recorded TRAIN likelihood objectives. Their poor A accuracy and large A NLL therefore do not establish an inability to fit those supervised examples. This evidence weighs against a simple insufficient optimization explanation for the references; it does not identify a cause, prove overfitting, or rescue the failed G0 comparison.

## Actual training loss evidence

The means below use fixed windows from the complete recorded train mode traces. They are existing loss scalars with dropout, not new prediction scores or deterministic TRAIN evaluations.

| Reference | Mean CE updates 1 to 16 in S/R | Mean CE updates 385 to 400 | Mean CE updates 2285 to 2300 | Pool NLL updates 2285 to 2300 |
|---|---:|---:|---:|---:|
| Native member 0 and SINGLE | 1.508170 | 0.061035 | 0.012394 | Not logged |
| Native member 1 | 1.503590 | 0.059094 | 0.011908 | Not logged |
| Native member 2 | 1.518357 | 0.068534 | 0.013023 | Not logged |
| Native member 3 | 1.526479 | 0.059616 | 0.012889 | Not logged |
| Ordinary own | 1.413597 | 0.115873 | 0.006144 | 0.003723 |
| Ordinary own plus pool | 1.394452 | 0.117515 | 0.005874 | 0.002957 |

Native W acquisition also learns: its last 16 mean CE values are 0.196455, 0.194532, 0.197034, and 0.139192. The source switches from the local to global readout after update 200; all four traces record that exact switch and the expected temporary loss increase. S/R begins on a new disjoint supervised population. Initial S/R CE around 3.04 to 3.15 and its subsequent reduction are not a discontinuity in one fixed evaluation loss.

Every native member has 400 W and 2,300 S/R entries. Each ordinary trace has 2,300 entries, and its recorded optimized loss matches the specified own CE or half own CE plus half probability pool NLL. No truncation, missing update, nonfinite scalar, or incorrect loss combination was found. All 15,400 entries were parsed; no model was run.

Each native member supplies individual TRAIN likelihood evidence. Ordinary own CE averages over members, so it does not supply separate member accuracy or separate route losses. No TRAIN accuracy or confidence was logged. Low CE supports strong training fit; it is not a measurement of held member competence. The previous frozen comparison still reports only 41.98% candidate accuracy versus 46.30% ENS4, 45.37% SINGLE, and 45.32% own plus pool.

## Configuration audit

The actual native RUN and RESULT, common origin RUN, and both ordinary RUN and RESULT files match their sealed byte and hash bindings. Native, common, and ordinary runs share public graph, preprocessing edge, label role, and public B manifest identities. Ordinary runs bind the exact same common checkpoint and origin RUN; source and executed objective assignments match. No discrepancy was found among these checked metadata and source commitments.

Native metadata records independent seeds 17, 1026, 2035, and 3044, fresh W acquisition per member, W200 local plus W200 global updates, all W400 states frozen before S/R access, and SR2300 global updates. Completion accounting agrees for every member. Ordinary metadata records SR2300 from the common W400 state and the streamed exact implementation. Its prior normal qualification reports complete gradient, Adam, and RNG replay parity. No qualification or model computation was repeated here.

Source checks confirm Adam rate 0.001, zero weight decay, native dropout and architecture commitments, sorted S/R targets, supervised populations W4898 and S/R4899, and exclusion of W and A from S/R losses. These are intentional recipe choices, not discovered configuration mistakes. Native parameter clocks are stage dependent: at the final state the dormant local predictor retains 200 steps, global modules have 2,500, and continuously active modules have 2,700. A local predictor clock of 200 must not be mistaken for an incomplete run. The clock rule is source verified; checkpoint tensors were not read.

Input identity agreement does not independently validate every feature and label's semantic correctness. The intended role restriction also means these references are not an unrestricted all TRAIN benchmark recipe. Training hardware and optional extension differences remain as disclosed, and common serving does not remove them. G0 H16 and native or ordinary SR2300 are different optimization and supervision histories.

## Actionable conclusions

1. Preserve the G0 gate failure and all prior A outcomes. High reference A NLL is not a reason to discard their higher accuracy or declare candidate superiority.
2. Treat the references as capable of fitting their specified training likelihood losses. These traces do not support blaming their poor A results on failed backpropagation, incomplete declared horizons, or persistently high terminal TRAIN CE.
3. The remaining issue is transfer from the intended supervised roles to A, with substantial confidence errors. This is a description of the observed discrepancy, not a diagnosed training cause. Low TRAIN CE and poor A outcomes are consistent with failed generalization, but CE decrease alone cannot establish when or why it arose.
4. Any further configuration investigation should first check whether the deliberate W then S/R role policy, preprocessing, native constructor, dropout, and regularization match the intended scientific comparator. The present audit found consistency with the admitted recipe. It does not authorize checkpoint selection, retuning, additional labels, or a new fit.

No required trace remains missing. Root fetched exact scalar text copies; this agent executed only the local scalar parser. No server call, checkpoint or prediction payload, assessment labels, new scores, or forwards were used. One seed configuration and split remain insufficient for confirmation, and this G0 result does not reject all GNNM. The distinct 77 ranking objective control is unchanged.
