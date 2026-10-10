# Native SAGE: GNCL x within-route SupCon

**Prospective source only.** One compact2x2 learning comparison reuses the native `Family` constructor, routes, optimizer, selector, restoration, outputs and acquisition lifecycle. The separate GNCL-only proposal is preserved unchanged. No new framework, owner, launcher, reader, teacher, projector, grid or scientific execution is supplied. Root owns configuration, criteria, qualification, anchor admission and launch after the frozen distribution family closes.

## Exact learning contrast

| Shared-bank cell | Learning objective | Acquisition |
|---|---|---|
| Base | F, mean own-member TRAIN CE | Archived own-only shared bank; reuse pending root custody check |
| A | .5F+.5L | Fresh `shared4_GNCL` |
| B | F+.05SC | Fresh `shared4_own_SupCon` |
| A+B | .5F+.5L+.05SC | Fresh `shared4_GNCL_SupCon` |

L is CE of the **actual probability mean** served, evaluated by live log-softmax/member-logsumexp, without a clamp or detached responsibilities. A gives direct supervised credit to the served distribution. B gives each route additional same-class discrimination; it is not a cross-route diversity target. Either can affect both member quality and coverage, and their interaction is unknown.

For the one factual stochastic forward used by CE, use all580 TRAIN rows of each route's actual output-head input H. `NativeModelAdapter.native` and the shared wrapper already expose this live H. The code consumes their returned values; it installs no persistent hook and makes no second native pass. Singles use the original native graph object and boolean TRAIN-mask CE order. Each independently fitted I4 body has one route and its own full CE+.05SC, without a1/4 ensemble divisor.

For each route, normalize H with epsilon1e-12, compute cosine scores divided by tau=.2, exclude exact self from every denominator and positive set, and use every other same-label TRAIN node as a positive. The scalar is **SupCon Eq2's outside-log positive average**:

`SC = mean_(route,anchor) [logsumexp_(all nonself TRAIN candidates) score - mean_(nonself same-label positives) score]`.

At least two examples of each of the ten TRAIN classes are required before fitting; a failed support check is not repaired by changing the loss. Gram multiplication is batched within routes, never across them. There are no cross-route negatives or extra views. SC is differentiated through the live upstream shared/private representation parameters; the output head, including its factors, receives CE only because H is captured before that head. Coefficient.05, tau.2 and epsilon1e-12 are literal fixed choices.

SupCon2004.11362v5 Eq2 and BotSCL2306.07478v1 supply ancestry. This single-view joint-CE use of native H differs from their complete original pipelines. The saved `within_route_supervised_geometry_SAGE_assessment_20261010_v1/REPORT.md` states the existing adverse Polynormer/retrieval evidence and the bounded missing native-SAGE comparison. GNCL2011.02952v2 and learner-collusion2301.11323v1 supply own/pool ancestry. No novelty, benefit or competence guarantee follows.

## Fixed intended roster and reuse barrier

Seeds are7301/7403/7507. The six fresh arms are the three shared cells above plus `ordinary_M1_SupCon`, `factorized_M1_SupCon` and `genuine_factorized_I4_SupCon`: **18 fresh banks/27 native optimizer-body fits**. The equipped genuine I4 fits and selects all four complete factorized bodies independently, then probability-pools their own selected outputs.

The five archived own-only arms are ordinary M1/I4, factorized M1/I4 and shared4: **15 banks/33 selected-fit records**. Root must establish exact source, constructor/factor starts, config, graph/features/ordered roles, seed/RNG streams, native loss/update law, horizon/selector, selected-state custody and complete costs before comparative opening. This acquisition entry reads no anchor payload. Its completion explicitly says **fresh acquisition only**, with `full_comparative_family_complete=false`;18 new banks cannot substitute for the required15 archived banks. Invalid reuse requires a separately fixed successor decision, not selective replacement. The complete joined scope would be33 banks/60 fit records, with27 new fits charged separately from historical work.

The intended inherited recipe is SAGE depth2/width128/dropout.2, native AdamW lr.001/decay0, maximum1000updates/patience300, full11701-node native graph and TRAIN580/VALID5274. Root binds exact configuration and role bytes. Selection remains the first strict full VALID probability-mean accuracy maximum; ties retain the earliest state regardless of NLL. Model/optimizer/member streams restore together. VALID is encountered selection data; three paired optimizer seeds do not make it untouched confirmation.

Genuine independent I4 is a practical quality/cost reference with four bodies, unscaled own objectives, separate optimizers and own selectors. A jointly optimized untied bank would couple four bodies under J and use one coherent selected state. It is a different reference and is not added here. This study tests the known learning ingredients and their complete served interaction; it does not isolate a causal tying-by-learning effect.

## Outputs and paid work

Native probability serving, float64 reported log-softmax/logsumexp NLL from float32 logits, every member's accuracy/NLL, coverage and losses in pooling, raw selected/error archives, complete fresh repairs/harms and inclusive native acquisition/selection/checkpoint/serving costs are preserved. Trace fields distinguish own CE, actual pool CE, unweighted mean/routewise SC, weighted SC and total objective. No joint I4 TRAIN pool loss is invented.

Every update still has one native forward per route, one scalar backward and one AdamW step. SupCon adds normalization, cosine/Gram products, masks, logsumexp and their backward from the existing H. Source counters record actual update-scaled batched Gram calls, route Grams, normalized coordinates, cosine entries, anchor rows and Gram multiply-accumulates using the observed H width. Inclusive acquisition/GPU peaks charge these operations and tapes. A shared4 FP32 Gram alone has1,345,600 entries/5,382,400bytes before scores, masks and autograd storage; this is not a measured peak-memory claim.

At the1000-update ceiling:27000 optimizer/backward calls,54000 TRAIN and54000 VALID-selection native trajectories,54 selected trajectories,42000 SupCon route Grams and1,808,486,400,000 Gram multiply-accumulates at D128. Actual stopping, width and measured work remain authoritative. There is no extra native pass or serving SC. Root records concurrency and historical costs separately.

Root's complete reader should compare A, B and A+B with the admitted common base, retain the paired interaction, equipped capable references, mean/worst competence, acquired-and-served alternatives, common-rival repairs, full repairs/harms and raw NLL. An interaction, geometry change or stronger member alone cannot pass worse final quality. Numerical criteria and the reader are root decisions, not implemented here. Syntax was checked with stdlib AST only; no model/data/numerical import, raw/partial outcome, remote/TEST access, canonical edit or optional parity/proof packet occurred.
