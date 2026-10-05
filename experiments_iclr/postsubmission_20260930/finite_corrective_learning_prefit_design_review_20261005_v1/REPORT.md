# Pre-fit design for finite corrective learning

5 October 2026. **Recommendation: prospectively separate warm W, support S and outer R.** This is a design recommendation for a new sealed protocol, not an amendment to sealed V1, a fit authorization, a numerical result or a novelty claim.

## What the two designs test

| Design | Question it can answer | Limitation |
| --- | --- | --- |
| V1: 400 updates on all B, then fixed S/R episodes, with one warm-state response check | Can the prescribed private own-CE step produce relative margin responses and nonuniform responsibilities after all B labels have already trained the model? | S and R were both supervised during acquisition. This is further correction on previously seen labels; the check cannot create a test of newly introduced labels. |
| New design: warm on W=1/2 of B; S=1/4 and R=1/4 excluded from acquisition losses | Does the finite private response to newly introduced S supervision support corrective learning and shared-core credit from initially unseen R supervision? | Only the initial state has this label-exposure boundary. Repeated episodes expose S and R; R is training supervision, not untouched evaluation. |

The reported near-zero prior FIT errors motivate a saturation concern but do not prove that continuous margin responses are negligible. Those earlier fits are not the fresh, fixed 400-update state. With G0 epsilon=.001, the actual response scale must be recorded. No FIT payload was opened for this review.

## Explicit one-time warm-state check

For option 1, record the **already prescribed first live episode's preliminary probe and Q at the immutable update-400 state**, before its core update. Reuse that computation; add no diagnostic episode, checkpoint choice or alternate warmup. A remains closed. Live and stop_q have the same forward response at this common state; later trajectories can diverge.

For every supported class pair, retain both label counts, centered raw response RMS r, scale s=sqrt(.001^2+r^2), .001/s, normalized cost RMS r/s, Q relative RMS and maximum deviation from 1/4, entropy, minimum Q and marginal residuals. Retain each member's probe own-CE change. A common CE improvement can coexist with zero centered response because allocation uses relative member learning.

The fixed analytic boundary r<.001 means epsilon contributes more than half of s^2; r/s then falls below 1/sqrt(2). Report this as scale information, not a learned pass threshold. Exactly zero centered costs give uniform Q. Positive raw contrast alone does not establish useful assignment movement: balance, entropy and the fixed eight steps can suppress it. Conversely, uniform Q values alone do not certify that the outer derivative through Q is zero. Read scale, actual assignments and the live/stop_q contrast together.

This check can expose a vacuous warm-state assignment; it neither establishes transfer utility nor proves that all later H16 responses will remain vacuous. Do not tune epsilon, steps, coefficients, warm state or horizon to rescue the result. Use the same initial-state diagnostic in the proposed separated design.

## Recommended prospective roles

Keep V1's label-independent A order and boundary. Order B once by `(sha256(UTF8(amazon-response-G0|split=0|seed=17|WSR|<node_id>)), node_id)`. Assign the first floor(|B|/2) IDs to W, the next floor(|B|/4) to S and the remainder to R. Source-declared |B|=9797 implies **W=4898, S=2449, R=2450**; these are prospective counts, not derived data. Do not stratify, reseed or rebalance after inspecting labels. A missing S class is a retained qualification failure.

Use W labels only for the same local200/global200 acquisition. Freeze its last state once for all six V1 arms, including live and stop_q. S and R labels must be absent from acquisition loaders, losses, selectors and learned preprocessing. Full public graph/features remain transductive, so their nodes are not graph-unseen. S first supplies private/probe supervision after acquisition; R first supplies the outer own/pool objective. Subsequent episodes repeatedly use both fixed sets, and R labels update theta. A is excluded from every W/S/R loss and selector and opens only after all six immutable H16 endpoints exist under the retained scoring rules.

Keep the native model, complete graph forwards, G0, serving pool, six controls and H16. Preserve the declared charges: 400 common updates/1600 member forwards, 96 continuation episodes/3456 callback forwards/1536 private first-derivative constructions/192 assignment maps, and seven endpoint evaluations/28 member forwards. Smaller S changes assignment storage and arithmetic, not full-node model context or these callback counts; report actual time/memory and make no equal-wall-time or cheaper-full-graph claim.

This design directly addresses initial label saturation without guaranteeing nonvacuity or useful learning. Its lower warm-label coverage also changes initial competence, so it is a prospective representative test rather than a causal comparison to V1. If the frozen gate later passes, the capable same-forward all-branch direct-pool or ordinary own+pool baseline remains essential. Exact same-graph, same-operator relabeling is an equality check, not a baseline to outperform.

## Review boundary

Read only sealed V1 protocol/metadata, the operator's saved mathematical contract and relevant disabled-source normalization/diagnostic sections. No new literature, data, outcomes, checkpoints, imports of the operator, execution, compilation, SSH or fits. No predecessor files were edited. Native safety, higher-order feasibility and predictive qualification remain separate pending requirements.
