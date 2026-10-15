# Factual context affinities

The two prospective methods train existing route preheads to match similarities
computed from permitted TRAIN context. Common gives every route the same target
mixture. Split assigns two routes to each context. Native task risk remains in
both. This tests information assignment; it does not reward arbitrary route
distance or assume that disagreement is useful.

The exact task targets are defined in
`../expanded_study_design_v1/TASK_CONTEXT_SPEC.md`. `core.py` implements the
bounded tensor criterion with explicit native auxiliary reductions, authenticated
object/relationship identity, detached factual targets and support reporting.
`check_core.py` checks the information contrast, source group scale, gradient
support and unsupported/zero-context cases. These are semantic checks only.

Native task extraction, target construction/cache, optimizer bridge,
checkpoint reconstruction and runtime/resource qualification remain pending.
No native forward, fit, dataset result or novelty claim follows from this source.
The giant campaign remains disabled pending preparation and the user's GO.

The ingredients have supervised metric-learning and graph-context ancestry.
This package is not a claimed reproduction of SupCon, MoCL, AMCL or GraphDIVE.
Sources and scoped prior conclusions remain in the expansion literature ledger.
