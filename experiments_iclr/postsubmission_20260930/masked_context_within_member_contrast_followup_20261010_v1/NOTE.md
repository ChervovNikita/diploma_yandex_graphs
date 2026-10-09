# Contrast within members and useful differences between members

10 October 2026. Method-development follow-up only. Existing sources, experiments and the sealed CORE proposal remain unchanged. No new primary scope, model, tensor, server operation or fit is added.

## A constructive interpretation

Keep the backbone shared **and trainable**. Each private BE route supplies its own task-facing representation; its own classification loss and contrastive loss update both the shared weights and that route's factors through the ordinary scalar objective.

| Contrast | What is separated | Intended benefit and limit |
|---|---|---|
| Within a member | Different-class TRAIN examples; same-class examples across views are positives | Stronger class discrimination. Canonical SupCon already specifies this operation. Every member can learn the same competent decisions. |
| Between members at one layer | Representations of the same input from different routes | Distinct representations. Separation can occupy directions the classifier ignores or oppose useful class information. It does not establish complementary correct decisions. |

For the within-member interpretation, compare examples **inside each route**, using the existing task-facing capture point, own CE and canonical same-class positive/different-class negative rules. Do not make another route's representation a negative merely because its member index differs. No new projector, layer-location search or strength grid is needed to state this rule. The [saved SupCon scope](../contrastive_BE_canonical_SupCon_graph_comparator_scope_20261008_v1/READ_SCOPES.json) already records Eq. 2 and the internal-BE adaptation; this is an attributed reference, not a newly invented learning rule. Applying it at every layer would change the recipe and budget.

The [saved decision-visible assessment](../shared_backbone_decision_visible_contrast_assessment_20261007_v1/NOTE.md) gives an explicit construction where class alignment and member separation coexist with identical logits. That is a warning about interpretation, not evidence that the running models actually suffered latent collapse. Evaluate mean/worst member quality, correct alternatives and net pooled repairs, alongside any embedding statistic.

## What the fixed masked-context recipe actually supplies

It supplies **different conditional auxiliary inputs**, not new class information. A route's feature-masked graph is a deterministic transformation of the same allowed `(X,A)`. The hash partition is label blind and carries no guarantee of different task-relevant evidence. Factual serving gives every route the same complete inputs.

CORE contrasts each missing-feature reconstruction with its raw target and treats other masked reconstructions as negatives. Some negatives can belong to the same class. Therefore this is instance restoration/discrimination, not the within-member class discrimination described above. Filtering negatives by class or adding SupCon would be a changed learning rule; neither is silently added here.

A common decoder reduces private-decoder freedom but cannot prevent a shortcut: all routes can share a good class code while reconstruction uses a separate feature subspace that the classifier ignores. Masked own CE provides a task-facing gradient and makes useful contextual learning plausible, but the same shared class function can satisfy it across members. Thus the recipe must earn its contribution through factual decisions. It does not already provide member-specific useful information or satisfy a between-member separation guarantee.

## Closest complete-rule ancestry

The closest **project training-rule collision** is [the saved accuracy-first assigned-view bank](../accuracy_first_graph_ensemble_direction_20261004_v1/DECISION.md), with its [exposure amendment](../accuracy_first_graph_ensemble_direction_20261004_v1/AMENDMENT_LABEL_VISIBILITY_AND_EXPOSURE.md): live shared/private trajectories, factual plus assigned-view CE, persistent/shuffled ownership, complete factual serving and capable single/untied controls. Those parts cannot be claimed anew.

The controlled addition is full-feature masking plus CORE's raw-target reconstruction. The [already indexed CORE scope](../adaptive_sharing_one_new_gap_scout_20261010_v1/READ_SCOPES.json) and [saved GCMAE/DGE scopes](../context_target_recipient_new_primary_literature_20261008_v1/READ_SCOPES.json) own reconstruction/contrast and different graph-evidence ancestry. Their inspected complete methods do not establish the exact four-route supervised masked recipe. That bounded difference permits an attributed composition; it does not clear global novelty. The reconstruction increment and its useful consequence are the claim to test. A practical win over native independent models would not by itself isolate weight tying, decoder sharing or initialization as the cause.

## Simplify admission, preserve the full positive comparison

Use root's proposed **prospective staged screen** on the same complete graph, full native horizon and all three fixed seeds:

1. **Stage 1: nine records.** Own-only shared4, fixed masked-context candidate shared4, and ordinary genuine independent4. Freeze worthwhile factual accuracy/NLL and mean/worst member safeguards before execution. Complete all seeds before reading comparisons. Stop if the complete candidate fails the whole-population/competence gate; retain every failure and introduced error.
2. **Stage 2 only after a pass: eighteen records.** Execute the original fixed masked-CE-only, updatewise ownership-permutation, auxiliary-only rewiring, ordinary native single, all-view objective-matched single, and objective-matched independent4 conditions. These controls remain mandatory before a restoration, persistence, local-graph or sharing-quality explanation. A match to the relevant simpler control removes that explanation.

The positive family still has 27 comparison records. The saving is avoiding six expensive conditions after a failed nine-record screen. Unexecuted conditions are recorded as unexecuted, never hidden or treated as favorable. There is no post-outcome coefficient, mask, backbone or seed search.

A standalone ordinary single could reuse a **predeclared** independently trained member only if its exact source, initialization, objective, optimizer history and native selection contract match. Do not choose the best member afterward. An independent auxiliary member seeing only one mask is not equivalent to the all-four-view single. No such reference identity is established here.

PubMed remains a development nomination. [Saved exposure metadata](../continuous_method_gap_search_v1/round5_alignment_followup_v1/GRAPH_EXPOSURE_METADATA.json) records earlier PubMed proposals and explicitly cannot certify an unused task. The [HeaRT preparation](../pubmed_heart_acquisition_native_plan_20261004_v1/READ_SCOPES_AND_LIMITS.json) and [raw-feature equivalence preparation](../pubmed_planetoid_raw_feature_equivalence_source_20261004_v1/READ_SCOPES_AND_CUSTODY.json) concern LP/source interfaces; they do not qualify a modern node-classification reference or its score reuse. Root must establish history, native capability, exact bindings and resources before admission. No new study directory creates fresh confirmation.

## Canonical memory handoff

The exact current pointer is `/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/literature_memory/CURRENT_SUPPLEMENT.json`. It now points to `literature_memory/ACTIVE_SUPPLEMENTS_20261010_v39.json` (SHA256 `31409c7d33b6fb0190b6b273366b6b53d0f0d2fc271537aceea6c4986e699f2a`). Root has indexed CORE once. This follow-up adds **zero** paper/method/code-reading credit and does not mutate that pointer.

The existing supplement-update helper remains root-owned. Its exact path was requested from root but was not located in the permitted `literature_memory`/`coordination` helper search; no old index adopter is misidentified or invoked as that updater. Root can attach this note through its existing update procedure without rereading or recounting CORE, CorDA or PathBoost.
