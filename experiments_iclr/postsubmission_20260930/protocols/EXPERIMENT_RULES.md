# Rules for new experiments

All studies in this phase are new experiments. They must not overwrite existing score files or redefine previous reported numbers.

## Pilot and confirmation

A pilot is exploratory. It uses at least three real public graph tasks where feasible, all specified before outcomes, paired optimization seeds, a fixed recipe or a small equal selection budget, and a relevant baseline. A small data subset is used only for code preflight. It cannot support an empirical research claim.

Before a pilot starts, save a protocol with a hypothesis, precise model operation, nearest prior methods, arms, input data source and bytes/tensor hashes, split definitions, preprocessing, optimizer, cap, checkpoint rule, seed list, evaluation, resource measurements, and continue/stop criteria. Freeze source hashes, environment and hardware metadata. Save every run, including failures. Changes require an amendment and a new study identifier.

Method and hyperparameter choices use training and validation only. New confirmation tasks remain unscored until the candidate and protocol are locked. Confirmation must include independent tasks, adequate optimizer seeds, standard splits and representative baselines. Do not describe multiple overlapping masks as independent graphs. Small pilot evidence does not justify a universal performance claim.

## Comparators and cost

Separate complete-method comparisons from component interventions. Specify whether initialization and checkpoint selection match. Include BASE, an explicit ensemble, original GNNM, a simple shared-stack multihead model, and at least one relevant efficient ensemble when the claim requires those comparators. An ablation should remove or replace the specific new mechanism while retaining its selection budget.

Measure stored parameters, trainable parameters, training time, inference time and peak device allocation at stated batch/graph sizes. If a deployment benefit is claimed, measure that benefit directly. Parameter count alone does not establish faster computation. Report both quality and costs, including parameter- or latency-matched baseline operating points justified before test scoring.

## Evidence

Each selected model must preserve its checkpoint, validation trace, input identity, member or pooled logits adequate for replay, selected step, training timing and source/environment manifest. A verifier rebuilds the selected model and checks predictions and scores. Run a genuinely fresh new-study reproduction when needed rather than calling setup checks a reproduction.

## Hypothesis promotion

Promote a candidate only if its proposed mechanism exists in code, its literature distinction is defensible, its pilot is stable across representative tasks, and relevant controls support the stated claim. A promising observation remains exploratory until independent confirmation. Favor an idea with a clear falsifiable prediction over a large hyperparameter grid.

## Independent review

Prepare a separate manuscript/evidence snapshot only after audited evidence supports a contribution. Each review process starts without authoring history, sees no target verdict, and receives the complete bounded claims and evidence. Preserve every review. Do not repeat an unchanged manuscript solely to seek a favorable score.
