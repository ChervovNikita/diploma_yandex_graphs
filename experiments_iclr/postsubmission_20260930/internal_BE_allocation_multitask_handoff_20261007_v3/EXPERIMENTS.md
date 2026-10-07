# Shared-backbone ensemble experiments

The experiment code is ready on the authorized one-GPU allocation. Preparation does not require the other Mac. New training on 18.77 will start after its connection is restored and its existing detached jobs are inspected.

## Method being tested

Each example passes through four learned paths inside one shared GNN. Each path has its own small trainable scaling vectors, called BatchEnsemble factors, around the backbone's linear maps. The main weight matrices are shared. Each path learns from the labels; the final prediction uses the declared average of its four outputs.

The first experiment adds two training objectives. One aligns representations across two stochastic views within a path. On WikiCS and MolHIV, positives include all examples with the same TRAIN label; on Collab they are records of the same canonical target edge. The other encourages differences between paths after subtracting their TRAIN-class mean representations. The comparison asks whether this combination improves predictions. Greater embedding distance alone is insufficient: it can leave the predictions unchanged or weaken useful members. MolHIV's shared binary label does not imply chemical equivalence, so the alignment assumption also needs testing.

A separate experiment changes which parameters receive the ensemble's supervised loss. Shared weights and the input/output factors can continue learning from each member's own loss, while internal factors receive a mixture of member and pooled-prediction losses. Four policies isolate the internal factors, prediction boundaries and shared core. This builds on GNCL; the proposed restriction needs an empirical advantage over its simpler alternatives.

## Three different tasks

| Task | Dataset and split | Backbone | Served prediction | Primary metric | Complete horizon |
|---|---|---|---|---|---|
| Node classification | WikiCS, split 0 | Polynormer | Mean class probabilities | Accuracy | 100 local + 1,000 global epochs |
| Link prediction | ogbl-collab, official temporal split | GCN + NCN | Mean link logits | Hits@50 | 100 epochs over all TRAIN events |
| Molecular graph classification | ogbg-molhiv, official scaffold split | Bond-aware GINE + virtual node | Mean graph logits | ROC AUC | 100 epochs over all TRAIN molecules |

These backbones and tasks are implemented explicitly. The code does not establish support for every GNN or heterogeneous graph architecture. Collab keeps repeated temporal events and the official negatives. MolHIV keeps categorical atom and bond features. WikiCS uses one transductive graph; optimizer seeds on split 0 are not additional graphs.

Wiki probability pooling gives more supervised credit to a member contributing more correct-class probability. Collab and MolHIV use mean logits, so their pool term supplies the same output residual to each member. Different internal computations may respond differently, but the Wiki responsibility explanation cannot be transferred to them.

## Comparisons and interpretation

The initial eight-arm family separates the single model, independent ensemble, shared factors, factor initialization and the auxiliary package. Three paired seeds are development evidence. The current 24-fit WikiCS family stays unchanged, and its comparative scores remain closed until the whole family finishes.

The separate O/I/P/G comparison holds initialization, views, alignment permissions, coefficient and checkpoint selection fixed. It tests internal ensemble supervision, protection of input/output boundaries, and protection of the shared core. Each task requires its own explicit loss and serving contract.

Two narrow comparators address remaining attribution gaps:

- Shared alignment alone, with the same parameter permissions as the original auxiliary package, isolates the residual diversity term.
- An own-only untied ensemble with joint pool checkpoint selection separates training changes from checkpoint-selection changes. The ordinary independently selected ensemble remains a practical baseline.

These follow-ups are separate prospective comparisons. Their source preparation does not authorize a large combined grid. Select one constructor and one coefficient, record the paired seeds before opening their outcomes, and use the original full horizons. Conditional confirmation must use verified unused evidence. Preserve unsuccessful and missing runs.

Assess the final served metric, each member's quality, errors common to all members, corrected errors and newly introduced errors. A reduction in common errors cannot rescue a worse final metric. Report paired differences and uncertainty at the seed level, with the limits of each graph/split. Charge actual training, reverse passes, validation, checkpoint selection and serving work.

## Execution checks

All three public converters passed complete ordered equality against the audited allocation TRAIN/VALID roles. The portable MolHIV single, shared contrastive model and ordinary independent ensemble passed a real update on the fixed maximum-node batch, complete 4,113-molecule validation, selection, checkpoint reload and finite serving on the allocation GPU.

The original three private-gradient modes passed the corresponding allocation CPU check. The new P/G paths separately passed on 7 October in 65.32 seconds, with 6.34 GB peak RSS. P made three actual gradient collections; G made two. Both updated all intended parameter groups, used one native Adam transition, and completed validation and reload serving. The coefficient 0.5 was a runtime fixture, not an adopted scientific strength.

These are bounded execution checks. They are not full training results or accuracy gains. The changed O/I paths, portable WikiCS/Collab training paths, and execution on 18.77 retain their stated runtime limits. Comparative quality values and TEST were not opened during these checks.

## Conditional structural alternative

A source-only WikiCS alternative adds a degree-sensitive channel to the first local aggregation, with either a shared scalar or centered member coefficients. PNA supplies the degree-scaling prior. Same-access native single and independent-ensemble controls are included. This remains conditional: cardinality blindness has not been diagnosed on the actual data, and the new channel has not undergone model or training qualification.

## Allocation source locations

Under `/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930`:

- `portable_internal_be_public_interface_20261007_v2`: public conversion and complete training for all three tasks.
- `portable_internal_be_family_queue_source_20261007_v1`: fixed MolHIV 24-fit queue.
- `public_internal_be_private_steering_complete_interface_20261007_v1`: original three private-gradient modes.
- `public_internal_be_allocation_controls_20261007_v1`: complete O/I/P/G training interfaces.
- `public_internal_be_auxiliary_selection_controls_20261007_v1`: alignment-only and own-only jointly selected untied controls.
- `internal_BE_cross_task_attribution_design_20261007_v1`: exact task-specific comparisons, readouts and prospective decisions.
- `wikics_first_local_cardinality_channel_source_20261007_v1`: conditional structural channel and same-access controls.

The readiness receipt binds the source versions and measured execution scope. The additional comparator interface passed allocation CLI help; its model execution remains pending. Five new seeds on the same split can confirm repeatability, but independent heldout evidence needs a separate frozen scoring protocol before a stronger claim. No new 18.77 job has started during this preparation. On restored access, inspect existing jobs, sync the committed source, verify the runtime and devices, and perform a same-host real-data check before launching a new family. Notify the user when that training actually starts.
