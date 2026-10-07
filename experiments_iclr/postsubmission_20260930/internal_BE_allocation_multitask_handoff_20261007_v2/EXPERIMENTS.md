# Internal BatchEnsemble experiments prepared on the allocation

The complete training code is ready for node classification, link prediction and graph classification. Each example goes through four learned paths inside one shared backbone. Every path learns from the correct labels. The experimental auxiliary loss encourages agreement between two views of the same example within a path and differences between the four paths after removing each TRAIN class's mean representation. At inference, the four predictions are averaged using the declared task rule.

The scientific question is whether these private paths learn useful differences that improve the combined prediction. A larger distance between representations does not establish this. We will compare the final predictions, each member's accuracy or ranking quality, and mistakes shared by all four members.

## Tasks and full recipes

| Task | Data | Backbone | Primary metric | Full training horizon |
|---|---|---|---|---|
| Node classification | WikiCS, split 0 | Polynormer | Accuracy | 100 local epochs followed by 1,000 global epochs |
| Link prediction | ogbl-collab | GCN with the NCN link predictor | Hits@50 | 100 epochs, all TRAIN events |
| Graph classification | ogbg-molhiv, scaffold split | Bond-aware GINE with a virtual node | ROC AUC | 100 epochs, all TRAIN molecules |

These are three different prediction tasks. WikiCS uses one graph and one development split. Additional seeds on that split do not establish generalization across graphs. Collab retains official temporal events and negatives. MolHIV retains the official scaffold split and categorical atom and bond features.

## Initial comparisons

Each separately adopted task family has eight methods and three paired development seeds:

1. An ordinary single model.
2. The single model with the applicable view/class alignment objective.
3. Four independently trained models, each selecting its own checkpoint.
4. Four separate models trained with the coupled auxiliary objective.
5. Shared BatchEnsemble with identity factors.
6. Shared BatchEnsemble with the declared first-factor initialization.
7. Identity-initialized shared BatchEnsemble with the auxiliary objective.
8. The declared initialization with the auxiliary objective.

The primary comparison keeps initialization, task, backbone, labels, training horizon and serving rule fixed, then adds the auxiliary package. Comparing methods 8 and 6 tests that package. The other methods show whether a change comes from initialization, ordinary ensembling or the auxiliary supervision. Methods 3 and 4 have different training and checkpoint selection rules. They are separate baselines.

The initialized factors follow the saved TabM-inspired recipe. Rank-one factors follow BatchEnsemble. Alignment, residual diversity and member/pool-risk objectives have existing literature ancestry. This experiment is a hypothesis about useful graph predictions, not evidence of methodological novelty.

## Checks completed

All three public raw-data converters passed full ordered equality against the audited TRAIN and VALID arrays on the authorized one-GPU allocation. The public MolHIV single, shared contrastive model and ordinary independent ensemble also passed one real update on the maximum-node batch, complete VALID evaluation, checkpoint selection, reload and finite serving. That check used 3.82 GB peak GPU memory and took 29.80 seconds in total.

This verifies the stated execution scope. Full MolHIV training, the portable WikiCS/Collab model paths and execution on 18.77 remain unverified. It is not a measured accuracy gain.

At 06:54 UTC on 7 October, the allocation's fixed WikiCS family had finished five full fits and was training `be_init_6101`. Comparative scores remain closed until the adopted 24-cell family finishes. The complete-family rule also retains failed and unlaunched cells.

## Prepared follow-up controls

The separate agent prepared three single-model controls with eight supervised stochastic paths per update. They test whether any benefit can instead be explained by more training views. Their corrected prospective comparison includes the ordinary independent ensemble on the same five reserved confirmation seeds. These controls are not added to the running family. Confirmation is conditional on a useful development result and verified unused seeds.

A separate callable adapter compares alignment-only private gradients, hidden-diversity private gradients and an attributed GNCL member/pool-risk mixture restricted to internal factors. The shared weights and private input/output boundaries continue to learn from each member's own labels. Its coefficient remains explicit and unadopted. A separate complete-training wrapper now supplies the CLI for these controls while reusing the original full driver. Both packets have been staged on the allocation, and CLI help passed. Real model/gradient execution of this new control remains unverified. No coefficient, comparison family or new optimization principle is adopted.

## Locations and next execution

The source root on the allocation is:

`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930`

- `portable_internal_be_public_interface_20261007_v2/` contains conversion and complete training entry points.
- `portable_internal_be_family_queue_source_20261007_v1/` contains the fixed MolHIV family queue.
- `internal_BE_portable_reproduction_readiness_report_20261007_v1/` contains the recorded checks and prospective 18.77 command.
- `internal_be_single8_attribution_controls_source_20261007_v2/` contains the later view-count controls and confirmation plan.
- `public_internal_be_private_steering_complete_interface_20261007_v1/` contains the additional full-training interface for the three private-gradient controls.

Preparation and existing allocation training do not need the other Mac. When access to 18.77 is restored, first inspect existing detached jobs, sync the committed source, verify its actual runtime and GPU, and perform the stated real-data check there. Then start the new matched family and notify the user of its actual start. Source readiness does not mean that a new 18.77 job has started.
