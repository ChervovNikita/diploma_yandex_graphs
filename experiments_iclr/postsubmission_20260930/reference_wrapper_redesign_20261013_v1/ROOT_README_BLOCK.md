## Revised wrapper benchmark — 13 October 2026

**The earlier 80,850-fit proposal is superseded.** The new proposed study keeps
the reference backbone, features, preprocessing, splits, evaluator and native
optimizer presets unchanged. Its main comparison is a native single model,
four genuinely independent native models, and a four-route GNNM with one shared
native backbone and private factors at explicitly mapped input/output boundaries.
There is no learning-rate or weight-decay search.

| Task | Native architectures | Datasets | Fixed presets |
| --- | --- | --- | ---: |
| Node classification | PolyFormer, separately reported Mono/Cheb source settings | CiteSeer, PubMed, Coauthor CS, Coauthor Physics | 8 |
| Molecular graph classification | GCN, GCN+virtual, GIN, GIN+virtual | MolHIV, MolBACE | 16 |
| Link prediction | NCN, NCNC, BUDDY | OGBL Collab | 3 |

That is **7 datasets, 8 native architectures, 27 fixed presets and 2,480 planned
cold fits**: 270 native source-horizon checks plus 2,210 main comparison fits,
with ten paired trials per preset. The single reuses exactly member zero of the
ordinary ensemble. Original-paper GNNM experiments are not rerun.

The additional GNNM candidates change only training on the same wrapper:
mixed member/ensemble loss; ensemble-loss credit confined to private factors;
fixed private example responsibilities; and a within-route supervised contrastive
loss on eligible node tasks. None is claimed successful or novel from this design.

Published scores and our source-horizon checks are distinct from the main
longer-training comparison. Main caps/minimums/patience are 10,000/2,000/100
for PolyFormer and 1,000/100/50 for molecular and link presets. Native LR,
WD, optimizer groups and constant schedules remain fixed. Curves, stopping
reasons, failures, per-trial scores and paired uncertainty will be retained.

OGB's two published dropout presets are separately reported because the paper
does not identify each winning value. Conflicting Polynormer paper/release rows
and unavailable HGB hidden-test protocols remain explicitly deferred/excluded.

**Design and metadata checks are complete; the scientific runner is not ready.**
The portable boundary core and loss rules have passed static checks. Exact
source-specific categorical/parallel/reused-boundary adapters, data/trainer
integration, checkpoint/evaluator integration and physical resource admission
are still pending. No new fits have started and no GO has been received.
The old public v29 launcher refuses execution. Current preparation accesses
only the authorized one-GPU allocation; three-GPU assignments are prospective.

- [Complete revised design and methods](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/README.md)
- [Exact proposed counts](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/SUMMARY.json)
- [Fixed source settings and published score cells](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/SOURCE_CELLS.csv)
- [Exact fit roster](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/FIT_ROSTER.csv)
- [Native-model preservation contract](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/methods/PRESERVATION_CONTRACT.md)
- [Portable core and pending source ports](experiments_iclr/postsubmission_20260930/reference_wrapper_redesign_20261013_v1/runtime/README.md)

The dated proposals and research outcomes below are preserved history.
