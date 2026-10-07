# Internal BatchEnsemble steering: code ready, full training started

The experiment code is prepared for three distinct tasks: WikiCS node classification with Polynormer; temporal collaboration link prediction with GCN+NCN; and molecular graph classification with bond-aware GINE and a virtual node. Official TRAIN/VALID exports have passed their complete data audits. No original manuscript score has changed.

The mechanism learns private BatchEnsemble factors inside the GNN while the shared weights learn. The experiment separates initialization, the contrastive auxiliary objective and sharing, using single models and ordinary independent ensembles as controls. Every member receives its own supervision. Hidden separation is a hypothesis about useful predictive differences, not a result.

## Actual execution

At 2026-10-07 04:02 UTC the fixed 24-cell WikiCS family started on the authorized singleton allocation. It comprises eight arms and three paired seeds, each with the original 1100-epoch horizon. Controller PID510850/start ticks6015502511, supervisor510853 and native scientific worker510855 were observed. The first single6101 fit reached epoch8 by04:03UTC. This is full scientific training, not a resource trajectory. VALID selects checkpoints through the frozen procedure; comparative results and TEST remain closed.

Eleven same-arm/seed resource cases had already passed. The resource-only controller was superseded at an observed idle boundary so those successes can be reused and qualified full fits can proceed immediately. Only that exact engineering controller was signalled. Existing initialization and growth experiments continue. Prior OOM and data-assumption failures, source versions and costs are preserved. No resource checkpoint initializes a fit.

The shared four-path workload peaks near81GB. Admission waits for its measured peak plus2GiB headroom; unrelated jobs are not stopped. A failure is retained with the declared roster. This can constrain concurrency even on an80GiB card.

## Next tasks and limits

Collaboration and molecular source are ready, while their launch packages and full-workload resource checks are being completed separately. Neither task has been adopted as a scientific family or trained by this new suite yet. The maximal72-cell design is not an automatic campaign.

The dedicated method agent is completing those two task launch preparations. A literature agent is assessing learnable diversity strength and closest prior. A third agent is preparing complete-cohort scientific analysis of the existing Citeseer initialization and growth studies. No new18.77 jobs have started; the other Mac may remain off during allocation preparation.

The fixed post-fit TRAIN intervention panel measures whether different members respond differently to removed edges and features. Its source is prepared, but runtime hooks and resource qualification remain pending. A later eight-view single control is required to distinguish shared training from extra stochastic supervision. Task quality, competent controls, prior-method comparisons and unused confirmation are required before superiority or novelty claims.

The goal remains active and unmet. Source approvals are not manuscript acceptance recommendations.
