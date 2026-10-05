# Amazon/Polynormer: one retrospective logits-only fusion protocol

**Frozen specification, 5 October 2026.** This packet specifies a later development study; it performs no payload access, fitting, scoring or execution admission. All constants below are fixed before Amazon fusion outcomes. The purpose is useful attributed prediction quality, with a diagnostic test of graph-local member-error moments. Reused VALID results remain development-biased.

## 1. Complete bank and native reference

Use the existing V6 source and registry order, all three split/block identities, and both four-member families:

| Official split | Block seed | GNNM family | Independent member seeds, order 0–3 |
|---|---:|---|---|
| 0 | 17 | `gnnm_boundary_4` | 17, 1026, 2035, 3044 |
| 1 | 29 | `gnnm_boundary_4` | 29, 1038, 2047, 3056 |
| 2 | 43 | `gnnm_boundary_4` | 43, 1052, 2061, 3070 |

The independent family is `independent_author_4_same_width`; `single_author` is its member-0 alias, with no new fit. Complete custody requires the existing 3 GNNM and 12 independent physical fits. Preserve each fit's own selected checkpoint/stage, member order 0–3, node rows 0–24491 and class columns 0–4. Never choose a later epoch, force a global-stage replay, omit an unfinished member, or pool checkpoints across splits. Physical payload integrity and availability were not checked by this packet.

The native four-member predictor is **mean member probabilities**, not softmax of mean logits. Carry its exact FP32 native accuracy calculation and stable FP64 log-softmax/log-sum-exp mixture NLL. Use FP64 probabilities and arithmetic for new postprocessing/Brier calculations; report any native FP32/FP64 argmax discrepancy as a numerical diagnostic. Mean raw logits is a separate parameter-free control. Also report uncorrected member-0 reference metrics from the independent alias.

## 2. Labels and exact folds

For each split separately, fusion development uses only its 6123 official VALID labels. FIT labels (9795/9796/9795), TRAIN-control labels (2451/2450/2451), and TEST labels remain excluded from every fusion fit, moment seed, residual, label reset, posterior field and score. Those roles and their original contracts remain unchanged.

Order VALID node IDs by ascending SHA256 of UTF-8 `amazon-moment-v1|split={s}|outer|node={v}`; break hash ties by numeric node ID. Assign sorted rank modulo 3: three outer folds of 2041 nodes. For scored fold D, B=VALID\D has 4082 fusion-fit/anchor nodes. The **entire D** is absent from every label-derived computation, including global moments and fallback class frequencies. D/all-node predictions and label-free topology may participate in transductive context. Removing only a node's own Gram entry is insufficient.

**Honest stacker training features:** sort B by SHA256 of `amazon-moment-v1|split={s}|outer={f}|inner|node={v}`, ties by ID, then rank modulo 2. Each 2041-node inner fold receives features computed with only the other inner fold's labels. Recompute local and global moments, posterior, anchor mass/pair summaries and the analytic skip for that seed set. Stitch the two feature sets; fit one head on all B target labels. Serve D using context from all B. No inner head fits are needed. The C&S stage occurs after this uncorrected head and uses B only; C&S outputs are not head training features.

Tune each bank/split independently on its three outer OOF folds. No cross-split fusion fit, mixed-label feature cache, shared selected hyperparameter, or numerical pooling of another split's labels is allowed. Report all three complete blocks. Overlapping official roles in independently trained splits do not by themselves invalidate a correctly isolated per-split pipeline; the graph and overlapping splits do limit independence claims.

The base checkpoints already used full VALID for selection. These folds cross-fit the aggregator's label-derived feature construction, **not the complete predictor**. Fold exclusion prevents direct fusion target leakage, but cannot undo base selection reuse or later OOF tuning bias.

## 3. One graph and one diffusion

Use the canonical V6 graph (`to_undirected`, remove old loops, add one loop/node), logical edge hash `229a8a787ef9120a4d7a1dcdd6481b973619a1c7f44a4abe9ed69d05c256e550`, source shape [2,210592]. For each bank/split, keep a non-loop edge only when the two nodes' **frozen native pooled predicted classes** agree; retain every self-loop. Argmax ties choose the smallest class. This mask uses predictions only: no true labels, correctness indicators, fold outcomes or tuned stacker predictions. The mask is fixed across methods/folds/settings.

Let T be the row-normalized retained adjacency. Define H operationally by F0=X and F(k+1)=0.2X+0.8TF(k), for exactly 20 updates. Apply the same H to scalar/vector fields; never form a dense N×N matrix. H is nonnegative and row stochastic. Graph masking remains development-dependent through the already selected base predictions; it is not an independent-validation construction.

## 4. Moments and nine operators

For permitted anchors l, p_m(l)=softmax(z_m(l)), e_m=p_m−onehot(Y_l), and G_l=E_l E_l^T (an **uncentered** error second moment). Let s=H1_B, R(v)=H(1_B G)(v)/s(v); if s=0, use R0=mean_B G. Define a=max(trace(R0)/4,1e-12), eta=0.5, epsilon=0.05 and rho in {0.01,0.1}. The candidate matrix is Rtilde=0.5R+0.5R0+rho*a*I. Minimize w^T Rtilde w with sum(w)=1 and w_m>=0.0125; pool q=sum_m w_m p_m. Use a deterministic convex solver with primal/stationarity tolerance 1e-10; a numerical failure is reported and fixed, not scored as a successful arm.

| ID | Operator | Fixed settings |
|---|---|---|
| 1 | Native mean probabilities | one |
| 2 | Softmax(mean raw logits) | one |
| 3 | Global positive scalar temperatures + dense simplex weights | two regularizers |
| 4 | Global full moments: R0+rho*a*I | two rho values |
| 5 | Local diagonal moments: diag(diag(0.5R+0.5R0))+rho*a*I | two rho values |
| 6 | **Local full moments, primary analytic candidate** | two rho values |
| 7 | Score/context residual MLP, native probability skip | two regularizers |
| 8 | Same-information residual MLP, **exact operator-6 skip** | two coupled settings |
| 9 | Fixed graph-label posterior projection into target member hull | one |

For four members, R requires ten scalar Gram fields plus anchor mass. Equivalently R_mn=(R_mm+R_nn−D_mn)/2, using all six **same-anchor, same-H** pairwise squared probability distances. Do not run a duplicate reconstructed-matrix arm. An off-diagonal gain is an attribution to this statistic/estimator, not extra independent label information.

Operator 9 diffuses permitted one-hot labels and divides by the same s; zero-mass fallback is B class frequency. At each target node minimize ||sum_m w_m p_m(v)−q_H(v)||² under the same density constraint. The projected probability vector is unique; any weight nonuniqueness uses minimum Euclidean weight norm. No temperature/ridge grid is added. At exact conditional moments, simplex risk equals ||P(v)^T w−q(v)||²+1−||q(v)||². Transported anchor Grams use P(l), not P(v): differences from this projection can reflect predictor drift as well as local competence/shared error.

## 5. Competent matched stackers and a bounded budget

Score features X_s have 69 coordinates: member probabilities (20), stable member log-probabilities (20), native probabilities and log-probabilities (5+5), H(native probabilities) (5), six target pairwise distances, six anchor-local pairwise distances, log(1+retained degree), and anchor mass. X_f adds the ten raw local and ten global upper-triangle moments and five diffused posterior coordinates: 94 coordinates. All seed-dependent training coordinates use the whole-inner-fold exclusion in Section 2. Standardize using stitched B feature rows only, std floor 1e-6.

Let h=A X+V ReLU(WX+b)+c. Both heads serve q_out,c=q_anchor,c*exp(h_c−max(h))/sum_j[q_anchor,j*exp(h_j−max(h))], equivalently a log-probability residual readout. Prediction outputs are not contaminated; zero residual recovers q_anchor exactly. Operator 7 has width 44 (3650 parameters) and native skip; operator 8 has width 32 (3675 parameters) and the corresponding rho candidate skip. Direct linear and nonlinear readouts can change the predicted class. Zero A,V,c recovers its analytic skip exactly, so the full-information control genuinely contains the candidate. No weak scalar gate is substituted for it.

Initialize A,V,c,b=0; initialize W by Xavier-uniform with the first eight SHA256 bytes, big-endian modulo 2^31, of `amazon-moment-v1|bank={name}|split={s}|outer={f}|operator={id}|setting={j}`. Use exact family names, outer indices 0–2 and setting indices 0–1. Final refit uses `outer=refit`. Fit multiclass Brier plus L2 mean squared displacement from initialization, r in {1e-4,1e-2}. Operator-8 settings couple (rho,r)=(0.01,1e-4) and (0.1,1e-2); there is no Cartesian grid.

Operator 3 uses q=sum_m w_m softmax(z_m/T_m), T_m in [0.25,4], w_m=0.0125+0.95*softmax(a)_m, initialized T=1/a=0. Minimize Brier plus r*[mean((log T)^2)+sum(w−1/4)^2]. Clip log T after each update. For all fitted controls use exactly 150 full-batch Adam updates, beta=(0.9,0.999), optimizer epsilon 1e-8; head learning rate 0.01, operator-3 rate 0.03, FP64, one CPU thread. Use final iterate, with no epoch selection, restarts, extra seeds or adaptive expansion.

There are **nine operators and fifteen configurations per bank/split**, not a multi-axis search. Across six bank/splits and three outer folds: 72 MLP fits and 36 global calibration fits; moment/projection operations reuse cached fields. This is the entire development fitting budget. Raw and corrected outputs are both reported; C&S is fixed, not a second tuning axis. Select each operator's setting within each bank/split by mean outer OOF corrected Brier; exact ties choose the first listed setting. The processed finalist is the lowest corrected OOF Brier among the nine selected operators; exact ties follow ID order. This bounded selector itself is part of the declared development pipeline.

## 6. One fixed C&S adaptation for every operator

Given uncorrected q on all nodes and permitted B labels, compute correction C=H[1_B(onehot(Y)−q)] and qc=Euclidean-simplex-project(q+C). Set S0=onehot(Y) on B and qc elsewhere; serve qcs=H(S0). Use the graph, restart and iteration count in Section 3, scale 1, no autoscaling, no second label clamp or extra search. The sign is Y−q followed by addition. This is a declared C&S adaptation, not a reproduction of every native C&S normalization/autoscale option. All nine operators receive the same B supervision and corrector. Scored D labels are absent from correction and smoothing seeds/reset inputs. In-sample B anchor predictions are permitted correction inputs; they are not scored.

For new NLL calculations apply fixed uniform contamination delta=1e-12 before taking logs; retain the original stable native uncorrected log-mixture NLL as its reference. Keep uncorrected results as attribution diagnostics; choose configurations/finalists only on corrected Brier.

## 7. Metrics, uncertainty and explicit go/no

Primary: mean multiclass Brier sum_c(q_c−Y_c)^2 (range 0–2, no division by C). Required secondary metrics: NLL in nats and top-1 accuracy with smallest-class ties. Report every configuration, all outer folds, each complete split, both banks, raw/corrected metrics, native/member-0 references and all paired differences. Pool the three equally sized folds within a split; never present nine folds as nine independent bank replicates. Record graph retention/degree, zero-mass fraction, anchor mass, weight concentration and classwise metrics as diagnostics, with no outcome-selected subgroup gate.

Report the three split-level paired differences individually, their mean and range. A mean±4.303*SD/sqrt(3) interval may be shown **only as a descriptive three-block t summary**; overlapping splits, the common graph and OOF tuning violate an iid confirmation interpretation. No node-iid bootstrap, confirmatory p-value or claim that aggregator OOF removed development bias.

**Quality screen:** per split/bank, the strongest cheap comparator is the smallest corrected OOF Brier among IDs 1–5 after their own bounded tuning. A bank's complete per-split finalist route is eligible to propose untouched confirmation only if, across all three splits, against **both that cheap comparator and the uncorrected native probability pool**, (i) mean Brier improvement >=0.002; (ii) positive Brier improvement in at least two splits; (iii) mean accuracy improvement >=0.25 percentage points; (iv) no split accuracy loss >0.5 points; and (v) mean NLL harm <=0.01 nats. These are fixed practical screening constants, not power calculations or significance thresholds. If no bank passes, **no-go for this new fusion confirmation**; report the useful cheaper rule if it wins. Do not acquire hidden states or retrain a bank to rescue a failed screen within this protocol.

**Moment attribution:** to advance a specific off-diagonal/local-moment claim, operator 6 must additionally improve mean corrected Brier by >=0.001 over each of IDs 4 and 5, with no split Brier harm >0.002 against either. If ID 8 matches operator 6 within 0.001 Brier or wins, keep useful attributed moments/stacking but make no special analytic-rule superiority claim. If ID 9 matches/wins, graph-label posterior transport is a sufficient practical explanation; a unique covariance-mechanism claim is unsupported. If gains exist only on fit anchors/subgroups, use extra labels/tuning, or disappear against matched independent processing, the claimed explanation fails. An eligible quality route can still merit confirmation even when a novelty/mechanism claim fails.

## 8. Freeze and untouched confirmation boundary

Before any later authorized final score, seal each split's actual selected settings/operator, all head weights, numerical environment, graph/fold hashes, complete provenance and the entire selector/refit recipe. Refit the selected pipeline on all that split's VALID: moment/C&S anchors are full VALID; learned-head training features are stitched from the three already defined outer exclusions, then one final head is fitted. No configuration is chosen using another split's labels/outcomes. Only a separately satisfied final-label contract can admit evaluation of the new estimand. Preserve the original native pool endpoint and original complete-cohort evaluator.

A correctly isolated, frozen per-split pipeline may use that split's TRAIN/VALID development labels and be assessed on its untouched TEST, despite role overlap in another independently developed split. Record that separation; no cross-split fit/cache/tuning can exploit such overlap. A fresh graph/population is scientifically stronger but is not a mandatory globally-disjoint-node condition for describing that isolated per-split estimate. If final target outcomes influenced the rule/selector, fresh confirmation is required. Existing TEST/control access is neither supplied nor changed here; TRAIN-control cannot silently become fusion training or a new final endpoint. A later confirmation must report both complete banks and all three splits, including a stronger processed independent bank; do not promote only a favorable shared-bank result.

## 9. Costs and overhead to omit

No hidden export, neural training, GPU request or backbone replay is needed once existing logits custody/replay admission is satisfied. Promised unique raw logits total 11,756,160 tensor bytes; no physical availability or speed claim is made. Cache each graph and label-free field once; ten Gram fields, five posterior fields and mass need 3,134,976 FP64 bytes per seed context, before buffers/graph/outputs. Inner construction adds field propagation, not inner head fits. Stream contexts rather than retaining all-node fields for every configuration. Batch the 4-variable convex solves; global moment weights need one solve per anchor set. Account for all sparse propagation and C&S across the fifteen configurations rather than calling their cost free.

Record CPU wall time, peak RSS, fit counts, bytes/cache I/O and amortized per-node serving cost, separately from historical bank acquisition. Every member still runs in dense serving. A quality screen does not require being cheaper than native inference, but eventual quality/cost choice must include correction/head overhead.

Omit hidden features, a GETS/raw-feature reproduction, a context-permutation grid, duplicate pairwise reconstruction, extra graph/restart/shrinkage searches, per-configuration portable neural replay, nested optimizer searches and new execution-release schemas. Existing cohort/integrity gates suffice; mask assertions and an implementation audit are necessary. Heavy extra gates cannot make reused VALID independent. Real leakage is forbidden fold labels entering any moment/global/fallback/residual/reset/head feature, mixed split supervision, or final outcomes influencing selection.

## 10. Evidence and scope

Source and release identities are inherited from the completed source-only feasibility packet and recorded in SOURCE_BINDINGS.json. Scientific context comes from the saved graph-moment scout, label-reuse amendment, root synthesis and GETS closure note; their hashes are bound here. The scout's earlier stricter validation condition is superseded by the saved amendment and this protocol. The exact conditional-posterior projection interpretation and cross-split isolation clarification were supplied by the parent analysis and are disclosed as analytical additions. No Amazon scores were accessed or used to set constants. No new literature retrieval, model import, source execution, payload read, SSH/allocation contact or source/gate/registry/canonical mutation occurred.
