# Prospective conditional graph-response pilot recommendation

3 October 2026. Immutable recommendation, not a scientific freeze or launch. Root reported that all eight native v4 CPU engineering families passed on77's Torch2.7.1/PyG2.7.0 declared runtime. That evidence establishes neither Amazon competence, actual-data feasibility, CUDA equivalence, efficiency nor predictive utility.

## Scope

Treat this as supervised graph regularization. The hypothesis is that a fixed conditional graph-evidence response penalty can improve a competent ordinary ensemble through existing private intermediate factors, under fixed competence/energy constraints. Label-derived removals are finite supervised probes, not identified causal interventions. Lower redundancy, different responses or accepted updates are not success criteria.

Keep the author Amazon `heter_fixed_splits` body unchanged: `mask.permute(1,0)[split]`, equivalent to `mask[:,split]`. Pair official splits0/1/2 with optimizer seeds17/29/43. Within each official TRAIN class, hash-order node IDs, take floor(4*n_class/5) for fit and the remainder for control. No forced minimum, dropped targets or retries. Ordinary losses use fit only; control labels supervise guards and are not heldout evidence.

Block prefix is `condresp-amazon-v1|split=<j>|opt=<s>`. Use suffixes `|roles`, `|edges`, `|null-permutation`, `|null-mask` with the qualified packet's fixed hash-priority algorithms. Freeze before data preparation.

## Native and probe graphs

On the unchanged author/PyG-native reciprocal simple unweighted graph, eligible units are distinct unordered nonloop edges whose endpoints are both fit targets. Separately remove floor(E_same/10) same-class units and floor(E_other/10) different-class units by fixed hash priority. Remove both reciprocal records, retain original loops, exclude loops from grouping, and independently rebuild native gcn_norm/SciPy/float32 mono tokens for each complete view. Masks remain fixed through continuation. Unsupported graph semantics stop admission; no silent repair or new sampling seed.

Exact null A preserves every node's per-view removed degree and needs a qualified exact oracle. Exact null B matches integer endpoint native-degree pairs; feasibility on actual label-restricted Amazon edges is unknown. Neither can be presumed feasible or causal.

For a representative regularization pilot, recommend a separately declared log-degree-stratified permutation null from the outset. Define d_v=1+native nonloop degree and b(d)=d.bit_length()-1. Match original removal counts exactly in (view,min(b(d_u),b(d_v)),max(b(d_u),b(d_v))) strata, using one fixed class-count-preserving fit-label permutation and fixed hash order. This is not exact raw-degree or per-node matching and is not implemented by v4's exact-B helper. Report raw endpoint-degree histograms/total-variation imbalance and per-node removed-degree differences. Insufficient capacity, an unchanged relation assignment or a duplicate functional mask pair stops this screen; no permutation retries, bin substitution or selective favorable inclusion.

## Groups, objective and common guards

Use true class × native nonloop fit-neighbor same-class fraction >=1/2, <1/2, or no fit-labeled neighbors. Keep raw fit cells with >=32 targets; merge only rare cells within class, then remaining rare targets globally. A terminal fallback below32 stops. Control guards include global plus supported complete-class and raw cells, minimum16, deduplicating identical memberships. Unsupported targets remain in global supervision.

Candidate objective is native mean-member CE + mean two-probe CE +0.1D, where D is the equal-group/equal-unordered-pair squared normalized inner product of centered concatenated member-probability responses. Warm norm floor is1e-5, common active coverage >=50%, inclusive response norm band [0.5,2] times warm, and every member/view/control-cell CE <= own warm reference+0.01 nat. No epsilon or guard-tolerance relaxation.

Freeze common conditional active IDs from the original-mask warm function. Null-arm reference values use that same copied warm function on its declared probes; any required member norm below floor stops admission without intersections/retries. Class-only D uses complete-class groups with the same rare-class/global support rule and equal active-cell weighting. Conditional energy groups remain common to all arms; no union guard is added. This ablation changes centering and equal-cell weighting together and cannot isolate neighborhood conditioning alone.

## Bounded warm recipe comparison before intervention

Keep source Adam, maximum2000 epochs, patience250, strict validation-accuracy improvement and earliest ties. NLL is a secondary metric at that same selected checkpoint. Retain both prospectively specified native-single recipes, all three blocks, even if one is weak:

| Setting | Source CLI defaults | Pinned Roman-empire mono recipe |
| --- | --- | --- |
| hidden /FFN /heads /layers /K |64 /128 /1 /1 /10 |256 /64 /16 /3 /14 |
| q /multi /dropout /dprate |1 /1 /.2 /.5 |1 /2 /.5 /.1 |
| ordinary lr /weight decay |.0005 /.00005 |.0001 /.001 |
| attention lr /weight decay |.0005 /.0001 |.001 /.0001 |

Compare mean selected validation accuracy across the three blocks; strict improvement wins and default wins an exact tie. This is bounded development selection using shared native split blocks. Neither recipe is authored or empirically certified for Amazon-ratings. No independently sourced comparable published native Amazon reference has been established in this source record, so a scientific competence threshold is explicitly unresolved. A chosen recipe is not automatically competent; ordinary M4 warm competence still needs review before intervention. No threshold is invented here.

For later M4 warm acquisition, initialize every existing R/S factor before Stage A with independently keyed Rademacher signs, magnitude1, and fixed separate member dropout streams. This is an explicit BE adaptation. No post-wrap reset, new Stage B factor or reinitialization. The present native-single warm study fixes all adapter factors at1 and trains only native common parameters; it does not perform this M4 acquisition.

## Later continuation and serving

Copy one selected warm bank per block into all eight original comparison arms. Freeze all common parameters and boundary factors; only existing intermediate attention/FFN R/S train. Stage B dropout is off. Use fresh identical AdamW state, chosen recipe site-group rates/decays and no scheduler. Attempt200 proposals including rejections. Try complete-displacement scales1,1/2,1/4,1/8, commit full proposed moments once on nonzero acceptance, otherwise restore all state and retain work costs. Fewer than20 accepted proposals fails the existing10% mechanism screen. This is modified AdamW, not unchanged native AdamW.

Use the same strict validation-accuracy selector for all arms, including copied warm step0; this explicitly supersedes the earlier recommendation's inaccurately called native validation-NLL selector before any pilot scientific freeze. Serve native graph mean raw logits then softmax. Mean-probability pooling is secondary. DICE/FoRDE implementation, source recipes and auxiliary rollback remain separate blocked work; unavailable controls cannot be weak stand-ins.

## Retained comparison family, screen and costs

The original fixed eight arms remain native-only CE, augmentation-only native+mean-probe CE, candidate+0.1D, class-only D, normalized hidden-representation cosine repulsion, source-qualified DICE, source-qualified FoRDE, and candidate under the separately declared label-permutation null. They share copied warm banks, fixed true-label supervision, private-only continuation permissions, common competence/conditional-energy guards and the accuracy selector. The log-degree-bin proposal changes the null's stated scope; it does not certify the original exact-degree falsifier. Neither RBF energy nor the exactly duplicate degree-two kernel is added as a ninth fit. Complete strong-control source/auxiliary-state qualification is still required.

Retain the original development utility screen at the common accuracy-selected states: pooled validation NLL improves by at least0.01 nat over augmentation-only and qualified diversity controls, with the same improvement sign in all three blocks; mean accuracy and fixed-class macro-F1 lose at most0.2 percentage point, mean-member NLL increases at most0.01 nat, and worst-member NLL increases at most0.02 nat. Intervals are descriptive. Lower D without pooled benefit fails; matching class-only or permuted-mask behavior limits the corresponding mechanism claim. Fewer than20 accepted proposals, or cheaper matching utility references, also fails the intended screen. No post-outcome rescue through extra sites, groups, masks, objectives or seeds.

Utility references remain competent native PolyFormer single, capacity-matched PolyFormer single, native TFE-GNN single and M4 independently trained PolyFormer ensemble with competent packing. Each needs separate source/competence/resource admission; none is implemented here. Keep the original proposed33 base member-fit equivalents, three M4 warm banks and24 M4 continuation banks as a source plan, not a measured total. Charge the new six-fit warm recipe study separately, plus actual checkpoint/replay work, all graph/token construction, complete member paths, DICE estimator, FoRDE derivatives, guards/backtracking and rejected proposals. Do not infer equality of budgets or hardware efficiency from allocated parameter counts.

Only after a passing development screen would a separate unchanged mechanism/control freeze for complete Roman-empire with native splits and fresh predetermined seeds become eligible for review. No confirmation is authorized by this recommendation.

## Exploratory limits and remaining admission

D is an attributed degree-two functional redundancy energy, not MI, independence or a new repulsion primitive. Exact kernel matching duplicates it and does not warrant an extra fit. TRAIN-control guards can overfit and do not protect TEST. Recipe selection and intervention evaluation reuse development labels. Three overlapping split/seed blocks on one graph are not independent graphs, and descriptive uncertainty does not establish confirmatory generalization. Source Adam, its coupling of L2 weight decay, strict accuracy selection and native graph serving are preserved in the bounded native-single study; TRAIN80/20 roles and selecting an Amazon configuration are explicit adaptations. No paper result is an interchangeable competence reference unless architecture, release, splits, labels, budget and selection are comparable.

Before any intervention, root still needs actual official-data custody, two-recipe resource/whole-study closure, native/M4 competence disposition, actual mask/null feasibility, strong source-reviewed controls and a separate prospective pilot freeze. This proposal does not authorize data access, training, evaluation, resource acquisition or release and amends no frozen study. No project outcome informed these choices.
