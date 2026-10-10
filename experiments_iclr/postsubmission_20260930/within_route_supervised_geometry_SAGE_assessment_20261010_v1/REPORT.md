# Within-route supervised geometry: one SAGE decision

10 October 2026. Saved conclusions and complete reports were read first. Two saved primary methods were revisited: SupCon and graph-specific BotSCL. No new primary method, whole-paper reading, author-code audit, scientific runtime, data/prediction payload, remote operation or TEST access. This is an inactive assessment; root owns admission and execution. Existing sources and decisions remain unchanged.

## Decision

**Retain one attributed SAGE comparison: native own CE versus own CE plus supervised contrast on each route's actual preclassifier representation, with native serving.** This is a useful existing ingredient to test, not a new primitive. A gain on SAGE can be sufficient; passing every architecture is not required. The current frozen distribution family closes first under its unchanged rules.

The rationale is narrower than “more diverse embeddings.” Mean own CE trains the classifier but does not explicitly require a difficult same-class TRAIN node to resemble every other same-class example. SupCon supplies that signal. Sharing can amortize its representation learning across private views; it does not guarantee different correct predictions or an advantage over independently trained models.

## What was actually tested

| Operation | Verified saved status | Consequence for this question |
|---|---|---|
| Within-member canonical SupCon | **Complete:** Wiki12/SupCon15, Polynormer, three seeds, full 1,100 epochs, two stochastic views, 512 auxiliary TRAIN identities/all 580 own-CE labels, no projector or cross-member contrast, coefficient .05/temperature .2. | This operation is not globally untested. SupCon minus plain: −.195930 pp mean accuracy/+ .485586 NLL; minus class-full cross-view alignment: +.050563 pp/+ .556452 NLL. Preserve the adverse selected-development result, including the late selected seed. It does not answer the native SAGE comparison. |
| Residual separation and specialist credit | **Complete:** the Wiki R/C arms and private-CMCL18. Private CMCL loses .668017 pp accuracy and worsens NLL versus own_floor; correct-member coverage increases at two seeds but pooling loss/member harm overcome it. | Cross-route separation and assigned CE/nonowner uniformization are different objectives. Their failure does not establish failure of class discrimination inside a route. |
| Live nonlocal similarity/label retrieval | **Complete:** all 63 fits/81 new-plus-reference banks. SAGE live minus detached accuracy −.107445 pp; native-plus-memory serving changes the readout. | This already tested supervised representation geometry through retrieval. Its exact learning increment is unsupported; it cannot be called unexecuted. It is not the outside-log, all-positive metric below. |
| Direct single-view, all-TRAIN SupCon on native SAGE H, native serving | **No completed exact test established in the inspected sources/status records.** Original SAGE and paired-native fitting use own CE; the saved within-member follow-up is a proposal. | Eligible as one missing comparison, with explicit ancestry and this bounded execution finding. No claim of universal historical nonexecution. |

These statuses follow the complete result reports, not old launch/proposal timestamps. Current distribution predictions were not accessed or used.

## Closest primary methods and mechanism difference

**SupCon**, Khosla et al., arXiv:2004.11362v5, §§3.1–3.2.2/Eq.2, contrasts examples inside one predictor: same-label examples are positives, different labels are negatives, exact self excluded, positive log ratios averaged **outside** the logarithm. The original method uses two augmented views, a discarded projector and later frozen-encoder linear classification. A joint-CE, single-view native-H arm is therefore an adaptation, not an original-recipe reproduction.

**BotSCL**, Wu et al., arXiv:2306.07478v1, §§4.1–4.3/Eqs.12–14, supplies graph-specific supervised contrast: class-aware TRAIN feature swaps and edge removal; a signed, channel-specific graph encoder; a two-MLP projector; same-class positives across the opposite graph view, with all opposite-view candidates in the denominator. It later classifies concatenated input/final embeddings using class-weighted logistic regression. Its method gives concrete graph ancestry, without transferring its performance claims or heterophily explanation to SAGE. This proposal imports its class-discrimination principle, not those augmentations, encoder or two-stage pipeline.

For one route/anchor, write scores s_j=cos(h_i,h_j)/tau, allowed candidates A, same-label positives P. Then:

    SupCon:   logsumexp_A(s) − mean_P(s)
    retrieval class CE: logsumexp_A(s) − logsumexp_P(s)

The latter is the completed label-memory class-mass operation on its Q/S support. Its positive gradient target is the current within-class softmax and can concentrate on an already-close exemplar. SupCon uses a uniform target over positives and explicitly pulls difficult same-label examples. The displayed difference compares equal candidate sets algebraically; the completed retrieval recipe additionally changes masking, coefficient, temperature and serving. It is not an isolated historical loss contrast.

All contrastive pairs in the proposed arm belong to **one route**. Another route's representation is never a negative because its member index differs. Same-class attraction can make all routes agree on competent decisions; that is allowed. Larger cross-route hidden distance can occur without improved class decisions. Assess actual true-versus-rival predictions, member quality and full-population repairs/harms. No native rotational-symmetry theorem or measured representation-collapse claim is made.

Expected harm: forcing multiple same-label graph modes together can erase useful distinctions or amplify misleading labeled examples; the auxiliary may harm native optimization or confidence. No new graph information is supplied. The poor Polynormer result and unsupported SAGE retrieval increment lower confidence; they motivate a bounded test rather than a coefficient search.

## One decisive inactive comparison

Use the existing SAGE graph/config, seeds 7301/7403/7507, depth 2/width 128/dropout .2, native AdamW .001/weight decay 0, 1,000-update maximum/300 patience and original strict selector. Capture the existing output-head input from the **same** factual stochastic forward used by CE. Use all 580 TRAIN nodes; check at least two examples per class before fitting. Normalize with epsilon 1e−12; tau=.2. For each route i, P(i) is every other same-label TRAIN node; A(i) is every other TRAIN node. Average Eq.2 across anchors and routes, then use:

    J = mean native own CE + .05 × mean routewise SupCon

This single-view operator deliberately omits the counterpart and same-view/cross-view split of the old two-view arm. It isolates class-pair geometry on the current native forward without extra native passes, augmentation or serving changes. Every ordinary trainable upstream shared/private parameter receives its scalar derivative; the output head receives native CE. There is no projector, selective ownership, nonowner-uniformization term, retrieval decoder, cross-route repulsion or loss grid.

One family has four fresh conditions: **shared4 + SupCon, ordinary M1 + SupCon, factorized M1 + SupCon, stronger genuine factorized I4 + SupCon**. Each I4 body is fitted and selected independently under its own objective. Retain both raw ordinary/factorized I4 anchors and the two raw singles from the complete native SAGE family; the equipped factorized I4 was the stronger raw accuracy reference. Exact own-only reference reuse requires root's source/config/RNG/selector/custody match before outcomes. Invalid reuse requires a separately fixed successor roster, not selective replacement.

With eligible own-only reuse: **12 new banks/21 optimizer-body fits**, plus 15 existing own-only banks from 33 acquisitions. No new parameters or serving operation. A shared4 float32 TRAIN Gram matrix has 1,345,600 entries, about 5.13 MiB before autograd buffers; a single has about 1.28 MiB. At the maximum horizon the fresh family has 21,000 optimizer updates and 30,000 native member trajectories, plus validation/restoration work; the Gram products add at most about 1.292e12 multiply-accumulates at D128. Actual time/memory require measurement; no throughput estimate is inferred from concurrent historical jobs.

Root should freeze worthwhile accuracy/proper-risk criteria prospectively. Report shared candidate minus own-only, versus both equipped singles and equipped I4, and the shared-versus-I4 improvement contrast. Raw and any equally calibrated risk remain separate; a calibration policy requires prospective binding and cannot hide raw harms. Preserve mean/worst member quality, coverage gained/lost, native common-rival repairs, aggregation-only rescues, repairs minus harms, every seed/class and inclusive cost. A geometry statistic cannot pass the experiment.

**Refutation:** no worthwhile native pooled improvement, repairs outweighed by harms, or the capable equipped references explaining the proposed shared advantage closes this exact operating point. Comparable tied/untied gains can still establish a useful generic metric ingredient, without a special sharing claim. Favorable exploration gives one unchanged SAGE pipeline on an unused graph/split priority; the history/exposure check precedes that nomination. Three optimizer seeds on encountered WikiCS are not confirmation. No GCN/GAT gate is added.

## Reading boundary

SupCon's retained PDF4–5 text and BotSCL's retained method/preliminaries and second-stage paragraph were read. SupCon's result table was incidentally present and not adopted. No new source retrieval, supplement/proof reading, author implementation audit, whole-paper certification or new literature-memory credit. Saved STGCN conclusions were visible in a prior report; its primary body was not reread. Hash bindings and exact scopes accompany this assessment. It authorizes no fit and changes no frozen packet, canonical state or manuscript.
