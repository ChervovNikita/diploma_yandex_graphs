# Heldout confirmation on IMDB and molecular lipophilicity

Recommend **HGB IMDB and ogbg-mollipo** for a compact confirmation after a worthwhile existing pilot. They test heterogeneous multilabel node prediction and inductive graph regression with an official scaffold split. This broadens the target and evaluation settings beyond Tolokers, WikiCS and MolHIV. The second domain remains molecular; it supplies less domain diversity than an image or peptide task.

The recommendation is conditional on freezing and qualifying the task adapters. It admits no acquisition or training. The current binary, static Tolokers runner cannot run either task unchanged.

## Eligibility and reserved alternatives

The saved local research ledger, context, confirmation registrations and filename inventory were checked before choosing candidates. Source registration, feature acquisition and development exposure preserve the possibility of heldout confirmation. TEST targets and outcomes determine whether a reserved split remains available.

| Task | Saved exposure status | Scientific relevance and cost | Priority |
| --- | --- | --- | --- |
| HGB IMDB | No acquisition, fit or TEST use record found in the scoped inventory. Root must reconcile the complete current inventory. | Typed movie, actor, director and keyword graph; multiple simultaneous genres. Small complete graph and a strong inspected public typed reference. | First compact task. |
| ogbg-mollipo | No acquisition, fit or TEST use record found. Input overlap with previously used MolHIV is unverified. | Experimental scalar property on 4,200 small molecules; graph pooling and scaffold generalization. A contemporary maintained public regression reference is available. | Second compact task, conditional on a frozen bond-aware regression port. |
| HGB ACM | Features and development labels acquired; five development splits registered; the saved record says TEST label payload closed and fits not started. | Closest source-compatible alternative: heterogeneous multiclass nodes and existing HGT, Simple-HGN and SeHGNN source work. Similar task family to IMDB. | Eligible reserve with explicit development exposure. Prefer it if the regression port is scientifically unsettled before choosing the confirmation pair. |
| Peptides-func | Already registered as confirmation for a different sharing-policy branch; saved record says unacquired and unscored. | Biological graph-level multilabel prediction is attractive, with explicit GINE and reassessed GINE recipes. Much more full-task propagation than mollipo. | Eligible reserve when its mechanism relevance justifies the larger budget. Root must reconcile the existing registration and multiplicity before reuse. |

PPI and MNIST have saved use and data-freeze records. The original five heterophily benchmarks, WikiCS, Planetoid, arxiv, Amazon/Coauthor and previously studied link tasks supply development evidence. Their existing use must be declared. The inventory snapshot is a scoped local audit; it cannot certify the absence of every external exposure.

Choose the pair before any confirmation outcome. A failed confirmation cannot be replaced by an alternative dataset to obtain a favorable result.

## IMDB protocol and public reference

[SeHGNN Appendix Table 7](https://arxiv.org/html/2207.02547v3) reports **21,420 nodes, four node types, 86,642 edges and five labels**. These are published statistics. Its author source comments describe 4,932 movies with 3,489 attributes, director and actor attributes, and keyword identity features. A separate comment lists 1,097 TRAIN, 274 VALID, 3,202 TEST and 359 unlabeled movies. All release geometry, membership and class-schema counts still require verification from the selected complete official archive.

The [HGB README](https://github.com/THUDM/HGB/blob/ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe/README.md) announces complete public TEST data on 2 March 2023 while retaining older competition instructions. Bind the complete released archive prospectively and resolve its authority; do not silently mix the legacy public-half/dummy-label protocol with the complete release.

Retain official development and TEST membership. For seeds **1–10**, use the literal author split: shuffle the sorted development IDs with the seeded NumPy stream, allocate the first floor(20%) to VALID and the remainder to TRAIN, then sort both partitions. Freeze each descriptor once and use it for every arm. Preserve every predeclared split, including poor ones. Only TRAIN labels may enter label propagation.

The inspected [SeHGNN command](https://github.com/ICT-GIMLab/SeHGNN/blob/e92bd37d0b803457339555684f139b4c8f3e160d/hgb/Readme.md) fixes 200 epochs, width/embed 512, four feature hops, four label hops, two feature projection layers, four task layers, dropout 0.5, input dropout 0 and no residual flag. Source defaults give Adam LR 0.001, decay 0, batch 10,000 and patience 50. The actual stop is `epoch-best_epoch > 50`; selection uses strictly lower full VALID BCE, with the earliest exact tie. The literal command governs discrepancies with paper tables. Charge metapath preprocessing, all feature/label caches and the complete model pipeline.

Use **macro F1 at a fixed 0.5 sigmoid threshold** as the primary official task metric; retain micro F1, Bernoulli BCE and every label's diagnostics. Multilabel outputs need independent Bernoulli logits. Freeze the shared typed input projection, relation representation and four-member sigmoid probability mean before admission.

The stock SeHGNN loader imports TEST truth, its driver scores TEST each epoch, and `check_acc(show_test=False)` still computes TEST losses and equality checks. A future TRAIN/VALID-only provider and removed TEST diagnostics are required. Hiding printed output does not provide that isolation. The identified source is a public reference candidate; it has not been numerically qualified here.

## Lipophilicity protocol and public references

[MoleculeNet Section 3.1.6](https://arxiv.org/html/1703.00564v2) defines **4,200 ChEMBL compounds with experimental octanol/water logD at pH 7.4**. Pinned PyG provider documentation gives approximately 27 nodes and 59 directed edges per molecule. The official OGB registry defines `ogbg-mollipo` as one-target regression with **scaffold splitting and RMSE**. Keep its supplied split indices for all ten paired initialization seeds. Counts 3,360/420/420 are an 80/10/10 planning assumption, not archive measurements.

Use the official evaluator's RMSE in raw logD units, plus descriptive MAE and scaffold-level error differences. The original MoleculeNet lipophilicity experiments used a random split. Their published scores cannot be adopted as competence evidence on this OGB scaffold split. Chemprop's independently generated `scaffold_balanced` split also cannot replace official OGB indices.

The contemporary inspected reference is **Chemprop v2 D-MPNN**, [DOI 10.1021/acs.jcim.5c02332](https://doi.org/10.1021/acs.jcim.5c02332), published online 26 December 2025 and appearing in the 2026 issue. Its pinned [author source](https://github.com/chemprop/chemprop/tree/db0bb86fbad6f6ec3c414202810390d8acc8baa0) is dated 1 September 2026. The package is contemporary; directed message passing has earlier ancestry.

The literal defaults are directed bond messages, depth 3, message width 300, one width-300 hidden FFN layer, dropout 0, native norm aggregation by 100, batch 64 and 50 epochs. Regression uses MSE and TRAIN-fitted target scaling; predictions return to raw units. The learning rate warms up for two epochs from 0.0001 to 0.001 and decays to 0.0001. Restore the best VALID loss checkpoint. Supply official custom split indices and separate TEST targets from fitting; disable pretrained/foundation models and additional descriptors for this reference. No hyperparameter search is proposed.

The [official OGB GIN-virtual example](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/graphproppred/mol/main_pyg.py) supplies an additional conventional reference: five layers, width 300, dropout 0.5, batch 32, 100 epochs, Adam 0.001, full atom/bond features and mean graph pooling. It evaluates TEST every epoch and does not implement the required selected model-state replay. Preserve its model/optimizer recipe while adding owned checkpoint restoration and TRAIN/VALID-only access.

The proposed method needs a predeclared bond-aware port, graph weighting/readout, dynamic topology handling, a scalar regression head without `log_softmax`, TRAIN-only scaling, mean-member MSE and mean raw-output serving. Its classifier-response screen becomes a regression-output response screen. Those are material adaptations and must be settled using existing development evidence before confirmation admission. Count graph/index reconstruction and every screen batch. Input-only canonical-molecule overlap with used MolHIV must be reported; a distinct property label does not prove an entirely unseen graph corpus.

## Paired comparison and uncertainty

Transfer one winner frozen from the existing pilot. The compact primary family is **winner, identical bank with identity initialization, and hash-random choice from the same admissible pool**. Keep the same native arguments, losses, optimizer, directions, step and screening access. If the live initializer is selected, preserve its common 100-epoch warmup and exactly 400 continuation epochs, eight directions, both signs and step 0.01. On molecular minibatches, a screen means 17 complete TRAIN-population sweeps, rather than 17 individual graph calls.

Use **ten paired seeds 1–10** with identical role partitions and checkpoint opportunities. Include a competent independent four-model reference and the native public references above for quality/cost context. Keep each independent member's own selected state. External architectures retain their fixed native recipes and report their actual budgets. This establishes a conditional transfer test within the frozen bank; broad superiority needs the external reference comparisons and their competence evidence.

For the matched IMDB family, select the smallest complete VALID Bernoulli BCE with the earliest exact tie. For regression, select the smallest complete VALID RMSE with the earliest exact tie. Evaluate the complete VALID population each epoch, retain the common warmup's eligible best state, and restore each arm's own selected parameters, optimizer and streams. A selected epoch at or before 100 supplies no observed initializer benefit. These task-specific selectors are prospective adaptations of the binary pilot and must be frozen before admission.

Register four primary contrasts: winner versus random and versus identity on each task. Suggested practical margins for root review are **+0.005 macro F1** and **−0.05 RMSE in raw logD units**. These are prospective choices, with no demonstrated power. Freeze their final values before confirmation outcomes. Report every paired vector, mean, SD, paired 95% interval, signs and leave-one-seed-out means.

Treat the four contrasts as one family with Holm correction at 0.05. An exact paired sign-flip analysis has 1,024 assignments for ten pairs and requires an explicit symmetry/exchangeability assumption. Secondary metrics and contextual reference comparisons remain descriptive unless registered in the inferential family before opening TEST. Ten seeds assess training/split randomness on one graph or corpus; they are not ten independent graph domains.

Use paired scaffold-cluster bootstrap sensitivity for mollipo. For IMDB, an input-defined director grouping can probe sensitivity to dependence; shared actors and keywords still connect those groups, so it supplies no independent-graph uncertainty guarantee. Label categories and overlapping splits also cannot inflate the task count. Wide intervals or incomplete work remain inconclusive. Keep all failures and all adverse outcomes; do not add TEST-selected seeds or replace a dataset.

## Full task work and resource admission

For ten seeds, the three matched initializer arms require at most **52,000 full native TRAIN paths and 52,000 VALID paths**, from one common 100-epoch warmup and three 400-epoch continuations per seed, each with four members. IMDB adds 170 screen paths, selected restoration and serving. Independent references and external backbones add their complete separate costs.

Under the mollipo planning counts and batch 128, TRAIN has 27 minibatches. The same matched programme requires about **1,404,000 TRAIN batch paths**, plus complete VALID/serving paths and 4,590 screen batch calls. Native Chemprop at batch 64 requires approximately 26,500 total TRAIN steps across ten 50-epoch fits; OGB GIN at batch 32 requires approximately 105,000 across ten 100-epoch fits. These are planning work counts, not measured seconds.

The saved Peptides-func plan has 10,873 expected TRAIN graphs and a published mean of 150.94 nodes: about **1.64 million TRAIN nodes per epoch**, versus roughly 90,720 for mollipo, about an eighteenfold difference before depth, channels and preprocessing. Its controlled GINE plan uses 100 epochs with a predeclared 250-epoch ceiling. The stronger reassessed reference uses eight layers, width 160, 20-step random-walk features and 250 epochs. This supports reserving Peptides for a more compelling biological or multilabel mechanism question; it remains a valid prospective option.

Request up to **48 GPU hours per chosen task** as provisional resource bounds after the pilot gate, on an explicitly owned GPU with at least 24 GiB subject to actual qualification. These are reservations, not ETA predictions. A full-data native step and resource profile must establish whether the complete ten-seed programme fits. Include all common setup, preprocessing, graph construction, four member paths, backward/Adam, checkpoint/restoration/serving, allocator/RSS peaks and failures. Report shared setup both standalone and equally allocated. If the budget is inadequate, retain incomplete evidence and settle resources before TEST; a successful three-seed subset does not complete this confirmation.

## Release conditions

Root must reconcile the current dataset/TEST inventory and competing registrations, freeze the worthwhile pilot winner and task ports, verify official archive/split authority using input-only metadata, and bind all sources, runtime, roles, seeds, arms, selected-state rules, evaluators, statistics, margins and costs. All selected checkpoint identities must be final before one separately authorized TEST opening. No data archive, split archive, array, checkpoint, running result or research host was accessed for this recommendation.

The evidence includes two new bounded primary paper scopes, public source/protocol scopes and Chemprop title/date metadata. No complete paper was read. The ACS Chemprop paper receives bibliographic credit only; the model/protocol assessment comes from its pinned public code. Failed retrievals and one rejected wrong paper identifier receive no reading credit. These scopes identify usable references without establishing an exhaustive contemporary leaderboard search or numerical competence.
