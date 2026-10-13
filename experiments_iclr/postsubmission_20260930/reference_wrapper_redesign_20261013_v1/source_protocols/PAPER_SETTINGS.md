# Frozen paper and author settings for the reference-wrapper redesign

This is a local document/source audit. It defines a finite preparation roster and reported references, without training, remote access, HPO, or a score-reproduction claim.

## Closed roster

The primary proposal has **31 fixed setting cells**:

- **8 PolyFormer:** CiteSeer, PubMed, Coauthor CS, Coauthor Physics × Mono/Cheb.
- **16 OGB:** MolHIV/MolBACE × GCN, GCN+virtual node, GIN, GIN+virtual node × two predeclared dropout presets,0 and0.5; full features only.
- **4 Polynormer:** Photo/WikiCS × pure no-ReLU/Polynormer-r, using the paper-stated horizon and core dimensions.
- **3 Collab:** author NCN, NCNC, and BUDDY native benchmark presets.

The roster also retains **4 separate Polynormer release-horizon reference cells** and **2 HGB score-evidence cells excluded from primary comparisons**. Thus the CSV/JSON has37 rows. The four release-horizon references are not quietly added to the primary31, and HGB is not an executable primary admission.

Every primary setting compares only native M1, original boundary GNNM, and genuine ordinary I4. F1, internal all-map/factor variants, jointly trained untied banks, equipped models, and capacity controls are removed. Independently selected ordinary bodies define I4; a joint cohort optimizer/selector is not an independent ensemble.

All new runs on `roman-empire`, `amazon-ratings`, `minesweeper`, `questions`, and `tolokers` are excluded for every method. The setting roster is not an all-models × all-datasets expansion. Additional source-supported datasets and models found in archived scripts are outside this closed proposal.

`paper_setting_roster.csv` and `paper_setting_roster.json` contain exact per-cell parameters, epochs, patience, warmup, seed/split law, source entry points, reported references, and comparability limits. Every cell remains `preparation_pending`; that label does not certify a working wrapper, matched provider, numerical parity, or resource qualification.

## Exact settings and reported references

### PolyFormer: eight original-task author presets

The literal node-classification script fixes ordinary/attention learning rates and decay, hidden width, classifier dropout, PolyAttn/FFN dropout, heads, FFN width, q, multi, degree and blocks separately per dataset/basis. Training uses grouped coupled Adam, maximum2,000 epochs and native patience250. There is no warmup or new search. The author source uses random60/20/20 roles and ten fixed split seeds; global seed42 initializes an advancing RNG across the ten repeats.

| Dataset | Mono reference | Cheb reference |
|---|---:|---:|
| CiteSeer |82.37 ±0.65|81.80 ±0.76|
| Coauthor CS |96.49 ±0.09|96.49 ±0.17|
| PubMed |91.01 ±0.41|90.68 ±0.31|
| Coauthor Physics |98.42 ±0.16|98.08 ±0.27|

These are accuracy percentages from PolyFormer paper Table5; **± is a95% confidence interval over ten runs, not standard deviation**. Exact script settings are copied into the CSV/JSON. No broadened practical grid or v29 default is used.

Source: `coordinate_source_independent_review_v1/strong_backbones_v1/sources/polyformer_code/node_classification/node_classification.sh` and `training.py`; paper: `sources/polyformer_paper.txt`, Table5. Changing these original roles into SELECT/CAL, training on fewer labels, or changing the ten-repeat RNG law prevents an original-benchmark protocol claim.

### OGB: two fixed published dropout presets, unknown paper-winning configuration

Both source-supported molecule tasks use full categorical atom/bond features, native BatchNorm, hidden300, five layers, batch32,100 full epochs, Adam LR0.001/decay0, residualfalse, JKlast and mean pooling. The native default dropout is0.5. The OGB paper explicitly tuned dropout over `{0,0.5}` but the retained paper does **not** disclose the selected dropout per reported table cell. Therefore the redesign declares two fixed named presets, not an HPO task, and does not choose a winning preset or claim either one exactly matches the paper-winning configuration.

| Architecture | MolHIV ROC-AUC % | MolBACE ROC-AUC % |
|---|---:|---:|
| GCN |76.06 ±0.97|79.15 ±1.44|
| GCN+virtual node |75.99 ±1.19|68.93 ±6.95|
| GIN |75.58 ±1.40|72.97 ±4.00|
| GIN+virtual node |77.07 ±1.49|73.46 ±5.24|

Values are mean ± standard deviation over ten runs: saved OGB primary paper Table15/full HIV and Table23/full BACE, uncertainty statement text1805, dropout statement text7194. They are paper references to a tuned-between-two result, not predictions for each fixed preset. Full/simple feature configurations remain separate; only full features are adopted here.

Preserve official scaffold TRAIN, **whole VALID**, and TEST. Do not substitute a SELECT/CAL subdivision. Native source selects the first validation ROC-AUC argmax after100 epochs and stores the corresponding metric, without a selected-model checkpoint. The wrapper needs a source-preserving selected-state artifact for the boundary/I4 comparison; implementing that artifact is pending and must not change the original selection rule. Source singleton skipping, observed-label BCE, native shuffled TRAIN evaluation and its RNG advancement remain part of the recipe.

The source provides no native seed or repeat flag. Ten externally declared independent seeds are an explicit comparison/repetition adaptation. They cannot be called a published seed protocol merely because the numeric paper references average ten runs.

### Polynormer: paper and release are explicit different presets

The paper distinguishes **Polynormer** without ReLU from **Polynormer-r** with ReLU. The release README says to remove the ReLU functions for the former. This is not a new ELU substitution; an ELU replacement preset is not admitted from these table rows.

| Dataset / variant | Paper accuracy % ± std,10 runs | Paper horizon | Literal release horizon |
|---|---:|---|---|
| Photo / pure |96.11 ±0.23|warmup200 / total1,000|local200 + global1,000 =1,200|
| Photo / ReLU |96.46 ±0.26|warmup200 / total1,000|local200 + global1,000 =1,200|
| WikiCS / pure |79.53 ±0.83|warmup100 / total500|local100 + global1,000 =1,100|
| WikiCS / ReLU |80.10 ±0.67|warmup100 / total500|local100 + global1,000 =1,100|

Paper Table5/core text fixes effective hidden512, eight heads, LR0.001; Photo has local/global layers7/2 and dropout0.7, WikiCS7/2 and0.5. The release fixes Photo per-head64 ×8, WD0.00005, input dropout0.2; WikiCS per-head512 ×1, WD0, input dropout0.5. The latter differs from the paper's eight-head statement as well as its horizon. Release global dropout defaults to local dropout when unspecified. Beta-1, sharedQ/K and these auxiliary settings are named source defaults, not new tuned choices.

The primary four rows follow the paper's core settings/horizon, with auxiliary settings explicitly labeled as release-sourced. They remain preparation pending rather than certified full-paper equality. The four release-native rows separately preserve its different head/horizon choices and can support the source-horizon reference track. Longer three-arm runs are separate experiments; their outcomes must not be compared to paper reference numbers as if the training schedule matched.

Photo's release reuses archived split0 for Amazon/Coauthor datasets; bind the exact60/20/20 archive before admission. WikiCS source enumerates all official train-mask columns, uses `val_mask OR stopping_mask` for validation and shared official TEST. Preserve the official mask axis, without substituting one handpicked mask. A ten-seed comparison and the official-mask protocol need distinct identities; do not silently replace masks with optimizer seeds or infer that there are only ten official masks.

Paper score locator: `sources/polynormer_paper.txt`, Table1; schedules: AppendixH/Table5. The README's96.67 Photo and81.20 WikiCS are first-run ReLU examples, not the paper's mean over ten runs.

### Collab: retain the native published benchmark topology

**NCN/NCNC:** exact author collab command fixes GCN encoder H64/L1, encoder LR0.0082, predictor LR0.0037, dropout fields x0.25/tail0.05/encoder-edge0.25/predictor-edge0/predictor0.3/encoder0.1, batch65,536 and100 epochs. Preserve source normalization, residuals, maskinput, source decoder/completion, whole-minibatch target removal and dropped tail. `--runs10` calls `set_seed(run)`, so source seeds0–9 are explicit.

The native command's `--use_valedges_as_input` adds VALID edges at TEST. Current complete-TRAIN-only TEST variants and HeaRT CiteSeer tasks are protocol adaptations, not this native Collab reference. NCN is `cn1`; NCNC is `incn1cn1`. No numeric NCN/NCNC table cell was verified in retained bounded sources, so mean/std are null with a reason. The README points to paper2302.00890; missing numeric values are not filled from memory or an unrelated leaderboard.

**BUDDY:** exact retained author command uses year2007, native H1024/batch1024/100epoch defaults, Adam0.02/WD0, feature dropout0.05, label dropout0.1, normalized features and cache reuse; other native parser defaults are recorded explicitly. `--reps10` calls `set_seed(rep)`, seeds0–9. Native data code filters TRAIN to year≥2007 and uses TRAIN+VALID topology for TEST. Its native evaluation batch1,000,000 and extraction batch11,000,000 differ from the earlier bounded-memory preparation choices. The old year0/training-only-test packet is not this source protocol. Any necessary batch change will be a separately labeled preparation adaptation.

The retained BUDDY README describes the repository as a reimplementation with evolving defaults; freeze commit `3562d94a07d1166faa0949030824bf75ad9bb2c4`. It points to ICLR2023 paper `m1oqEOAozQU` but contains no exact numeric result table. Reference mean/std stay null with a reason. No leaderboard submission or value retrieval occurred.

## HGB evidence is excluded from primary head-to-head comparisons

The saved SeHGNN paper Table3 reports IMDB macro-F1 67.11 ±0.25 / micro-F1 69.17 ±0.43. Its HGT DBLP baseline row reports macro-F1 93.01 ±0.23 / micro-F1 93.49 ±0.25; this is a secondary HGB baseline reference, not a primary HGT paper result. The bounded passage does not establish the exact type of ±, so the CSV stores a quoted dispersion rather than inventing an SD label.

HGB reserves70% of target nodes for hidden online TEST. SeHGNN paper averages five local partitions of the public30%, split24% TRAIN/6% VALID. A local inner holdout from that30%, a SELECT/CAL split, or transferred role labels cannot reproduce the online TEST score. These two rows are evidence only and excluded from primary comparisons until official full evaluation is obtainable. No submission is authorized or performed here.

Source HGT's parsed LR0.005/dropout0.5 do not describe effective training: optimizer uses AdamW decay0.0001 plus OneCycle(maxLR0.001,total300,pct_start0.05), and the HGT layer constructor keeps its dropout0.2 default. Source DBLP uses H64/L3/heads8, cap300/patience30. SeHGNN's paper confirms H512, Adam0.001/WD0 and IMDB25 feature/12 label channels; retained source supplies cap200/stale>50 and native AMP. These are useful frozen-source facts without an online-score equivalence claim.

## Transfer-only or unsupported old cells are not admitted

The v29 experiments were standardized development comparisons, not a paper roster. Their default practical LR/dropout grids, altered roles and longer source continuations do not become paper settings by removing HPO records.

- SAGE/GCN/GAT generic H128/L2/LR0.001/dropout0.2 ports on new node tasks: no exact published dataset/preset established by these records. A baseline score in someone else's paper does not recover the producing configuration.
- Source MeanSAGE, endpoint FiLM, Rank1-BNN and private label/response/control families: native/source mechanics are documented, but the current datasets/roles or operators do not establish the requested original published backbone/task/preset cell.
- Practical residual/LayerNorm GINE H256/L5 on MolHIV/MolBACE is distinct from the adopted official OGB BatchNorm GIN/GCN release.
- General/bundle/Bayesian sheaf on Twitch DE is a transfer task relative to the retained original paper source datasets; the current cold-geometry/control presets are not original published task settings.
- PatternTwin/cardinality and GT-sep capacity designs are private/prototype/control recipes rather than the adopted published native backbone roster.
- Torch UniMPv2 on Photo retains disclosed Paddle-mask/provider adaptations; an exact original paper task/score/settings match is unresolved in the bounded retained evidence.
- All equipped, F1, all-map, joint-untied and capacity-control rows are removed by the user's redesign, independently of whether their engineering was complete.

These may retain historical source evidence; they are not fallback entries in the closed primary roster. Unsupported settings stay unsupported rather than receiving invented paper numbers or a new default grid.

## Seed and reporting rule

Use ten paired comparison cohorts. Preserve explicit source seed IDs where given: NCN/NCNC and BUDDY0–9. Where independent paper optimizer seed IDs are not published, ten deterministic default IDs are an explicitly labeled adaptation. PolyFormer's source seed42/ten advancing repeats and literal split seeds, Polynormer's default42 and official masks, and HGB's five source partitions are documented separately rather than being rewritten as the same RNG law.

Hold architectural settings and optimizer/LR/WD/dropout fixed across the three primary arms for each named preset. Attribute original boundary scaling and declare any necessary source-task/optimizer/seed adaptation. Retain separate source-horizon replication and longer same-backbone comparison labels. Do not select a winning dropout/preset, pool disparate schedules into one score, or claim a measured paper reproduction before actual results exist.

`SOURCE_AUDIT_RECEIPT.json` binds the source documents by bytes/SHA. `build_roster.py` reconstructs this finite metadata roster. No package imports, models, numerical forwards, fits, scoring, remote operations, or Git mutations were performed by this source audit.
