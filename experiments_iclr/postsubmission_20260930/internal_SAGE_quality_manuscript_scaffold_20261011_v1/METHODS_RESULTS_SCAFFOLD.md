# Internal SAGE sharing: prediction-quality scaffold

**Priority:** better learned ensemble decisions are the primary contribution criterion. Weight storage and measured execution costs are secondary. This sentence supersedes any earlier storage-use-case pitch. The historical manuscript and canonical status remain untouched.

This is concrete draft text and an evidence map for a later revision. Penn94's full 70-bank confirmation and the endpoint-VALUE qualifier are pending, with no outcome read here. Their completion must determine the corresponding claims.

## 1. Evidence synthesis and manuscript flow

The broad conclusion remains conditional: sharing can improve predictive quality in a specified graph/backbone/training procedure. F strengthens the accuracy evidence under capable matched references. It changes the explanatory emphasis: better members explain Roman SAGE, while broader correct-alternative coverage explains the F selected predictions despite weaker average members. Neither pattern identifies a general learning mechanism.

Affinity and fixed keys do not change that broad conclusion into a successful auxiliary-specialization claim. They change and sometimes improve individual decisions, but their declared private procedures do not improve the unchanged shared baseline. Their I4 wins remain visible; the shared baseline already explains the main advantage. Keep these failed increments rather than promote another objective from mixed controls.

Recommended flow:

1. Prediction-quality motivation and original boundary GNNM definition.
2. Original five-dataset benchmark as the central broad empirical result, unchanged.
3. Roman sharing analysis, followed by the internal SAGE method and complete F evidence under stronger references.
4. One prospective Penn94 confirmation section after complete release, separating recipe utility, factor placement and fixed-state pooling.
5. Confidence, negative auxiliary results and measured costs; detailed failures and seed/class records in the appendix.
6. Conclusion limited to the quality evidence actually obtained. Endpoint-VALUE enters only after its definition, qualification and relevant empirical evidence are available.

## 2. Methods text

### Original and internal factor placement

Original GNNM creates member-specific input and output projectors around a shared graph stack. Each member enters the graph layers with its own hidden states and produces its own scores. The five-dataset benchmark uses this boundary design and averages raw logits. Its original results and procedures are retained.

The internal SAGE extension also applies member factors to dense maps inside the native graph and feed-forward blocks. It retains one set of shared native weight matrices and biases, with private factor vectors distinguishing the four routes. These factors allow routes to transform their states differently during propagation. Every route still makes a complete graph pass. Established factorization supplies the construction; the empirical issue is the predictive quality of the resulting learned bank.

### Training and stronger references

The F shared bank uses native two-block SAGE, width 128, LayerNorm and dropout 0.2. It trains the mean of four ordinary member cross-entropy losses and serves the mean of member probabilities. Pooled cross-entropy contributes no gradient, and no context auxiliary is present. Native construction, factor starts, dropout streams and the finite acquisition/selection procedure are declared with the experiment.

The matched joint untied control has four complete bodies, with initial route functions, factor/dropout starts, the mean-member objective and coherent pooled checkpoint selection matched to Shared4. The genuine independent factorized ensemble instead acquires four complete bodies with separate native starts, optimizers, unscaled own losses and own checkpoint selectors. Ordinary and factorized singles provide capable one-model references. These comparisons distinguish sharing from acquiring and averaging either matched joint bodies or independently selected bodies. Parameter ownership, optimization histories and selected updates remain part of the complete procedures.

The prospective Penn94 study separately compares the internal bank with the original boundary recipe and a boundary control matched to the current native procedure. The original comparison includes initializer, bias, binary representation, RNG, selector and acquisition differences. The matched boundary comparison holds the common native procedure fixed while changing factor placement. Alternate logit/probability fusion uses the same selected states and supplies a serving comparison, not a second independently selected training pipeline. The final frozen details belong in its protocol appendix.

### Prediction diagnostics

Coverage counts nodes where at least one member predicts the correct class. Pooling losses count covered nodes the served average gets wrong; aggregation-only corrections count correct pooled decisions without a correct individual member. Paired repairs and harms compare the two complete selected procedures on the same evaluation nodes. These quantities describe useful decisions. They neither estimate covariance by themselves nor show that weaker members have collapsed.

Keep the original essential factor equation. No new basic loss or averaging equations are needed for these paragraphs.

## 3. Results text already supported

### Original five-dataset result

The original benchmark compares one model, four independent ordinary models and boundary GNNM across eight backbones on Roman Empire, Amazon Ratings, Minesweeper, Questions and Tolokers. Its recorded means favor GNNM over the four-model ensemble in 28 of 40 comparisons and over a single model in 33. SAGE and both graph-transformer variants favor GNNM on every dataset, while other backbones include losses. Preserve the complete heatmap and absolute tables, the accuracy/ROC-AUC distinction, intended validation-selection procedures and historical provenance limits. The cells reuse five graphs and are not 40 independent tasks.

### F: a useful shared bank under stronger controls

In the complete encountered-WikiCS F study, Shared4 reaches 80.6219% raw accuracy. It gains 0.720516 percentage points over matched joint untied4 and 0.537227 points over genuine factorized I4, with positive differences at all three paired seeds. It also exceeds both capable singles. The frozen raw accuracy gates pass.

The member pattern differs from the original Roman SAGE analysis. F Shared4 has lower mean member accuracy than both ensemble controls, but covers more correct alternatives. Across three repeated readouts, it has 13,731 covered cases versus I4's 13,506. The additional 225 covered cases are partly offset by 141 additional pooling losses; one aggregation-only correction leaves 85 more correct pooled decisions. Direct pairwise counts are 645 repairs and 560 harms. Thus weaker average members coexist with better served predictions. These counts repeat the same 5,274 development nodes and are descriptive, not independent node trials.

The confidence result remains adverse. Mean calibrated NLL harm to I4 is 0.021448, exceeding the frozen 0.02 cap; every per-seed cap passes. The full frozen accuracy-and-confidence conjunction therefore fails. Favorable calibrated Brier does not replace that NLL rule. Present the accuracy gain and confidence tradeoff together. F is three optimizer seeds on encountered development, with calibration after native checkpoint selection; it is not unused whole-pipeline confirmation.

### Auxiliary additions did not establish an increment

Continuous private affinity scores 80.565036% versus its unchanged shared baseline's 80.571356%: −0.006320 points with mixed seed signs. It also loses to the common mixture by 0.025281 points. Its raw I4 wins are real, but mean calibrated NLL harms of 0.036786 and 0.036650 exceed the protection caps. Private affinity acquires 310 new covered cases and removes 316; most of its repairs use alternatives the baseline already possessed. The net raw decision change is 333 repairs and 334 harms. This does not establish a net acquisition or private-assignment advantage.

Private fixed keys score 80.533434% versus their unchanged shared baseline's 80.571356%: −0.037922 points, negative at every seed. All five attribution checks fail. Against BASE, 163 repairs are outweighed by 169 harms; net coverage falls by seven while pooling losses improve by only one. I4/single accuracy wins remain, with failed mean NLL protection against both equipped I4s. The small mixed moment/peer controls are not frozen successful methods.

These auxiliary studies use their own declared baseline records; their numbers do not replace F's selected shared state or form a new cross-study ranking. Close the exact additions without a rescue grid. Their failures do not diagnose projection absorption, gradient conflict or hidden-state collapse.

## 4. Penn94 results placeholders

Replace these only after all declared acquisition, calibration and scoring records close. Preserve failures and every block.

| Planned comparison | Raw accuracy effect / adjusted inference | Confidence | Interpretation |
| --- | --- | --- | --- |
| Internal shared4 vs original boundary recipe | **PENDING** | **PENDING** | Whole-recipe utility; initializer/bias/binary/RNG/selector/budget differences disclosed |
| Internal shared4 vs genuine I4 | **PENDING** | **PENDING** | Complete bank versus separately acquired independent bodies |
| Internal shared4 vs ordinary M1 | **PENDING** | **PENDING** | Quality beyond a capable ordinary single |
| Internal shared4 vs factorized M1 | **PENDING** | **PENDING** | Quality beyond a capable factorized single |
| Internal vs matched boundary factors | **PENDING** | **PENDING** | Trained placement procedure under common native conditions |
| Same-state probability vs logit fusion | **PENDING** | **PENDING** | Serving effect on fixed acquired predictions |
| Fixed LINKX context | **PENDING** | **PENDING** | Task context; no optimized-paper reproduction claim |

Report all ten paired bank-level differences under the frozen inference/multiplicity rule. This conditions on one graph and allocation, not a population of unseen graphs. An observed practical mean gain and positive-effect inference do not establish that the true gain exceeds the practical margin. Failed confidence protection accompanies an accuracy/confidence tradeoff; member weakness creates no extra quality veto.

**Endpoint-VALUE:** qualifier result **PENDING**; predictive/mechanistic result **NOT ESTABLISHED**. Successful implementation qualification alone would not fill a prediction-quality or mechanism gap.

## 5. Concrete scientific issues for the revision

- **Supported contribution:** state the quality comparison that closes successfully. Current F supports an accuracy-oriented sharing signal; Penn94 must determine its prospective scope and original-recipe comparison. The main contribution cannot yet be a successful new auxiliary or general mechanism.
- **Attribution:** keep original-versus-current recipe effects separate from matched placement and same-state serving. F's matched sharing result does not isolate factor placement or explain why its optimizer finds the selected predictions.
- **Cleaner decisions:** report acquired/removed alternatives where available and repairs/harms, including new common wrong rivals. Large coverage alone is insufficient if pooling loses the added alternatives or creates more errors elsewhere.
- **Confidence:** preserve NLL/Brier disagreement and the exact failed rules. Strong accuracy with weaker probability quality is a material scientific result, not a hidden success or a reason to erase the accuracy evidence.
- **Original breadth:** retain all five-dataset results, Roman member improvement, GAT's mixed result and the unfavorable partial-sharing/settings checks. New single-graph evidence cannot rewrite them or establish universal transfer.
- **Resources:** weight savings remain measured secondary evidence. F's recorded acquisition is greater than its ensemble controls; retain acquisition, restoration, serving and memory scopes separately, with no latency promise.

Read basis: saved paper map, complete F and fixed-key reports, and complete continuous-affinity report. No reviewer invocation, blanket literature reread, live Penn94 output, data array/checkpoint, remote or model execution occurred. This new subfolder contains draft planning text only.
