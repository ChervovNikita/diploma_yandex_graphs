# Minimum competent label-aware comparisons

**For root, 8 October 2026.** Future implementation plan only. No fit, scoring, retrieval, installation, running-screen change or canonical mutation. Current first-screen metrics remain unopened. Reuse saved paper/code conclusions; no new paper identity/full-paper credit.

## Minimum references

| Reference | Purpose | Minimum implementation boundary |
| --- | --- | --- |
| **Native feature predictor + C&S** | Tests whether ordinary label-residual propagation/smoothing explains gains over feature-only serving. Lowest additional acquisition cost. | Use the independently prescribed native-own-best checkpoint from each full native seed; do not choose a base epoch from candidate-corrector performance. Apply pinned author residual/correction/smoothing operations with TRAIN-only anchors. |
| **Source-faithful UniMP original operation** | Strong end-to-end label+feature message-passing single; the current late label-only corrector is not a substitute. | Preserve label-conditioned Q/K/values, masked-label supervision, gated graph-transformer layers and native all-TRAIN-label inference. Port graph/dimensions/roles explicitly or use a compatible isolated original runtime. |
| **True ordinary independent GNN4 + same correction** | Tests the broad claim against independent backbone acquisition. | Four genuinely initialized/trained native models, optimizers and coherent member-specific selectors; source same-operation correctors. Shared-B U4 does not supply this reference. |

The current four-head joint-single/U4 controls test shared-route utility within the same native trajectory. A gain over the feature-only model establishes neither that utility nor superiority to label-aware methods. Published leaderboard numbers are not experimental results for this project.

## C&S: precise role-safe port

Pinned source: `Chillee/CorrectAndSmooth`, commit `b910314a59270984f5e249462ee3faa815fc9a0c`.

Cached code: [correct_smooth_outcome_correlation.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/continuous_method_gap_search_v1/round15_graph_route_initialization/primary/author_source/correct_smooth_outcome_correlation.py:133>) — residual/outcome initializers at lines133–158; autoscale/fixed operations at lines186–236. Cached [README](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/continuous_method_gap_search_v1/round15_graph_route_initialization/primary/author_source/correct_smooth_README.md:1>) documents validation tuning, autoscale guidance and possible smoothing-only benefit on already smooth GNNs.

**Critical default:** `double_correlation_autoscale/fixed(..., train_only=False)` can concatenate TRAIN and VALID indices for both residual anchors and label resets. Here require **TRAIN only**. Never insert development/TEST truth into propagation, residuals, reset fields or class-count inference.

Build dense residuals as `one_hot(y_A,C)−p0[A]` on permitted TRAIN IDs A and zero elsewhere. Build smoothing input by resetting only A to true one-hot labels. C=10 comes from the declared schema/output width; the cached helper's full-array `labels.max()+1` must not require a heldout truth array. A thin explicit-C/TRAIN-label adapter is preferable to passing full labels to an unmodified stock driver. Do not invert the residual sign from ambiguous paper prose: pinned code uses `Y−probability` followed by addition.

Extract p0 for **all graph nodes**, including TRAIN anchors needed for residuals, from the trusted native-own-best state in eval mode. The current four-bank `reconstruct_for_serving` accepts arm-final/engineering kinds, not native-own-best: a small separately qualified feature-only reconstruction adapter is pending. Never relabel a checkpoint kind to bypass that guard or replace it with a candidate-selected state.

Retain the author's graph-normalization, propagation restart/clamping and autoscale safeguards once an exact preset is bound. Source normalization may differ from the native predictor's; declare it as method preprocessing on the same allowed graph, not hidden new data. Include both correction and smoothing work. If smoothing-only is included, name/freeze it before scoring rather than add it after an unfavorable C&S result.

**Pending, not invented:** the cached scopes contain no qualified WikiCS-specific alpha1/alpha2, normalization1/normalization2, propagation-count or scale preset. Before any evaluation, root must bind an authoritative preset adapted explicitly to WikiCS, or a small prospectively bounded VALID-only selection budget sufficient for competence. Log every configuration/cost; choose without TEST scores. No preset/grid is admitted here.

C&S returns class scores that need not sum to one. Native accuracy uses their argmax. Before NLL/Brier comparison, declare a fixed probability map—for example row normalization of nonnegative final smoothed scores with uniform zero-row fallback, preserving argmax ties. Record zero probabilities/infinite NLL honestly; do not fit an undisclosed temperature or clip away bad risk. Current frozen screen metrics/gates remain unchanged.

## UniMP: actual method and fair inputs

Authority: saved [author-code scope report](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/unimp_author_code_reference_scope_20261008_v1/REPORT.md>) and `READ_SCOPES.json`; cached [main_arxiv.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/unimp_author_code_reference_scope_20261008_v1/source/main_arxiv.py>) and [model.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/unimp_author_code_reference_scope_20261008_v1/source/model.py>). Original PGL implementation is pinned to `6dbb47c4559352ea1b1e327ee0039c47095583af`, linked by the official OGB leaderboard.

Original `main_arxiv.py` exposes `floor(0.625*n_train)` shuffled TRAIN labels and supervises the complementary TRAIN targets each epoch; inference uses all TRAIN labels. The model embeds labels into features and propagates them through three nonlinear gated transformer layers, so Q/K/values become label-conditioned. There is no inverse-inclusion scaling. Preserve these method choices in a faithful reference; do not force the GNNM corrector's half-mask/feature-only-score/scaling operation onto it and call that UniMP.

Known original arxiv CLI: width128/head, two heads, three layers, dropout0.3, 2,000 epochs, Adam0.001 and L2 regularization0.0005. The inspected call does not pass the CLI attention-dropout argument; helper default is zero. These are source arxiv settings, **not** a proven WikiCS recipe. Explicitly adapt 300 feature dimensions/10 classes and graph/node order; do not keep hardcoded128-feature/40-class assumptions silently. A role/dimension/backend port requires source qualification and competent fitting, not reproduction claims for the author's numbers.

**Critical evaluator boundary:** stock source prints TEST scores each epoch and tracks best TEST alongside best VALID. Remove TEST scoring entirely. Model label arrays must contain only allowed TRAIN labels at permitted indices; development truths belong solely to the evaluator. For this exploratory WikiCS comparison use the same full `(val_mask|stopping_mask)[:,0]` development role and strict first-max selection, declaring the role adaptation. Keep all targets hidden from literal label input when they are supervised masked queries. Full allowed graph/features remain transductive inputs.

Use native complete schedules and a predeclared modest VALID-only competence budget; do not weaken references by arbitrarily truncating to1,100epochs or changing masking. Different label exposures, steps and selection opportunities must be reported. The same human-labeled population is required; byte-identical optimization across different methods is not assumed.

UniMP_v2 is a stronger reference obligation if claims extend to strong label-aware methods: saved interfaces add virtual nodes and attention-based APPNP. Its graph-transformer/APPNP/virtual-node helpers remain unaudited, so v2 is not execution-ready or interchangeable with original UniMP. GAMLP's saved reliable-label scopes provide another strong family if a supported direction warrants broader comparisons; no new implementation is prescribed here.

## Selection and conclusion limits

Give C&S/UniMP the same permitted TRAIN label population and complete development IDs. Bind their actual fitting/selection budgets before scoring, preserve failures and full costs, and never use candidate-favorable cohorts to rank them. A same-selected-state C&S diagnostic may explain a correction effect, but cannot replace the primary native-own-best C&S pipeline by selecting from candidate epochs.

If C&S or a competent UniMP matches gains, generic label context is sufficient to explain them. Shared-route claims additionally require the current joint-single/U4 contrasts and genuine independent GNN4 acquisition. Any lead on this consumed development role still needs genuinely unused confirmation after role/history audit, then fresh independent review without a requested verdict. Neither citations, embedding distances nor incomplete references establish novelty or acceptance.

Scope: cached notes and bounded cached C&S implementation regions consulted; no papers reread, downloads or runtime. Previously saved code's propagation-loop details were inspected locally for implementation planning; no new unique method identity or cumulative literature count was created.
