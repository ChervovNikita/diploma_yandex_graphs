# CMCL concise selected primary excerpts and formula transcriptions

Source: official ICML 2017 / PMLR 70 lee17b PDF, https://proceedings.mlr.press/v70/lee17b/lee17b.pdf. Exact downloaded body SHA256 5dcc65f0cc0a5a421b40d9ef2ff4822aaae86775b6d2c8af05fe4976bea85ac6. PDF pages 3–5 visually checked; page numbers below are physical PDF pages. This file retains selected excerpts/transcriptions, not the complete paper or all read pages.

## PDF page 3, §2.2: deployment problem

“For choosing a single output ... one can average the output probabilities from ensemble members trained by MCL, but the corresponding top-1 classification error rate is often very high ... This is because each model trained by MCL is overconfident for its non-specialized classes.”

The printed Fig1/2 and accompanying empirical discussion were exposed on read pages; their numerical claims are not adopted here.

## PDF page 3, §3.1, Eq3a–c: exact loss transcription

L_C = min_v sum_i sum_m [v_i^m ell(y_i,P_theta_m(y|x_i)) + beta(1-v_i^m) KL(U(y)||P_theta_m(y|x_i))],
subject to sum_m v_i^m=1 and v_i^m in{0,1}.

The paper's prose calls this a divergence “from the predictive distribution to the uniform one.” The displayed equation, complete Algorithm1 and §4.2 derivation use KL(U||P). The algorithm/formulas' orientation is the qualified operation; KL(P||U) is not substituted.

## PDF page 4, §3.2 / complete Algorithm1

The complete extracted algorithm is saved separately as cmcl_ALGORITHM1_EXCERPT.txt. The assigned candidate score is

L_i^m = ell(y_i,P_m) + beta*sum_(n!=m) KL(U||P_n).

For each batch, choose the lowest-score model for each item; take its true-label loss gradient and the nonwinning models' beta*KL(U||P_m) gradient (version0), or beta-weighted random-uniform-label CE gradient (version1). Update parameters once per batch. The stated alternating procedure does not wait for model convergence before reassignment.

## PDF page 5, §4.1: feature-sharing transcription

h_m^ell(x) = activation(W_m^ell [h_m^(ell-1)(x) + sum_(n!=m) sigma_nm^ell * h_n^(ell-1)(x)]),
sigma_nm^ell ~ Bernoulli(lambda), elementwise, lambda=.7 in the reported experiments.

“Instead of sharing the whole units of a hidden feature, we introduce random binary masks determining which units to be shared with other models.” The paper recommends lower-layer sharing and in its CNN examples uses features just before the first pooling layer. This is a masked SUM, not feature concatenation or a shared weight matrix. The mathematical hidden features couple members; author-code gradient ownership and test-time mask handling are outside this read scope.

## PDF pages 5–6, §4.2: stochastic-label gradient transcription

KL(U||P_theta) = sum_y U(y)logU(y) - sum_y U(y)logP_theta(y|x).
grad_theta KL(U||P_theta) = -E_(y~U)[grad_theta logP_theta(y|x)].

The method estimates this with uniform label samples and treats the resulting random-label CE gradient as both an implementation device and stochastic regularization. No forward finite own-CE probe, post-probe margin bank, disjoint S/R meta-loss or live derivative through assignments appears in these complete method sections.
