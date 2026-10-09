# Prospective Bernoulli-marginal amendment

This is an explicit amendment for a verified multilabel task; the earlier categorical proposal and frozen/running recipes are unchanged. `Config.score_mode` must be declared as `bernoulli_marginal_logits`; no label shape or dataset name automatically selects it. Categorical logits and native normalized categorical log probabilities remain separate modes.

Let z_m[i,c] be native logits, p_m[i,c]=sigmoid(z_m[i,c]), and y[i,c]∈{0,1} a complete observed TRAIN outcome matrix. The served prediction is the arithmetic **per-label probability** mean `q[i,c]=mean_m p_m[i,c]`. Its proper score is

`R(q)=mean_(i,c) [-y[i,c]*log q[i,c] − (1−y[i,c])*log(1−q[i,c])]`.

No probabilities are normalized across labels. No product over labels is formed before mixing members. In particular, `−log(mean_m prod_c likelihood_m,c)` is a different objective and is not implemented.

For each entry define the native observed-outcome probability `b_m = y*p_m + (1−y)*(1−p_m)`. Its stable log value is `−BCEWithLogits(z_m,y,reduction='none')`, equivalently log-sigmoid(z_m) for y=1 and log-sigmoid(−z_m) for y=0. Member-axis log-sum-exp followed by the final mean across query/label entries computes the exact pooled mean BCE.

For assigned recipient m and family a, the two marginal pools use m's factual or a-removed prediction and the other members' **same-family a-removed frozen references**. Their observed-outcome probability is `B=(b_m+sum_other b_k)/M`. Define

`rho_m = b_m/(M*B)`.

Equivalently this is positive responsibility `p_m/(M*q)` when y=1 and negative responsibility `(1−p_m)/(M*(1−q))` when y=0. Both lie in [0,1]. For one entry the exact native-logit source derivative is

`dJ/dz_full = rho_supply*(p_full−y)`,

`dJ/dz_probe = −rho_absent*(p_probe−y)`.

Apply the same query/label mean and active-recipient mean to the derivative. The implementation differentiates the actual mixture log score. It does not stop-gradient a recomputed responsibility inside an alternative scalar loss. Peer predictions are explicitly frozen as the named state-dependent surrogate requires.

Every own, factual-pool, assigned-probe and absent-pool competence guard uses the **same mean across observed query/label entries**. `J_m=S_m−A_m` still gives `Delta S_m=Delta J_m+Delta A_m<=0` under the declared J/absent anchors. This is an aggregate TRAIN/reference fact, not a heldout or per-query guarantee.

The native integration must verify that the benchmark's label matrix represents hard observed outcomes, that all TRAIN query/label entries are supervised under this score, and that native full-input own supervision uses the declared reduction. Unknown/missing labels, soft targets, positive weighting or focal loss are unsupported contracts and require a separate amendment. The code does not infer that a missing value is negative. Loss/prior scaling and native optimizer matching must be qualified before any run; the categorical constants are not a demonstrated multilabel optimum.

This amendment uses ordinary Bernoulli proper scoring and mixture responsibilities. It establishes no new likelihood principle, posterior interpretation, semantic-source advantage or scientific result. Synthetic positive/negative entry checks passed; native Torch/backbone/data qualification remains pending.
