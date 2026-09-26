# Initial cross-member graph-gradient diagnostic

This is a post hoc, train-label-only calculation from all 12 already exported
four-member graph-gradient arrays in `initial_update_raw_gradients`. The
complete 4×4 Gram matrices, source hashes, and per-member quantities are in
`INITIAL_MEMBER_GRADIENT_GRAM.json`. No model was trained or test label read
for this calculation.

Let (L_m) be member (m)'s training cross-entropy, (g_m=\nabla_\theta L_m)
for the graph-stack parameters at the matched initial state, and
(L=\frac14\sum_m L_m). With an infinitesimal SGD step of size (\eta),
the graph-only first-order changes in average training loss are

\[
\Delta L_{\rm tied}/\eta=-\frac1{16}\left\|\sum_m g_m\right\|^2,
\qquad
\Delta L_{\rm untied}/\eta=-\frac1{16}\sum_m\|g_m\|^2.
\]

Thus the difference is
(-\sum_{m\ne n}g_m^\top g_n/16\). The untied expression uses the
mean-loss gradient (g_m/4) for each private graph stack, at the **same**
learning rate. This comparison also changes the effective per-copy graph
step scale. Shared boundary-parameter contributions are present in both
systems at initialization and are omitted from this graph-only expression.

| Graph | Mean off-diagonal cosine, 3 seeds | Tied/untied graph-only initial descent magnitude, 3-seed mean | Negative pairs / 18 |
|---|---:|---:|---:|
| Cora | 0.098 | 1.29 | 0 |
| WikiCS | 0.120 | 1.36 | 7 |
| Actor | 0.514 | 2.42 | 0 |
| Filtered Chameleon | 0.559 | 2.68 | 0 |

All 48 member-specific graph-only total directions are descending at this
initial point. These numbers quantify local gradient alignment; they do not
predict held-out accuracy or finite-step AdamW dynamics. The original study
uses AdamW, so this calculation should be presented as an explanatory
diagnostic, not evidence for a new optimizer or a causal performance claim.
