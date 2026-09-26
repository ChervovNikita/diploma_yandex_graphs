# Initial AdamW update order: train-label-only diagnostic

This separate diagnostic was specified before opening the 96-cell SYNC study's
validation or test outcomes. It is not a model-selection experiment. Run all
four frozen graphs (Cora, WikiCS, Actor, official filtered Chameleon), seeds
0, 1, and 2, exactly once on CPU with the default learning rate 0.001, AdamW
betas (0.9, 0.999), epsilon 1e-8, and zero weight decay. Use the frozen
width-128, two-block SAGE graph and split-0 training labels. The loader reads
the full label vector and test mask to check the frozen dataset fingerprint,
but excludes test indices and labels from the returned model bundle. The
diagnostic uses only training labels for gradients and statistics. Keep the
model in training mode, including dropout. Match the tied and SYNC
constructors and the RNG state before their first forward passes.

For one graph-weight coordinate, let `g_m` be the gradient of member `m`'s
unscaled training cross-entropy. The TIED first AdamW graph update is
`d_T = -eta * mean(g_m) / (abs(mean(g_m)) + epsilon)` at fresh moments.
The SYNC graph update averages four separately normalized first updates:
`d_S = mean_m[-eta * g_m / (abs(g_m) + epsilon)]`. The SYNC code's mean-loss
autograd gradient is `g_m/4`, so it multiplies only private graph gradients
by four before AdamW. Both formulas here use zero decay. These are graph
parameter update vectors, not validation or test scores.

For every graph/seed row, record graph-parameter coordinate count, cosine
similarity of `d_T` and `d_S`, fraction of all coordinates with strictly
opposite nonzero signs, fraction with zero SYNC update, both graph-update L2
norms, the SYNC/TIED norm ratio, relative L2 update difference, and the
graph-parameter components of the local first-order
training-loss directional products `mean(g_m) dot d_T` and
`mean(g_m) dot d_S`. Also perform real matched
one-step TIED and SYNC AdamW updates and compare their graph-parameter
changes with the formulas. Require the maximum absolute formula-versus-real
error for each arm to be at most 1e-5 and matched post-step RNG states.
Report all 12 rows. No row is chosen according to outcome.

The directional product is a local first-order calculation, not a guarantee
about the finite-step loss change. The diagnostic describes update
arithmetic at initialization and cannot explain later generalization or
predict which arm will win on held-out nodes. AdaTask already uses separate
Adam moments for tasks; this is an application to four graph ensemble paths,
not an optimizer novelty claim.
