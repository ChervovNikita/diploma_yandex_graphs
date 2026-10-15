# Frozen six-fit alternative

Status: source prepared for root review; no qualification or scientific fits
executed by this package. PubMed remains the next planned study.

## Question and six cells

Does learning private down-weight noise scales improve the deterministic
mean-served residual-only/private-BN MolBACE procedure over fixed noise and
the existing point procedure under one declared prior?

| Arm | Seeds | New fits |
| --- | --- | ---: |
| Fixed sigma = .1/sqrt(300) | 101, 202, 303 | 3 |
| Learned log sigma, same initialization | 101, 202, 303 | 3 |
| Paid residual-only F point reference | 101, 202, 303 | 0 |
| Native M1/I4 and original GNNM | original identities | 0 |

The six descriptors change only ID, method ID/settings and extension binding
relative to the paid F descriptors. M1 is exact I4 body0; I4 keeps four
independent original selectors and offsets 0/100003/200003/300007. No equipped
control substitution. Existing scores, negative outcomes and costs remain.

## Exact objective and native contract

At the complete native AtomEncoder output:

`h_m = h + U_m GELU((mu_m + exp(rho_m)*epsilon_m) h)`.

Four routes, rank8, width300. Each route draws one full down matrix per native
TRAIN minibatch, shared across its atoms and graphs. U is deterministic and
starts at zero; mu is the existing isolated-RNG orthonormal frame V0. The
entire shared native GIN remains trainable. No extra zero gate. Prior
`N(V0, s0² I)`, `s0=.1/sqrt(300)`, fixed prospectively. Fixed arm freezes sigma;
learned arm trains rho with original adapter Adam group options.

`KL_m = .5 sum_2400 [(sigma²+(mu−V0)²)/s0² − 1 + 2 log(s0/sigma)]`.

`J = mean_4(native minibatch BCE_m) + sum_4(KL_m)/(4*1210)`.

KL is added once per eligible update, with no parameter, minibatch-size or
batch-count divisor. Fixed-scale KL reduces to the mean-anchor quadratic.
One ordinary backward of J and one native Adam transition train every block.
The stable FP32 variance implementation is documented in README/METHOD.

Native five-layer width300 GIN, dropout .5, full categorical atom/bond inputs,
graph topology/operators, residual=False, JK=last, mean graph pooling,
batch32, native observed BCE masks/reduction, singleton skip law, Adam .001
defaults, and persistent paired sampler are unchanged. Private BN v3 buffers
update under each route's noisy training forward; affine parameters stay
shared. Mean serving reads the original route buffers without recalibration.

Whole official VALID151 selects the first strict maximum raw-logit-mean AUC;
no checkpoint0, TRAIN evaluation each epoch, calibration, thresholds,
stochastic-serving panels or TEST access. Minimum100/patience50/cap1000 is the
existing matched controller, not an exact paper-reproduction schedule.

## Controls and data evidence

CONTROL_COMPATIBILITY records equal common descriptors and current source
hashes, original result identities and scalar evidence. The historical native
controls lack a contemporaneous processed-cache hash. Their descriptor/source
compatibility and aligned saved VALID IDs/labels support reuse at the original
evidence strength; no historical byte-level TRAIN identity is invented.
New runtime construction verifies actual provider cache, TRAIN/VALID role-file
hashes and override absence, and records native split/source identities.
This limitation does not add reruns or change the six-fit design.

## Finite execution and reporting

Root owns publication, qualification admission and any later scientific
execution admission. Qualify both arms first with the prepared bounded CPU
lifecycle. Run admitted fits serially on the sole authorized allocation GPU,
in seed order101/202/303, fixed then learned within each seed. Maximum6,000
complete epochs and228,000 eligible optimizer transitions at38 updates/epoch.
Actual skips, epochs, transitions, failures, time, peaks and bytes must be
recorded; these ceilings are not runtime estimates. First failure halts later
admission; no automatic retry, seed pruning, horizon extension or rescue.
Cap without patience plateau is unresolved convergence.

Preserve all paired seed AUC and proper-loss contrasts against fixed, point,
M1, I4 and original GNNM. Report mean and sample SD only with all three seeds;
no route/molecule pseudoreplication, small-N significance claim or fresh VALID
confirmation claim. This is an already-exposed scaffold split. Keep native
BCE and uniform-graph saved-logit BCE reduction labels distinct. Compare mean
member and pooled AUC, NLL/BCE, Brier and measured cost without choosing new
aggregation, hyperparameters or best-seed panels.

The candidate needs learned mean-served AUC to beat both fixed-noise and point
references; the broader goal also requires native M1/I4 comparisons. More
disagreement or NLL improvement alone does not meet that ranking objective.

## Interpretation and risks

Learned versus fixed measures the ability to adapt scales under this prior,
variance KL and optimizer contract, including subsequent shared/mean/BN
changes. Point versus either stochastic arm changes both noise and mean
anchoring, so it cannot identify a noise-only benefit. This is an adaptation
of stochastic adapter ideas, not faithful BLoB, a full-network posterior
correctness result, or evidence of a causal advantage from sharing.

Training BN sees sampled V while mean evaluation uses mu. The resulting
distribution mismatch is part of the procedure; no recalibration rescue is
allowed. U can grow or rescale with V to compensate for regularization;
small down variance does not bound output perturbations. No clipping,
annealing or scale/prior grid is introduced.

Learned scale adds9,600 FP32 parameters (38,400 bytes), 76,800 bytes of dense
Adam moments plus gradients, and total private learned28,800 parameters.
Fixed has19,200. Both add immutable prior storage and four noise generator
states; one batch of CPU draws occupies38,400 bytes, with device copies and
autograd storage accounted by actual peaks. Four native route forwards and
one backward remain per eligible update. Record actual checkpoint and
optimizer state bytes separately from dense estimates and historical costs.
