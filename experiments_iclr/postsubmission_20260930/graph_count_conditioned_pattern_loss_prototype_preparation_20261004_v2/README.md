# Count-conditioned pattern loss prototype — source only

This fresh packet implements the conditional-Bernoulli auxiliary algebra and
comparison controls from the sealed literature assessment. It does not change
any existing science/qualification source, the frozen P0/J_P/F_P/C_mu screen,
or its completed outcomes. There is no fit or dispatch here. Candidate modules
and CPU oracles have not been imported, compiled or executed during preparation.

## Post-TEST status

All choices in this successor prototype were made after root reported official
TEST values: private67.29%, native64 66.44%, independent4 67.63%, and primary
private-minus-pooled+0.2236percentage points with uncertainty. These values are
root reports, not calculations or heldout file reads by this source task. This
is post-TEST exploratory development. Reusing that consumed TEST later would
not provide fresh independent confirmation. No new arm or broader training is
added to the frozen screen by this packet.

## Attributed ingredients and claim limits

Conditional Bernoulli weighted subsets, elementary-symmetric normalization and
cardinality inference are classical: the saved scoped Chen and Liu (1997)
and Tarlow et al. conclusions are reused. GRAN is the closest saved graph
whole-pattern mixture/responsibility prior. Contextual/latent/autoregressive
graph laws from Graphite, SIG-VAE, SeeGera, DiGress and KREPE, and NCNC-informed
refinement from PIFM, remain explicit ancestry. The proposed single is an
attributed binary fixed-count adapter, not a native reproduction of those
models. No paper retrieval or new primary reading occurred. Source programming
or fabricated numerical QA is not methodological novelty evidence.

## Conditional loss implementation

`conditional_loss.py` supplies a differentiable finite-state logaddexp ESP
normalizer, using complement degree min(k,n-k). Every stored recurrence state
is reachable; it never differentiates logaddexp(-inf,-inf). Conditional NLL
centers each member/side at its first slot and complements the likelihood
itself, avoiding cancellation between two large total-logit sums. At k0 or kn,
including the empty side, an empty-slice sum gives connected exact zero loss
and gradient even if summing the finite input logits would overflow. There is
no probability clipping, epsilon likelihood, sampling or subset truncation.

Mathematical common member/side logit offsets cancel. Finite precision remains
finite precision: huge mixed logits can still overflow a centered subtraction
or log-weight sum, and autograd has no universal large-support accuracy bound.
The fabricated shift checks use dyadic fixtures so adding the declared common
shifts does not silently change float32 relative logits through rounding.

The ragged adapter consumes exact existing `pattern_forward` details['t']
(log odds from the normalized native completion clamp), split into native left
and right residual slots. It accepts their original query rows and one detached
1D TRAIN observation-membership pattern from the existing ObservationTeacher.
It derives k_left/k_right internally, preserves original row/candidate order,
and adds no mask, pool, negative sampler or teacher-count input argument.

Outputs are J_K (one uniform member mixture across both sides), J_K_sep
(independent whole-side member mixtures) and W_K (mean component conditional
NLL). J_K_sep matches the whole per-side conditional laws at fixed parameters.
W_K is a responsibility-learning contrast, not the analogue of frozen F_P.
Conditioning old unconditioned J_P would reweight members by count evidence;
that law is not the uniform J_K implemented here. Fixed-count conditioning
removes a count-only C_mu potential and its direct signal; actual frozen C_mu
is unchanged and is not offered as a capable conditional-identity single.

Each query divides by max(n_left+n_right,1). The caller retains every query,
including empty/extreme cases, its all-query mean and coefficient1. A future
admitted integration must preserve its positive/negative query reductions,
complete native neural calls and optimizer schedule. Zero conditional loss
does not authorize skipping native scorer/empty calls or changing the budget.

TRAIN zeros mean unobserved in TRAIN, not verified latent nonlinks. Non-TRAIN
teacher authorities, member-specific label axes and nonbinary/differentiable
labels are rejected. The low-level label contract is not a cryptographic proof
of provenance: an actual adapter still needs root-authenticated TRAIN data and
mask authority before use. No VALID/TEST count or member label pattern may feed
the loss. TRAIN masking/count conditioning is absent from downstream link
inference. Native target BCE, count-free completion and raw-logit serving keep
their own route; conditional pattern probabilities are not served link scores.

## Capable conditional single core

`conditional_single.py` defines one auxiliary MLP,523->64->1 with ReLU and33601
additional parameters. It takes193visible slot features,194visible query
features,128prefix identity features and8progress/count scalars. Native unary
context is first-slot centered per query/side without using labels. Other
features are one native width64 predictor's visible endpoint/candidate vectors,
endpoint product and visible support means/sizes. Counts and previous teacher
bits occur only in `training_nll`, never its label-free visible-context builder
or a target-serving input.

The exact law visits ascending unique native CSR candidate IDs, all left slots
then all right slots, under the original endpoint orientation. Prefix features
summarize previously selected identities; no current/future teacher bit identity
is an input. It uses logistic choices unless the remaining budget is zero or
all remaining positions must be selected, when the unique choice has probability1.
Every slot still evaluates the head. This teacher-forced product is normalized
on every feasible count-constrained support. It can express the matching-side
example that a count-only potential cannot. Canonical ordering is disclosed;
no endpoint-exchange invariance or orientation averaging is claimed.

`CONTRACT.json` freezes auxiliary dimensions, initialization, optimizer-group
proposal, feature construction, order and normalization before any CPU outcome.
One native single backbone/target integration remains separate and unqualified.
The head is not a prediction ensemble. Its extra parameters and sequential work
are charged, so competence/information equality is not a parameter/wall budget
match. No inference API requiring teacher counts is introduced.

## Disabled fabricated CPU qualification source

`oracles.py` defines exhaustive small-support checks independent of the ESP
recurrence: explicit feasible subsets produce log normalizers, marginals and
analytic pi-z gradients. It includes508float32/64 CB pattern cases (supports0-6,
small/large finite logits), normalizer gradients, shift invariance, exact
extreme zeros including overflowing-sum inputs, uniform log-binomial baselines
and an ESP Hessian versus exhaustive subset covariance. It defines675two-sided
mixture pattern cases (supports0-3, M1/2/4), analytic responsibility/side/mean
loss gradients, bounds, collapsed components, unique-side degeneration, ragged
empty/extreme/all-query reduction and teacher-role/member-axis rejection.

For the single core it enumerates225feasible patterns in100support/count groups,
checks total mass1 and zero normalization gradient for all head parameters,
checks exact forced-case zero gradients and visible-unary shift invariance.
A separate manually assigned fabricated head matches the shared-mixture2x2/k1
law and its analytic parameter gradient; independent side mixtures remain1/4.
These are engineering algebra/implementation checks, not representative graph
experiments, native-code parity, predictive tests or methodological validation.

`qualify_cpu.py` imports numerical modules only after an exact external root
release, independent source-only PASS, source/closure hashes and CPU runtime pin
are authenticated. Templates are DISABLED with no authorized stages. Its one
fresh prospective CPU attempt has900seconds and4GiB RSS caps,2CPU threads,
no optimizer, no data/state loader, no GPU API and no retry. It hides CUDA only
in that future CPU process. Failure retains incurred comparison/case receipts.
No external release or execution is created by this source packet.

## Resource obligations

The rolling ESP table uses O(M*r) explicit forward entries per side, but training
autograd retains an O(M*sum n*r) recurrence graph plus Python/Tensor/node metadata.
The current reference loops over queries/slots/counts and performs host `.item`
counts and schema/finite reductions; it is not a vectorized GPU implementation.
The single executes a width64 head sequentially for every slot, including forced
ones, and retains native neural graphs and prefix dependencies.

`PROSPECTIVE_COMPLEXITY.json` gives arithmetic scenarios for65536queries per
population, without reading a graph/count histogram. For n100/k50 on both sides,
M4 requires about1.98billion scalar recurrence updates per population. Naive
two-vector saved-value accounting alone is about15.8GB float32, before graph
metadata/native work. For100total slots/query, the single has6.55million head
calls and about220billion affine MACs per population/update, plus context arrays
and native work. These are illustrative formulas, not measured allocator bounds
or evidence the protocol fits. Positive and negative populations both pay costs.

Small CPU oracle success, if later obtained, would not establish full65536-record
feasibility. Exact native empty-call/gradient/state proof, complete batch resource
qualification, a separate frozen scientific successor and independent data-role
release remain necessary. No batch/degree/splitsize/epoch/precision/tolerance
change is authorized to make the prototype fit.

## Disabled separately sealed v2 successor

This successor groups exact conditional-loss arithmetic by(n,r=min(k,n-k)), with connected r0 zeros, explicit categorical r1/complement and vector reachable ESP cells for higher r. It preserves the attributed J_K/J_K_sep/W_K laws, all-query normalization, teacher/slot/serving contracts and native complete schedule. Source grouping is not a full-batch speed/memory certificate. PROSPECTIVE_COMPLEXITY.json requires a complete native census and inclusive resource qualification before that claim.

F02 is corrected by one negative/out-of-range endpoint guard; the remaining single AST is v1-exact. The finite capable-single head and sequential law are unchanged. L01 receives meaningful unexecuted source coverage: actual visible-context and progress/prefix reconstruction, float32/64 ragged per-pattern parameter/h/unary gradients, full manual head derivatives, normalization and endpoint rejection. The v1 oracles are byte-identical; their existing staged attempt is untouched. New QA is disabled at a fresh execution-root-v2 and requires separate exact review/supervisor/root/runtime admission. No numerical code, data, model/state/array or server was accessed by this source task, and frozen science remains unchanged.
