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
The head is not a prediction ensemble. Its extra parameters and full-slot head work
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

The CB batch path is byte-identical grouped core-v2: each side is grouped by
(n,r=min(k,n-k)); r0 is connected zero, r1 is categorical/complement, and r>=2
uses reachable [M,g,r+1] rolling tables. It still dispatches once per slot in
each genuine group and retains differentiable recurrence operands. This
successor changes S_K teacher forcing, with local per-query segmented cumulative
feature sums grouped by visible support length, tensor count/progress assembly,
segment-start unary centering and one unchanged batched MLP. Every slot row,
including forced choices, remains evaluated. Head MACs and all-query reduction
remain unchanged; fewer Python calls do not establish fit.

The scoped actual TRAIN-census metadata audit reports positive coupling with at
least one genuine subset at3.4635057636% of query occurrences, and both genuine
at1.3599171358%. The unchanged CB implementation still entails1413-1605genuine
groups and121369-139724Python group-slot iterations per positive full batch.
This audit checks root projection/receipt arithmetic, not raw tensors, useful
association, predictive quality or numerical/native resource behavior. The
negative conditional identity signal is nearly absent; all reductions/native
negative work stay mandatory.

The prior illustrative single array/MAC figures in PROSPECTIVE_COMPLEXITY remain
applicable to evaluated rows, while millions of scalar MLP calls are historical.
One batched 523-feature input for6.55million rows is about13.7GB float32 before
prefix/group buffers, hidden activations, gradients and the retained native
neural graph. Local group cumsums avoid floating subtraction of unrelated query
totals, but their floating reduction order and batched GEMM/segment reductions
still require the unchanged tolerances and native working-dtype assurance.
Distinct support-size dispatch and grouping buffers remain charged.

Small fabricated CPU QA, if later admitted, cannot establish full65536-record
native feasibility. Complete native producer/teacher/empty-call/gradient/state
proof and inclusive forward/backward/wall/RSS/no-reset CUDA allocation/reservation
qualification are required before science. No batch/degree/splitsize/epochs/
precision/tolerance change or forced-slot omission is authorized to fit a cap.

## Historical separately sealed v2 successor

This successor groups exact conditional-loss arithmetic by(n,r=min(k,n-k)), with connected r0 zeros, explicit categorical r1/complement and vector reachable ESP cells for higher r. It preserves the attributed J_K/J_K_sep/W_K laws, all-query normalization, teacher/slot/serving contracts and native complete schedule. Source grouping is not a full-batch speed/memory certificate. PROSPECTIVE_COMPLEXITY.json requires a complete native census and inclusive resource qualification before that claim.

F02 is corrected by one negative/out-of-range endpoint guard; the remaining single AST is v1-exact. The finite capable-single head and sequential law are unchanged. L01 receives meaningful unexecuted source coverage: actual visible-context and progress/prefix reconstruction, float32/64 ragged per-pattern parameter/h/unary gradients, full manual head derivatives, normalization and endpoint rejection. The v1 oracles are byte-identical; their existing staged attempt is untouched. New QA is disabled at a fresh execution-root-v2 and requires separate exact review/supervisor/root/runtime admission. No numerical code, data, model/state/array or server was accessed by this source task, and frozen science remains unchanged.

## Disabled separately sealed complete core-v3 successor

Core-v2 CB laws/source, v1 oracles and v2 ragged QA remain byte-identical.
S_K teacher forcing now constructs all features through exact integer count
prefixes and local floating selected-feature cumsums, retaining left-before-right
conditioning and source slot order. One unchanged523->64->1head evaluates every
slot, including forced rows; parameter count33601and mean loss remain frozen.
One new source-only QA supplement compares the sealed sequential single and full
manual head/h/unary gradients on both dtypes, checks head-row coverage/forced
zeros, and closes right/both genuine-subset/mixed categorical CB coverage. The
inherited QA and new supplement would run once at a fresh disabled execution
root-v3 under unchanged900s/4GiB caps/tolerances after exact independent core,
physical-supervisor/runtime/root admission. No predecessor numerical run is
repeated or silently adopted. No numerical source has been imported, compiled or
executed, no data/state/array/server accessed, and frozen science is unchanged.
