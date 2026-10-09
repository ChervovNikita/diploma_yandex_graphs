# Inactive live-action initialization patch

This packet prepares one initializer comparison using the **exact V2 geometry-only bank and loop**. It adds a hook/screen helper and an auditable derivative of `shared_fit.py`; it adds no runner, supervisor, model family, decoder or source-response auxiliary loss. All public entries default inactive, and the separate root release template is disabled. No dataset, metric, checkpoint, heldout array, numerical module/model or research host was accessed in preparation. Source/AST/hash inspection is not numerical qualification.

## Fixed comparison and prospective amendments

The source inherits V2's `d4_f16_L4`, native arguments, Adam/lr/decay and selector, and base seeds **7409/8501/9607**. The earlier sealed optional pilot's1103/2207/3301 is preserved; using V2's existing architecture/seeds is a prospective amendment, not hidden reuse of historical fits or fresh confirmation. The same official Tolokers split0 is an original benchmark.

For each seed:

1. Construct one V2 bank through `make_bank` and the existing native placed factory. Freeze **every** private incidence r/s at exactly1 with `requires_grad=False`. Call the original `shared_fit.train_update` for exactly100 epochs: four full native dropout/jitter views, streamed own NLL/4 backwards at old parameters, one Adam. Identity checks and absence of private Adam state are required after every step. Adam skips those gradient-free factors, including coupled decay; slow native moments continue normally.
2. Save the exact epoch100 CPU model/optimizer/RNG snapshot. Preserve the ordinary best warmup checkpoint and its selected outputs as a common preinitialization comparator. This is not a warmup-selected start: **all arms start from epoch100**.
3. Reconstruct one fresh screen bank from epoch100 and run **eight directions × two signs = sixteen candidate native forwards**, plus **one baseline = seventeen**. Hash generation and scoring use no model RNG; the caller RNG is restored after the screen.
4. Reconstruct each of four branches from the same epoch100 model/optimizer/RNG. Re-enable all private incidence gradients. Only first-layer input r values change for nonidentity choices. Output s and all later r/s start at1. No slow state or moment is reset, and no private moment existed to remove.
5. Run **exactly400 additional native epochs101–500** for each arm. Patience200 is retained as inherited metadata/diagnostic but cannot truncate the requested comparison. All original `train_update`, `optimizer`, `evaluate`, full-VALID selector, selected-state reconstruction, fresh probability-mean serving and output-materiality behavior are reused. A selected epoch<=100 records `initializer_effect_selected=False`; it supplies no observed initializer benefit.

The four choices are identity, hash-random from the common admissible pairs, live off-node action independence, and final classifier-response independence. Identity pays the same allocated screen work and label opportunity. One common screen is physically computed and shared, rather than repeated four times; every arm reports one-quarter allocated setup cost and the complete standalone setup cost. The whole comparison pays one warmup and one screen per seed. A failed warmup/screen or fewer than two common admissible directions leaves the entire four-arm comparison unqualified; do not report identity or other survivors as a completed panel.

## Exact original operator and value interception

Original `models/disc_models.py` lines275–283 execute:

`L, trans_maps = laplacian_builder(maps)`;

`V = left_right_linear(..., lin_right_weights[layer])`;

`message = torch_sparse.spmm(L[0], L[1], ..., V)`.

The original builder is an `nn.Module`. `FirstActionCapture` temporarily registers a forward hook on **that builder** and takes the first `output[0]`. In `laplacian_builders.py` lines319–335 this pair has already passed native block-degree/SVD normalization, +I augmentation, eval jitter policy, clamp, fixed-map assembly and sparse merges. `output[1]` is the detached unnormalized transport diagnostic and is ignored. No normalizer is reimplemented or guessed.

A forward hook on the original first `lin_right_weights[0]` receives its actual output. Both left/right weights are enabled in the frozen native preset. This tensor is the return value of `left_right_linear` immediately before the author sparse multiplication, including the actual preceding stem/left transform. It is not reconstructed from a guessed feature map. Hook callbacks return `None`, never replace native outputs, and are removed even when observation fails. They are installed only on the single probed native member during an eval/no-grad screen; the author forward source is unchanged.

The baseline actual V is copied once on device. Every trial uses that common captured V for its measurement. The trial's actual V shape/dtype/device and floating materiality relative to baseline are recorded. Algebraically it is unchanged because only the incidence learner's r changes; no new byte-equality/tiny-drift acceptance gate is introduced. This action comparison is in the fixed common native value coordinates, not an independent node-frame quotient.

Filter the **returned** sparse entries by `row//final_d != col//final_d` for measurement and call the exact `torch_sparse.spmm` provider in the original forward's globals. The author receives its original full L unchanged. Off-node filtering does not delete deployed edges, rebuild degrees, reconstruct topology, inspect `.L`, or create a dense N-by-N matrix. All final_d stalk rows/channels for **all TRAIN nodes** enter the action vector. The common transformed value/support still cover the full graph.

The original interception is therefore source-feasible. Floating capture, extra sparse kernels, GPU residency and candidate eligibility have **not** been numerically qualified; the future separate root release explicitly requires capture qualification. It does not activate or replace the pending V3 engineering assessor.

## Sealed candidate generation and screens

`INITIALIZATION_TEMPLATE_DISABLED.json` fixes every screen constant:

- Eight directions, indexed0..7, in the first incidence input-factor space. SHA256 low bits produce Rademacher signs; native-dtype unit-L2 normalization follows. There is no adaptive direction generation or factor/output perturbation.
- Step0.01, both signs. Trial parameters are `r=1 +/- 0.01*u`; every trial starts from the same slow/other-fast state.
- Finite original operators, V, actual off-node action, complete binary logp and all-TRAIN NLL.
- Each sign's NLL is at most `NLL_base + 0.001*max(NLL_base,0.01)`.
- Each sign's action and class-margin difference has relative norm>1e-6; denominator is `max(baseline_vector_norm,1)`.
- Both half-plus-minus action and classifier vectors pass that same relative floor before **any** choice uses the shared admissible pool. The all-TRAIN class margin is `logp[:,1]-logp[:,0]`; its score is a generic finite functional screen, not native FoRDE.
- Pair score is `1-cosine^2` on the appropriate half-plus-minus vectors. Hashes settle exact-score ties and random order. A zero best independence score remains a recorded rank-deficient result, not a reason to search.

The off-node measurement stays in native dtype/provider. Detached CPU float64 norms/dots are measurement arithmetic only. The native propagation remains unchanged. Native/runtime/nonfinite/hook exceptions abort the common screen and retain partial costs; finite trials that fail the fixed loss/visibility criteria are rejected while the remaining predeclared trials continue. No retry, reseeding, strength increase or extra candidates are supplied. The screen reads x/TRAIN indices/labels only; it never reads VALID indices/labels or computes a heldout score.

Random hashing chooses a direction subset; ascending direction IDs determine member0/1 and2/3 assignments for every nonidentity choice. If choices coincide, their initial states coincide and this is disclosed; no action-statistic distinction can be inferred from that seed. The requested four branches are not silently replaced by fewer runs.

## Cost and interpretation

Per successful seed, the training work is400 native warmup paths plus four arms×400 epochs×four native paths. Ordinary full-VALID evaluation costs the same four-member calls each epoch, including the common warmup. The common screen adds17 complete native eval paths, their original per-layer map/SVD/propagation work, and **17 extra off-node sparse multiplications**, filtered sparse copies, baseline V storage, detached all-TRAIN vectors, CPU norms/dots/hash selection and serialization. Counters, actual wall/CPU time, cumulative RSS, CUDA allocated/reserved peaks, copied/vector bytes and failed partial work are recorded. These costs are not inferred from parameter count or called free.

Common warm state and screen metadata are server-only future artifacts. Branch selected checkpoints/fresh serving references retain the original V2 reporting contract. Construction, selected checkpoint writes and reconstruction remain charged; common setup costs are reported separately and equally allocated across the four branches. Full TRAIN and full VALID populations are retained throughout ordinary training/selection. TEST truth never enters the six-key role contract.

When transporting the common warm snapshot between owned processes, the caller augments the loaded `WARM_STATE.pt` with its external SHA256/byte binding and actual cost from `WARMUP.json`. Those fields are attached after serialization in the direct return value and are deliberately not a circular self-hash inside the checkpoint. The screen and every branch bind that exact epoch100 artifact as well as the seed/identity.

Action visibility and local TRAIN-loss bounds do not prove useful graph sources, safe future training, strong final individual paths, irreducible geometry or improved pooling. Keep all own-member/pooled curves and selected fresh quality. The all-VALID served-source ablation/permutation assays from the sealed prior remain a separate requirement before any graph-source utility claim. Proper global Rank-1 Bayesian/structured-latent, independent, frame/pointwise and capable joint-capacity controls remain necessary for broader attribution. BSNN, BatchEnsemble, generic functional/gradient diversity and initialization are explicit collisions; no novelty claim is made.
