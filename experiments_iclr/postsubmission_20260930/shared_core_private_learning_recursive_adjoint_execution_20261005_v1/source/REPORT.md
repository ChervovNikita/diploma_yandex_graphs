# Exact recursive-adjoint repair: practical source plan

**Prepared source only. No wrapper import/model construction, numerical execution, server action or fit was performed for this packet. The job is disabled.**

The preceding single native CPU diagnostic is preserved in `shared_core_private_learning_native_derivative_execution_20261005_v1` (manifest SHA256 `22de2538170bbfc7d13a1808c6923bd9f8eb8d64b2f1e1865a9be404f198976e`). It demonstrated absent dense-input second-order graphs on all four nonempty native supports. Both full mixed FD checks were separately unstable/inconclusive; the internally consistent partial chain decomposition did not qualify the operator.

## Repair and invariant forward

`recursive_adjoint.py` defines a factory without importing torch. At future qualification runtime it receives the reviewed torch module and **saved original native `spmm_add`**. Its custom Function uses:

```text
forward(K, Z)  = saved_native_spmm_add(K, Z)
backward(q)   = (None, same_custom_Function(K.T, q))
```

For fixed K, this is the exact linear adjoint. The recursive application records dependence on the incoming cotangent when autograd requests higher derivatives. Sparse structure/values receive no gradients; learnable sparse values are rejected. No dense operator, approximate Hessian or finite-difference update is inserted.

The donor files and exact 47-name shared/private partition remain unchanged. The qualifying script changes only the process-local donor module's `native.spmm_add` alias, inside its `main` numerical qualification. It verifies the initial alias is the original `torch_sparse.matmul.spmm_add`, and restores it in `finally`, including failures. The torch_sparse package and host/global environment are not patched. This local alias covers both donor `PureConv` and the head's explicit `native.spmm_add` calls.

## One separately authorized qualification

Keep the prior full TRAIN/features, native architecture, partition, seed/factor seed `20261005`, CPU runtime, two-thread profile, disjoint query IDs, captured RNG, objectives and SGD steps. The proposed numerical child remains bounded by a 300-second soft check and external 360-second hard timeout, with no retry or reduced fixture.

1. Compare the original native alias and recursive alias on the **full inner training-mode and outer evaluation-mode bundles**, replaying identical RNG and parameter/buffer values. Require bitwise equality of every route logit and every first-gradient tensor across all 47 parameter names. Report maximum gradient error; a discrepancy fails qualification.
2. Run the same nonempty sparse quadratic identities. Expected `KᵀKd` comes from two calls to the saved original native forward, while the HVP uses the recursive adjoint. Keep `rtol=3e-4`, `atol=3e-5`, finite/nonzero reference and inner CN coverage gates. Empty individual supports remain unexercised.
3. Run the full live functional private step and shared mixed gradient, chain decomposition and **unchanged** two-direction FP32 FD criteria. Keep steps `2^-8`, `2^-10`, agreement `1e-5+.02×scale` and stability `2e-5+.05×scale`. Unstable/nonsmooth or disagreeing checks remain unqualified. No tolerance is loosened to obtain a pass.
4. Only after all gates pass, make the discarded shared update and recompute the private step from original φ at θ⁺ with exact RNG replay. Compare functional serving with a deep-copied committed model whose serving call explicitly uses the original native alias. Confirm original weights/buffers/support/features unchanged and the native alias restored.

The disabled template additionally requires a separately reviewed wrapper hash and `constant_adjacency_recursive_adjoint_authorized=true`. Its ordinary review, authenticated input and external-bound flags remain false. The preceding native failure is never overwritten.

## If FP32 FD remains unstable: a separately declared oracle

The optional next source gate would compare the **complete analytic mixed pullback vector and complete shared meta-gradient**, covering all 1,409,026 shared scalars, between recursive native SpMM and ordinary differentiable dense-constant-adjacency multiplication at **float64 full-network precision**. It would not infer correctness from unstable finite differences.

Regenerate the same prospectively fixed FP32 initialization and exact query/support/RNG fixture, then cast a private diagnostic copy of features, parameters, buffers and sparse constant values to float64. Implicit ones become explicit float64 constants so encoder normalization also uses the declared precision. Construct dense K from those exact constants and use a separate local donor alias only for the oracle evaluation. Both branches use identical float64 point, loss, support, dropout RNG and private step. First verify float64 forward/first-gradient agreement, then compare all mixed/meta-gradient coordinates directly with prospectively declared tolerances. Preserve disagreement or numerical ambiguity.

This oracle would qualify the float64 diagnostic operator and repair implementation. It would not retroactively pass failed FP32 FD gates, replace serving, supply an update, admit a final FP32/Adam recipe, or establish predictive utility. It is a separate proposal, not implemented or authorized for execution in this packet.

## Honest cost and scope

The recursive qualification adds four full bundle forward/first-gradient evaluations to the prior source. A completely passing path has at most **19 full bundle forwards plus one encoder-only warm forward**, five sparse identities and the existing higher-order/FD work. The prior incomplete native diagnostic took 11.48 seconds and peaked at 3.10 GB resident memory. The repaired derivative graph retains additional paths; its complete time/memory are unknown and must be measured.

The optional dense float64 oracle is materially more expensive: a full 3327×3327 constant matrix occupies 88,551,432 bytes; each 1024×3327 CN matrix occupies 27,254,784 bytes. Keeping all five consumes 197,570,568 bytes before features or derivative graphs. One full bundle forward would perform about 9.81 billion dense multiply-accumulate terms at these shapes; higher-order backward adds work. It should be bounded separately and evaluated sequentially to avoid retaining two full higher-order graphs together. No current CPU time or memory guarantee is claimed.

This is an implementation repair for the shared/private learning rule. It contributes no method-novelty claim, endpoint episode evidence, predictive result or fit admission.
