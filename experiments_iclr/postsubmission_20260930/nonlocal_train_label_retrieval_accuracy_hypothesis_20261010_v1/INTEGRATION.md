# Existing model interface integration

Use the current caller, optimizer, member RNG scopes and checkpoint selector.
The helper owns no state or parameters. Its `torch` argument is the caller's
already admitted provider. No existing sealed file is edited by this packet.

For shared4, call the existing wrapper with the factual label-free batch and **no
`ids` gather**, so it returns full `(native_logits[M,N,C], H[M,N,D])`. For an
ordinary M1/body, construct the existing `NativeModelAdapter`, call
`adapter.native(batch)` under its existing member RNG scope and unsqueeze the two
outputs to M=1. Genuine ordinary I4 runs the same operation separately for its
four independently initialized/trained bodies. Do not use the current
`Family.logits` helper: it discards H. No second native forward is necessary.

The caller supplies only role-restricted TRAIN arrays A/y_A and one mask
generator independent of native/dropout streams. Reuse the same Q draw at each
paired update across routes and retrieval arms; retain its seed/IDs or digest in
the existing study record. Fit Q is drawn by `common_half_query` once per update.
The fixed per-class counts qualify every class in Q and S; no heldout label may
choose the draw. The caller must guarantee row identities and label-role custody.

```python
Q = memory.common_half_query(torch, A, y_A, C, generator=existing_mask_generator)
native_logits, H = existing_full_native_forward()  # same current forward
result = memory.label_memory(torch, native_logits, H, A, y_A, Q,
                             mode='fit', stop_attention_gradient=detached_control)
native_ce = F.cross_entropy(native_logits[:, A].flatten(0, 1), y_A.repeat(M))
# y_Q is gathered in Q order from the caller's restricted TRAIN truth field.
retrieval_ce = F.nll_loss(result['retrieval_log_probs'].flatten(0, 1), y_Q.repeat(M))
loss = native_ce + retrieval_ce
# Use the existing zero_grad/backward/step and complete-task selector.
```

For selection/serving, call the same helper with `mode='serve'`, query rows equal
to the whole permitted evaluation role, all TRAIN A/y_A, and under the caller's
evaluation/no-grad convention. Select and restore by the existing complete-role
served accuracy rule. Treat `mixed_log_probs` as log probabilities, not native
raw logits; probability pooling is `exp(mixed_log_probs).mean(0)` and pool NLL
uses `logsumexp(mixed_log_probs, dim=0)-log(M)`. Report native, retrieval and mixed
outputs distinctly. No TEST label is accepted by the helper interface.

Root must numerically qualify normalization, same-forward capture, live query
and support gradient paths, detached-control gradients, common-Q exclusion,
unchanged fixed-Q/S predictions when Q truths are altered without changing roles,
and device/RNG integration. This packet verifies syntax and source contracts
only; it does not certify a runnable admitted study.
