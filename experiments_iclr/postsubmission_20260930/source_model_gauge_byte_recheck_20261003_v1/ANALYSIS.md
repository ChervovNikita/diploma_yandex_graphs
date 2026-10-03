# Strict byte identity recheck of the fixed gauge witness

## Correction to the original evidence

The preserved `source_model_gauge_probe_20261003_v1/probe.py` uses `torch.equal` to set its two “bitwise identical” fields. That is a numerical equality check. It does not establish dtype equality or raw floating point byte identity, and it accepts positive and negative zero as equal. The original checker therefore did not justify its strict byte wording. The original source, witness, analysis and fetch receipt remain unchanged; their exact hashes are bound in `INPUT_BINDINGS.json` and checked again in `VERIFICATION.json`.

This successor independently enforces identical dtype and shape before comparing `detach().cpu().contiguous().reshape(-1).view(torch.uint8)`. Equality on those uint8 tensors compares the representation bytes. It uses no tolerance. Numerical equality is recorded in a separate field for every comparison. A real Torch signed-zero control passed: the float tensors compare numerically equal but have different bytes. Dtype and shape mismatch controls were rejected, and a noncontiguous tensor matched its contiguous copy after canonicalization.

## Executed result

**Strict byte identity survives on the original fixed fixture.** Normal CPU execution on the verified `shmelev@192.168.18.77` project repository finished on 3 October 2026 at 12:27:05 UTC, exit 0, in approximately 5.93 seconds. Torch was 2.7.1. CUDA remained uninitialized. No environment overrides, isolation wrapper, installation or base environment change was used.

The fixed seed 20261003, sealed width64/M4 model, nine-node random feature fixture, 12 predeclared edges, six queries, graph variants, positive scales 32 and 1/32, three parameter changes, evaluation mode and two routing modes are unchanged. The original hidden-scaling `rtol=1e-5, atol=1e-6` is unchanged and is used only for that diagnostic. The byte comparisons have no tolerance. `STATIC_CHECK.json` binds 21 source checks, including AST identity of the model initialization, transformation, fixture, run helper, routing/graph iterators and response expressions. It also confirms that the original execution command contains exactly the preserved original probe source.

The successor checked:

- Six complete float32 member-logit tensors, shape `[6, 4]`: three graphs under each of private and pooled-after-clamp routing.
- Each member separately in those tensors: 24 comparisons, shape `[6]`.
- Six served mean-logit vectors as an additional explicit check.
- The two originally claimed concatenated graph-response tensors, shape `[12, 4]`, plus their eight member columns.
- Each of the two edge-removal responses separately under both modes, including all 16 member vectors.

Every comparison above has identical dtype, shape, byte count and SHA256, zero differing bytes, numerical equality and maximum numeric difference zero. The response is the original memberwise expression `sigmoid(z_native) - sigmoid(z_variant)`, concatenated over the two graph variants. It is not a newly introduced response definition. All six original hidden cosine diagnostics reproduced exactly: on the native graph, mean squared hidden cosine changed from `0.9999995827674866` to `0.000005734027126891306` under either routing mode.

`RECHECK.json` SHA256 is `15ff43a30126bae7302f5ef5a2ae50af3d6d3ff407292716d8d649abd14be9fa`, 69,404 bytes. `FETCH_RECEIPT.json` verifies all five remote execution files against their exact local bytes. The MacLink wrapper's command custody receipts are copied into this successor; the wrapper also retains its standard transport command and receipt entries in its own existing command directory.

## Analytic argument and scope

The separate exact-arithmetic argument is unchanged: positive member-specific diagonal scaling of the final private LayerNorm affine scale and bias, compensated by inverse scaling of the final input factor, commutes with evaluation dropout and ReLU and preserves the factorized final linear product. Applied at every recursive scoring call, it preserves the member functions and completion weights in exact arithmetic.

The finite precision evidence here is a fixed CPU float32 witness, not a universal floating point byte-invariance theorem. Power-of-two scales and this fixture do not establish strict bytes for arbitrary inputs, parameter states, scales, architectures, hardware, runtimes or training masks. The initial unit factors also intentionally make the four functions identical. The witness shows that this internal hidden cosine separation can occur without changing these member predictions or graph responses.

No dataset, checkpoint, label or predictive metric was accessed. No GPU computation, training, scientific novelty or predictive superiority claim was made. No original paper score, public status, ledger or preserved packet was edited.
