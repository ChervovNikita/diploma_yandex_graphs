# Exact-native NCN shared/private derivative qualification

**Status: source prepared and statically checked; numerical execution has not occurred. No fit is admitted.**

This packet answers a bounded feasibility question: can the pinned Citeseer NCN/private-frame bank differentiate one functional private SGD step through the shared parameters, and then realize its committed weights without inference adaptation? It does not establish useful episode structure, novelty, predictive improvement, an Adam training recipe, or permission to change the frozen screen.

## Shortest meaningful next gate

After root source review and authenticated TRAIN/features/runtime binding, run `qualify_train_only_sgd.py` once on an approved CPU host. The script uses the actual full `[3327, 3703]` feature tensor and full node population, two disjoint native TRAIN query batches of 1024 positives and 1024 negatives each, and all four member routes. It makes one shared SGD candidate and recomputes one private SGD candidate; every state is discarded. The current `JOB_TEMPLATE.json` is deliberately disabled and incomplete.

The external supervisor must enforce a 360-second hard process bound and preserve its exit/timeout evidence. The script checks a 300-second soft bound between stages. An incomplete run is unqualified. There is one prospective seed and no automatic retries, reseeding, reduced graph, or substitute backend.

Required gates, in order:

1. Authenticate native source, exact parameter/buffer map, full TRAIN geometry, feature authority, hostname and CPU runtime versions.
2. For the native encoder sparse support and each active inner/outer CN support, verify the quadratic second-order identity below. An empty individual CN support is explicitly unexercised; at least one inner CN support must be active.
3. Diagnose the live functional private step and exact full-network mixed pullback, including its chain-rule decomposition and two directional finite-difference checks. This diagnosis is attempted even after a sparse identity failure, when the runtime permits it.
4. Only if all derivative gates pass, apply the discarded shared SGD candidate and recompute the private update from the original private state at the updated shared weights with exactly replayed RNG.
5. Compare functional route logits and mean raw logits with a real deep-copied model loaded with the committed candidate parameters. Verify original weights, buffers, feature bytes and sparse support contents remain unchanged.

The diagnostic produces metadata only (`RESULT.json` or `FAILURE.json`). No logits, checkpoints, accuracy, ranking metrics, epochs or selected model are produced. A supervisor timeout must remain visible even if the interrupted script cannot write its final receipt.

## Pinned architecture and partition

Author donor: HeaRT commit `c447cbff4c493b60d14b6544c3c39d3b9c5ddff0`; unchanged NCN `model.py` SHA256 `4f83f0dfe051808354e62230d2c84f737ccb7507810b9ec5a6b173d683a28e49`; unchanged `util.py` SHA256 `d4c6857f51c9c6fdcaaa077182dffae8083c1ca33dff03ee004003eb31ba74ff`.

The donor lives in the existing `citeseer_heart_ncn_trainval_runner_source_20261005_v1` packet. Its `heads.py` SHA256 is `49f2a61825d5dbcbe977926ddd3e2ea35eb02c6603b79833092e4e560fbab52f`. The existing `run.py` is authenticated as source and is **not imported**: its loader would also open VALID/pool files.

Encoder construction is exactly:

```python
GCN(3703, 256, 256, 1, .3, True, False, -1, "puregcn", True, 0.,
    xdropout=.4, taildropout=0., noinputlin=False)
```

The head is `make_predictor(..., arm="private_frame_f4", members=4, width=256, factor_seed=...)`: native full CN support, `xlin`, LayerNorm, `tailact=True`, `twolayerlin=True`, dropout `.3`, zero edge dropout, and four trainable private Householder vectors. This NCN architecture has no recursive completion.

| Parameter role | Names | Scalars |
| --- | ---: | ---: |
| Shared encoder projection and JK scalar | 3 | 948,225 |
| Shared head base Linear weights/biases | 16 | 460,801 |
| All shared | 19 | 1,409,026 |
| Private route factors, LayerNorm affine parameters, beta and frames | 28 | 26,632 |
| Total | 47 | 1,435,658 |

`PARAMETER_PARTITION.json` lists every name and shape. `encoder.adjdrop.ratio` and `predictor.dropadj.ratio` are scalar constant buffers. Runtime instantiation must match the map exactly; it has not yet been verified by execution.

TRAIN metadata specifies 3,870 nonself unique undirected positives. Removing the union of the two positive query batches leaves 1,822 undirected support edges (3,644 symmetric entries) while retaining all 3,327 nodes. Negatives use the pinned native PyG sampler/`PermIterator` convention. Only query edge IDs are disjoint; endpoint-separated episodes are not claimed. This operator fixture does not qualify the proposed episode distribution.

## Differentiated operator

Let θ be the complete shared partition and φ the complete private partition. The inner loss is the native balanced mean log-sigmoid loss over query/member axes, with native training dropout and captured CPU RNG replay. With `M=4`, `α=4×0.001` converts that mean-member gradient into a per-route SGD step of `0.001`:

\[
\phi'=\phi-\alpha\nabla_\phi L_I(\theta,\phi).
\]

The outer objective uses evaluation mode and is half the same loss on mean raw logits plus half the mean route loss. Full `torch.func.functional_call(..., strict=True)` supplies every parameter and buffer. The qualifying fast map remains live in autograd.

With `q=∂L_O/∂φ'`, the diagnosis verifies

\[
g_{\rm meta}=g_{\rm direct}-\alpha H_{\theta\phi}^{T}q,
\quad H_{\theta\phi}^{T}q=\nabla_\theta\left[(\nabla_\phi L_I)^Tq\right].
\]

Only the diagnostic direct partial and fixed cotangent detach tensors. The qualifying meta gradient uses the complete live fast map. Unused or unavailable derivatives raise/fail; they are not zero-filled.

For alternating-sign normalized directions in `encoder.xemb.1.bias` and `predictor.xlin.ops.0.bias`, compare the full native mixed projection with central finite differences of the contracted inner private gradient, holding `q` and RNG fixed. Steps are `2^-8` and `2^-10`. The smaller-step agreement tolerance is `1e-5 + .02 × scale`; step-to-step stability tolerance is `2e-5 + .05 × scale`. These are two full-network directional checks, not exhaustive Jacobian verification. Disagreement or unstable/nonsmooth behavior leaves the operator unqualified. Finite differences never supply an update.

After a pass:

\[
\theta^+=\theta-0.001g_{\rm meta},
\qquad \phi^+=\phi-\alpha\nabla_\phi L_I(\theta^+,\phi).
\]

The recomputation starts from original φ, replays the same dropout RNG and must reproduce exactly. Serving uses the committed θ⁺,φ⁺ in evaluation mode; there is no private adaptation at inference.

## Native sparse second-order risk

Reference-only public `torch_sparse` 0.6.18 sources are saved with retrieval receipts. Their equivalence to an installed runtime is **not asserted**.

- `matmul.py:9–32` routes `spmm_add → spmm_sum → torch.ops.torch_sparse.spmm_sum`.
- `spmm.cpp:55–114` implements custom `SPMMSum`; its dense-input backward at lines 104–108 invokes raw `spmm_fw` on `grad_out` and transposed sparse structure.
- `spmm_fw` at lines 22–35 dispatches directly to CPU/CUDA kernels. The dense adjoint is not expressed as a recursively recorded `SPMMSum` operation in this reference source.

This suggests a cut in higher-order dependence on `grad_out`. Some mixed branches may still have gradients, so obtaining a nonzero meta gradient or encountering no exception is insufficient evidence of correctness. Native sparse products occur both in parameter-free `PureConv` and in CN aggregation before private head branches.

The explicit gate uses the installed native `spmm_add` with constant K and native dense width 256:

\[
Q(Z)=\tfrac12\|KZ\|^2,\qquad H_Qd=K^TKd.
\]

Autograd's HVP must match two native forward sparse products (`rtol=3e-4`, `atol=3e-5`), with a finite nonzero expected HVP. A missing second-order graph, partial/zero derivative, error, or disagreement fails the gate. No dense sparse replacement, double-backward wrapper, stop-gradient approximation, or finite-difference Hessian surrogate is admitted here.

If the installed native path fails, preserve its evidence as an implementation limitation for this exact operator/runtime. That failure does not establish that the scientific shared/private learning direction is infeasible. Any equivalent wrapper or Hessian surrogate would need separate authorization and qualification.

## Cost and remaining admission

The source avoids materializing a Hessian. A complete successful path has at most 15 full bundle forwards plus one encoder-only warm forward, five sparse identity gates, native first/second-order backward work, and eight inner finite-difference evaluations already included in those 15 forwards. Full features alone occupy 49,279,524 float32 bytes; one full hidden tensor occupies 3,406,848 bytes. Activation/derivative storage and runtime remain unmeasured; final metadata records inclusive wall time and peak resident memory.

The committed-copy bug found during source review is fixed by `copy.deepcopy(model)`; a new closure-based `Bundle()` would have reused and mutated the original modules. Static AST parsing and file/partition hashing provide source preparation evidence only. No numerical/model import, prepared-script execution, data/outcome read, server/GPU action, scientific fit, canonical edit or frozen-screen change was performed for this packet.

Root must review this packet and the existing screen state before any execution. A later passing CPU qualification establishes only this discarded SGD operator; a final Adam recipe, useful endpoint episode rationale, controls, complete training cost and predictive tests remain separate scientific admission questions.
