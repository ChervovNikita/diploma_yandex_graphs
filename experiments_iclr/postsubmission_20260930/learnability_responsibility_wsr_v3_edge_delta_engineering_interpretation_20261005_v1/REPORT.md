# W/S/R V3 edge repair and engineering interpretation

5 October 2026. **Static V3 repair PASS; native numerical gate remains unqualified.** No new source blocker was identified in this narrow delta. The blocked V2 review and failed synthetic qualifier remain preserved. This reviewer read source, metadata and the two explicitly authorized engineering receipts; no imports/runs, data labels, scientific outcomes, SSH, fits or new agents.

## V3 source delta

The actual accessor/worker/evaluator differences match sealed `CHANGES.patch`. Queue and warm diagnostic helper are byte-identical to V2. All V3 manifest entries and external source bindings match their recorded sizes/hashes.

The shared `_public_context` now invokes `_native_edges` before device transfer. It copies the pinned stored int64 edges to a CPU tensor and performs the original V6 statements in the original order:

```python
edge = to_undirected(edge)
edge, _ = remove_self_loops(edge)
edge, _ = add_self_loops(edge, num_nodes=24492)
```

The helper verifies the frozen V6 recipe/producer source and the normal imported PyG helper file bytes against the saved undirected/loop source pins. It records actual loaded helper paths/hashes, input/output shapes and logical edge hashes, the public NPZ descriptor, raw feature convention and original 24492-node identity. Both W and B accessors use this same label-free path; no label/mask read is added. This closes the V2 source-conformance defect without claiming any measured graph output in this review.

The worker binds the processed edge hash in the common recipe and requires complete warm/continuation preprocessing identities to match. The evaluator matches the same processed hash before prediction or A access and saves preprocessing provenance. The unchanged public file hash also binds the raw features. Thus the evaluator cannot accept a different processed topology under the reviewed caller contract.

W4898/S2449/R2450, V1 A, seed17, W-only local200/global200, fixed G0/H16/six arms, theta+ recompute/commit, metric formulas/thresholds, and seven immutable states plus predictions before final A labels are unchanged. The V2 scientific protocol retains SHA256 `27f8a8d85aec4e81114ed9ecd8554fcfaa6dd4ccf1eaf7678cc0f5eeeb477d99`. The extra diagnostic remains +16 member forwards/+8 private gradients; total planned member forwards remain **5100**. All four guards remain false. Reviewed enabled successors, safe projection, actual graph/runtime identity and whole-process/native qualification are still required before execution.

## What the two engineering receipts establish

The 15-node CPU float64 synthetic qualifier has status **FAIL_ENGINEERING**. Before failing, it completed sparse product/feasibility checks, all four full native member output/first-gradient reconstructions, dense/sparse assignments and cost derivatives including gamma0, permutation orientation, and dense/sparse native response/outer-gradient parity. Reported member reconstruction errors are zero; complete shared gradient parity error is 3.61e-15. These support those executed paths on one artificial state/context only.

Its fixed shared directional FD failed at epsilon=.001: numerical .0285137476 versus analytic .00387111866, absolute error .0246426289 versus tolerance 8.87112e-6. The worker stopped there. It did **not** finish that check's unused-tangent/independent stop-Q checks or reach the later complete public episode/recompute and state-restoration checks. A prior completed value/gradient comparison cannot replace these missing checks.

The separate receipt has status **COMPLETE_DERIVATIVE_DIAGNOSIS_NOT_QUALIFICATION**. Its source reconstructs the private partial with independent equal-valued private proxies and ordinary `autograd.grad(create_graph=True)`, preserving the Q dependence for the outer derivative. Query values match, and every shared coordinate agrees with `torch.func` to 1.11e-16. The measured live-minus-stop-Q gradient norm is .22318048, and this diagnostic reports native/input/RNG restoration.

| epsilon | right FD | left FD | central FD |
| --- | ---: | ---: | ---: |
| 3e-5 | .0038710284 | .0196021742 | .0117366013 |
| 1e-5 | .0038710886 | .1837984625 | .0938347756 |

The analytic value is .0038711187. The two smallest right differences are close; left/central differences remain irregular. Agreement between two AD constructions supports chain ownership at this state and argues against a mismatch specific to the `torch.func` construction. Both still use the same native operations and their derivative rules. It does not prove a two-sided derivative exists, establish the asymptotic left limit, identify the responsible branch/operation, or exclude an error shared by both AD paths. Branch sensitivity/nonsmoothness is a hypothesis requiring the pending source diagnosis.

**No native-gate clearance or tolerance change follows.** The receipts neither qualify the real 24492-node graph, complete public commit/resource path, initially label-unseen W/S/R behavior, predictive usefulness, novelty or paper acceptance. Preserve the failed qualifier and investigate the actual branch cause under a separately fixed engineering plan.

## Exact identities

V3 accessor `acda1004e2ab7bb522978111307985ef6458a642ea450404b5dab8c48a752bb0`; worker `68c7902f340432fe3e7d04593244f6af84ebe6c30968af4fac93ce38fd53d53a`; evaluator `3b1a3cd0b4c36e1bbd4e1601fbd71f9faf868dff408ea4c3937b7c5eefd6a000`. Full packet, receipt, source and preserved-review hashes/read scopes are in `PROVENANCE.json`.
