# Native V2 metadata delta — independent source audit

Accepted in the focused source scope. No concrete source blocker remains in this delta. Root owns actual native V2 qualification.

The sealed runner differs from V1 by exactly one replacement in runtime identity construction:

```python
versions = {name: str(value.__version__) for name, value in providers.items()}
```

This is the narrow repair for the root-reported TorchVersion metadata failure. An isolated stdlib expression fixture using an inherited string subclass yields an exact built-in string. This audit does not execute Torch serialization or claim arbitrary-object support.

The native model, engine, propagation/train helpers, owned state helpers, protocol, disabled release templates, source bindings, original verifier, integration contract, license and schema binding remain byte-identical. The exact V1/V2 manifests, seals, own payloads and fixed dependency hashes passed. `torch.load(weights_only=True)` remains unchanged. The prior independent V1 source review is retained as an immutable input.

## Exact successor binding

- Manifest: `946d3f6ea31f9c40952747cb414f98f0844ccb81c6626e6d64dc42475cc60aee`.
- Seal: `5ca6dff2c1f64bf37e2642fff7101e4377a48e087d2b21082a8d46887dc3654e`.
- Runner: `26c2636a30cbb151f4eacdc69a87baf63c7547a6a9db0da19a61cae8722e4aa7`.

V1 failure evidence cannot qualify V2. The new root qualification must observe actual full native execution and safe selected reconstruction. No numerical providers/models, scientific data/labels/arrays/checkpoints/outcomes, server actions, installations, orchestration or float campaigns were used. No sealed source or evidence was modified; no predictive or paper verdict is supplied.
