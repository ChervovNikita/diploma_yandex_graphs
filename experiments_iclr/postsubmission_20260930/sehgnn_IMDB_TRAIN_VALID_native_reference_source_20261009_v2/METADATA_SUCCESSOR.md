# Metadata-only native successor

Root reported that the v1 full-input qualifier completed native preprocessing and one full TRAIN forward/backward/Adam update, then failed fresh selected reconstruction at `torch.load(..., weights_only=True)`. The runtime `versions` record contained `torch.torch_version.TorchVersion`, a str subclass. JSON serializes its text successfully, while Torch checkpoint pickling preserves its foreign class and weights-only loading rejects that global. This source author did not open the scientific data, checkpoint or failure outcome; the causal path was verified from source and root's failure report.

The sole executable v2 change is in `runner.runtime`:

```python
versions = {name: str(value.__version__) for name, value in providers.items()}
```

The resulting record is checked against the same root release versions and placed in the same runtime identity before `engine.snapshot` calls `torch.save`. Thus provider version values become exact built-in strings before serialization. No unsafe allowlist, arbitrary metadata import or unrestricted pickle loading is added. `torch.load(weights_only=True)` is unchanged.

The entire model, engine, native helpers, owned state helpers, protocol/recipe, source bindings and disabled releases remain byte-identical to v1. This retains native initialization, architecture, propagation, TRAIN roles, AMP/Adam, strict full VALID selector, patience, master scaler lifecycle, owned state restoration, costs and practical drift gate. Public native dispatch still installs no adapter. The v1 source seal and failed qualifier/cost evidence remain immutable.

`verify_metadata_successor.py` checks that the executable source differs by exactly this one text replacement, verifies unchanged payload bytes and weights-only loading, and executes only the isolated dictionary-comprehension metadata expression against a stdlib str-subclass descriptor. It imports no numerical provider/model and uses no floats, tensors, scientific arrays, labels, checkpoints or server. The inherited native source verifier repeats exact author/helper AST and recipe/source checks.

V2 remains inactive and requires a new root external release bound to its new seal. The old failed receipt cannot qualify v2. Root owns actual requalification and any native reference fits. The source fix does not itself establish native competence or numerical success.
