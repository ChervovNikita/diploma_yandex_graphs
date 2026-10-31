"""CPU checkpoint envelope/state validation, required before final test access."""
from guards import read_json, HERE


def validate_selected_checkpoint(path, expected_metadata):
    import torch
    from models import make_model
    # These locally generated envelopes need only tensors and primitive metadata.
    envelope = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(envelope, dict) or set(envelope) != {"metadata", "model_state"} or envelope["metadata"] != expected_metadata:
        raise RuntimeError("Selected checkpoint envelope/selection identity mismatch.")
    if expected_metadata["torch_version"] != torch.__version__:
        raise RuntimeError("Selected checkpoint runtime differs from qualification.")
    with torch.random.fork_rng(devices=[]):
        model = make_model(read_json(HERE / "CONFIG.json"), expected_metadata["arm"], expected_metadata["seed"])
    state = envelope["model_state"]
    reference = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(reference):
        raise RuntimeError("Selected checkpoint state keys mismatch.")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != reference[name].shape
                or value.dtype != reference[name].dtype or not torch.isfinite(value).all()):
            raise RuntimeError(f"Invalid selected checkpoint tensor: {name}")
        if name.endswith("running_var") and (value < 0).any():
            raise RuntimeError(f"Negative BatchNorm running variance: {name}")
        if name.endswith("num_batches_tracked") and (value < 0).any():
            raise RuntimeError(f"Negative BatchNorm batch counter: {name}")
    model.load_state_dict(state, strict=True)
    return envelope
