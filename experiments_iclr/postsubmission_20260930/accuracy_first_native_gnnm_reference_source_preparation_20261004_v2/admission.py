"""Reference-side input, backend and output admission; no numerical imports."""
from pathlib import Path

INPUT_FIELDS = ("role", "native_edges_sha256", "feature_identity", "validation_labels_sha256")


def validate_paired_inputs(paired, split, expected):
    """Compare this actual split with the caller-owned paired master."""
    if type(split) is not int or split not in (0, 1, 2):
        raise ValueError("Predeclared paired split required")
    by_split = paired.get("input_bindings_by_split")
    entry = by_split.get(str(split)) if isinstance(by_split, dict) else None
    if not isinstance(entry, dict):
        raise ValueError("Paired protocol requires input_bindings_by_split[str(split)]")
    for field in INPUT_FIELDS:
        if field not in entry or entry[field] != expected[field]:
            raise ValueError("Paired split input differs or is absent: " + field)


def reference_device(rt, device, resolver):
    """Admit only backends represented in the inherited RNG inventory."""
    selected = rt.torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Reference devices are limited to CPU and selected-device CUDA")
    return resolver(rt, selected)


def create_output_directory(output):
    """Fix the caller-relative spelling once, at directory creation."""
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    return output
