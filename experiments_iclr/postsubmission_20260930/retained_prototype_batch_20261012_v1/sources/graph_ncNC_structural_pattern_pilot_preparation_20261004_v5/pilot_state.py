"""Own trusted checkpoints, complete fit RNG and atomic epoch journals."""
from pathlib import Path
from hashlib import sha256
import copy
import os
import tempfile
from pilot_common import atomic_json, file_sha, require, utc


def rng_state():
    import random
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(value):
    import random
    import numpy as np
    import torch
    require(torch.cuda.device_count() == len(value["torch_cuda"]), "Fit RNG device topology differs")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in value["torch_cuda"]])


def rng_digest(value):
    return state_digest(value)


def state_digest(value):
    """Stable typed value hash, independent of storage IDs and Torch ZIP IDs."""
    import json
    import numpy as np
    import torch
    digest = sha256()
    def visit(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"torch:" + str((array.shape, array.dtype)).encode())
            digest.update(memoryview(array).cast("B"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy:" + str((array.shape, array.dtype)).encode())
            digest.update(memoryview(array).cast("B"))
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b":" + json.dumps(item, allow_nan=False).encode() + b";")
        else:
            raise TypeError("Unsupported trusted state value type: " + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def cpu_clone(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_clone(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_clone(v) for v in value)
    if isinstance(value, list):
        return [cpu_clone(v) for v in value]
    return copy.deepcopy(value)


def flags(model):
    parts = model if isinstance(model, tuple) else (model,)
    return [{name: module.training for name, module in part.named_modules()} for part in parts]


def snapshot(model, optimizer, *, rng=None):
    parts = model if isinstance(model, tuple) else (model,)
    return {"models": [{name: value.detach().cpu().clone() for name, value in part.state_dict().items()} for part in parts],
            "optimizer": cpu_clone(optimizer.state_dict()), "rng": cpu_clone(rng if rng is not None else rng_state()),
            "flags": flags(model)}


def restore_snapshot(model, optimizer, value, *, restore_random=True):
    parts = model if isinstance(model, tuple) else (model,)
    require(len(parts) == len(value["models"]) == len(value["flags"]), "Checkpoint model cardinality differs")
    for part, state, saved_flags in zip(parts, value["models"], value["flags"]):
        part.load_state_dict(state, strict=True)
        named = dict(part.named_modules())
        require(set(named) == set(saved_flags), "Checkpoint flags/module tree differs")
        for name, flag in saved_flags.items():
            named[name].training = flag
    # Adam's noncapturable CPU step tensors can otherwise alias a retained best
    # snapshot and mutate it during a resumed run.
    optimizer.load_state_dict(cpu_clone(value["optimizer"]))
    if restore_random:
        restore_rng(value["rng"])


def atomic_torch(path, value):
    import torch
    path = Path(path)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}


def write_journal(output, context, unit, seed, state):
    output = Path(output)
    payload = {"schema": "ncnc-pilot-own-trusted-state-v1", "identity": context["identity"],
               "unit": unit, "seed": seed, "state": state}
    # Alternate slots preserve the old journal's valid state until new JSON commit.
    import json
    prior = json.loads((output / "JOURNAL.json").read_text()) if (output / "JOURNAL.json").exists() else None
    revision = 0 if prior is None else prior["revision"] + 1
    slot = "STATE_SLOT_" + str(revision % 2) + ".pt"
    receipt = atomic_torch(output / slot, payload)
    journal = {"schema": "ncnc-pilot-private-epoch-journal-v1", "identity": context["identity"],
               "unit": unit, "seed": seed, "epoch": state["epoch"], "revision": revision, "state_file": receipt,
               "UTC": utc(), "project_prediction_visibility": "sealed_until_family_lock"}
    atomic_json(output / "JOURNAL.json", journal)
    return journal


def read_journal(output, context, unit, seed):
    import json
    import torch
    output = Path(output)
    journal = json.loads((output / "JOURNAL.json").read_text())
    require(journal["schema"] == "ncnc-pilot-private-epoch-journal-v1" and journal["identity"] == context["identity"], "Resume contract differs")
    require(journal["unit"] == unit and journal["seed"] == seed, "Resume fit identity differs")
    pin = journal["state_file"]
    require(pin["path"] in ("STATE_SLOT_0.pt", "STATE_SLOT_1.pt"), "Resume is not an own journal slot")
    path = output / pin["path"]
    require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Resume state custody differs")
    # Only this driver's own bound journal slot is an admissible pickle source.
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(value["schema"] == "ncnc-pilot-own-trusted-state-v1" and value["identity"] == context["identity"] and value["unit"] == unit and value["seed"] == seed, "Resume payload identity differs")
    require(value["state"]["epoch"] == journal["epoch"], "Resume epoch differs")
    return value["state"]


def write_selected(output, context, label, unit, seed, candidate, states):
    payload = {"schema": "ncnc-pilot-validation-selected-checkpoint-v1", "identity": context["identity"],
               "unit": unit, "seed": seed, "arm": label, "serving_pool": "mean_raw_logits",
               "validation_graph": "complete_TRAIN_only", "test_stage_supported": False,
               "selection": candidate, "states": states, "resource_state_donor": False,
               "fresh_scientific_initialization": True}
    receipt = atomic_torch(Path(output) / ("SELECTED_" + label + ".pt"), payload)
    # Metric values remain only in private selection/state artifacts until lock.
    atomic_json(Path(output) / ("PRIVATE_SELECTION_" + label + ".json"), {"schema": payload["schema"],
        "identity": context["identity"], "arm": label, "seed": seed, "selection": candidate, "checkpoint": receipt})
    return receipt
