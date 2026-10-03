"""Source-bound synthetic replay helpers. No dataset/heldout accessors."""
import copy
import hashlib
import io
import random
from pathlib import Path
import numpy as np
import torch
from prototype import CompletionTwin, Recipe, configuration, native_optimizer


def code_hashes():
    here = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(here.glob("*.py"))}


def manifest_sha():
    return hashlib.sha256((Path(__file__).resolve().parent/"MANIFEST.json").read_bytes()).hexdigest()


def rng_state():
    return {"python":random.getstate(), "numpy":np.random.get_state(),
            "torch_cpu":torch.get_rng_state().clone(),
            "torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state["torch_cuda"]):
            raise RuntimeError("CUDA RNG topology differs from selected state")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def selected_bytes(model, optimizer, mode, selection_reference):
    if mode not in ("private", "pooled_after_clamp") or not selection_reference:
        raise ValueError("Require exact twin identity and explicit selection provenance")
    payload = {"schema":"ncnc-completion-selected-state-v1", "code_hashes":code_hashes(),
               "prototype_manifest_sha256":manifest_sha(),
               "recipe":configuration(model), "completion_mode":mode,
               "serving_pool":"mean_raw_logits", "selection_reference":copy.deepcopy(selection_reference),
               "model":copy.deepcopy(model.state_dict()),
               "optimizer":copy.deepcopy(optimizer.state_dict()), "rng":rng_state(),
               "training_flags":{name:module.training for name,module in model.named_modules()}}
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def restore_selected_bytes(blob, device="cpu"):
    # Only locally produced, source-bound blobs are supported. This helper is
    # not an importer for untrusted files or artifacts of another study.
    payload = torch.load(io.BytesIO(blob), map_location=device, weights_only=False)
    if payload["schema"] != "ncnc-completion-selected-state-v1" or payload["code_hashes"] != code_hashes():
        raise RuntimeError("Selected source binding mismatch")
    if payload["prototype_manifest_sha256"] != manifest_sha():
        raise RuntimeError("Selected prototype manifest mismatch")
    if payload["serving_pool"] != "mean_raw_logits" or payload["completion_mode"] not in ("private","pooled_after_clamp"):
        raise RuntimeError("Selected serving/completion identity mismatch")
    # Constructors consume RNG; restore the selected RNG only after complete
    # model and Adam restoration. Preserve the saved floating dtype explicitly.
    floating = next(value for value in payload["model"].values() if value.is_floating_point())
    model = CompletionTwin(Recipe(**payload["recipe"])).to(device=device, dtype=floating.dtype)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = native_optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    named = dict(model.named_modules())
    if set(named) != set(payload["training_flags"]):
        raise RuntimeError("Selected module tree mismatch")
    for name, flag in payload["training_flags"].items():
        named[name].training = flag
    restore_rng(payload["rng"])
    return model, optimizer, payload
