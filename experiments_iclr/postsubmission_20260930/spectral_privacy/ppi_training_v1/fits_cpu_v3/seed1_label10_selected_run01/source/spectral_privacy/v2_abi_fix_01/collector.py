"""Read-only, train-only named-neighbor matrix gradient collection."""

import copy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import random
import sys

import torch
from torch.nn import functional as F

if __package__:
    from .contracts import (SCHEMA, code_hashes, json_hash, pack_float64, seal_record,
                            state_hash, tensor_fingerprint)
    from .spectral_core import RULE, choose_bases, serialize_selection
else:
    from contracts import (SCHEMA, code_hashes, json_hash, pack_float64, seal_record,
                           state_hash, tensor_fingerprint)
    from spectral_core import RULE, choose_bases, serialize_selection


FOLD_SCHEMA = "spectral-P10-D4-E4-F2-v2"
LOSS_RULE = "unscaled own-member mean finite observed-entry BCE; equal weight per complete graph"


@dataclass(frozen=True)
class GradientUnit:
    unit_id: str
    graph: object
    features: torch.Tensor
    targets: torch.Tensor
    observed_mask: torch.Tensor | None = None
    split: str = "train"


def partition_manifest(fit, discovery, screen, audit, *, fit_seed=700, score_seed=704):
    value = {"schema": FOLD_SCHEMA, "fit": list(fit), "discovery": list(discovery),
             "screen": list(screen), "audit": list(audit),
             "fit_partition_seed": fit_seed, "score_partition_seed": score_seed}
    validate_partition(value)
    return value | {"sha256": json_hash(value)}


def validate_partition(manifest):
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    expected = {"schema", "fit", "discovery", "screen", "audit",
                "fit_partition_seed", "score_partition_seed"}
    if set(payload) != expected or payload["schema"] != FOLD_SCHEMA:
        raise ValueError("Unsupported partition manifest")
    if "sha256" in manifest and manifest["sha256"] != json_hash(payload):
        raise ValueError("Partition hash mismatch")
    counts = {"fit": 10, "discovery": 4, "screen": 4, "audit": 2}
    seen = set()
    for group, count in counts.items():
        ids = payload[group]
        if not isinstance(ids, list) or len(ids) != count:
            raise ValueError("V2 first pilot requires P10/D4/E4/F2")
        if any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("Training-unit IDs must be nonempty strings")
        if len(set(ids)) != count or seen.intersection(ids):
            raise ValueError("Duplicate/overlapping training-unit IDs")
        seen.update(ids)
    if payload["fit_partition_seed"] != 700 or payload["score_partition_seed"] != 704:
        raise ValueError("Partition seeds differ from the frozen v2 proposal")


def observed_bce(logits, unit):
    target = unit.targets
    if logits.shape != target.shape or not target.is_floating_point():
        raise ValueError("BCE requires matching floating logits/targets shapes")
    observed = torch.isfinite(target)
    mask = unit.observed_mask
    if mask is not None:
        if mask.dtype != torch.bool or mask.device != target.device:
            raise ValueError("Observed mask must be boolean on the target device")
        if mask.shape == target.shape:
            observed = observed & mask
        elif mask.ndim == 1 and target.ndim >= 1 and len(mask) == target.shape[0]:
            observed = observed & mask.reshape((-1,) + (1,) * (target.ndim - 1))
        else:
            raise ValueError("Observed mask shape mismatch")
    count = int(observed.sum())
    if count == 0:
        raise ValueError("A score graph has no finite observed targets")
    selected = target[observed]
    if not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError("BCE targets must be binary or missing")
    loss = F.binary_cross_entropy_with_logits(logits[observed], selected.to(logits.dtype),
                                              reduction="mean")
    return loss, count


def unit_fingerprint(unit):
    value = {"unit_id": unit.unit_id, "split": unit.split,
             "features": tensor_fingerprint(unit.features),
             "targets": tensor_fingerprint(unit.targets),
             "observed_mask": None if unit.observed_mask is None else tensor_fingerprint(unit.observed_mask)}
    for name in ("edge_index", "edge_attr", "pos", "batch"):
        tensor = getattr(unit.graph, name, None)
        if tensor is not None:
            value[name] = tensor_fingerprint(tensor)
    return value


class PreservedProbe:
    """Preserve values, .grad objects/values, modes, and initialized RNGs."""

    def __init__(self, model):
        self.model = model
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self.state_sha256 = state_hash(model)
        self.grads = [(parameter, parameter.grad,
                       None if parameter.grad is None else parameter.grad.detach().clone())
                      for parameter in model.parameters()]
        self.modes = [(module, module.training) for module in model.modules()]
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        self.python_rng = random.getstate()
        self.numpy = sys.modules.get("numpy")
        self.numpy_rng = (copy.deepcopy(self.numpy.random.get_state())
                          if self.numpy is not None else None)

    def matches(self):
        unchanged = self.state_sha256 == state_hash(self.model)
        for parameter, old_reference, old_value in self.grads:
            unchanged = unchanged and parameter.grad is old_reference
            if old_value is not None:
                unchanged = unchanged and torch.equal(parameter.grad, old_value)
        unchanged = unchanged and torch.equal(self.cpu_rng, torch.get_rng_state())
        unchanged = unchanged and self.python_rng == random.getstate()
        if self.cuda_rng is not None:
            current = torch.cuda.get_rng_state_all()
            unchanged = unchanged and len(current) == len(self.cuda_rng)
            unchanged = unchanged and all(torch.equal(a, b) for a, b in zip(current, self.cuda_rng))
        if self.numpy_rng is not None:
            current = self.numpy.random.get_state()
            unchanged = unchanged and current[0] == self.numpy_rng[0]
            unchanged = unchanged and self.numpy.array_equal(current[1], self.numpy_rng[1])
            unchanged = unchanged and current[2:] == self.numpy_rng[2:]
        return bool(unchanged)

    def restore(self):
        with torch.no_grad():
            current = self.model.state_dict()
            for name, value in self.state.items():
                if not torch.equal(current[name], value):
                    current[name].copy_(value)
            for parameter, reference, value in self.grads:
                parameter.grad = reference
                if reference is not None and not torch.equal(reference, value):
                    reference.copy_(value)
        for module, training in self.modes:
            module.training = training
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        random.setstate(self.python_rng)
        if self.numpy_rng is not None:
            self.numpy.random.set_state(self.numpy_rng)


def environment_contract():
    pyg = sys.modules.get("torch_geometric")
    if pyg is not None:
        pyg_version = str(pyg.__version__)
    else:
        try:
            pyg_version = version("torch-geometric")
        except PackageNotFoundError:
            pyg_version = None
    return {"torch_version": str(torch.__version__), "python_version": sys.version,
            "torch_geometric_version": pyg_version,
            "torch_configuration_sha256": json_hash(torch.__config__.show()),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "byteorder": sys.byteorder,
            "selection_device": "cpu", "selection_dtype": "torch.float64"}


def collect_record(model, score_units, manifest, *, source_sha256, dataset_identity,
                   optimizer_seed, checkpoint_epoch=25, random_control_seed=800):
    """Collect only D/E labels; P/F are IDs in the partition manifest.

    Model must be SourceSAGEEnsemble at a tied state. Caller attests that
    its checkpoint was fitted on P labels only. Metadata is not proof of
    that provenance. No downloader, optimizer, checkpoint loader, or task
    scoring is implemented here.
    """
    validate_partition(manifest)
    if checkpoint_epoch != 25 or model.members != 4 or not model.is_tied_at_state():
        raise ValueError("Require the epoch25 tied M4 probe")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("Require the frozen source models.py SHA256")
    if not isinstance(dataset_identity, dict) or not dataset_identity:
        raise ValueError("Require explicit new-study dataset/mask identity metadata")
    by_id = {}
    for unit in score_units:
        if unit.split != "train" or unit.unit_id in by_id:
            raise ValueError("Only unique training graph units are accepted")
        by_id[unit.unit_id] = unit
    ordered_ids = manifest["discovery"] + manifest["screen"]
    if set(by_id) != set(ordered_ids):
        raise ValueError("Supply exactly D/E graph units, excluding P/F labels")
    targets = model.named_neighbor_matrices()
    if len(targets) != 3 or any(p.shape != (128, 128) for p in targets.values()):
        raise ValueError("Require the three fixed h128 neighbor matrices")
    if len({id(p) for p in targets.values()}) != len(targets):
        raise ValueError("Named target matrices must not overlap")
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    fingerprints = [unit_fingerprint(by_id[key]) for key in ordered_ids]
    snapshot = PreservedProbe(model)
    keys, parameters = tuple(sorted(targets)), tuple(targets[key] for key in sorted(targets))
    raw = {group: {key: [] for key in keys} for group in ("discovery", "screen")}
    graph_metadata = {}
    try:
        model.eval()
        with torch.enable_grad():
            for group in ("discovery", "screen"):
                for unit_id in manifest[group]:
                    unit = by_id[unit_id]
                    rows, losses, counts = {key: [] for key in keys}, [], []
                    for member in range(model.members):
                        logits = model.forward_member(unit.graph, unit.features, member)
                        loss, count = observed_bce(logits, unit)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("Nonfinite own-member graph loss")
                        # Own graph/member mean loss: no fold or member division.
                        gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
                        for key, gradient in zip(keys, gradients):
                            value = gradient.detach().cpu().double().contiguous()
                            if not bool(torch.isfinite(value).all()):
                                raise ValueError("Nonfinite named matrix gradient")
                            rows[key].append(value)
                        losses.append(float(loss.detach()))
                        counts.append(count)
                    if len(set(counts)) != 1:
                        raise ValueError("Members used different observed-target counts")
                    graph_metadata[unit_id] = {"observed_entries": counts[0], "member_loss": losses}
                    for key in keys:
                        raw[group][key].append(torch.stack(rows[key]))
        raw = {group: {key: torch.stack(rows) for key, rows in group_rows.items()}
               for group, group_rows in raw.items()}
        bases, details, statistics = choose_bases(raw["discovery"], raw["screen"],
                                                  n0=RULE["n0"], rank_max=RULE["rank_max_per_target"])
        selection = serialize_selection(bases, details, statistics, raw["discovery"], random_control_seed)
        if not snapshot.matches():
            raise RuntimeError("Probe forward/gradient collection changed state, .grad, or RNG")
        if fingerprints != [unit_fingerprint(by_id[key]) for key in ordered_ids]:
            raise RuntimeError("Probe mutated supplied training graph units")
        record = {
            "schema": SCHEMA, "source_models_sha256": source_sha256,
            "dataset_identity": dataset_identity, "optimizer_seed": int(optimizer_seed),
            "checkpoint_epoch": checkpoint_epoch, "members": model.members,
            "model_state_sha256": snapshot.state_sha256,
            "partition": manifest, "loss_rule": LOSS_RULE,
            "gradient_scale": "unscaled own-member own-graph mean loss",
            "fold_weighting": "equal_complete_graph",
            "input_fingerprints": fingerprints, "input_identity_sha256": json_hash(fingerprints),
            "graph_metadata": graph_metadata,
            "targets": {key: {"actual_parameter_name": parameter_names[id(targets[key])],
                               "shape": list(targets[key].shape), "dtype": str(targets[key].dtype)}
                        for key in keys},
            "groups": {group: {"unit_ids": manifest[group],
                                "matrix_gradients": {key: pack_float64(value) for key, value in rows.items()}}
                       for group, rows in raw.items()},
            "selection": selection, "code_sha256": code_hashes(),
            "environment": environment_contract(),
            "state_grads_modes_and_rng_preserved": True,
            "provenance_boundary": "Caller supplies P-only probe and training-only D/E units; no F labels supplied",
        }
        return seal_record(record)
    finally:
        snapshot.restore()
