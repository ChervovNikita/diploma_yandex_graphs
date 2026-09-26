"""Post hoc, prospectively frozen 72-cell all-layer SAGE factor ablation.

The primary 432-cell source and outputs are read only. Training never receives
test indices or labels. Independent selection is in verify_all_layer_factor.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import tuning as primary
from models import SAGEModule, TABMModel


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "all_layer_factor_results"
FREEZE = OUT / "FROZEN_ALL_LAYER_STUDY.json"
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
SEEDS = (0, 1, 2)
CANDIDATES = primary.CANDIDATES
DEFAULT = primary.DEFAULT
EPOCHS = primary.EPOCHS
MEMBERS = primary.MEMBERS
TRACE_FIELDS = primary.TRACE_FIELDS
SOURCE_FILES = ("all_layer_factor_study.py", "verify_all_layer_factor.py",
                "ALL_LAYER_FACTOR_PROTOCOL.md")
PRIMARY_LOCK_SHA = "176c68b855835a625086a1287a35e76cff43ecf144e7dbe19297ebb2d3ef6f99"
CONTROL_DIR = ROOT / "same_runtime_tied36_results"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


class MemberFactorLinear(nn.Module):
    """Identity-initialized member factors around an existing shared map."""

    def __init__(self, shared: nn.Module):
        super().__init__()
        weight = getattr(shared, "weight", None)
        require(weight is not None and weight.ndim == 2,
                "Expected a two-dimensional shared linear weight")
        outputs, inputs = weight.shape
        self.shared = shared
        self.R = nn.Parameter(weight.new_ones((MEMBERS, inputs)))
        self.S = nn.Parameter(weight.new_ones((MEMBERS, outputs)))
        self.B = nn.Parameter(weight.new_zeros((MEMBERS, outputs)))
        self.active_member: int | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.active_member
        require(m is not None and 0 <= m < MEMBERS, "Select a member before forwarding")
        return self.shared(x * self.R[m]) * self.S[m] + self.B[m]


class AllLayerFactorModel(nn.Module):
    def __init__(self, bundle, device: torch.device):
        super().__init__()
        self.base = TABMModel(**primary.common_kwargs(bundle),
                              tabm_inits=MEMBERS, device=device).to(device)
        factors = []
        require(len(self.base.residual_modules) == primary.DEPTH,
                "Unexpected residual depth")
        for residual in self.base.residual_modules:
            sage = residual.module
            require(isinstance(sage, SAGEModule), "Expected residual SAGE module")
            for owner, name in ((sage.conv, "lin_l"), (sage.conv, "lin_r"),
                                (sage.feed_forward_module, "linear_1"),
                                (sage.feed_forward_module, "linear_2")):
                wrapped = MemberFactorLinear(getattr(owner, name))
                setattr(owner, name, wrapped)
                factors.append(wrapped)
        require(len(factors) == 4 * primary.DEPTH, "Wrong factorized map count")
        self.factor_layers = tuple(factors)  # Registered only inside self.base.

    def forward(self, graph, x, tabm_seed: int):
        require(0 <= tabm_seed < MEMBERS, "Invalid member")
        for layer in self.factor_layers:
            layer.active_member = tabm_seed
        return self.base(graph, x, tabm_seed=tabm_seed)


def shared_state(model: AllLayerFactorModel) -> dict[str, torch.Tensor]:
    output = {}
    for name, value in model.base.state_dict().items():
        if name.endswith((".R", ".S", ".B")) and ".shared." not in name and \
                "residual_modules" in name:
            continue
        output[name.replace(".shared.", ".")] = value
    return output


def make_model(bundle, device: torch.device):
    model = AllLayerFactorModel(bundle, device)
    params = list(model.parameters())
    require(len({p.untyped_storage().data_ptr() for p in params}) == len(params),
            "Unexpected aliased parameter storage")
    return model


def expected_freeze() -> dict:
    primary_sha = primary.check_freeze()
    require(primary.sha256_file(ROOT / "VALIDATION_SELECTION_LOCK.json") ==
            PRIMARY_LOCK_SHA, "Original 432-cell selection lock changed")
    source = {name: primary.sha256_file(ROOT / name) for name in SOURCE_FILES}
    base_source = {name: primary.sha256_file(ROOT / name)
                   for name in ("tuning.py", "models.py", "FROZEN_STUDY.json")}
    base = json.loads((ROOT / "FROZEN_STUDY.json").read_text())
    return {
        "protocol": "all_layer_factor_placement_posthoc_v1",
        "posthoc_choice_after_primary_outcomes": True,
        "primary_frozen_study_sha256": primary_sha,
        "primary_selection_lock_sha256": PRIMARY_LOCK_SHA,
        "source_sha256": source,
        "primary_source_sha256": base_source,
        "graphs": base["graphs"],
        "matrix": {
            "datasets": list(DATASETS), "seeds": list(SEEDS),
            "candidates": [{"lr": lr, "weight_decay": wd} for lr, wd in CANDIDATES],
            "default_candidate": list(DEFAULT), "epochs": EPOCHS,
            "members": MEMBERS, "width": primary.WIDTH, "depth": primary.DEPTH,
            "dropout": primary.DROP,
            "objective": "mean of four member cross-entropies",
            "checkpoint_rule": "highest pooled validation accuracy, then lowest CE, then earliest epoch",
            "candidate_rule": "highest three-seed mean validation accuracy, then lowest mean CE, then lower LR/decay",
            "test_rule": "after 72-cell all-layer, 36-cell same-runtime TIED, and matching primary 432-cell validation locks; selected/default only",
            "complete_audit_target_utc": "2026-09-26T06:30:00Z",
        },
    }


def check_freeze() -> str:
    require(FREEZE.is_file(), "Missing prospective all-layer freeze")
    saved = json.loads(FREEZE.read_text())
    require(saved == expected_freeze(), "All-layer source/data/matrix differs from freeze")
    return primary.sha256_file(FREEZE)


def identity_audit(model, bundle, seed: int, device: torch.device) -> dict:
    post_python, post_numpy = random.getstate(), np.random.get_state()
    post_cpu = torch.get_rng_state().clone()
    post_cuda = (torch.cuda.get_rng_state(device).clone()
                 if device.type == "cuda" else None)
    model.eval()
    with torch.no_grad():
        actual = torch.stack(primary.member_logits(model, "all_layer", bundle))
    require(torch.equal(post_cpu, torch.get_rng_state()) and
            (post_cuda is None or torch.equal(post_cuda, torch.cuda.get_rng_state(device))),
            "Candidate eval changed RNG state")
    primary.seed_all(seed)
    reference, canonical_sha = primary.make_model("tied", bundle, device)
    reference_post_python, reference_post_numpy = random.getstate(), np.random.get_state()
    reference_post_cpu = torch.get_rng_state().clone()
    reference_post_cuda = (torch.cuda.get_rng_state(device).clone()
                           if device.type == "cuda" else None)
    require(torch.equal(post_cpu, reference_post_cpu) and
            (post_cuda is None or torch.equal(post_cuda, reference_post_cuda)),
            "Post-construction RNG differs from TIED")
    require(post_python == reference_post_python and
            post_numpy[0] == reference_post_numpy[0] and
            np.array_equal(post_numpy[1], reference_post_numpy[1]) and
            post_numpy[2:] == reference_post_numpy[2:],
            "Post-construction Python/NumPy RNG differs from TIED")
    reference.eval()
    mapped = shared_state(model)
    ref_state = reference.state_dict()
    require(set(mapped) == set(ref_state) and
            all(torch.equal(mapped[k], ref_state[k]) for k in ref_state),
            "Shared initial tensors differ from TIED")
    with torch.no_grad():
        expected = torch.stack(primary.member_logits(reference, "tied", bundle))
    diff = float((actual - expected).abs().max().item())
    require(diff <= primary.INITIAL_LOGIT_TOL and
            torch.equal(actual.argmax(-1), expected.argmax(-1)),
            f"Initial member logits/decisions differ: {diff}")
    random.setstate(post_python)
    np.random.set_state(post_numpy)
    torch.set_rng_state(post_cpu)
    if post_cuda is not None:
        torch.cuda.set_rng_state(post_cuda, device)
    return {"canonical_tied_state_sha256": canonical_sha,
            "initial_member_logits_max_abs_diff": diff,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "factor_parameter_count": sum(p.numel() for layer in model.factor_layers
                                          for p in (layer.R, layer.S, layer.B)),
            "initial_member_logits_sha256": primary.tensor_sha(actual),
            "post_construction_cpu_rng_sha256": primary.tensor_sha(post_cpu),
            "post_construction_cuda_rng_sha256":
                primary.tensor_sha(post_cuda) if post_cuda is not None else None}


def gradient_preflight(bundle, device: torch.device) -> dict:
    primary.seed_all(0)
    reference, _ = primary.make_model("tied", bundle, device)
    primary.seed_all(0)
    candidate = make_model(bundle, device)
    reference.train(); candidate.train()
    def backward(model, arm):
        model.zero_grad(set_to_none=True)
        primary.seed_all(782913)
        for member in range(MEMBERS):
            logits = model(bundle.graph, bundle.x, tabm_seed=member)
            (F.cross_entropy(logits.index_select(0, bundle.train_idx),
                             bundle.train_y) / MEMBERS).backward()
    backward(reference, "tied")
    backward(candidate, "all_layer")
    candidate_grads = {}
    for name, parameter in candidate.base.named_parameters():
        if ".shared." in name:
            candidate_grads[name.replace(".shared.", ".")] = parameter.grad
        elif not (name.endswith((".R", ".S", ".B")) and
                  "residual_modules" in name):
            candidate_grads[name] = parameter.grad
    ref_grads = {name: p.grad for name, p in reference.named_parameters()}
    require(set(candidate_grads) == set(ref_grads), "Shared gradient keys differ")
    maximum = max(float((candidate_grads[k] - ref_grads[k]).abs().max().item())
                  for k in ref_grads)
    require(maximum <= 1e-5 and all(torch.isfinite(v).all() for v in candidate_grads.values()),
            f"Matched first-step shared gradient mismatch: {maximum}")
    factor_grads = {name: parameter.grad for name, parameter in
                    candidate.base.named_parameters()
                    if "residual_modules" in name and
                    name.endswith((".R", ".S", ".B"))}
    require(len(factor_grads) == 12 * primary.DEPTH and
            all(g is not None and torch.isfinite(g).all() for g in factor_grads.values()) and
            any(torch.count_nonzero(g).item() > 0 for g in factor_grads.values()),
            "Added factors lack finite nonzero first-step gradients")
    factor_gradient_audit = {
        name: {"finite": bool(torch.isfinite(g).all().item()),
               "nonzero_count": int(torch.count_nonzero(g).item()),
               "l2_norm": float(g.norm().item())}
        for name, g in factor_grads.items()}
    return {"max_shared_gradient_abs_diff": maximum,
            "factor_gradient_tensors": len(factor_grads),
            "factor_gradients": factor_gradient_audit}


def preflight(device: torch.device) -> None:
    freeze_sha = check_freeze()
    report = {"freeze_sha256": freeze_sha, "device": str(device), "graphs": {}}
    for dataset in DATASETS:
        bundle, _ = primary.load_graph(dataset, device)
        records = []
        for seed in SEEDS:
            primary.seed_all(seed)
            model = make_model(bundle, device)
            records.append(identity_audit(model, bundle, seed, device))
        report["graphs"][dataset] = records
    cora, _ = primary.load_graph("cora", device)
    report["first_gradient"] = gradient_preflight(cora, device)
    primary.write_json(OUT / f"PREFLIGHT_{device.type.upper()}.json", report)
    print(json.dumps({"preflight": "PASS", "device": str(device),
                      "freeze_sha256": freeze_sha,
                      "first_gradient": report["first_gradient"]}), flush=True)


def cell_dir(dataset: str, lr: float, wd: float, seed: int) -> Path:
    return OUT / "results" / dataset / primary.candidate_name(lr, wd) / f"seed{seed}"


def train_one(dataset, bundle, freeze_sha, lr, wd, seed, device):
    final = cell_dir(dataset, lr, wd, seed)
    if final.exists():
        row_path, trace_path, checkpoint, valid_predictions = (final / name for name in
            ("result.json", "validation_trace.csv", "checkpoint.pt", "validation_predictions.npz"))
        require(row_path.is_file() and trace_path.is_file(), "Incomplete existing cell")
        row = json.loads(row_path.read_text())
        require(row["freeze_sha256"] == freeze_sha and
                row["validation_trace_sha256"] == primary.sha256_file(trace_path) and
                row["dataset"] == dataset and row["lr"] == lr and
                row["weight_decay"] == wd and row["seed"] == seed and
                (row["failure"] is not None or
                 (checkpoint.is_file() and row["checkpoint_sha256"] ==
                  primary.sha256_file(checkpoint) and
                  valid_predictions.is_file() and
                  row["validation_predictions_sha256"] ==
                  primary.sha256_file(valid_predictions))),
                "Existing cell fails identity/hash gate")
        return
    work = final.with_name(final.name + ".inprogress")
    require(not work.exists(), f"Interrupted cell needs inspection: {work}")
    work.mkdir(parents=True)
    primary.seed_all(seed)
    model = make_model(bundle, device)
    initial = identity_audit(model, bundle, seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_acc, best_ce, best_epoch, best_state = -math.inf, math.inf, 0, None
    trace, failure = [], None
    if device.type == "cuda": torch.cuda.synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for member in range(MEMBERS):
            logits = model(bundle.graph, bundle.x, tabm_seed=member)
            loss = F.cross_entropy(logits.index_select(0, bundle.train_idx), bundle.train_y)
            if not torch.isfinite(loss):
                failure = f"nonfinite training CE at epoch {epoch}"; break
            (loss / MEMBERS).backward()
        if failure: break
        optimizer.step()
        va, vc, _, _ = primary.evaluate(model, "all_layer", bundle,
                                       bundle.valid_idx, bundle.valid_y)
        if not (np.isfinite(va) and np.isfinite(vc)):
            failure = f"nonfinite validation at epoch {epoch}"; break
        improved = va > best_acc or (va == best_acc and vc < best_ce)
        if improved:
            best_acc, best_ce, best_epoch = va, vc, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        trace.append((epoch, va, vc, int(improved)))
    if device.type == "cuda": torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    with (work / "validation_trace.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(TRACE_FIELDS); writer.writerows(trace)
    row = {"protocol": "all_layer_factor_placement_posthoc_v1",
           "freeze_sha256": freeze_sha, "dataset": dataset, "arm": "all_layer",
           "lr": lr, "weight_decay": wd, "seed": seed,
           "epochs_required": EPOCHS, "epochs_completed": len(trace),
           "training_seconds": elapsed, "initialization": initial,
           "validation_trace_sha256": primary.sha256_file(work / "validation_trace.csv"),
           "failure": failure}
    if failure is None:
        require(len(trace) == EPOCHS and best_state is not None,
                "Incomplete finite training trace")
        checkpoint = {"state_dict": best_state, "epoch": best_epoch,
                      "dataset": dataset, "arm": "all_layer", "lr": lr,
                      "weight_decay": wd, "seed": seed, "freeze_sha256": freeze_sha}
        torch.save(checkpoint, work / "checkpoint.pt")
        model.load_state_dict(best_state, strict=True)
        va, vc, pooled, member = primary.evaluate(model, "all_layer", bundle,
                                                  bundle.valid_idx, bundle.valid_y)
        require(abs(va - best_acc) <= 1e-7 and abs(vc - best_ce) <= 1e-6,
                "Selected validation checkpoint replay differs")
        np.savez_compressed(work / "validation_predictions.npz",
                            pooled_logits=pooled.numpy(), member_logits=member.numpy(),
                            valid_indices=bundle.valid_idx.detach().cpu().numpy(),
                            valid_labels=bundle.valid_y.detach().cpu().numpy())
        row.update({"selected_epoch": best_epoch,
                    "selected_valid_accuracy": best_acc,
                    "selected_valid_ce": best_ce,
                    "selected_valid_pooled_logits_sha256": primary.tensor_sha(pooled),
                    "validation_predictions_sha256":
                        primary.sha256_file(work / "validation_predictions.npz"),
                    "selected_state_sha256": primary.state_sha(best_state),
                    "checkpoint_sha256": primary.sha256_file(work / "checkpoint.pt")})
    primary.write_json(work / "result.json", row)
    work.rename(final)
    print(json.dumps({"completed": str(final.relative_to(OUT)),
                      "seconds": elapsed, "failure": failure}), flush=True)


def run_dataset(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    bundle, _ = primary.load_graph(dataset, device, include_test=False)
    for lr, wd in CANDIDATES:
        for seed in SEEDS:
            train_one(dataset, bundle, freeze_sha, lr, wd, seed, device)


def score(dataset: str, device: torch.device) -> None:
    freeze_sha = check_freeze()
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    primary_lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    control_lock_path = CONTROL_DIR / "VALIDATION_SELECTION_LOCK.json"
    control_freeze_path = CONTROL_DIR / "FROZEN_STUDY.json"
    require(lock_path.is_file() and primary_lock_path.is_file() and
            control_lock_path.is_file() and control_freeze_path.is_file(),
            "All-layer, same-runtime TIED36, and original validation locks are required")
    lock = json.loads(lock_path.read_text())
    primary_lock = json.loads(primary_lock_path.read_text())
    control_lock = json.loads(control_lock_path.read_text())
    require(lock["freeze_sha256"] == freeze_sha and len(lock["cells"]) == 72 and
            lock["primary_selection_lock_sha256"] == PRIMARY_LOCK_SHA ==
            primary.sha256_file(primary_lock_path) and
            len(primary_lock["cells"]) == 432 and
            primary_lock["freeze_sha256"] == primary.check_freeze(),
            "Validation locks differ from frozen studies")
    require(control_lock["protocol"] == "same_runtime_tied36_v1" and
            len(control_lock["cells"]) == 36 and
            control_lock["freeze_sha256"] == primary.sha256_file(control_freeze_path) and
            control_lock["original_validation_lock_sha256"] == PRIMARY_LOCK_SHA,
            "Same-runtime TIED36 validation lock is incomplete or changed")
    bundle, _ = primary.load_graph(dataset, device, include_test=True)
    chosen = tuple(lock["selections"][dataset]["selected_candidate"])
    for lr, wd in dict.fromkeys((chosen, DEFAULT)):
        for seed in SEEDS:
            cell = cell_dir(dataset, lr, wd, seed)
            key = f"{dataset}/{primary.candidate_name(lr, wd)}/seed{seed}"
            frozen = lock["cells"][key]
            require(primary.sha256_file(cell / "result.json") == frozen["result_sha256"] and
                    primary.sha256_file(cell / "checkpoint.pt") == frozen["checkpoint_sha256"],
                    f"Cell changed after lock: {key}")
            out = OUT / "scores" / key
            require(not out.exists(), f"Refusing overwrite of score: {key}")
            primary.seed_all(seed)
            model = make_model(bundle, device)
            saved = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"], strict=True)
            va, vc, vp, _ = primary.evaluate(model, "all_layer", bundle,
                                              bundle.valid_idx, bundle.valid_y)
            old = json.loads((cell / "result.json").read_text())
            require(abs(va - old["selected_valid_accuracy"]) <= 1e-7 and
                    abs(vc - old["selected_valid_ce"]) <= 1e-6,
                    f"Validation replay differs: {key}")
            ta, tc, tp, tm = primary.evaluate(model, "all_layer", bundle,
                                              bundle.test_idx, bundle.test_y)
            out.mkdir(parents=True)
            np.savez_compressed(out / "predictions.npz",
                                valid_pooled_logits=vp.numpy(),
                                test_pooled_logits=tp.numpy(),
                                test_member_logits=tm.numpy())
            primary.write_json(out / "score.json", {
                "freeze_sha256": freeze_sha,
                "validation_selection_lock_sha256": primary.sha256_file(lock_path),
                "primary_selection_lock_sha256": primary.sha256_file(primary_lock_path),
                "same_runtime_tied36_validation_lock_sha256": primary.sha256_file(control_lock_path),
                "dataset": dataset, "arm": "all_layer", "lr": lr,
                "weight_decay": wd, "seed": seed,
                "selected_candidate": (lr, wd) == chosen,
                "predeclared_default": (lr, wd) == DEFAULT,
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "valid_accuracy": va, "valid_ce": vc,
                "test_accuracy": ta, "test_ce": tc,
                "test_predictions_sha256": primary.sha256_file(out / "predictions.npz"),
            })
            print(json.dumps({"scored": key, "test_accuracy": ta}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("check-freeze")
    sub.add_parser("preflight").add_argument("--device", default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--device", default="cuda")
    scoring = sub.add_parser("score")
    scoring.add_argument("--dataset", choices=DATASETS, required=True)
    scoring.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not FREEZE.exists(), "Refusing to overwrite prospective freeze")
        primary.write_json(FREEZE, expected_freeze())
        print(json.dumps({"freeze_sha256": primary.sha256_file(FREEZE)}), flush=True)
    elif args.command == "check-freeze":
        print(json.dumps({"freeze_sha256": check_freeze()}), flush=True)
    elif args.command == "preflight":
        preflight(torch.device(args.device))
    elif args.command == "run":
        run_dataset(args.dataset, torch.device(args.device))
    elif args.command == "score":
        score(args.dataset, torch.device(args.device))


if __name__ == "__main__":
    main()
