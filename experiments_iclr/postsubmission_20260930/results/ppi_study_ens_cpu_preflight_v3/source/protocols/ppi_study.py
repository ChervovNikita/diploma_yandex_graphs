"""Prospective PPI development fits and selected-checkpoint replay.

Only official train/validation loaders exist. CPU defaults to one-graph
engineering preflight; full fits require an idle explicitly named GPU.
The layer-selection study binds all three train-only decision locks before
any validation outcomes are opened. Earlier engineering runs are preserved.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
AUTHORIZED_PHASES = [Path("/Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930").resolve(),
                     Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930").resolve()]
candidate = os.environ.get("GNNM_PHASE_ROOT")
if candidate is None:
    candidate = next((p for p in AUTHORIZED_PHASES if p == CODE or p in CODE.parents), None)
if candidate is None or Path(candidate).resolve() not in AUTHORIZED_PHASES:
    raise ValueError("Require the explicit authorized phase root; filesystem root is never allowed")
PHASE = Path(candidate).resolve()
sys.path.insert(0, str(CODE / "protocols"))
import ppi_pilot as impl
import torch
from torch_geometric.datasets import PPI
from torch_geometric.nn import SAGEConv

RECIPE = {"width": 128, "layers": 3, "members": 4, "hidden_dim_multiplier": 1,
          "dropout": 0.2, "optimizer": "AdamW", "learning_rate": 0.001,
          "weight_decay": 0.0, "epsilon": 1e-8, "epochs": 300,
          "dtype": "float32", "amp": False, "threads": 1,
          "batch": "one complete training graph per optimizer step",
          "loss": "equal graph mean of mean member observed-entry BCE",
          "order": "shuffle20 with Random(19709+1000*seed+epoch)",
          "mask": "nested node randperm with seed9701+officialtrainindex",
          "pooling": "mean member raw logits; binary threshold0",
          "selection": "maximum pooled validation microF1, minimum pooledBCE, earliest epoch"}
SHARING = ["gnnm", "first_private", "middle_private", "last_private", "untied", "base", "ens", "multihead", "packed_full"]
ARCHITECTURE = ["centered", "no_recycling", "independent_narrow", "uncentered_feedback",
                "packed_channels", "packed_parameters", "base_parameters", "base", "multihead", "gnnm", "ens", "packed_full"]


def inside(path):
    path = Path(path).resolve()
    if PHASE not in path.parents:
        raise ValueError("Inputs/outputs must stay inside the permitted research phase")
    return path


def sha(path):
    return impl.sha(path)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_paths():
    names = ["protocols/ppi_study.py", "protocols/ppi_pilot.py", "protocols/baseline_adapters.py",
             "efficient_paths/centered_paths.py", "source_snapshot/models.py", "new_tasks/shareable_freeze.py"]
    return {name: CODE / name for name in names}


def sources():
    return {name: sha(path) for name, path in source_paths().items()}


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "compiled_cuda": torch.version.cuda,
            "packages": dict(sorted((d.metadata["Name"], d.version)
                             for d in importlib.metadata.distributions() if d.metadata["Name"])),
            "runtime_class_sha256": {c.__name__: hashlib.sha256(inspect.getsource(c).encode()).hexdigest()
                                     for c in (PPI, SAGEConv)},
            "dtype": str(torch.get_default_dtype()),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "no_cuda_context_required": True}


def data_freeze(manifest):
    manifest = inside(manifest)
    data = json.loads(manifest.read_text())
    if data["dataset"] != "ppi" or data["scoring_performed"] or data["confirmation_data_accessed"]:
        raise ValueError("Require the unscored development PPI data freeze")
    for row in data["raw_and_processed_files"]:
        if sha(inside(manifest.parent / row["relative_path"])) != row["sha256"]:
            raise ValueError("Frozen provider bytes changed")
    for row in data["provider_sources"]:
        if sha(inside(row["path"])) != row["sha256"]:
            raise ValueError("Frozen provider source text changed")
        if row["class"] == "PPI" and hashlib.sha256(inspect.getsource(PPI).encode()).hexdigest() != row["sha256"]:
            raise ValueError("Installed PPI provider differs from frozen tensor provider")
    return data


def confinement_gate():
    names = ["TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS",
             "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
             "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR"]
    values = {name: os.environ.get(name) for name in names}
    if any(value is None or not Path(value).resolve().is_relative_to(PHASE) for value in values.values()):
        raise ValueError("Source the verified explicit repository environment wrapper before execution")
    return values


def mask(graph, index, fraction):
    generator = torch.Generator().manual_seed(9701 + index)
    selected = torch.randperm(graph.num_nodes, generator=generator)[:max(1, int(graph.num_nodes * fraction))]
    result = torch.zeros(graph.num_nodes, dtype=torch.bool)
    result[selected] = True
    return result


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}, sort_keys=True).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mask_contract(graphs, fraction):
    return [{"training_index": i, "nodes": g.num_nodes, "observed_nodes": int(mask(g, i, fraction).sum()),
             "mask_sha256": tensor_sha(mask(g, i, fraction))} for i, g in enumerate(graphs)]


def selector_locks(fraction, manifest):
    # Bind immutable choices and their source records, without reading any
    # validation evidence. Fresh checkpoint gradient replay is a prerequisite.
    result = {}
    data_identity = sha(manifest)
    data = json.loads(manifest.read_text())
    if "manifest_amendment" in data:
        parent = manifest.parent / "frozen_manifest.json"
        from importlib.util import spec_from_file_location, module_from_spec
        path = CODE / "new_tasks/shareable_freeze.py"
        spec = spec_from_file_location("ppi_manifest_amendment", path)
        amendment = module_from_spec(spec)
        spec.loader.exec_module(amendment)
        if data != amendment.project(json.loads(parent.read_text()), sha(parent), sha(path)):
            raise ValueError("Shareable manifest does not derive exactly from preserved freeze")
        data_identity = sha(parent)
    for seed in (0, 1, 2):
        label = int(fraction * 100)
        run = PHASE / "sharing_policy" / f"ppi_probe_v2_seed{seed}_label{label}_cpu"
        paths = [run / "contract.json", run / "result.json", run / "selector_lock.json"]
        paths += [run / f"probe_epoch{e}.json" for e in (0, 10, 25)]
        replay = PHASE / "sharing_policy" / f"PPI_PROBE_v2_seed{seed}_label{label}_AUTOGRAD_REPLAY.json"
        lock = json.loads(paths[2].read_text())
        fit = json.loads(paths[1].read_text())
        contract = json.loads(paths[0].read_text())
        checked = json.loads(replay.read_text())
        if (fit["status"] != "completed_train_only_probe" or checked["status"] != "PASS"
                or checked["private_indices"] != lock["private_indices"]
                or checked["contract_sha256"] != sha(paths[0])
                or fit["selector_lock_sha256"] != sha(paths[2])
                or contract["data_manifest_sha256"] != data_identity
                or contract["seed"] != seed or contract["label_fraction"] != fraction
                or lock["policy"] != "recurrence-budget-v2"):
            raise ValueError("Prospective selector lock or fresh replay differs")
        for epoch in (0, 10, 25):
            name = f"probe_epoch{epoch}.json"
            if lock["input_record_sha256"][name] != sha(run / name):
                raise ValueError("Selector input record changed")
        result[str(seed)] = {"private_indices": lock["private_indices"],
                             "sha256": {str(p.relative_to(PHASE)): sha(p) for p in paths + [replay]}}
    return result


def freeze(args):
    data_freeze(args.data_manifest)
    graphs = list(PPI(root=str(args.data_manifest.parent / "provider"), split="train"))
    arm_list = SHARING if args.kind == "sharing" else ARCHITECTURE
    lock = selector_locks(args.label_fraction, args.data_manifest) if args.kind == "sharing" else None
    protocol = {"schema": "gnnm-ppi-development-v1", "status": "prospective_not_fit",
                "kind": args.kind, "arms": arm_list, "seeds": [0, 1, 2],
                "label_fraction": args.label_fraction, "recipe": RECIPE,
                "source_sha256": sources(), "environment": environment(),
                "data_manifest_sha256": sha(args.data_manifest),
                "mask_contract": mask_contract(graphs, args.label_fraction), "confinement": confinement_gate(),
                "selector_locks": lock, "test_loader_accessed": False,
                "continue_stop": ("Selected private layout must lower graph-mean member validation BCE relative to tied in at least2/3seeds, distinguish rejected placements, then transfer to a second independent task. No significance or general claim from3seeds."
                                  if args.kind == "sharing" else
                                  "Require useful quality/resource frontier versus parameter/channel-matched ENS and BASE. Centering must improve over uncentered mean-feedback paths; packed small ENS domination stops the architecture claim."),
                "primary_mechanism_practical_BCE_tolerance": 0.001,
                "primary_practical_F1_tolerance_points": 0.5,
                "creation_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    output = inside(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, protocol)
    print(json.dumps({"status": "prospective_protocol_saved", "sha256": sha(output), "training": False}))


def artifact_manifest(output):
    write_json(output / "artifact_manifest.json", {str(p.relative_to(output)): sha(p)
               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})


def run(args, training):
    output = inside(args.output)
    if output.exists():
        raise ValueError("Refuse to overwrite an existing run")
    data_freeze(args.data_manifest)
    src, env = sources(), environment()
    confinement = confinement_gate()
    cpu_graphs = list(PPI(root=str(args.data_manifest.parent / "provider"), split="train"))
    protocol = None
    if training:
        if args.device != "cuda:0" or args.protocol is None:
            raise ValueError("Full fits require a prospective protocol and explicit idle GPU")
        protocol = json.loads(inside(args.protocol).read_text())
        required = {"schema": "gnnm-ppi-development-v1", "status": "prospective_not_fit",
                    "source_sha256": src, "environment": env,
                    "recipe": RECIPE, "data_manifest_sha256": sha(args.data_manifest),
                    "label_fraction": args.label_fraction,
                    "mask_contract": mask_contract(cpu_graphs, args.label_fraction), "confinement": confinement}
        if any(protocol.get(k) != v for k, v in required.items()):
            raise ValueError("Source/data/environment/recipe differs from prospective freeze")
        if args.arm not in protocol["arms"] or args.seed not in protocol["seeds"]:
            raise ValueError("Run outside registered arm/seed family")
        if protocol["kind"] == "sharing" and protocol["selector_locks"] != selector_locks(args.label_fraction, args.data_manifest):
            raise ValueError("Selector evidence changed after freeze")
    impl.guard_gpu(args)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    graphs, masks, operators = [], [], []
    for index, graph in enumerate(cpu_graphs):
        observed = mask(graph, index, args.label_fraction)
        safe = graph.clone()
        safe.y[~observed] = float("nan")
        safe = safe.to(device)
        graphs.append(safe)
        masks.append(observed.to(device))
        operators.append(impl.MeanOperator.from_edges(safe.edge_index, safe.num_nodes, dtype=safe.x.dtype, device=device))
    model, mode, budget = impl.make_model(args.arm, RECIPE["layers"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0, eps=1e-8)
    output.mkdir(parents=True)
    for name, path in source_paths().items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(args.data_manifest, output / "data_manifest.json")
    write_json(output / "environment.json", env)
    if training:
        shutil.copyfile(args.protocol, output / "protocol.json")
    torch.save({"state_dict": model.state_dict(), "masks": [m.cpu() for m in masks],
                "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate()}, output / "initial.pt")
    contract = {"arm": args.arm, "seed": args.seed, "label_fraction": args.label_fraction,
                "device": args.device, "source_sha256": src, "recipe": RECIPE,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "budget": budget, "environment": env, "confinement": confinement,
                "data_manifest_sha256": sha(args.data_manifest),
                "initial_sha256": sha(output / "initial.pt"),
                "protocol_sha256": sha(args.protocol) if training else None,
                "mask_contract": mask_contract(cpu_graphs, args.label_fraction),
                "hidden_labels": "Physically replaced byNaN before supplying training graph objects",
                "test_loader_accessed": False, "training_fit": training}
    write_json(output / "contract.json", contract)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if not training:
            optimizer.zero_grad(set_to_none=True)
            logits = impl.forward(model, mode, graphs[0], operators[0])
            loss = impl.mean_member_loss(logits, graphs[0].y, masks[0])
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Inactive or nonfinite registered trainable parameter")
            optimizer.step()
            impl.synchronize(device)
            write_json(output / "preflight.json", {"status": "passed", "optimizer_steps": 1,
                       "complete_train_graph": 0, "nodes": graphs[0].num_nodes,
                       "observed_nodes": int(masks[0].sum()), "logits_shape": list(logits.shape),
                       "finite_initial_loss": float(loss.detach()), "seconds": time.perf_counter() - started,
                       "validation_or_test_scored": False, "not_research_performance_evidence": True})
        else:
            val = [g.to(device) for g in PPI(root=str(args.data_manifest.parent / "provider"), split="val")]
            val_ops = [impl.MeanOperator.from_edges(g.edge_index, g.num_nodes, dtype=g.x.dtype, device=device) for g in val]
            torch.save([g.y.cpu() for g in val], output / "validation_targets.pt")
            evidence_dir = output / "validation_epochs"
            evidence_dir.mkdir()
            best = None
            with (output / "validation_trace.csv").open("x", newline="") as stream:
                fields = ["epoch", "micro_f1", "pooled_bce", "mean_member_bce", "graph_mean_member_bce", "train_graph_mean_BCE", "fit_epoch_seconds", "elapsed_seconds"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for epoch in range(1, 301):
                    order = list(range(20))
                    random.Random(19709 + 1000 * args.seed + epoch).shuffle(order)
                    model.train()
                    impl.synchronize(device)
                    epoch_start, loss_sum = time.perf_counter(), 0.0
                    for i in order:
                        optimizer.zero_grad(set_to_none=True)
                        logits = impl.forward(model, mode, graphs[i], operators[i])
                        loss = impl.mean_member_loss(logits, graphs[i].y, masks[i])
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite training loss")
                        loss.backward()
                        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise ValueError("Inactive or nonfinite parameter gradient")
                        optimizer.step()
                        loss_sum += float(loss.detach())
                    impl.synchronize(device)
                    epoch_seconds = time.perf_counter() - epoch_start
                    metrics, logits = impl.evaluate(model, mode, val, val_ops)
                    torch.save(logits, evidence_dir / f"epoch{epoch:03d}.pt")
                    row = {"epoch": epoch, **metrics, "train_graph_mean_BCE": loss_sum / 20,
                           "fit_epoch_seconds": epoch_seconds, "elapsed_seconds": time.perf_counter() - started}
                    writer.writerow(row)
                    stream.flush()
                    key = (metrics["micro_f1"], -metrics["pooled_bce"], -epoch)
                    if best is None or key > best:
                        best = key
                        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "epoch": epoch, "validation": metrics, "rng_cpu": torch.get_rng_state(),
                                    "rng_python": random.getstate(), "rng_cuda": torch.cuda.get_rng_state(device),
                                    "contract_sha256": sha(output / "contract.json")}, output / "selected.pt")
                        torch.save(logits, output / "selected_validation_member_logits.pt")
                        # Mutable selection files within this unfinished run.
                        (output / "selected.json").write_text(json.dumps(row, indent=2) + "\n")
                    if epoch == 1 or epoch % 25 == 0:
                        print(json.dumps({"epoch": epoch, **metrics}), flush=True)
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng_cpu": torch.get_rng_state(), "rng_python": random.getstate(),
                        "rng_cuda": torch.cuda.get_rng_state(device)}, output / "final.pt")
            write_json(output / "result.json", {"status": "completed_development_fit",
                       "selected": json.loads((output / "selected.json").read_text()),
                       "total_seconds": time.perf_counter() - started,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                       "test_scored": False, "source_sha256": src})
        artifact_manifest(output)
        print(json.dumps({"output": str(output), "status": "completed_fit" if training else "passed_preflight"}))
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                   "elapsed_seconds": time.perf_counter() - started, "test_scored": False})
        artifact_manifest(output)
        raise


def metrics_from_logits(logits, targets):
    tp = fp = fn = entries = 0
    pooled_loss = member_loss = graph_member_loss = 0.0
    from torch.nn.functional import binary_cross_entropy_with_logits as bce
    for pred, target in zip(logits, targets):
        pooled = pred.mean(0)
        truth, guessed = target.bool(), pooled >= 0
        tp += int((truth & guessed).sum()); fp += int((~truth & guessed).sum()); fn += int((truth & ~guessed).sum())
        pooled_loss += float(bce(pooled, target, reduction="sum"))
        risk = float(bce(pred, target[None].expand_as(pred), reduction="sum")) / pred.shape[0]
        member_loss += risk; graph_member_loss += risk / target.numel(); entries += target.numel()
    return {"micro_f1": 2 * tp / max(1, 2 * tp + fp + fn), "pooled_bce": pooled_loss / entries,
            "mean_member_bce": member_loss / entries, "graph_mean_member_bce": graph_member_loss / len(targets)}


def replay(args):
    run_path = inside(args.run)
    output = inside(args.output)
    if run_path == output or run_path in output.parents or output.exists():
        raise ValueError("Replay must be new and outside immutable fit")
    manifest = json.loads((run_path / "artifact_manifest.json").read_text())
    actual = {str(p.relative_to(run_path)): sha(p) for p in run_path.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
    if actual != manifest:
        raise ValueError("Artifact coverage or hashes changed")
    contract = json.loads((run_path / "contract.json").read_text())
    if sources() != contract["source_sha256"] or not contract["training_fit"]:
        raise ValueError("Use exactly the archived fitted source to replay a completed fit")
    data_freeze(args.data_manifest)
    if sha(args.data_manifest) != contract["data_manifest_sha256"]:
        raise ValueError("Different data freeze")
    result = json.loads((run_path / "result.json").read_text())
    selected = json.loads((run_path / "selected.json").read_text())
    if result["status"] != "completed_development_fit" or result["selected"] != selected:
        raise ValueError("Result/selection disagree")
    torch.set_num_threads(1)
    targets = torch.load(run_path / "validation_targets.pt", map_location="cpu")
    with (run_path / "validation_trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["epoch"]) for r in rows] != list(range(1, 301)):
        raise ValueError("Incomplete validation trajectory")
    for row in rows:
        logits = torch.load(run_path / "validation_epochs" / f"epoch{int(row['epoch']):03d}.pt", map_location="cpu")
        values = metrics_from_logits(logits, targets)
        if any(abs(values[k] - float(row[k])) > 2e-5 for k in values):
            raise ValueError("Retained logits do not replay reported epoch metrics")
    best = max(rows, key=lambda r: (float(r["micro_f1"]), -float(r["pooled_bce"]), -int(r["epoch"])))
    checkpoint = torch.load(run_path / "selected.pt", map_location="cpu")
    if checkpoint["epoch"] != int(best["epoch"]) or selected["epoch"] != checkpoint["epoch"]:
        raise ValueError("Wrong selected epoch")
    if checkpoint["contract_sha256"] != sha(run_path / "contract.json"):
        raise ValueError("Checkpoint belongs to another contract")
    for k in checkpoint["validation"]:
        if abs(checkpoint["validation"][k] - float(best[k])) > 1e-12 or abs(float(selected[k]) - float(best[k])) > 1e-12:
            raise ValueError("Checkpoint/selected metrics do not match selected trace row")
    kept = torch.load(run_path / "selected_validation_member_logits.pt", map_location="cpu")
    epoch_logits = torch.load(run_path / "validation_epochs" / f"epoch{checkpoint['epoch']:03d}.pt", map_location="cpu")
    if len(kept) != len(epoch_logits) or any(not torch.equal(a, b) for a, b in zip(kept, epoch_logits)):
        raise ValueError("Selected logits differ from selected epoch evidence")
    torch.manual_seed(contract["seed"])
    model, mode, _ = impl.make_model(contract["arm"], 3)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    val = list(PPI(root=str(args.data_manifest.parent / "provider"), split="val"))
    if len(val) != len(targets) or any(not torch.equal(g.y, y) for g, y in zip(val, targets)):
        raise ValueError("Validation target identity changed")
    operators = [impl.MeanOperator.from_edges(g.edge_index, g.num_nodes, dtype=g.x.dtype) for g in val]
    metrics, fresh = impl.evaluate(model, mode, val, operators)
    maximum = max((a - b).abs().max().item() for a, b in zip(fresh, kept))
    if any(not torch.allclose(a, b, atol=2e-4, rtol=2e-4) for a, b in zip(fresh, kept)):
        raise ValueError("Fresh selected checkpoint predictions drift")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"status": "PASS", "scope": "CPU checkpoint/selection replay; not fresh training reproduction",
               "epoch": checkpoint["epoch"], "metrics_cpu": metrics, "maximum_logit_error": maximum,
               "contract_sha256": sha(run_path / "contract.json"), "sources": sources(), "test_scored": False})
    print(json.dumps({"status": "PASS", "maximum_logit_error": maximum}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "preflight", "train", "replay"], default="preflight", nargs="?")
    p.add_argument("--data-manifest", type=Path, default=PHASE / "new_tasks/data/ppi/frozen_shareable_manifest_v2.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=["sharing", "architecture"], default="sharing")
    p.add_argument("--arm", choices=impl.ARMS, default="gnnm")
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--label-fraction", type=float, choices=[0.1, 1.0], default=0.1)
    p.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    p.add_argument("--physical-gpu", type=int)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--run", type=Path)
    args = p.parse_args()
    args.data_manifest = inside(args.data_manifest)
    if args.command == "freeze": freeze(args)
    elif args.command == "replay": replay(args)
    else: run(args, args.command == "train")


if __name__ == "__main__":
    main()
