"""Prospective grouped MolHIV transfer: freeze, P-only probe, replay.

Reads only a hashed official-train-only bundle. Fits tied GINE on P for25
epochs; Q gradients at0/10/25 feed the UNCHANGED recurrence-budget-v2 rule.
No predictive evaluation, validation loader or test loader exists here.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from common import HERE, PHASE, confined, sha, source_hashes, write_json
import torch
from torch_geometric.data import Data, Batch
from gine_adapters import MolecularProjectorEnsemble, gine_options
from recurrence_collector import TrainingUnit, tensor_hash
from replay_recurrence import replay_and_decide
from gine_collector import collect_record, block_contract, equal_graph_bce
from group_folds import grouped_folds, validate_grouped


RECIPE = {
    "members": 4, "hidden_dim": 128, "num_layers": 3, "hidden_dim_multiplier": 2,
    "output_dim": 1, "dropout": 0.1, "optimizer": "AdamW", "lr": 0.001,
    "weight_decay": 0.0, "epsilon": 1e-8, "precision": "float32", "AMP": False,
    "batch_size": 128, "threads": 1, "num_workers": 0, "probe_epochs": [0, 10, 25],
    "optimizer_seeds": [0, 1, 2], "label_fractions": [0.1, 1.0], "primary_fraction": 0.1,
    "adverse_fraction": 1.0, "HPO_candidates": 1,
    "label_masks": "unchanged MolHIV largest-remainder class quotas; nested tenpercent permutation101+class",
    "fitting": "P observed graphs only; mean-member equal-graph finite-target BCE",
    "gradient_scoring": "unscaled own-member equal-graph mean; Q only; three grouped A/B partitions",
    "grouping": "canonical molecule OR stereo-free Bemis-Murcko scaffold union; empty scaffold kept together",
    "selector": "UNCHANGED recurrence-budget-v2; may abstain; rank(S-N)/(3*block_count)",
    "budget": "at most one full GINE block, including its BondEncoder;3*p0 added parameters",
    "epoch_order": "Random(19709+1000*optimizer_seed+epoch) shuffle sorted P unit IDs",
    "later_training": "restart canonical paired initialization after locks; selected/tied/first/middle/last equal-capacity controls",
    "warmup_not_free": "all probe fit/gradient/checkpoint costs must count toward adaptive method cost",
    "dataset_status": "MolHIV development; reused official train/valid are not independent confirmation",
    "validation_rule": "no validation until all six P-only selector locks finalized; no depth chosen by validation",
}


def environment():
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    from torch_geometric.nn import GINEConv
    return {"python": sys.version, "torch": torch.__version__, "compiled_cuda": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "torch-geometric", "ogb", "numpy")},
            "runtime_class_source_sha256": {cls.__name__: hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
                                            for cls in (AtomEncoder, BondEncoder, GINEConv)}}


def new_model():
    return MolecularProjectorEnsemble(gine_options(output_dim=1, hidden_dim=128,
        num_layers=3, dropout=0.1, hidden_dim_multiplier=2), members=4)


def load_bundle(path):
    path = confined(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != "gnnm-molhiv-official-train-only-bundle-v1"
            or payload.get("dataset") != "ogbg-molhiv"
            or payload["export_scope"]["test_target_indexing_or_summary"] is not False
            or payload["export_scope"]["scores_read"] is not False):
        raise ValueError("Require an unscored official training-only bundle")
    if payload.get("source_sha256") != source_hashes():
        raise ValueError("Training-bundle export sources differ from prospective packet")
    ids, raw = payload["official_train_indices"], payload["graphs"]
    mapping = payload["group_by_official_train_id"]
    if len(ids) != len(set(ids)) or len(ids) != len(raw) or set(mapping) != {str(i) for i in ids}:
        raise ValueError("Invalid train-only bundle coverage")
    graphs = [Data(**row) for row in raw]
    if any(graph.y.shape != (1, 1) for graph in graphs):
        raise ValueError("MolHIV transfer expects a single graph-level target")
    return graphs, ids, mapping, payload


def label_masks(graphs):
    labels = torch.cat([graph.y for graph in graphs]).reshape(-1)
    if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("Expected observed official training binary labels before mask freeze")
    groups = [(labels == label).nonzero().view(-1) for label in (0, 1)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Require both classes in official training")
    exact = [len(group) * 0.1 for group in groups]
    quotas, total = [int(value) for value in exact], int(len(graphs) * 0.1)
    for label in sorted((0, 1), key=lambda k: (-(exact[k] - quotas[k]), k))[:total - sum(quotas)]:
        quotas[label] += 1
    ten = torch.zeros(len(graphs), dtype=torch.bool)
    for label, group in enumerate(groups):
        generator = torch.Generator().manual_seed(101 + label)
        chosen = group[torch.randperm(len(group), generator=generator)[:quotas[label]]]
        ten[chosen] = True
    return {"0.1": ten, "1.0": torch.ones(len(graphs), dtype=torch.bool)}


def identity(official):
    return f"molhiv:train:{official:08d}"


def prepare(path, fraction):
    graphs, ids, mapping, payload = load_bundle(path)
    masks = label_masks(graphs)
    observed = masks[str(fraction)]
    unit_to_group = {identity(official): mapping[str(official)]
                     for official, yes in zip(ids, observed.tolist()) if yes}
    folds = grouped_folds(unit_to_group)
    units = []
    for graph, official, yes in zip(graphs, ids, observed.tolist()):
        # Physical erasure, including every wholly hidden graph; never multiply NaN by0.
        graph.y = graph.y.float() if yes else torch.full(graph.y.shape, float("nan"))
        if yes:
            units.append(TrainingUnit(identity(official), graph, graph.x, graph.y))
    return units, folds, {"mask": tensor_hash(observed), "observed_graphs": int(observed.sum()),
                           "all_graphs": len(graphs), "hidden_graphs": int((~observed).sum()),
                           "masks_nested": True, "hidden_target_handling": "physically replaced by NaN before graph use"}, payload


def freeze_value(bundle):
    with torch.random.fork_rng(devices=[]):
        model = new_model()
        blocks = block_contract(model)
    budgets = {}
    for fraction in (0.1, 1.0):
        _, folds, mask, payload = prepare(bundle, fraction)
        budgets[str(fraction)] = {"folds": folds, "label_mask": mask}
    return {"schema": "gnnm-molhiv-sharing-transfer-v1", "recipe": RECIPE,
            "source_sha256": source_hashes(), "environment": environment(),
            "train_bundle_sha256": sha(bundle), "data_manifest_sha256": payload["data_manifest_sha256"],
            "grouping": payload["grouping"], "budgets": budgets, "block_parameters": blocks,
            "private_parameter_budget": 3 * blocks["0"]["count"],
            "validation_loaded_or_scored": False, "test_loaded_or_scored_by_probe": False,
            "export_all_target_storage_materialization": payload["export_scope"]["all_target_storage_materialized_by_torch_load"],
            "confirmation_status": "development_only; no independent confirmation", "predictive_fit_performed": False}


def gpu_guard(args):
    if args.device != "cuda:0" or args.physical_gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError("P-only fitting requires explicit idle physical GPU exposed as cuda:0")
    base = ["nvidia-smi", "-i", str(args.physical_gpu)]
    query = subprocess.run(base + ["--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
    row = [item.strip() for item in query.stdout.strip().split(",")]
    if len(row) != 4 or int(row[2]) > 100 or int(row[3]) != 0:
        raise RuntimeError("Selected GPU is occupied; model remains on CPU")
    apps = subprocess.run(base + ["--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True)
    if apps.stdout.strip():
        raise RuntimeError("Selected GPU has an active compute context")
    return {"uuid": row[0], "name": row[1], "physical_gpu": args.physical_gpu,
            "memory_MiB": int(row[2]), "utilization_percent": int(row[3]), "compute_contexts": 0}


def artifact_manifest(output):
    write_json(output / "artifact_manifest.json", {str(p.relative_to(output)): sha(p)
               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"})


def probe(args):
    bundle, protocol_path = confined(args.train_bundle), confined(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    if freeze_value(bundle) != protocol:
        raise ValueError("Source/bundle/folds/masks/environment changed after freeze")
    units, folds, mask, payload = prepare(bundle, args.label_fraction)
    resource = gpu_guard(args)
    output = confined(args.output)
    if output.exists():
        raise ValueError("Refuse to overwrite a probe run")
    output.mkdir(parents=True)
    for rel in protocol["source_sha256"]:
        target = output / "archive" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / rel, target)
        if sha(target) != protocol["source_sha256"][rel]:
            raise ValueError("Source changed during archival")
    shutil.copyfile(protocol_path, output / "protocol.json")
    torch.set_num_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = new_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0, eps=1e-8)
    by_id = {unit.unit_id: unit for unit in units}
    contract = {"scope": "MolHIV training-only recurrence-v2 transfer", "seed": args.seed,
                "label_fraction": args.label_fraction, "recipe": RECIPE, "folds": folds,
                "mask": mask, "protocol_sha256": sha(protocol_path),
                "source_sha256": protocol["source_sha256"], "train_bundle_sha256": sha(bundle),
                "data_manifest_sha256": payload["data_manifest_sha256"], "device": args.device,
                "resource_gate": resource, "private_parameter_budget": protocol["private_parameter_budget"],
                "validation_or_test_accessed": False, "warmup_counts_toward_adaptive_method_cost": True}
    write_json(output / "contract.json", contract)
    dataset_identity = {"dataset": "ogbg-molhiv-development", "contract_sha256": sha(output / "contract.json"),
                        "train_bundle_sha256": sha(bundle), "label_fraction": args.label_fraction,
                        "grouped_folds_sha256": hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()}
    records, events = [], []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    def checkpoint(epoch):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        checkpoint_path = output / f"probe_epoch{epoch}.pt"
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "seed": args.seed, "label_fraction": args.label_fraction,
                    "protocol_sha256": sha(protocol_path), "rng_cpu": torch.get_rng_state(),
                    "rng_python": random.getstate(), "rng_cuda": torch.cuda.get_rng_state(device)}, checkpoint_path)
        # Fold batches are moved lazily, not the entire official training set.
        record = collect_record(model, units, folds["probe_fit_unit_ids"], folds["partitions"],
                                epoch, args.seed, dataset_identity, batch_size=128)
        record["probe_checkpoint_sha256"] = sha(checkpoint_path)
        record["probe_checkpoint_filename"] = checkpoint_path.name
        write_json(output / f"probe_epoch{epoch}.json", record)
        records.append(record)
        torch.cuda.synchronize(device)
        events.append({"epoch": epoch, "checkpoint_and_collection_seconds": time.perf_counter() - start})
    try:
        checkpoint(0)
        with (output / "fit_trace.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("epoch", "mean_graph_bce", "fit_graphs", "fit_steps", "seconds"))
            writer.writeheader()
            for epoch in range(1, 26):
                order = list(folds["probe_fit_unit_ids"])
                random.Random(19709 + 1000 * args.seed + epoch).shuffle(order)
                model.train()
                start = time.perf_counter()
                loss_sum, steps = 0.0, 0
                for offset in range(0, len(order), 128):
                    batch_ids = order[offset:offset + 128]
                    batch = Batch.from_data_list([by_id[key].graph for key in batch_ids]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = equal_graph_bce(model(batch, batch.x), batch.y.float())
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Nonfinite P-only loss")
                    loss.backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise RuntimeError("Nonfinite P-only gradient")
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(batch_ids)
                    steps += 1
                torch.cuda.synchronize(device)
                writer.writerow({"epoch": epoch, "mean_graph_bce": loss_sum / len(order),
                                 "fit_graphs": len(order), "fit_steps": steps, "seconds": time.perf_counter() - start})
                stream.flush()
                if epoch in (10, 25):
                    checkpoint(epoch)
        lock = replay_and_decide(records, protocol["private_parameter_budget"])
        lock.update({"scope": "train-only selector; no predictive gain demonstrated",
                     "input_record_sha256": {f"probe_epoch{e}.json": sha(output / f"probe_epoch{e}.json") for e in (0, 10, 25)},
                     "policy_source_sha256": sha(HERE / "source/sharing_policy/replay_recurrence.py"),
                     "contract_sha256": sha(output / "contract.json"),
                     "fixed_placement_comparisons_required": ["first_private", "middle_private", "last_private"],
                     "restart_from_canonical_initialization_required": True})
        write_json(output / "selector_lock.json", lock)
        write_json(output / "result.json", {"status": "completed_train_only_probe", "private_indices": lock["private_indices"],
            "selector_lock_sha256": sha(output / "selector_lock.json"), "events": events,
            "fit_graph_steps": 25 * len(folds["probe_fit_unit_ids"]),
            "fit_minibatch_steps": 25 * math.ceil(len(folds["probe_fit_unit_ids"]) / 128),
            "elapsed_seconds": time.perf_counter() - begin, "validation_or_test_scored": False,
            "peak_allocated_bytes_including_probe_collection": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes_including_probe_collection": torch.cuda.max_memory_reserved(device)})
        artifact_manifest(output)
    except Exception as error:
        write_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                    "elapsed_seconds": time.perf_counter() - begin, "events": events, "validation_or_test_scored": False})
        artifact_manifest(output)
        raise


def replay(args):
    run = confined(args.run)
    artifacts = json.loads((run / "artifact_manifest.json").read_text())
    files = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"}
    if set(artifacts) != files or any(sha(run / name) != digest for name, digest in artifacts.items()):
        raise ValueError("Immutable probe artifact coverage/hash mismatch")
    contract = json.loads((run / "contract.json").read_text())
    if contract["source_sha256"] != source_hashes() or contract["train_bundle_sha256"] != sha(confined(args.train_bundle)):
        raise ValueError("Replay source or train bundle differs")
    units, folds, mask, _ = prepare(args.train_bundle, contract["label_fraction"])
    if folds != contract["folds"] or mask != contract["mask"]:
        raise ValueError("Replay grouped folds/masks differ")
    validate_grouped(folds)
    records = [json.loads((run / f"probe_epoch{e}.json").read_text()) for e in (0, 10, 25)]
    expected = replay_and_decide(records, contract["private_parameter_budget"])
    lock = json.loads((run / "selector_lock.json").read_text())
    for key in ("policy", "private_indices", "scores", "private_parameter_budget", "spent_private_parameters"):
        if expected[key] != lock[key]:
            raise ValueError("Stored selector lock does not replay")
    errors = []
    torch.set_num_threads(1)
    for epoch, record in zip((0, 10, 25), records):
        checkpoint = torch.load(run / f"probe_epoch{epoch}.pt", map_location="cpu", weights_only=False)
        if sha(run / f"probe_epoch{epoch}.pt") != record["probe_checkpoint_sha256"]:
            raise ValueError("Probe checkpoint binding differs")
        model = new_model()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        fresh = collect_record(model, units, folds["probe_fit_unit_ids"], folds["partitions"],
                               epoch, contract["seed"], record["dataset_identity"], batch_size=128)
        if fresh["training_unit_fingerprints"] != record["training_unit_fingerprints"]:
            raise ValueError("Safe training tensors differ")
        maximum = 0.0
        for original_part, new_part in zip(record["partitions"], fresh["partitions"]):
            for key in record["block_parameters"]:
                a, b = original_part["blocks"][key], new_part["blocks"][key]
                for scalar in ("C", "N"):
                    difference = abs(a[scalar] - b[scalar])
                    maximum = max(maximum, difference)
                    if difference > 1e-8 + 2e-4 * abs(a[scalar]):
                        raise ValueError("Cross-device autograd C/N replay failed")
                for kind in ("raw_gram", "centered_gram"):
                    for matrix in ("AA", "BB", "AB"):
                        if not torch.allclose(torch.tensor(a[kind][matrix], dtype=torch.float64),
                                              torch.tensor(b[kind][matrix], dtype=torch.float64), atol=1e-8, rtol=2e-4):
                            raise ValueError("Cross-device autograd Gram replay failed")
        errors.append({"epoch": epoch, "maximum_C_N_absolute_error": maximum})
    write_json(args.output, {"status": "passed", "scope": "train-only checkpoint autograd plus unchanged decision replay",
                            "autograd": errors, "cross_device_atol": 1e-8, "cross_device_rtol": 2e-4,
                            "selector_lock_changed": False, "predictive_score": None,
                            "validation_or_test_loaded": False, "artifact_manifest_sha256": sha(run / "artifact_manifest.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe", "replay"))
    parser.add_argument("--train-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--label-fraction", type=float, choices=(0.1, 1.0), default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "freeze":
        write_json(args.output, freeze_value(confined(args.train_bundle)))
    elif args.command == "probe":
        if args.protocol is None:
            parser.error("probe requires a prospective protocol")
        probe(args)
    else:
        if args.run is None:
            parser.error("replay requires immutable probe run")
        replay(args)


if __name__ == "__main__":
    main()
