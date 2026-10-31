"""CPU gate and frozen D4/E4 collection for six reproduced P-only probes."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
sys.path.insert(0, str(PHASE))

import torch
from torch_geometric.datasets import PPI

from freeze_inputs import (CODE_FILES, CORE, SCHEMA, SOURCE_SHA, checked,
                           fixed_ids, json_hash, old_folds, phase_path, reference,
                           sha, verify_contract, verify_core)
from spectral_privacy.v2_abi_fix_01.adapters import SourceSAGEEnsemble
from spectral_privacy.v2_abi_fix_01.collector import (GradientUnit, collect_record,
                                                    environment_contract,
                                                    partition_manifest,
                                                    unit_fingerprint)
from spectral_privacy.v2_abi_fix_01.contracts import (save_new_json, state_hash,
                                                    tensor_fingerprint)
from spectral_privacy.v2_abi_fix_01.replay import replay_record


GATE_SCHEMA = "spectral-ppi-precollection-gate-v1"


class CachedTrainPPI(PPI):
    def download(self):
        raise RuntimeError("No downloader is permitted; supply the complete frozen cache")

    def process(self):
        raise RuntimeError("No processing is permitted; supply existing train/val/test cache files")


def legacy_tensor_fingerprint(tensor):
    """Exact dtype/shape+bytes convention of the frozen probe tensor_hash."""
    value = tensor.detach().cpu().contiguous()
    metadata = {"dtype": str(value.dtype), "shape": list(value.shape)}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return metadata | {"sha256": digest.hexdigest()}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_lock(path):
    path = phase_path(path)
    lock = json.loads(path.read_text())
    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if lock.get("schema") != SCHEMA or lock.get("lock_sha256") != json_hash(payload):
        raise ValueError("Input lock schema/seal differs")
    if (lock["partition_ids"] != fixed_ids() or lock["epoch"] != 25
            or lock["n0"] != 1 or lock["rank_cap_per_target"] != 16
            or lock["random_control_seed"] != 800
            or lock["runtime"] != {"device": "cpu", "threads": 1, "interop_threads": 1}):
        raise ValueError("Frozen spectral integration recipe differs")
    if lock["spectral_prototype_manifest"] != verify_core():
        raise ValueError("Frozen spectral prototype identity differs")
    for name in CODE_FILES:
        if sha(HERE / name) != lock["integration_code_sha256"][name]:
            raise ValueError("Integration code changed after prospective freeze")
    for ref in lock["probe_sources"].values():
        checked(ref)
    for key in ("data_manifest", "synthetic_cpu_report", "reproduction_comparison"):
        checked(lock[key])
    if [(a["seed"], a["label_fraction"]) for a in lock["arms"]] != [
            (seed, fraction) for seed in (0, 1, 2) for fraction in (0.1, 1.0)]:
        raise ValueError("Require all six prospectively fixed arms")
    for arm in lock["arms"]:
        for name in ("contract", "label_masks", "checkpoint", "fit_trace"):
            checked(arm[name])
        contract = json.loads(checked(arm["contract"]).read_text())
        verify_contract(contract, arm["seed"], arm["label_fraction"], lock["data_manifest"]["sha256"])
        if arm["label_masks"]["sha256"] != contract["label_masks_sha256"]:
            raise ValueError("Mask-file identity differs")
    return lock


def load_train(lock):
    cache = phase_path(PHASE / lock["data_cache"])
    train = lock["train_processed_file"]
    if train["relative_path"] != "provider/processed/train.pt":
        raise ValueError("Only the processed training provider file is opened")
    path = phase_path(cache / train["relative_path"])
    if path.stat().st_size != train["bytes"] or sha(path) != train["sha256"]:
        raise ValueError("Processed training file differs from the probe data freeze")
    # Existence checks prevent PPI from invoking processing of other splits.
    # No validation/test file is hashed or deserialized by this script.
    for name in ("train.pt", "val.pt", "test.pt"):
        if not phase_path(cache / "provider/processed" / name).is_file():
            raise ValueError("Incomplete provider cache; processing is forbidden")
    dataset = CachedTrainPPI(root=str(cache / "provider"), split="train")
    if len(dataset) != 20:
        raise ValueError("Expected the official twenty training graph units")
    metadata = json.loads(checked(lock["data_manifest"]).read_text())
    rows = metadata["split_summaries"]["train"]["graph_tensor_fingerprints"]
    if len(rows) != 20:
        raise ValueError("Training metadata lacks all official graph indices")
    inputs = []
    for index in range(20):
        graph = dataset[index]
        row = rows[index]
        if row["split_index"] != index or row["num_nodes"] != graph.num_nodes:
            raise ValueError("Official training graph index/size differs")
        if (graph.x.shape != (graph.num_nodes, 50)
                or legacy_tensor_fingerprint(graph.x) != row["x"]
                or legacy_tensor_fingerprint(graph.edge_index) != row["edge_index"]):
            raise ValueError("Training input/topology differs from provider freeze")
        inputs.append({"unit_id": f"ppi:train:{index:02d}", "num_nodes": graph.num_nodes,
                       "features": tensor_fingerprint(graph.x),
                       "edge_index": tensor_fingerprint(graph.edge_index)})
    return dataset, inputs


def checked_masks(arm, contract, dataset):
    payload = torch.load(checked(arm["label_masks"]), map_location="cpu", weights_only=False)
    ids = [f"ppi:train:{i:02d}" for i in range(20)]
    if (set(payload) != {"masks", "label_fraction", "unit_ids"}
            or payload["label_fraction"] != arm["label_fraction"]
            or payload["unit_ids"] != ids or len(payload["masks"]) != 20):
        raise ValueError("Archived mask payload schema/fraction/IDs differ")
    manifest = []
    for index, mask in enumerate(payload["masks"]):
        nodes = dataset[index].num_nodes
        if mask.dtype != torch.bool or mask.device.type != "cpu" or mask.shape != (nodes,):
            raise ValueError("Archived mask shape/dtype/device differs")
        generator = torch.Generator(device="cpu").manual_seed(9701 + index)
        selected = torch.randperm(nodes, generator=generator)[:max(1, int(nodes * arm["label_fraction"]))]
        expected = torch.zeros(nodes, dtype=torch.bool)
        expected[selected] = True
        if not torch.equal(mask, expected):
            raise ValueError("Archived mask does not reproduce the frozen seed rule")
        manifest.append({"unit_id": ids[index], "total_nodes": nodes,
                         "observed_nodes": int(mask.sum()), "mask": legacy_tensor_fingerprint(mask)})
    if manifest != contract["mask_manifest"]:
        raise ValueError("Reproduced masks differ from P-only contract fingerprints")
    return payload["masks"], manifest


def score_units(dataset, masks, partition, metadata):
    rows = metadata["split_summaries"]["train"]["graph_tensor_fingerprints"]
    units = []
    for unit_id in partition["discovery"] + partition["screen"]:
        index = int(unit_id.rsplit(":", 1)[1])
        graph = dataset[index]
        if (graph.y.shape != (graph.num_nodes, 121)
                or legacy_tensor_fingerprint(graph.y) != rows[index]["y"]):
            raise ValueError("D/E training target bytes differ from the provider freeze")
        if not bool(torch.isfinite(graph.y).all()) or not bool(((graph.y == 0) | (graph.y == 1)).all()):
            raise ValueError("D/E training targets must be finite binary labels")
        safe = graph.clone()
        safe.y = graph.y.clone()
        safe.y[~masks[index]] = float("nan")
        # P/F target tensors are never supplied; hidden D/E labels are NaN.
        units.append(GradientUnit(unit_id, safe, safe.x, safe.y, masks[index], split="train"))
    return units


def checked_source(arm, contract, source_module, baseline_module, unit):
    with checked(arm["fit_trace"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if ([int(row["epoch"]) for row in rows] != list(range(1, 26))
            or any(int(row["fit_steps"]) != 10 for row in rows)):
        raise ValueError("P-only trace does not contain the fixed 250 fit steps")
    checkpoint = torch.load(checked(arm["checkpoint"]), map_location="cpu", weights_only=False)
    if (checkpoint["epoch"] != 25 or checkpoint["seed"] != arm["seed"]
            or checkpoint["label_fraction"] != arm["label_fraction"]
            or checkpoint["protocol_sha256"] != contract["protocol_sha256"]
            or checkpoint["rng_cuda"] is not None):
        raise ValueError("Checkpoint metadata does not identify the P-only CPU epoch25 state")
    state = checkpoint["state_dict"]
    if not state or any(not name.startswith("model.") for name in state):
        raise ValueError("Require tied source keys only; private blocks are forbidden")
    source_state = {name[len("model."):]: value for name, value in state.items()}
    options = baseline_module.model_options(50, 121, hidden_dim=128, layers=3, dropout=0.2)
    source = source_module.TABMModel(**options, tabm_inits=4, device="cpu")
    if set(source_state) != set(source.state_dict()):
        raise ValueError("Checkpoint does not map exactly to the original TABMModel")
    if any(value.device.type != "cpu" or value.dtype != torch.float32
           or not bool(torch.isfinite(value).all()) for value in source_state.values()):
        raise ValueError("Require the finite original float32 CPU source state")
    source.load_state_dict(source_state, strict=True)
    if any(not torch.equal(value, source.state_dict()[name]) for name, value in source_state.items()):
        raise ValueError("Strict source load did not preserve checkpoint tensor bytes")
    optimizer = checkpoint["optimizer"]
    parameters = list(source.parameters())
    group = optimizer["param_groups"]
    if (len(group) != 1 or group[0]["lr"] != 0.001 or group[0]["weight_decay"] != 0.0
            or group[0]["eps"] != 1e-8 or len(group[0]["params"]) != len(parameters)
            or len(optimizer["state"]) != len(parameters)
            or any(float(value["step"]) != 250 for value in optimizer["state"].values())):
        raise ValueError("Checkpoint optimizer does not match the fixed P-only AdamW250 fit")
    wrapper = baseline_module.SelectiveProjectorEnsemble(
        baseline_module.ProjectorEnsemble(options))
    wrapper.load_state_dict(state, strict=True)
    if wrapper.private_blocks or contract["folds"] != old_folds():
        raise ValueError("Only the tied P-only probe is accepted")
    source.eval()
    wrapper.eval()
    probe = SourceSAGEEnsemble(source)
    probe.eval()
    with torch.no_grad():
        for member in range(4):
            expected = wrapper.forward_member(unit.graph, unit.features, member)
            original = source(unit.graph, unit.features, tabm_seed=member)
            actual = probe.forward_member(unit.graph, unit.features, member)
            if not torch.equal(expected, original) or not torch.equal(original, actual):
                raise ValueError("Original source/tied wrapper/zero adapter function identity failed")
    if state_hash(source) != state_hash(probe.tied_source_copy()):
        raise ValueError("Rank-zero adapter does not preserve the original source state")
    return probe, {"arm_id": arm["arm_id"], "original_source_state_sha256": state_hash(source),
                   "adapter_state_sha256": state_hash(probe),
                   "strict_original_source_load": True, "exact_tied_forward_identity": True,
                   "optimizer_graph_steps": 250, "private_blocks": []}


def prepare(lock):
    source = load_module(checked(lock["probe_sources"]["source_snapshot/models.py"]),
                         "spectral_ppi_frozen_source")
    baseline = load_module(checked(lock["probe_sources"]["protocols/baseline_adapters.py"]),
                           "spectral_ppi_frozen_baseline")
    dataset, train_inputs = load_train(lock)
    metadata = json.loads(checked(lock["data_manifest"]).read_text())
    ids = lock["partition_ids"]
    partition = partition_manifest(ids["fit"], ids["discovery"], ids["screen"], ids["audit"])
    prepared, evidence = [], []
    for arm in lock["arms"]:
        contract = json.loads(checked(arm["contract"]).read_text())
        masks, mask_manifest = checked_masks(arm, contract, dataset)
        units = score_units(dataset, masks, partition, metadata)
        probe, source_evidence = checked_source(arm, contract, source, baseline, units[0])
        inputs = [unit_fingerprint(unit) for unit in units]
        evidence.append(source_evidence | {"mask_manifest_sha256": json_hash(mask_manifest),
                                           "score_input_sha256": json_hash(inputs),
                                           "score_input_fingerprints": inputs})
        prepared.append((arm, probe, units))
    gate = {"schema": GATE_SCHEMA, "status": "PRECOLLECTION_GATE_PASS",
            "input_lock_sha256": lock["lock_sha256"], "partition": partition,
            "environment": environment_contract(), "arms": evidence,
            "train_input_identity_sha256": json_hash(train_inputs),
            "train_inputs": train_inputs, "training_performed": False,
            "spectral_gradients_collected": False, "validation_loader_accessed": False,
            "test_loader_accessed": False, "F2_target_tensors_consulted": False,
            "hidden_DE_targets_are_NaN": True,
            "prior_environment_confinement_certified": False}
    gate["gate_sha256"] = json_hash(gate)
    return gate, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate", "collect"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--arm", choices=tuple(f"seed{seed}_label{percent}"
                                             for seed in (0, 1, 2) for percent in (10, 100)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = phase_path(args.output)
    if HERE not in output.parents or output.exists():
        raise ValueError("Use a new output directory inside this integration directory")
    output.mkdir()
    begin = time.perf_counter()
    try:
        lock = validate_lock(args.lock)
        gate, prepared = prepare(lock)
        if args.command == "gate":
            if args.arm is not None:
                raise ValueError("The prospective gate must bind all six arms")
            save_new_json(gate, output / "PRECOLLECTION_GATE.json", HERE)
            print(json.dumps({"status": gate["status"], "gate_sha256": gate["gate_sha256"],
                              "output": str(output), "spectral_gradients_collected": False}))
            return
        if args.gate is None:
            raise ValueError("Collection requires a prior successful --gate artifact")
        prior_path = phase_path(args.gate)
        prior = json.loads(prior_path.read_text())
        if prior != gate:
            raise ValueError("Precollection source/input/environment gate does not replay exactly")
        save_new_json({"input_lock_file": reference(args.lock),
                       "precollection_gate_file": reference(prior_path),
                       "input_lock_sha256": lock["lock_sha256"],
                       "gate_sha256": gate["gate_sha256"]}, output / "RUN_INPUTS.json", HERE)
        results = []
        for arm, probe, units in prepared:
            if args.arm is not None and arm["arm_id"] != args.arm:
                continue
            record = collect_record(probe, units, gate["partition"], source_sha256=SOURCE_SHA,
                                     dataset_identity={"dataset": "PPI", "split": "train",
                                                       "spectral_input_lock_sha256": lock["lock_sha256"],
                                                       "precollection_gate_sha256": gate["gate_sha256"],
                                                       "arm_id": arm["arm_id"], "label_fraction": arm["label_fraction"],
                                                       "checkpoint": arm["checkpoint"],
                                                       "data_manifest": lock["data_manifest"],
                                                       "label_masks": arm["label_masks"]},
                                     optimizer_seed=arm["seed"], checkpoint_epoch=25,
                                     random_control_seed=lock["random_control_seed"])
            identity = next(row for row in gate["arms"] if row["arm_id"] == arm["arm_id"])
            if (record["input_identity_sha256"] != identity["score_input_sha256"]
                    or record["model_state_sha256"] != identity["adapter_state_sha256"]):
                raise ValueError("Collection inputs/source differ from the prospective gate")
            replay = replay_record(record, expected_source_sha256=SOURCE_SHA,
                                   expected_probe_state_sha256=identity["adapter_state_sha256"])
            name = arm["arm_id"]
            record_path = output / f"{name}_MATRIX_RECORD.json"
            save_new_json(record, record_path, HERE)
            save_new_json(replay, output / f"{name}_EXACT_REPLAY.json", HERE)
            basis_lock = {"status": "SPECTRAL_BASES_LOCKED", "arm_id": name,
                          "input_lock_sha256": lock["lock_sha256"], "gate_sha256": gate["gate_sha256"],
                          "record_file_sha256": sha(record_path), "record_sha256": record["record_sha256"],
                          "original_source_state_sha256": identity["original_source_state_sha256"],
                          "adapter_state_sha256": identity["adapter_state_sha256"],
                          "added_storage": {"private_trainable_scalars": 3 * 128 * sum(replay["ranks"].values()),
                                            "basis_buffer_scalars": 128 * sum(replay["ranks"].values()),
                                            "contrast_buffer_scalars": 12,
                                            "ranks": replay["ranks"]},
                          "selection": record["selection"], "F2_opened": False,
                          "validation_or_test_scored": False,
                          "selected_projector_unbiasedness_claimed": False}
            save_new_json(basis_lock, output / f"{name}_BASIS_LOCK.json", HERE)
            results.append({"arm_id": name, "record_sha256": record["record_sha256"],
                            "record_file_sha256": sha(record_path), "replay_status": replay["status"],
                            "ranks": replay["ranks"]})
        status = ("SIX_ARM_COLLECTION_AND_EXACT_REPLAY_PASS" if args.arm is None
                  else "ONE_ARM_COLLECTION_AND_EXACT_REPLAY_PASS")
        save_new_json({"status": status, "results": results,
                       "input_lock_sha256": lock["lock_sha256"], "gate_sha256": gate["gate_sha256"],
                       "seconds": time.perf_counter() - begin, "training_performed": False,
                       "validation_or_test_scored": False, "F2_opened": False,
                       "predictive_improvement_claimed": False}, output / "RESULT.json", HERE)
        print(json.dumps({"status": status, "output": str(output)}))
    except Exception as error:
        save_new_json({"status": "FAIL", "type": type(error).__name__, "message": str(error),
                       "seconds": time.perf_counter() - begin}, output / "FAILURE.json", HERE)
        raise


if __name__ == "__main__":
    main()
