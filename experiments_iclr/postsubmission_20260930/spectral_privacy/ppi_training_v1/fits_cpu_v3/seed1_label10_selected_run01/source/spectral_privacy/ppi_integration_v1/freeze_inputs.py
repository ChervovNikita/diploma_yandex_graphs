"""Freeze six reproduced P-only checkpoints before spectral collection.

Standard library only. No model, data loader, gradient, or task score is read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random


HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
SCHEMA = "spectral-ppi-repoenv-input-lock-v1"
PROBE_NAME = "sharing_policy/ppi_probe.py"
PROBE_SHA = "29b67a5ead28fa9f6fbfd98ce51e6480c3d9f1d2dd498858e5abd9c88ce844b6"
SOURCE_SHA = "07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303"
CORE = "spectral_privacy/v2_abi_fix_01"
CORE_MANIFEST_SHA = "d682d44bfd49b357c40888965687c5a5a322c92c6d95ca0e59950af7818c4c39"
CODE_FILES = ("freeze_inputs.py", "collect_ppi.py", "PROTOCOL.md")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def json_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phase_path(path):
    value = Path(path).resolve()
    if PHASE not in value.parents:
        raise ValueError("Input/output must remain within the research phase")
    return value


def reference(path):
    path = phase_path(path)
    return {"path": str(path.relative_to(PHASE)), "sha256": sha(path)}


def checked(ref):
    path = phase_path(PHASE / ref["path"])
    if sha(path) != ref["sha256"]:
        raise ValueError("Frozen input bytes changed: " + ref["path"])
    return path


def fixed_ids():
    order = list(range(20))
    random.Random(700).shuffle(order)
    fit, score = sorted(order[:10]), sorted(order[10:])
    shuffled = list(score)
    random.Random(704).shuffle(shuffled)
    ids = lambda indices: [f"ppi:train:{i:02d}" for i in sorted(indices)]
    return {"fit": ids(fit), "discovery": ids(shuffled[:4]),
            "screen": ids(shuffled[4:8]), "audit": ids(shuffled[8:])}


def old_folds():
    order = list(range(20))
    random.Random(700).shuffle(order)
    fit, score = sorted(order[:10]), sorted(order[10:])
    parts = []
    for seed in (701, 702, 703):
        pair = list(score)
        random.Random(seed).shuffle(pair)
        ids = lambda values: [f"ppi:train:{i:02d}" for i in sorted(values)]
        parts.append({"partition_id": str(seed), "A": ids(pair[:5]), "B": ids(pair[5:])})
    return {"probe_fit_unit_ids": [f"ppi:train:{i:02d}" for i in fit],
            "score_unit_ids": [f"ppi:train:{i:02d}" for i in score],
            "partitions": parts, "fit_split_indices": fit,
            "raw_graph_id_rule": "training split index + 1"}


def verify_core():
    path = PHASE / CORE / "PROTOTYPE_MANIFEST.json"
    if sha(path) != CORE_MANIFEST_SHA:
        raise ValueError("Repaired spectral prototype manifest differs")
    manifest = json.loads(path.read_text())
    for name, digest in manifest["files_sha256"].items():
        if sha(path.parent / name) != digest:
            raise ValueError("Frozen spectral file differs: " + name)
    if sha(PHASE / PROBE_NAME) != PROBE_SHA or sha(PHASE / "source_snapshot/models.py") != SOURCE_SHA:
        raise ValueError("Original probe/source identity differs")
    return reference(path)


def verify_contract(contract, seed, fraction, data_sha):
    if (contract["scope"] != "PPI training-only sharing selection probe"
            or contract["seed"] != seed or contract["label_fraction"] != fraction
            or contract["folds"] != old_folds()
            or contract["data_manifest_sha256"] != data_sha
            or contract["validation_or_test_scored"] is not False
            or contract["device"] != "cpu"
            or contract["source_sha256"].get(PROBE_NAME) != PROBE_SHA
            or contract["source_sha256"].get("source_snapshot/models.py") != SOURCE_SHA):
        raise ValueError("Reproduced probe contract differs from the exact P-only protocol")
    recipe = contract["recipe"]
    expected = {"layers": 3, "hidden_dim": 128, "members": 4, "dropout": 0.2,
                "hidden_dim_multiplier": 1, "optimizer": "AdamW",
                "learning_rate": 0.001, "weight_decay": 0.0, "epsilon": 1e-8,
                "probe_epochs": [0, 10, 25], "optimizer_seeds": [0, 1, 2],
                "label_fractions": [0.1, 1.0], "fit_graph_count": 10, "score_graph_count": 10,
                "mask_seed_rule": "9701 + official training split graph index",
                "epoch_order_rule": "shuffle all20 with Random(19709+1000*optimizer_seed+epoch), then retain P10",
                "fit_loss": "native mean of 4 member BCEs, one complete P graph per step, observed nodes only"}
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError("Reproduced P-only fit recipe differs")
    for name, digest in contract["source_sha256"].items():
        if sha(phase_path(PHASE / name)) != digest:
            raise ValueError("Reproduced probe source differs: " + name)


def build_lock(args):
    core = verify_core()
    data = reference(args.data_manifest)
    metadata = json.loads(checked(data).read_text())
    if metadata["dataset"] != "ppi" or metadata["scoring_performed"] is not False:
        raise ValueError("Require exact unscored PPI freeze used by reproduced probes")
    train = next(row for row in metadata["raw_and_processed_files"]
                 if row["relative_path"] == "provider/processed/train.pt")
    synthetic = reference(args.synthetic_report)
    report = json.loads(checked(synthetic).read_text())
    if (report["status"] != "PASS" or report["source_models_sha256"] != SOURCE_SHA
            or report["v1_preserved"] is not True
            or report["checks"]["source_boundary_abi"]["boundary_member_count"] != 4):
        raise ValueError("Synthetic source-ABI CPU gate has not passed")
    comparison = reference(PHASE / "sharing_policy/PPI_REPOENV_REPRO_COMPARISON_v1.json")
    comparison_report = json.loads(checked(comparison).read_text())
    if (comparison_report["status"] != "PASS" or len(comparison_report["runs"]) != 6
            or comparison_report["prior_runs_cache_confinement_certified"] is not False
            or comparison_report["predictive_improvement_claimed"] is not False):
        raise ValueError("Require the six reproduced CPU artifacts comparison")
    arms = []
    for seed in (0, 1, 2):
        for percent, fraction in ((10, 0.1), (100, 1.0)):
            directory = PHASE / f"sharing_policy/ppi_probe_v2_seed{seed}_label{percent}_repoenv_repro_v1"
            contract_ref = reference(directory / "contract.json")
            contract = json.loads(checked(contract_ref).read_text())
            verify_contract(contract, seed, fraction, data["sha256"])
            masks = reference(directory / "label_masks.pt")
            if masks["sha256"] != contract["label_masks_sha256"]:
                raise ValueError("Archived masks differ from reproduction contract")
            checkpoint = reference(directory / "probe_epoch25.pt")
            trace = reference(directory / "fit_trace.csv")
            matching = [row for row in comparison_report["runs"]
                        if row["seed"] == seed and row["label_fraction"] == fraction]
            if len(matching) != 1:
                raise ValueError("Missing/duplicate reproduction comparison arm")
            row = matching[0]
            if (row["reproduction"] != str(directory.relative_to(PHASE))
                    or row["fresh_training_graph_steps"] != 250
                    or row["verified_cache_temp_paths"] != 17
                    or row["archived_sources_identical"] is not True
                    or row["validation_or_test_scored"] is not False
                    or row["new_fit_trace_sha256"] != trace["sha256"]
                    or any(row["identical_artifact_sha256"][name] != ref["sha256"]
                           for name, ref in (("contract.json", contract_ref),
                                             ("label_masks.pt", masks),
                                             ("probe_epoch25.pt", checkpoint)))):
                raise ValueError("Reproduction comparison does not bind supplied artifacts")
            for name, digest in contract["source_sha256"].items():
                if sha(phase_path(directory / "source" / name)) != digest:
                    raise ValueError("Reproduction archived source differs: " + name)
            arms.append({"arm_id": f"seed{seed}_label{percent}", "seed": seed,
                         "label_fraction": fraction, "contract": contract_ref,
                         "label_masks": masks,
                         "checkpoint": checkpoint, "fit_trace": trace,
                         "supervisor_completion_sha256": row["supervisor_completion_sha256"]})
    sources = {name: reference(PHASE / name) for name in json.loads(checked(arms[0]["contract"]).read_text())["source_sha256"]}
    return {"schema": SCHEMA, "scope": "six P-only epoch25 checkpoints; training-only D4/E4 spectral collection",
            "spectral_prototype_manifest": core, "synthetic_cpu_report": synthetic,
            "reproduction_comparison": comparison,
            "integration_code_sha256": {name: sha(HERE / name) for name in CODE_FILES},
            "probe_sources": sources, "data_manifest": data,
            "data_cache": str(phase_path(args.data_cache).relative_to(PHASE)),
            "train_processed_file": train,
            "partition_ids": fixed_ids(), "fit_partition_seed": 700, "score_partition_seed": 704,
            "within_group_order": "ascending official training split index",
            "epoch": 25, "n0": 1, "rank_cap_per_target": 16, "random_control_seed": 800,
            "runtime": {"device": "cpu", "threads": 1, "interop_threads": 1},
            "arms": arms, "holdout_loader_accessed": False, "spectral_collection_performed": False,
            "provenance_boundary": "Hashes and code audit bind supplied P-only artifacts; they do not prove an unobserved execution history."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--synthetic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_lock(args)
    value["lock_sha256"] = json_hash(value)
    path = phase_path(args.output)
    if HERE not in path.parents:
        raise ValueError("Input lock must be saved inside this integration directory")
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PROSPECTIVE_INPUTS_LOCKED", "path": str(path),
                      "lock_sha256": value["lock_sha256"], "file_sha256": sha(path),
                      "spectral_collection_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
