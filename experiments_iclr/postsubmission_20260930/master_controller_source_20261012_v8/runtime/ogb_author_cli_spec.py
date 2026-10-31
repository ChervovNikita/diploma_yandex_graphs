"""Pinned OGB molecular CLI contracts; validation imports only the standard library."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from author_cli_spec import receipt, validate_receipt, verify_receipt
from spec import confined, fingerprint

AUTHOR_FAMILY = "snap_stanford_ogb_molecular_batchnorm_v1"
SOURCE_COMMIT = "61e9784ca76edeaa6e259ba0f836099608ff0586"
SEEDS = (73101, 73103, 73107, 73109, 73121, 73127, 73133, 73141, 73147, 73151)
DATASETS = {"ogbg-molhiv": {"train_graphs": 32901, "max_train_batches": 1029},
            "ogbg-molbace": {"train_graphs": 1210, "max_train_batches": 38}}
MODELS = ("gin", "gin-virtual", "gcn", "gcn-virtual")
RAW_FILES = {"RELEASE_v1.txt", "mapping/README.md", "mapping/mol.csv.gz",
             "raw/edge-feat.csv.gz", "raw/edge.csv.gz", "raw/graph-label.csv.gz",
             "raw/node-feat.csv.gz", "raw/num-edge-list.csv.gz", "raw/num-node-list.csv.gz",
             "split/scaffold/train.csv.gz", "split/scaffold/valid.csv.gz", "split/scaffold/test.csv.gz"}
SOURCE_PINS = {
    "main_pyg.py": (6894, "26125c957250accc8b933ab274ba6fa5ec5e21f91cd61df785266852fd1a156f"),
    "gnn.py": (2580, "eeac03fd138e49f0799a6757fc06f024f97d6774f814cba4ddb7ae5ff333f8b9"),
    "conv.py": (8741, "a592513e7fb0f6ed308ab036a961fe99382d3b09bd1fe3a503d5d8e65ff2e936"),
    "README.md": (2653, "0c4efb13e7a3166025f743a977e87f1af9cefaa9c814f7d1aad8ad0ef3fc0b7a"),
}
SPEC_FIELDS = {"schema_version", "study_id", "stage", "hypothesis", "dataset", "data",
               "source", "runs", "comparisons", "release", "resources"}
RUN_FIELDS = {"run_id", "kind", "method", "split", "seed", "body", "cli", "config",
              "budget", "selection", "parents"}
BUDGET = {"unit": "epochs", "policy": "native_full_budget", "maximum": 100,
          "repeats": 1, "resume": False}
SELECTION = {"metric": "rocauc", "direction": "max", "ties": "first_np_argmax",
             "evaluation_interval": 1, "checkpoint0": False,
             "artifact": "source_validation_curve_no_model_checkpoint"}
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
SEED_POLICY = {"kind": "declared_replication_seed_policy",
               "seed_ids": list(SEEDS), "independent_process_per_seed": True,
               "native_seed_or_repeat_flags_exist": False,
               "seed_before_native_module_import": True,
               "seed_libraries": ["python_random", "numpy", "torch_cpu", "torch_cuda_all"],
               "deterministic_algorithms_forced": False,
               "published_seed_protocol_equality_claim": False}


def is_ogb_author_spec(value):
    return isinstance(value, dict) and value.get("source", {}).get("author_family") == AUTHOR_FAMILY


def native_argv(dataset, model):
    return ["main_pyg.py", "--dataset", dataset, "--gnn", model, "--drop_ratio", "0.5",
            "--num_layer", "5", "--emb_dim", "300", "--batch_size", "32", "--epochs", "100",
            "--num_workers", "0", "--feature", "full", "--filename", ""]


def seeded_argv(dataset, model, seed):
    return ["-u", "declared_seed_entry.py", "--declared-seed", str(seed), "--",
            *native_argv(dataset, model)]


def relative(value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("An OGB path must be nonempty and project-relative")
    return Path(value)


def safe_id(value):
    return isinstance(value, str) and bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


def positive(value):
    return type(value) in {int, float} and math.isfinite(value) and value > 0


def validate_ogb_author_spec(value):
    if not isinstance(value, dict) or set(value) != SPEC_FIELDS or value["schema_version"] != 1 or not is_ogb_author_spec(value):
        raise ValueError("Incomplete dedicated OGB author-family study")
    if value["stage"] not in {"engineering", "development"} or not value["hypothesis"] or not safe_id(value["study_id"]):
        raise ValueError("Declare safe development/engineering OGB identity")
    if value["dataset"] != "ogb-two-molecule-native-author-roster":
        raise ValueError("The OGB family contains exactly the two prospective molecular datasets")
    source = value["source"]
    packet_root = relative(source.get("packet_root"))
    if (source.get("repository") != "snap-stanford/ogb" or source.get("git_commit") != SOURCE_COMMIT
            or source.get("replication_seed_policy") != SEED_POLICY
            or source.get("published_score_equality_claim") is not False
            or source.get("native_CLI_modified") is not False or not source.get("files")):
        raise ValueError("Bind exact source pins and declared independent replication seeds")
    for row in source["files"]:
        validate_receipt(row)
    wrapper = source.get("seed_entry")
    validate_receipt(wrapper)
    if Path(wrapper["path"]).name != "ogb_declared_seed_entry.py":
        raise ValueError("Bind the reviewed declared seed entry program")
    data = value["data"]
    if (set(data.get("roles", {})) != {"train", "select", "evaluate"}
            or data.get("role_protocol") != "official_ogb_scaffold_train_whole_valid_test_no_cal"
            or not data.get("prior_exposure") or not data.get("evidence")
            or data.get("train_graph_count_authority") != DATASETS
            or set(data.get("dataset_custody", {})) != set(DATASETS)):
        raise ValueError("Bind official scaffold roles/count authority; no CAL or split replacement")
    for ds, custody in data["dataset_custody"].items():
        validate_receipt(custody["receipt"])
        if not custody.get("raw_files") or custody.get("processed_cache_policy") != "fresh_empty_per_process":
            raise ValueError("Bind raw-only official provider custody and fresh native processing")
        for row in custody["raw_files"]:
            validate_receipt(row)
        suffixes = set()
        for row in custody["raw_files"]:
            parts = Path(row["path"]).parts
            if ds.replace("-", "_") not in parts:
                raise ValueError("Raw custody must bind the exact native dataset directory")
            suffixes.add(str(Path(*parts[parts.index(ds.replace("-", "_"))+1:])))
        if suffixes != RAW_FILES or len(custody["raw_files"]) != len(RAW_FILES):
            raise ValueError("Bind all twelve official raw/mapping/scaffold files")
        for name in ["train", "valid", "test"]:
            if not any("/split/scaffold/" + name + ".csv.gz" in x["path"] for x in custody["raw_files"]):
                raise ValueError("Missing official scaffold split binding")
    release = value["release"]
    if (release.get("policy") != "whole_roster" or release.get("global_plan_gates_required") is not True
            or release.get("required_before_launch") != ["whole_search"]
            or release.get("author_output_gate") != "whole_author"):
        raise ValueError("OGB requires whole-master search and whole-author quarantine gates")
    if not value["comparisons"] or not positive(value["resources"].get("wall_seconds")):
        raise ValueError("Declare comparisons and finite scheduling bounds")
    runs = value["runs"]
    if len(runs) != 80:
        raise ValueError("Exactly eight frozen recipes times ten independent seeds are required")
    seen, cells, paths = set(), set(), set()
    for run in runs:
        if set(run) != RUN_FIELDS or run["kind"] != "author_cli_group" or run["parents"]:
            raise ValueError("One complete native seeded subprocess per author group; no fake adapter/checkpoint")
        config, cli = run["config"], run["cli"]
        dataset, model = config.get("dataset"), config.get("source_model")
        if dataset not in DATASETS or model not in MODELS or run["seed"] not in SEEDS or type(run["seed"]) is not int:
            raise ValueError("Only the frozen eight recipes and ten declared seeds are allowed")
        cell = (dataset, model, run["seed"])
        expected_id = dataset + "-" + model + "-seed" + str(run["seed"])
        if (run["run_id"] != expected_id or not safe_id(run["run_id"]) or run["run_id"] in seen
                or cell in cells or run["body"] != 0 or run["split"] != "official_scaffold"
                or run["budget"] != BUDGET or run["selection"] != SELECTION
                or config.get("stage") != "author_cli" or config.get("author_family") != AUTHOR_FAMILY):
            raise ValueError("OGB process identity/budget/selection or independent recipe roster changed")
        seen.add(run["run_id"]); cells.add(cell)
        if config.get("max_train_batches_per_epoch") != DATASETS[dataset]["max_train_batches"]:
            raise ValueError("Retain the declared native batch-count upper bounds")
        for key in ["reservation_mib", "estimated_seconds", "wall_seconds"]:
            if not positive(config.get(key)):
                raise ValueError("Positive finite scheduling bounds are required")
        if type(config["reservation_mib"]) is not int or not config.get("resource_evidence"):
            raise ValueError("Declare memory MiB and explicit resource qualification scope")
        if (config.get("source_code_path_replication") is not True
                or config.get("published_score_equality_claim") is not False
                or cli.get("argv") != seeded_argv(dataset, model, run["seed"])
                or cli.get("environment") != ENVIRONMENT or cli.get("protocol") != SEED_POLICY):
            raise ValueError("Native numeric arguments/environment/declared seed wrapper changed")
        for key in ["python_executable", "execution_repository", "runtime_versions", "runtime_provider_files",
                    "source_files", "raw_files", "quarantine", "fresh_processed_dir", "cwd"]:
            if key not in cli:
                raise ValueError("Missing OGB process custody binding: " + key)
        if set(cli["quarantine"]) != {"stdout", "stderr"} or not cli["runtime_versions"]:
            raise ValueError("Bind native runtime metadata and fresh TEST log quarantine")
        cwd = relative(cli["cwd"])
        if cwd != packet_root / "workdirs" / expected_id:
            raise ValueError("Each seeded cwd must belong to its dedicated source packet")
        processed = relative(cli["fresh_processed_dir"])
        if processed != cwd / "dataset" / dataset.replace("-", "_") / "processed":
            raise ValueError("Fresh native provider processed directory must belong to this process")
        for path in [str(cwd), str(processed), *cli["quarantine"].values()]:
            relative(path)
            if path in paths:
                raise ValueError("Every seeded process owns separate cwd/cache/quarantine paths")
            paths.add(path)
        if cli["quarantine"] != {"stdout": str(packet_root / "quarantine" / (expected_id + ".stdout.log")),
                                  "stderr": str(packet_root / "quarantine" / (expected_id + ".stderr.log"))}:
            raise ValueError("Quarantine paths must be private to the bound seeded process")
        source_rows = {Path(x["path"]).name: x for x in cli["source_files"]}
        if set(source_rows) != set(SOURCE_PINS) | {"declared_seed_entry.py"} or len(source_rows) != len(cli["source_files"]):
            raise ValueError("Exact native source set and one declared seed entry required")
        for filename, (size, sha) in SOURCE_PINS.items():
            row = source_rows[filename]; validate_receipt(row)
            if (row["bytes"], row["sha256"]) != (size, sha) or relative(row["path"]).parent != cwd:
                raise ValueError("Pinned upstream OGB source bytes changed")
        row = source_rows["declared_seed_entry.py"]; validate_receipt(row)
        if (row["bytes"], row["sha256"]) != (wrapper["bytes"], wrapper["sha256"]) or relative(row["path"]).parent != cwd:
            raise ValueError("Declared seed entry differs from the reviewed wrapper")
        raw = {}
        provider = cwd / "dataset" / dataset.replace("-", "_")
        for row in cli["raw_files"]:
            validate_receipt(row)
            path = relative(row["path"])
            if not path.is_relative_to(provider) or "processed" in path.relative_to(provider).parts:
                raise ValueError("Raw provider files must remain inside the private native cache")
            rel = str(path.relative_to(provider))
            if rel in raw:
                raise ValueError("Duplicate raw file binding")
            raw[rel] = (row["bytes"], row["sha256"])
        expected = {}
        for row in data["dataset_custody"][dataset]["raw_files"]:
            parts = Path(row["path"]).parts
            pivot = parts.index(dataset.replace("-", "_"))
            expected[str(Path(*parts[pivot+1:]))] = (row["bytes"], row["sha256"])
        if raw != expected:
            raise ValueError("Private cache differs from the bound official raw/scaffold provider files")
        for row in cli["runtime_provider_files"]:
            validate_receipt(row)
    expected_cells = {(ds, model, seed) for ds in DATASETS for model in MODELS for seed in SEEDS}
    if cells != expected_cells:
        raise ValueError("The finite OGB recipe/seed roster is incomplete")
    return value


def bind_ogb_author_files(root, value):
    validate_ogb_author_spec(value)
    for row in value["source"]["files"]:
        verify_receipt(root, row)
    verify_receipt(root, value["source"]["seed_entry"])
    for custody in value["data"]["dataset_custody"].values():
        verify_receipt(root, custody["receipt"])
    for run in value["runs"]:
        for row in run["cli"]["source_files"] + run["cli"]["raw_files"]:
            verify_receipt(root, row)


def load_ogb_author_spec(path, root, *, bind=True):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    value = validate_ogb_author_spec(json.loads(raw))
    if bind:
        bind_ogb_author_files(root, value)
    return value, fingerprint(value)


def ogb_author_counts(value):
    validate_ogb_author_spec(value)
    return {"author_cli_groups": 80, "author_independent_seed_trajectories": 80,
            "author_sequential_repeat_trajectories": 0,
            "author_registered_maximum_epochs": 8000,
            "author_registered_maximum_optimizer_updates": sum(100 * r["config"]["max_train_batches_per_epoch"] for r in value["runs"]),
            "author_native_actual_updates": None, "actual_author_groups_started": 0,
            "actual_author_fits": 0, "actual_author_optimizer_updates_executed_so_far": 0,
            "model_fit_groups": 0, "fake_model_checkpoints": 0,
            "cost_bound_policy": "Upper bound; native skipped minibatches make actual calls unresolved without admitted accounting."}
