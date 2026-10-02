"""Shared stdlib source/history guards; checkpoint validation is a separate gate."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def reject_constant(value):
        raise RuntimeError(f"Nonfinite JSON constant: {value}")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
    def verify_numbers(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise RuntimeError("Nonfinite JSON number.")
        if isinstance(item, dict):
            for child in item.values():
                verify_numbers(child)
        elif isinstance(item, list):
            for child in item:
                verify_numbers(child)
    verify_numbers(value)
    return value


def read_json(path):
    return strict_json(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite_number(value, name, lower=0.0, upper=None):
    if type(value) not in (int, float) or not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise RuntimeError(f"Invalid finite/ranged {name}: {value!r}")
    return value


def digest_value(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError(f"Invalid SHA256 for {name}")
    return value


def verify_source_pins(root=HERE):
    root = Path(root)
    pins = read_json(root / "SOURCE_PINS.json")
    result = {}
    if not pins.get("files"):
        raise RuntimeError("No pinned vendor sources.")
    for name, expected in pins["files"].items():
        if Path(name).name != name:
            raise RuntimeError("Unexpected vendor path.")
        actual = file_sha(root / "vendor" / name)
        if actual != digest_value(expected, name):
            raise RuntimeError(f"Pinned source changed: {name}")
        result[f"vendor/{name}"] = actual
    return result


def implementation_hashes(root=HERE):
    root = Path(root)
    vendor = verify_source_pins(root)
    names = sorted(p.name for p in root.glob("*.py")) + ["CONFIG.json", "SOURCE_PINS.json", "requirements-qualification-extra.txt"]
    return {**{name: file_sha(root / name) for name in names}, **vendor}


def selection_record_sha(record):
    # The checkpoint stores the selection metrics. Its byte hash is then added
    # to the ledger, avoiding a circular checkpoint/hash dependency.
    return canonical_sha({key: value for key, value in record.items() if key != "selected_checkpoint_sha256"})


def validate_epoch_record(record, epoch, selection=True):
    if not isinstance(record, dict) or type(record.get("epoch")) is not int or record["epoch"] != epoch:
        raise RuntimeError("Epoch ledger must contain consecutive integer epochs.")
    for name in ("train_bce", "train_seconds", "validation_forward_seconds"):
        finite_number(record.get(name), name)
    if selection:
        finite_number(record.get("validation_hits50"), "validation_hits50", upper=1.0)
        finite_number(record.get("validation_bce_sampled_pool"), "validation_bce_sampled_pool")


def validate_history(path, epochs):
    lines = Path(path).read_text().splitlines()
    if len(lines) != epochs or any(not line.strip() for line in lines):
        raise RuntimeError(f"A complete {epochs}-record epoch ledger is required.")
    records = [strict_json(line) for line in lines]
    best = None
    for epoch, record in enumerate(records, 1):
        validate_epoch_record(record, epoch)
        improved = best is None or record["validation_hits50"] > best["validation_hits50"]
        if improved:
            digest_value(record.get("selected_checkpoint_sha256"), "epoch checkpoint")
            best = record
        elif "selected_checkpoint_sha256" in record:
            raise RuntimeError("Only a strict improvement may select a checkpoint.")
    return best


def checkpoint_metadata(identity, record):
    return {
        "schema": "buddy-selected-checkpoint-v2",
        "identity_sha256": canonical_sha(identity),
        "arm": identity["arm"], "seed": identity["seed"],
        "cache_manifest_sha256": identity["cache_manifest_sha256"],
        "config_sha256": identity["config_sha256"],
        "implementation_hashes": identity["implementation_hashes"],
        "parameters": identity["parameters"], "torch_version": identity["torch_version"],
        "selected_epoch": record["epoch"],
        "selected_validation_hits50": record["validation_hits50"],
        "selected_record_sha256": selection_record_sha(record),
    }


def expected_parameters(arm):
    return {"native1024": 1185091, "single256": 99907, "factorized4": 106349, "independent4": 399628, "matched_single": 106457}[arm]


def validate_run_metadata(folder, arm, seed, cache_sha, config, sources):
    """Validate claims/bytes/history; never sufficient to unlock test by itself."""
    folder = Path(folder)
    completion = read_json(folder / "completion.json")
    identity = read_json(folder / "identity.json")
    expected = {"arm": arm, "seed": seed, "cache_manifest_sha256": cache_sha,
                "config_sha256": file_sha(HERE / "CONFIG.json"), "implementation_hashes": sources,
                "parameters": expected_parameters(arm), "optimizer_fits": 4 if arm == "independent4" else 1,
                "pooling": "mean_raw_logits"}
    if any(identity.get(key) != value or completion.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Cell identity mismatch: {folder}")
    if type(identity.get("seed")) is not int or type(completion.get("seed")) is not int:
        raise RuntimeError("Cell seed must be an integer.")
    if any(completion.get(key) != value for key, value in identity.items()):
        raise RuntimeError("Completion does not retain its run identity.")
    if (completion.get("status") != "training_complete" or type(completion.get("epochs_completed")) is not int
            or completion["epochs_completed"] != config["epochs"] or completion.get("test_loaded_or_scored") is not False):
        raise RuntimeError(f"Incomplete or test-contaminated cell: {folder}")
    finite_number(completion.get("selected_validation_hits50"), "selected_validation_hits50", upper=1.0)
    finite_number(completion.get("total_seconds"), "total_seconds")
    for name in ("peak_cuda_allocated", "peak_cuda_reserved"):
        if completion.get(name) is not None:
            finite_number(completion[name], name)
    best = validate_history(folder / "epochs.jsonl", config["epochs"])
    checkpoint_sha = file_sha(folder / "selected.pt")
    bindings = {"identity_sha256": file_sha(folder / "identity.json"),
                "epochs_sha256": file_sha(folder / "epochs.jsonl"),
                "selected_epoch": best["epoch"], "selected_validation_hits50": best["validation_hits50"],
                "selected_record_sha256": selection_record_sha(best), "selected_checkpoint_sha256": checkpoint_sha}
    if any(completion.get(key) != value for key, value in bindings.items()) or best["selected_checkpoint_sha256"] != checkpoint_sha:
        raise RuntimeError("Completion/checkpoint does not match the earliest maximum in its ledger.")
    if type(completion.get("selected_epoch")) is not int:
        raise RuntimeError("Selected epoch must be an integer.")
    run = {"arm": arm, "seed": seed, "run_directory": str(folder.resolve()),
           "completion_sha256": file_sha(folder / "completion.json"), **bindings}
    return run, checkpoint_metadata(identity, best)


def verify_family_metadata(lock, cache_sha, config_sha):
    sources = implementation_hashes()  # all vendor bytes checked first
    config = read_json(HERE / "CONFIG.json")
    value = read_json(lock)
    expected_cells = {(arm, seed) for arm in config["arms"] for seed in config["seeds"]}
    runs = value.get("runs", [])
    if not isinstance(runs, list) or any(not isinstance(run, dict) or type(run.get("seed")) is not int for run in runs):
        raise RuntimeError("Invalid locked cell list/seed type.")
    if (value.get("schema") != "buddy-family-lock-v2" or value.get("status") != "family_locked"
            or len(runs) != len(expected_cells) or {(run["arm"], run["seed"]) for run in runs} != expected_cells):
        raise RuntimeError("Final test requires every fixed arm/seed cell exactly once.")
    if value.get("cache_manifest_sha256") != cache_sha or value.get("config_sha256") != config_sha or value.get("implementation_hashes") != sources:
        raise RuntimeError("Family lock cache/recipe/source identity changed.")
    checkpoints = []
    for run in runs:
        actual, metadata = validate_run_metadata(run["run_directory"], run["arm"], run["seed"], cache_sha, config, sources)
        if run != actual:
            raise RuntimeError("Locked training artifacts/history changed.")
        checkpoints.append((Path(run["run_directory"]) / "selected.pt", metadata))
    return value, checkpoints
