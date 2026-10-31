#!/usr/bin/env python3
"""Metadata-only dependency planner. Never imports science code, transfers, or launches.

Use --release for the exact normalized release source closure. Without it, the
local source manifest is explicitly partial. --inventory-providers only hashes
existing provider supplemental files; it does not decode datasets or score TEST.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

E = "benchmark_expansion_20261015_v1"
PHASE = "experiments_iclr/postsubmission_20260930"
RELEASE_PIN = {"bytes": 31728102, "sha256": "d5c09b12cc5d73ec7207a9d8f2417e88fa37821ca1fdcaefdb23bc77546500a9"}
QUEUE_PINS = {
    "server51/gpu0": {"bytes": 3125199, "sha256": "d22eb84930f15f337e3b990449c99a9babf470d375338e7f087393dfc0ae5549"},
    "server51/gpu1": {"bytes": 3122401, "sha256": "657c22c04e98fed8732d3aa42e63d37d8a3a261381b170efb317560fc03178cd"},
}


def require(test, message):
    if not test:
        raise ValueError(message)


def pin(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": h.hexdigest()}


def read(path, expected=None):
    if expected:
        require(pin(path) == {k: expected[k] for k in ("bytes", "sha256")}, "Pin mismatch: " + str(path))
    return json.loads(Path(path).read_text())


def relative(path, phase_relative=False):
    value = str(path)
    if "/experiments_iclr/" in value:
        value = "experiments_iclr/" + value.split("/experiments_iclr/", 1)[1]
    elif phase_relative:
        value = PHASE + "/" + value
    p = PurePosixPath(value)
    require(not p.is_absolute() and ".." not in p.parts and value.startswith(PHASE + "/"),
            "Noncanonical or escaping path: " + value)
    return value


def row(path, value, phase_relative=False):
    rel = relative(path, phase_relative)
    require(type(value["bytes"]) is int and value["bytes"] >= 0, "Invalid size")
    require(len(value["sha256"]) == 64 and all(c in "0123456789abcdef" for c in value["sha256"]), "Invalid hash")
    return dict(source_relative=rel, target_relative=rel, bytes=value["bytes"], sha256=value["sha256"])


def deduplicate(rows):
    result = {}
    for r in rows:
        key = r["source_relative"]
        require(key not in result or result[key] == r, "Conflicting source identities: " + key)
        result[key] = r
    return sorted(result.values(), key=lambda r: r["source_relative"])


def coverage(receipt_files):
    found = set()
    for path in receipt_files:
        outer = read(path)
        if outer.get("returncode") != 0:
            continue
        try:
            result = json.loads(outer["stdout"])
        except (KeyError, ValueError):
            continue
        if result.get("status") != "EXACT_STREAM_DESTINATION_HASHES_VERIFIED":
            continue
        for r in result["files"]:
            if "/experiments_iclr/" in r["path"]:
                found.add((relative(r["path"]), r["bytes"], r["sha256"]))
    return found


def core_data(cells):
    files, uses, directories = [], defaultdict(list), set()
    for cell_id, cell in sorted(cells.items()):
        for role, value in sorted(cell["data_role_receipts"].items()):
            r = row(value["path"], value)
            files.append(r)
            uses[r["source_relative"]].append({"cell_id": cell_id, "dataset": cell["dataset"], "role": role})
            if "/processed/" in r["source_relative"]:
                directories.add(r["source_relative"].split("/processed/", 1)[0])
        if cell["family"] == "link":
            directories.add(relative(cell["data_root"]))
    return deduplicate(files), dict(uses), sorted(directories)


def supplemental_inventory(phase_root, directories):
    files = []
    for directory in directories:
        local = phase_root / directory.removeprefix(PHASE + "/")
        if not local.is_dir():
            continue
        # Whole raw folders are provider inputs, including PyG raw index files.
        # Separate split/test files, result directories and download archives are excluded.
        candidates = list((local / "raw").rglob("*")) if (local / "raw").is_dir() else []
        candidates += list(local.glob("RELEASE*"))
        candidates += [local / "processed" / name for name in ("pre_filter.pt", "pre_transform.pt")]
        for p in candidates:
            if p.is_file() and not p.is_symlink():
                files.append(row(p.relative_to(phase_root).as_posix(), pin(p), True))
    return deduplicate(files)


def source_rows(phase_root, release):
    native = read(phase_root / "reference_wrapper_redesign_20261013_v1/native_sources/MANIFEST.json")
    native_rows = [row(r["snapshot_path"], r, True) for r in native["files"]]
    if release:
        value = read(release, RELEASE_PIN)
        require(value.get("human_GO") is False, "Only the pinned unadmitted release is accepted")
        sources = [row(r["path"], r) for r in value["source_files"]]
        return deduplicate(sources + native_rows), native_rows, "pinned_release_declared_source_closure"
    sources = []
    for folder in ("normalization_portable_dispatch_v1", "composable_normalization_runtime_v1",
                   "composable_normalization_v2", "benchmark_reporting_normalization_rebind_v1",
                   "composable_normalization_final_scoring_v2", "portable_dispatcher_v3"):
        directory = phase_root / E / folder
        manifest = directory / "MANIFEST.json"
        sources.append(row(manifest.relative_to(phase_root).as_posix(), pin(manifest), True))
        for r in read(manifest)["files"]:
            path = r.get("path_relative_to_phase", E + "/" + folder + "/" + r["path"])
            sources.append(row(path, r, True))
        bindings = directory / "SOURCE_BINDINGS.json"
        if bindings.exists():
            sources.extend(row(r["path_relative_to_phase"], r, True) for r in read(bindings)["files"])
    return deduplicate(sources + native_rows), native_rows, "partial_local_current_manifest_closure"


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--queue", type=Path, action="append", default=[])
    parser.add_argument("--inventory-providers", action="store_true")
    args = parser.parse_args()
    phase = args.phase_root.resolve()
    cells_path = phase / E / "job_binding_preparation_v1/CELL_BINDINGS.json"
    cells = read(cells_path)["cells"]
    data, uses, directories = core_data(cells)
    sources, native, scope = source_rows(phase, args.release)
    descriptors, queue_workers = [], set()
    for path in args.queue:
        queue = read(path)
        worker = queue.get("worker")
        require(worker in QUEUE_PINS and worker not in queue_workers, "Unique server51 queue required")
        read(path, QUEUE_PINS[worker])
        require(queue["schema"] == "dormant-three-gpu-queue-v1" and queue["launch_admitted"] is False,
                "Only exact dormant queues accepted")
        queue_workers.add(worker)
        for task in queue["tasks"]:
            t = task["canonical_task"]
            if t["kind"] == "fit":
                descriptors.append(row(t["source_descriptor"]["path"], t["source_descriptor"]))
    descriptors = deduplicate(descriptors)
    receipts = sorted((phase / E / "planning/three_gpu_sync_20261015_v1").glob("*TRANSFER_RECEIPT.json"))
    covered = coverage(receipts)
    supplement = supplemental_inventory(phase, directories) if args.inventory_providers else []
    all_rows = deduplicate(data + sources + supplement + descriptors)
    not_receipted = [r for r in all_rows if (r["target_relative"], r["bytes"], r["sha256"]) not in covered]
    summary = {"schema": "campaign-dependency-plan-v1", "source_scope": scope,
               "cell_count": len(cells), "core_files": len(data), "core_bytes": sum(r["bytes"] for r in data),
               "native_snapshot_files": len(native), "native_snapshot_bytes": sum(r["bytes"] for r in native),
               "source_files": len(sources), "source_bytes": sum(r["bytes"] for r in sources),
               "descriptor_files": len(descriptors), "descriptor_bytes": sum(r["bytes"] for r in descriptors),
               "queue_workers": sorted(queue_workers),
               "supplemental_files": len(supplement), "supplemental_bytes": sum(r["bytes"] for r in supplement),
               "not_receipted_files": len(not_receipted), "not_receipted_bytes": sum(r["bytes"] for r in not_receipted),
               "fresh_destination_inventory": False, "provider_complete_claim": False,
               "complete_import_tree_claim": False, "full_campaign_admitted": False,
               "launches": 0, "transfers": 0, "dataset_payloads_decoded": False,
               "source_identity_conflicts": 0,
               "required_remaining": ["Exact pinned release source closure when local partial mode is used",
                   "Fresh destination file/size/hash comparison before transfer; never overwrite a conflicting identity",
                   "Exact source descriptor inventories from canonical server51 queues",
                   "Supplemental provider file inventory and current-source provider/import applicability",
                   "Explicit physical project/phase binding supplied by portability package",
                   "Current GPU/RAM/cgroup/disk admission; historical receipts are not reservations"]}
    args.output.mkdir(parents=False, exist_ok=False)
    write(args.output / "CORE_DATA.json", {"files": data, "uses": uses})
    write(args.output / "NATIVE_SNAPSHOT.json", {"files": native, "complete_import_tree_claim": False})
    write(args.output / "SOURCE_TRANSFER_PLAN.json", {"files": sources})
    write(args.output / "DESCRIPTOR_TRANSFER_PLAN.json", {"files": descriptors})
    write(args.output / "PROVIDER_INVENTORY_REQUEST.json", {"repository_relative_directories": directories,
          "allow": ["raw/** provider files", "processed/pre_filter.pt", "processed/pre_transform.pt", "RELEASE*"],
          "exclude": ["separate split/test files", "results", "download archives"],
          "raw_PyG_index_files_are_provider_inputs": True, "decode_payloads": False})
    write(args.output / "SUPPLEMENTAL_TRANSFER_PLAN.json", {"files": supplement})
    write(args.output / "NOT_RECEIPTED_TRANSFER_PLAN.json", {"files": not_receipted})
    write(args.output / "SUMMARY.json", summary)
    write(args.output / "INPUT_PINS.json", {"cell_bindings": pin(cells_path), "release": RELEASE_PIN if args.release else None,
          "saved_receipts": [{"name": p.name, **pin(p)} for p in receipts]})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
