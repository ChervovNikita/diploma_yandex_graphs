"""Project the audited raw profile to explicit anonymous public fields only.

The original profile remains author-only. This script does not measure new
timings or choose model checkpoints; it copies all 7,200 frozen samples.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "inference_profile_public_stage"
DATASETS = ("cora", "wikics", "actor", "chameleon_filtered")
ARMS = ("base", "ens", "tied", "private_first", "private_last", "untied")
COPY = ("VALIDATION_SELECTION_LOCK.json", "FROZEN_STUDY.json",
        "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv", "INFERENCE_PROFILE_AUDIT.json",
        "project_inference_profile_public.py", "verify_inference_profile_public.py",
        "profile_selected_inference.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def main() -> None:
    require(not STAGE.exists() and not STAGE.is_symlink(),
            "Refusing to overwrite prior public profile stage")
    raw_path = ROOT / "INFERENCE_PROFILE.json"
    audit_path = ROOT / "INFERENCE_PROFILE_AUDIT.json"
    lock_path = ROOT / "VALIDATION_SELECTION_LOCK.json"
    frozen_path = ROOT / "FROZEN_STUDY.json"
    for path in (raw_path, audit_path, lock_path, frozen_path):
        require(path.is_file(), f"Missing profile projection input: {path.name}")
    raw = json.loads(raw_path.read_text())
    audit = json.loads(audit_path.read_text())
    lock = json.loads(lock_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    require(raw["protocol"] == "selected_inference_profile_v1" and
            audit["status"] == "PASS" and
            audit["raw_profile_sha256"] == sha(raw_path) and
            audit["anonymous_table_sha256"] ==
            sha(ROOT / "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv") and
            audit["selection_lock_sha256"] == raw["selection_lock_sha256"] ==
            sha(lock_path) and
            raw["freeze_sha256"] == lock["freeze_sha256"] == sha(frozen_path) and
            raw["source_sha256"] == audit["source_sha256"] ==
            "427b828a8be63526922e1f1a98d1f4d5ca5c09bf95100388b1364e8cf1ec90d2" and
            len(lock["cells"]) == 432 and
            set(raw["graphs"]) == set(DATASETS) and
            raw["optimization_seed"] == 0 and raw["rounds"] == 3 and
            raw["warmups_per_block"] == 20 and
            raw["repetitions_per_block"] == 100,
            "Original independent audit or selection lock differs")
    sample = {
        "protocol": "selected_inference_profile_public_v1",
        "raw_profile_sha256": sha(raw_path),
        "projection_source_sha256": sha(Path(__file__)),
        "profiler_source_sha256": raw["source_sha256"],
        "selection_lock_sha256": sha(lock_path),
        "freeze_sha256": sha(frozen_path),
        "optimization_seed": 0, "rounds": 3,
        "warmups_per_block": 20, "repetitions_per_block": 100,
        "graphs": {},
    }
    count = 0
    for dataset in DATASETS:
        graph = raw["graphs"][dataset]
        require(graph["descriptor"] == frozen["graphs"][dataset]["descriptor"] and
                set(graph["arms"]) == set(ARMS),
                f"Raw graph profile differs from freeze: {dataset}")
        public_arms = {}
        for arm in ARMS:
            rec = graph["arms"][arm]
            selected = lock["selections"][dataset][arm]["selected_candidate"]
            lr, wd = selected
            key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed0"
            require(rec["selected_candidate"] == selected and
                    rec["checkpoint_sha256"] == lock["cells"][key]["checkpoint_sha256"] and
                    len(rec["blocks"]) == 3,
                    f"Raw selected profile differs from lock: {dataset}/{arm}")
            blocks = []
            for original in rec["blocks"]:
                block = {name: original[name] for name in
                         ("round", "position", "latency_ms",
                          "idle_allocated_bytes", "peak_allocated_bytes",
                          "peak_increment_bytes")}
                count += len(block["latency_ms"])
                blocks.append(block)
            public_arms[arm] = {
                "selected_candidate": selected,
                "checkpoint_sha256": rec["checkpoint_sha256"],
                "parameter_count": rec["parameter_count"],
                "parameter_bytes": rec["parameter_bytes"],
                "blocks": blocks,
            }
        sample["graphs"][dataset] = {"arms": public_arms}
    require(count == 7200 and audit["rows"] == 24 and audit["samples"] == 7200,
            "Public profile did not retain all 7,200 timings")

    STAGE.mkdir()
    for name in COPY:
        source = ROOT / name
        require(source.is_file() and not source.is_symlink(),
                f"Missing public profile source: {name}")
        shutil.copyfile(source, STAGE / name)
    require(sha(STAGE / "profile_selected_inference.py") == raw["source_sha256"],
            "Original profiler source differs from raw profile/audit")
    runtime = {
        "protocol": "selected_inference_profile_runtime_public_v1",
        "raw_profile_sha256": sha(raw_path),
        "python": raw["python"], "torch": raw["torch"],
        "pyg": raw["pyg"], "cuda": raw["cuda"],
        "device_class": raw["device"],
        "cpu_threads": raw["cpu_threads"],
        "memory_measurement": "torch.cuda.max_memory_allocated after one postwarmup forward per block; maximum of three blocks in table",
        "timing_measurement": "100 synchronized full-graph forward calls per block, three rotated blocks",
    }
    (STAGE / "INFERENCE_PROFILE_RUNTIME_ANONYMOUS.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    checkpoint_archive = ROOT / "PROFILE_SELECTED_CHECKPOINTS.tar.gz"
    require(checkpoint_archive.is_file() and sha(checkpoint_archive) ==
            "d206bdc316cb32e917bc6d258d00a2cd55dc93736949a8af34f6a4a25059c406",
            "Audited selected checkpoint transport archive differs")
    selected_results = {}
    with tarfile.open(checkpoint_archive, "r:gz") as archive:
        names = {member.name for member in archive if member.isfile()}
        for dataset in DATASETS:
            for arm in ARMS:
                lr, wd = lock["selections"][dataset][arm]["selected_candidate"]
                key = f"{dataset}/{arm}/lr{lr:g}_wd{wd:g}/seed0"
                source_name = f"results/{key}/result.json"
                require(source_name in names,
                        f"Selected training result missing from audited transport: {key}")
                source = archive.extractfile(source_name)
                require(source is not None, f"Cannot read selected training result: {key}")
                value = source.read()
                require(hashlib.sha256(value).hexdigest() ==
                        lock["cells"][key]["result_sha256"],
                        f"Selected training result differs from global lock: {key}")
                destination = STAGE / "selected_results" / dataset / arm / "result.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(value)
                selected_results[key] = sha(destination)
    sample_path = STAGE / "PUBLIC_PROFILE_SAMPLES.json"
    sample_path.write_text(json.dumps(sample, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    manifest = {
        "protocol": "selected_inference_profile_public_v1",
        "raw_profile_sha256": sha(raw_path),
        "samples_sha256": sha(sample_path),
        "anonymous_table_sha256": sha(STAGE / "INFERENCE_PROFILE_ANONYMOUS_TABLE.csv"),
        "audit_sha256": sha(STAGE / "INFERENCE_PROFILE_AUDIT.json"),
        "selection_lock_sha256": sha(STAGE / "VALIDATION_SELECTION_LOCK.json"),
        "frozen_study_sha256": sha(STAGE / "FROZEN_STUDY.json"),
        "projection_source_sha256": sha(STAGE / "project_inference_profile_public.py"),
        "verifier_source_sha256": sha(STAGE / "verify_inference_profile_public.py"),
        "profiler_source_sha256": sha(STAGE / "profile_selected_inference.py"),
        "runtime_sha256": sha(STAGE / "INFERENCE_PROFILE_RUNTIME_ANONYMOUS.json"),
        "selected_result_sha256": selected_results,
        "selected_checkpoint_transport_sha256": sha(checkpoint_archive),
        "rows": 24, "timings": count,
        "omitted": "Original raw GPU context, UUID, host details, checkpoints, graph files, and model logits",
    }
    (STAGE / "PUBLIC_PROFILE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Schema allowlists above and in the public verifier prevent raw context
    # fields from entering the projection. No raw profile file is copied.
    completed = subprocess.run(
        [sys.executable, str(STAGE / "verify_inference_profile_public.py")],
        check=True, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    require('"status": "PASS"' in completed.stdout,
            "Public profile verifier did not pass")
    print(json.dumps({"stage": STAGE.name,
                      "samples_sha256": sha(sample_path),
                      "manifest_sha256": sha(STAGE / "PUBLIC_PROFILE_MANIFEST.json"),
                      "files": len(tuple(STAGE.iterdir())),
                      "verifier": completed.stdout.strip()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
