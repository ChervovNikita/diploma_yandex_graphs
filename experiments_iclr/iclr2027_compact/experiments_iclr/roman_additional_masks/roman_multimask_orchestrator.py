"""Freeze, run, and audit the complete Roman four-mask/two-depth grid."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MASKS = (1, 2, 3, 4)
DEPTHS = (2, 5)
SEEDS = (0, 1, 2)
ARMS = ("tied", "untied_propagation")
STUDIES = [(mask, depth) for mask in MASKS for depth in DEPTHS]
PINNED_FILES = {"roman_multimask.py", "verify_roman_multimask.py",
                "roman_multimask_orchestrator.py", "roman_multimask_protocol.md",
                "models.py", "data/roman_empire.npz"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> str:
    path = ROOT / "ROMAN_MULTIMASK_FREEZE.json"
    frozen = json.loads(path.read_text())
    assert frozen["protocol"] == "roman_multimask_no_added_loops_depth2_5_1000_v1"
    assert frozen["masks"] == list(MASKS)
    assert frozen["depths"] == list(DEPTHS)
    assert frozen["seeds"] == list(SEEDS)
    assert frozen["arms"] == list(ARMS)
    assert frozen["expected_cells"] == 48
    assert set(frozen["input_sha256"]) == PINNED_FILES
    for rel, digest in frozen["input_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    return sha(path)


def run(script: str, mask: int, depth: int, gpu: int, *args: str) -> None:
    command = [PYTHON, "-u", script, "--dataset", "roman", "--mask", str(mask),
               "--depth", str(depth), "--device", "cuda:0", *args]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("START", mask, depth, "physical_gpu", gpu, script, " ".join(args), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def manifest_hashes() -> dict:
    import roman_multimask as study
    results = {}
    for mask, depth in STUDIES:
        path = ROOT / "results/roman" / f"mask{mask}" / f"depth{depth}" / "source_manifest.json"
        spec = json.loads(path.read_text())
        assert spec["protocol"] == study.PROTOCOL
        assert spec["split"] == mask
        assert spec["configuration"]["layers"] == depth
        assert spec["configuration"]["epochs"] == 1000
        assert spec["optimization_seeds"] == list(SEEDS)
        assert spec["arms"] == list(ARMS)
        assert spec["data_descriptor"]["undirected_edges"] == 65854
        assert spec["data_descriptor"]["explicit_self_loops_added"] is False
        for rel, digest in spec["source_sha256"].items():
            assert sha(ROOT / rel) == digest, rel
        results[f"mask{mask}/depth{depth}"] = sha(path)
    return results


def preflight() -> None:
    frozen_sha = verify_inputs()
    for mask, depth in STUDIES:
        run("roman_multimask.py", mask, depth, 0, "--preflight")
    manifests = manifest_hashes()
    for mask, depth in STUDIES:
        run("roman_multimask.py", mask, depth, 0, "--smoke")
    gate = {"status": "FROZEN_INPUTS_AND_GPU_SMOKE_PASS",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "freeze_manifest_sha256": frozen_sha,
            "source_manifest_sha256": manifests,
            "expected_cells": 48}
    path = ROOT / "results/roman/preflight_and_smoke_audit.json"
    path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print("ALL_PREFLIGHT_AND_SMOKE_PASS", len(STUDIES), flush=True)


def worker(depth: int, gpu: int) -> None:
    assert (depth, gpu) in ((2, 0), (5, 0))
    frozen_sha = verify_inputs()
    manifests = manifest_hashes()
    gate = json.loads((ROOT / "results/roman/preflight_and_smoke_audit.json").read_text())
    assert gate["status"] == "FROZEN_INPUTS_AND_GPU_SMOKE_PASS"
    assert gate["freeze_manifest_sha256"] == frozen_sha
    assert gate["source_manifest_sha256"] == manifests
    for mask in MASKS:
        study = ROOT / "results/roman" / f"mask{mask}" / f"depth{depth}"
        for seed in SEEDS:
            for arm in ARMS:
                result = study / f"seed{seed}" / arm / "result.json"
                if result.is_file():
                    print("ALREADY_COMPLETE", mask, depth, seed, arm, flush=True)
                    continue
                run("roman_multimask.py", mask, depth, gpu, "--seeds", str(seed),
                    "--arms", arm)
                assert result.is_file(), result
        run("verify_roman_multimask.py", mask, depth, gpu)
        audit = json.loads((study / "completion_audit.json").read_text())
        assert audit["status"] == "COMPLETE_CUDA_REPLAY_PASS"
        assert audit["mask"] == mask and audit["depth"] == depth
        assert len(audit["records"]) == 6
        print("STUDY_REPLAY_PASS", mask, depth, flush=True)
    print("DEPTH_COMPLETE", depth, flush=True)


def aggregate() -> None:
    frozen_sha = verify_inputs()
    manifests = manifest_hashes()
    records = []
    for mask, depth in STUDIES:
        study = ROOT / "results/roman" / f"mask{mask}" / f"depth{depth}"
        audit = json.loads((study / "completion_audit.json").read_text())
        assert audit["status"] == "COMPLETE_CUDA_REPLAY_PASS"
        assert audit["mask"] == mask and audit["depth"] == depth
        assert audit["source_manifest_sha256"] == manifests[f"mask{mask}/depth{depth}"]
        assert len(audit["records"]) == 6
        for row in audit["records"]:
            records.append({"mask": mask, "depth": depth, **row})
    assert len(records) == 48
    summary = {"status": "COMPLETE_48_CELL_CUDA_REPLAY_PASS",
               "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "freeze_manifest_sha256": frozen_sha,
               "source_manifest_sha256": manifests,
               "records": records}
    path = ROOT / "results/roman/complete_grid_audit.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("COMPLETE_48_CELL_CUDA_REPLAY_PASS", flush=True)


def production() -> None:
    worker(2, 0)
    worker(5, 0)
    aggregate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("preflight", "production", "worker", "aggregate"))
    parser.add_argument("--depth", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "production":
        production()
    elif args.stage == "worker":
        worker(args.depth, args.gpu)
    else:
        aggregate()
