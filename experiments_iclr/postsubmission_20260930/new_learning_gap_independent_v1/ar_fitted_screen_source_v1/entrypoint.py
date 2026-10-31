"""Unexecuted admitted-child bootstrap; only stdlib is loaded before checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
RESEARCH = BASE.parents[1]
EXPECTED_FREEZE = "5202fcb2b310377ee1363339a77711ff3447f35295f17b956e45fe61ea25d837"


def main():
    started = time.perf_counter_ns()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-freeze", type=Path, required=True)
    parser.add_argument("--root-freeze-sha256", required=True)
    parser.add_argument("--root-admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.root_freeze_sha256 == EXPECTED_FREEZE
    assert hashlib.sha256(args.root_freeze.read_bytes()).hexdigest() == EXPECTED_FREEZE
    freeze = json.loads(args.root_freeze.read_text())
    admission = json.loads(args.root_admission.read_text())
    assert admission["status"] == "admitted_by_root" and admission["scientific_freeze_sha256"] == EXPECTED_FREEZE
    manifest_path = BASE / "AUTHORING_MANIFEST.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == admission["authoring_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    hashes = {}
    for row in manifest["files"]:
        path = (BASE / row["path"]).resolve()
        assert path.is_relative_to(BASE) and path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        hashes[str(path)] = row["sha256"]
    for row in json.loads((BASE / "SOURCE_PINS.json").read_text())["files"]:
        path = (RESEARCH / row["research_relative_path"]).resolve()
        assert path.is_relative_to(RESEARCH) and path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        hashes[str(path)] = row["sha256"]
    spec = importlib.util.spec_from_file_location("ar_diag_bootstrap", BASE / "diagnostics.py")
    diagnostic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostic)
    meter = diagnostic.Meter(freeze["runtime"]["diagnostic_seconds"], args.output_dir.resolve())
    meter.phase_source_setup_seconds = (time.perf_counter_ns() - started) / 1e9
    instrumenter = diagnostic.Instrumenter(RESEARCH, hashes, meter)
    sys.meta_path.insert(0, instrumenter)
    try:
        runner = instrumenter.load_file("ar_fitted_instrumented_runner", BASE / "runner.py")
        return runner.main()
    finally:
        primary_exception = sys.exc_info()[0] is not None
        try:
            meter.save(final=True)
        except BaseException:
            if not primary_exception:
                raise


if __name__ == "__main__":
    raise SystemExit(main())
