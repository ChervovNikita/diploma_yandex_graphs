"""Run the sealed model-free qualification once and preserve its receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REVIEW = Path(__file__).resolve().parent
PHASE = REVIEW.parent
PACKET = PHASE / "new_learning_gap_independent_v1/ar_fitted_screen_source_v2"
PYTHON = "/Users/alex/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert sha(PACKET / "AUTHORING_MANIFEST.json") == "f866aac23e8b68df4633f6ec9a4938499153662711f0db82d66913af8a9c18ff"
assert sha(PACKET / "AUTHORING_SEAL.json") == "eafe038f4f5b5e9599a7ab85fdb2007b5baf142e02a837991bc9fccc48ea549a"
before = {str(path.relative_to(PHASE)): sha(path) for path in PACKET.rglob("*") if path.is_file()}
output = REVIEW / "model_free_test_run01"
output.mkdir(exist_ok=False)
argv = [PYTHON, "-B", str(PACKET / "validate_bootstrap.py"), "--result", str(output / "validation_result.json")]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
started = time.perf_counter_ns()
with (output / "stdout.txt").open("x") as stdout, (output / "stderr.txt").open("x") as stderr:
    completed = subprocess.run(argv, cwd=REVIEW, env=env, stdout=stdout, stderr=stderr, timeout=30)
elapsed = (time.perf_counter_ns() - started) / 1e9
after = {str(path.relative_to(PHASE)): sha(path) for path in PACKET.rglob("*") if path.is_file()}
record = {
    "schema": "independent-bootstrap-test-process-v1",
    "UTC": datetime.now(timezone.utc).isoformat(),
    "argv": argv,
    "cwd": str(REVIEW),
    "exit_code": completed.returncode,
    "process_wall_seconds": elapsed,
    "bytecode_disabled_in_argv_and_environment": True,
    "stdout_sha256": sha(output / "stdout.txt"),
    "stderr_sha256": sha(output / "stderr.txt"),
    "result_sha256": sha(output / "validation_result.json"),
    "sealed_packet_hashes_before": before,
    "sealed_packet_hashes_after": after,
    "sealed_packet_unchanged": before == after,
    "source_packet_runs_directory_exists": (PACKET / "runs").exists(),
    "SSH_model_import_training_or_scientific_draw_requested": False,
}
with (output / "process_receipt.json").open("x") as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({key: record[key] for key in ("exit_code", "process_wall_seconds", "sealed_packet_unchanged", "source_packet_runs_directory_exists")}))
raise SystemExit(completed.returncode)
