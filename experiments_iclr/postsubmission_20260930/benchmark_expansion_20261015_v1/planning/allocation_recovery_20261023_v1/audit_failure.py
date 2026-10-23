"""Audit the two reporting failures without training or opening TEST data."""
import datetime
import hashlib
import json
import socket
import subprocess
from pathlib import Path


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"
    ], text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    project = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
    phase = project / "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
    state_root = phase / "campaign_execution_20261016_v7/runs/allocation_gpu0"
    status = json.loads((state_root / "STATUS.json").read_text())
    assert status["phase"] == "full_failed" and not status["active_fit_ids"]
    expected = {
        "collab-buddy/long_convergence/split-0/seed-0/wrapper-aggregation_conditional_globalJ_v1/normalization-route-private-v1",
        "collab-buddy/long_convergence/split-0/seed-0/wrapper-aggregation_global_globalJ_v1/normalization-route-private-v1",
    }
    assert set(status["failed_fit_ids"]) == expected
    owner = json.loads((state_root / "OWNER.json").read_text())
    plan = json.loads(Path(owner["plan"]["path"]).read_text())
    tasks = {row["id"]: row for row in plan["tasks"]}
    # Checkpoint inspection is CPU deserialization of our saved artifacts only.
    import torch
    repairs = {}
    for fit_id in sorted(expected):
        state_path = state_root / "full" / hashlib.sha256(fit_id.encode()).hexdigest() / "STATE.json"
        state = json.loads(state_path.read_text())
        output = Path(state["output"])
        result = json.loads((output / "RESULT.json").read_text())
        assert result["status"] == "failed"
        assert result["failure"]["message"] == "Only genuine project source dependency bytes may be verified"
        assert "composed_authority.py" in result["failure"]["traceback"]
        assert result["TEST_accessed"] is False and result["completed_epochs"] == 149
        recovery = torch.load(output / "recovery.pt", map_location="cpu", weights_only=False)
        selected = torch.load(output / "selected.pt", map_location="cpu", weights_only=False)
        exported = torch.load(output / "selected_VALID.pt", map_location="cpu", weights_only=False)
        assert recovery["job_hash"] == selected["job_hash"] == exported["job_hash"] == result["job_hash"]
        assert result["job_hash"] == tasks[fit_id]["runtime_job_hash"]
        assert recovery["runtime_binding"] == selected["runtime_binding"] == result["runtime_binding"]
        assert exported["selected_checkpoint"] == pin(output / "selected.pt")
        assert exported["role"] == "VALID" and exported["TEST_accessed"] is False
        assert exported["selected_epoch"] == result["selected_epoch"]
        assert bool(torch.isfinite(exported["raw_logits"]).all())
        assert exported["complete_VALID_positive_records"] == 60084
        assert exported["complete_VALID_negative_records"] == 100000
        assert recovery["optimizer_calls"] == result["optimizer_calls"]
        job_path = Path(tasks[fit_id]["descriptor"]["path"])
        job = json.loads(job_path.read_text())
        curve = output / "CURVE.jsonl"
        with curve.open("rb") as stream:
            stream.seek(max(0, curve.stat().st_size - 100000))
            last = json.loads(stream.read().splitlines()[-1])
        assert last["selection"]["stop"] is True
        assert last["train"]["epoch"] == 149
        assert last["selection"]["stop_reason"] == "patience"
        repairs[fit_id] = {
            "failed_attempt_result": state["attempt_result"],
            "failed_native_result": state["native_result"],
            "descriptor": pin(job_path),
            "output": str(output),
            "artifacts": {name: pin(output / name) for name in (
                "RESULT.json", "selected.pt", "recovery.pt", "selected_VALID.pt", "PROGRESS.json", "CURVE.jsonl"
            )},
            "job_hash": result["job_hash"],
            "completed_epochs": 149,
            "selected_epoch": result["selected_epoch"],
            "optimizer_calls": result["optimizer_calls"],
            "recovery_selector": recovery["selector"],
            "stop_reason": "patience",
            "repair_scope": "reporting validation and finalization only; zero new training epochs",
        }
        del recovery, selected, exported
    config = {
        "schema": "explicit-allocation-reporting-recovery-v1",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "worker": "allocation/gpu0", "project_root": str(project),
        "plan": owner["plan"], "status_before": status,
        "repairs": repairs, "TEST_accessed": False,
        "checkpoint_tensor_inspection": "CPU only; existing own artifacts",
        "new_training_epochs_authorized_for_recovery": 0,
    }
    target = Path(__file__).with_name("RECOVERY_CONFIG.json")
    assert not target.exists(), "Keep the original audit immutable"
    target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "config": pin(target),
                      "recoverable_fits": len(repairs), "completed_epochs_each": 149,
                      "existing_exports_finite": True, "new_TRAIN_updates": 0}))


if __name__ == "__main__":
    main()
