"""Authorize only checkpoint-stopped reporting failures, without new training."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket

from repair_native import guard, pin, require
from reporting_patch import install

PLAN_SHAS = {
    "allocation/gpu0": "19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad",
    "server51/gpu0": "b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345",
    "server51/gpu1": "bc025b3209c0a7f5c2f10fd011fd54ca072c78269693b035ca3e53bbf243ad3c",
}
ALLOWED_FAILURES = {
    "Only genuine project source dependency bytes may be verified",
    "Only the exact pinned BUDDY README dependency is additionally allowed",
}


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    worker = plan["worker"]
    require(pin(plan_path)["sha256"] == PLAN_SHAS[worker], "Frozen worker plan changed")
    require(worker.split("/")[0] == ("allocation" if socket.gethostname() == "anogena-2-0" else "server51"), "Worker/host mismatch")
    project = Path(plan["project_root"]).resolve()
    target = Path(args.output).resolve()
    require(target.is_relative_to(project) and not target.exists(), "New repository-confined authority path required")
    root = Path(plan["state_root"])
    status = json.loads((root / "STATUS.json").read_text())
    require(status["phase"] == "full_failed" and not status["active_fit_ids"], "Only a fully stopped failed queue may be audited")
    owner = json.loads((root / "OWNER.json").read_text())
    require(owner["plan"] == pin(plan_path), "Owner/plan mismatch")
    tasks = {row["id"]: row for row in plan["tasks"]}
    import torch
    install()
    import composed_authority
    repairs = {}
    preflight = {}
    for fit_id in sorted(status["failed_fit_ids"]):
        state_path = root / "full" / hashlib.sha256(fit_id.encode()).hexdigest() / "STATE.json"
        state = json.loads(state_path.read_text())
        output = Path(state["output"])
        require(output.resolve().is_relative_to(root), "Failure output escapes queue")
        result = json.loads((output / "RESULT.json").read_text())
        require(result["status"] == "failed" and result["failure"]["message"] in ALLOWED_FAILURES,
                "Not an audited reporting dependency failure")
        require("observe_exports.py" in result["failure"]["traceback"], "Failure was not export observation")
        require(result["TEST_accessed"] is False and result["completed_epochs"] > 0, "Unexpected scientific state")
        recovery = torch.load(output / "recovery.pt", map_location="cpu", weights_only=False)
        selected = torch.load(output / "selected.pt", map_location="cpu", weights_only=False)
        exported = torch.load(output / "selected_VALID.pt", map_location="cpu", weights_only=False)
        require(recovery["job_hash"] == selected["job_hash"] == exported["job_hash"] == result["job_hash"]
                == tasks[fit_id]["runtime_job_hash"], "Checkpoint job identity mismatch")
        require(recovery["runtime_binding"] == selected["runtime_binding"] == result["runtime_binding"], "Runtime binding mismatch")
        require(exported["selected_checkpoint"] == pin(output / "selected.pt")
                and exported["role"] == "VALID" and exported["TEST_accessed"] is False
                and exported["selected_epoch"] == result["selected_epoch"], "Selected export mismatch")
        require(bool(torch.isfinite(exported["raw_logits"]).all()), "Existing VALID logits nonfinite")
        require(recovery["optimizer_calls"] == result["optimizer_calls"], "Recovery update ledger mismatch")
        descriptor = Path(tasks[fit_id]["descriptor"]["path"])
        require(pin(descriptor) == tasks[fit_id]["descriptor"], "Frozen descriptor changed")
        job = json.loads(descriptor.read_text())
        curve = output / "CURVE.jsonl"
        with curve.open("rb") as stream:
            stream.seek(max(0, curve.stat().st_size - 1000000))
            last = json.loads(stream.read().splitlines()[-1])
        require(last["selection"]["stop"] is True and last["selection"]["stop_reason"] == "patience"
                and last["train"]["epoch"] == result["completed_epochs"], "Training was not already stopped by patience")
        preset_path = Path(tasks[fit_id]["preset"]["path"])
        require(pin(preset_path) == tasks[fit_id]["preset"], "Reporting preset changed")
        preset = json.loads(preset_path.read_text())
        spec = preset["fit_specs"][fit_id]
        witness = composed_authority.runtime_witness(job, spec, result["runtime_binding"])
        preflight[fit_id] = {"job_hash": witness["job_hash"], "authority_sha256": witness["authority_sha256"],
                             "dependency_count": len(result["runtime_binding"]["dependency_sha256"])}
        repairs[fit_id] = {
            "failed_attempt_result": state["attempt_result"], "failed_native_result": state["native_result"],
            "descriptor": pin(descriptor), "output": str(output),
            "artifacts": {name: pin(output / name) for name in (
                "RESULT.json", "selected.pt", "recovery.pt", "selected_VALID.pt", "PROGRESS.json", "CURVE.jsonl")},
            "job_hash": result["job_hash"], "completed_epochs": result["completed_epochs"],
            "selected_epoch": result["selected_epoch"], "optimizer_calls": result["optimizer_calls"],
            "recovery_selector": recovery["selector"], "stop_reason": "patience",
            "repair_scope": "reporting validation and finalization only; zero new training epochs",
        }
        del recovery, selected, exported
    config = {"schema": "explicit-reporting-recovery-v2", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "worker": worker, "project_root": str(project), "plan": owner["plan"], "status_before": status,
              "repairs": repairs, "TEST_accessed": False, "checkpoint_tensor_inspection": "CPU only; existing own artifacts",
              "new_training_epochs_authorized_for_recovery": 0, "repaired_reporting_preflight": preflight}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "config": pin(target), "recoverable_fits": len(repairs),
                      "completed_epochs": {k:v["completed_epochs"] for k,v in repairs.items()},
                      "new_TRAIN_updates": 0, "preflight": preflight}))


if __name__ == "__main__":
    main()
