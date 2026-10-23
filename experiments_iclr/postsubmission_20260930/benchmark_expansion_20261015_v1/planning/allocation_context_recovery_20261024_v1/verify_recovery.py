"""Audit completed explicit finalization without any model execution."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    config_path = Path(__file__).with_name("RECOVERY_CONFIG.json")
    assert pin(config_path)["sha256"] == "89d53976c75f306257a235e3fd98e3c9e811fda010acdff466168bf50a04039d"
    config = json.loads(config_path.read_text())
    plan = json.loads(Path(config["plan"]["path"]).read_text())
    root = Path(plan["state_root"])
    states = {}
    for fit_id in config["repairs"]:
        states[fit_id] = json.loads((root / "full" / hashlib.sha256(fit_id.encode()).hexdigest() / "STATE.json").read_text())
    if any(state["status"] != "complete" for state in states.values()):
        print(json.dumps({"status": "WAIT", "states": {k:v["status"] for k,v in states.items()}}))
        return
    import torch
    rows = {}
    for fit_id, state in states.items():
        before = config["repairs"][fit_id]
        result = json.loads(Path(state["native_result"]["path"]).read_text())
        assert result["finalization_only_resume"] is True and result["training_epochs_this_invocation"] == 0
        assert result["completed_epochs"] == before["completed_epochs"]
        assert result["selected_epoch"] == before["selected_epoch"]
        assert result["optimizer_calls"] == before["optimizer_calls"]
        assert all(x == 0 for values in result["optimizer_calls_this_invocation"].values() for x in values)
        output = Path(state["output"])
        assert pin(output / "selected.pt") == before["artifacts"]["selected.pt"]
        assert pin(output / "recovery.pt") == before["artifacts"]["recovery.pt"]
        attempt = Path(state["attempt_result"]["path"]).parent
        preserved = attempt / "preserved_before_reporting_resume/selected_VALID.pt"
        old = torch.load(preserved, map_location="cpu", weights_only=False)
        new = torch.load(output / "selected_VALID.pt", map_location="cpu", weights_only=False)
        comparisons = {}
        for name in old:
            if torch.is_tensor(old[name]):
                assert torch.is_tensor(new[name]) and old[name].shape == new[name].shape
                row = {"exact_equal": bool(torch.equal(old[name], new[name]))}
                if old[name].is_floating_point():
                    row["max_abs_difference"] = float((old[name] - new[name]).abs().max()) if old[name].numel() else 0.0
                comparisons[name] = row
        rows[fit_id] = {"completed_epochs": result["completed_epochs"], "selected_epoch": result["selected_epoch"],
                        "new_epochs": 0, "new_optimizer_updates": 0, "checkpoint_bytes_unchanged": True,
                        "tensor_comparisons": comparisons, "result": state["native_result"],
                        "preserved_export": pin(preserved), "final_export": pin(output / "selected_VALID.pt")}
    report = {"status": "PASS", "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(), "TEST_accessed": False,
              "new_training_epochs": 0, "rows": rows}
    target = Path(__file__).with_name("FINALIZATION_AUDIT.json")
    assert not target.exists()
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
