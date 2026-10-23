"""Read compact live allocation progress and preserve the exact snapshot."""
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess


def main():
    assert socket.gethostname() == "anogena-2-0"
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]
    phase = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1")
    root = phase / "campaign_execution_20261016_v7/runs/allocation_gpu0"
    status = json.loads((root / "STATUS.json").read_text())
    value = {k:status.get(k) for k in ("phase", "counts", "active_fit_ids", "updated", "error", "failed_fit_ids")}
    value["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    value["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    value["children"] = []
    for fit_id in status["active_fit_ids"]:
        folder = root / "full" / hashlib.sha256(fit_id.encode()).hexdigest()
        state = json.loads((folder / "STATE.json").read_text())
        directory, output = Path(state["directory"]), Path(state["output"])
        item = {"fit_id": fit_id, "attempt": state["attempt"]}
        child = directory / "CHILD.json"
        if child.exists():
            identity = json.loads(child.read_text())["identity"]
            proc = Path("/proc") / str(identity["pid"])
            try:
                fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                same = int(fields[19]) == identity["start_ticks"] and proc.stat().st_uid == identity["uid"]
                same = same and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot_id"]
                item.update(pid=identity["pid"], same_process=same, process_state=fields[0],
                            title=(proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip() if same else None)
            except FileNotFoundError:
                item["same_process"] = False
        progress = output / "PROGRESS.json"
        if progress.exists():
            data = json.loads(progress.read_text())
            item.update(completed_epoch=data["train"]["epoch"], optimizer_calls=data.get("optimizer_calls"))
        log = directory / "native.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0,log.stat().st_size-1600))
                item["log_tail"] = stream.read().decode(errors="replace")
        value["children"].append(item)
    config = json.loads(Path(__file__).with_name("RECOVERY_CONFIG.json").read_text())
    value["reporting_recoveries"] = []
    for fit_id in config["repairs"]:
        state = json.loads((root / "full" / hashlib.sha256(fit_id.encode()).hexdigest() / "STATE.json").read_text())
        item = {"fit_id": fit_id, "status": state["status"], "attempt": state["attempt"]}
        if state["status"] == "complete":
            result = json.loads(Path(state["native_result"]["path"]).read_text())
            item.update({key:result.get(key) for key in ("completed_epochs", "selected_epoch", "training_epochs_this_invocation", "finalization_only_resume", "optimizer_calls_this_invocation", "selected_VALID_export")})
        value["reporting_recoveries"].append(item)
    value["tmux_alive"] = subprocess.run(["tmux", "has-session", "-t", "gnnm-allfit-allocation-gpu0-20261016"], capture_output=True).returncode == 0
    target = Path(__file__).with_name("STATUS-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
