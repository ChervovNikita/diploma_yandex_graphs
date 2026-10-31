"""Read queue state, owned process identities and storage without training."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

HOSTS = {
    "allocation": ("anogena-2-0", ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"],
        "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs", ["allocation_gpu0"]),
    "server51": ("olig", ["GPU-9c6c20fa-a620-479d-f72f-163ac74deef2",
        "GPU-30283973-140d-9e7b-e780-0fcc487dec54"],
        "/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git",
        ["server51_gpu0", "server51_gpu1"]),
}


def read(path):
    return json.loads(Path(path).read_text())


def process(identity):
    if not identity:
        return {"same_process": False}
    proc = Path("/proc") / str(identity["pid"])
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        same = (int(fields[19]) == identity["start_ticks"]
                and proc.stat().st_uid == identity["uid"]
                and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot_id"])
        value = {"pid": identity["pid"], "same_process": same, "state": fields[0]}
        if same:
            value["title"] = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        return value
    except FileNotFoundError:
        return {"pid": identity["pid"], "same_process": False}


def inspect_worker(expansion, worker):
    execution = expansion / "campaign_execution_20261016_v7"
    plan = read(execution / ("PLAN-" + worker + ".json"))
    root = execution / "runs" / worker
    status = read(root / "STATUS.json")
    rows, counts = [], Counter()
    retained = 0
    for task in plan["tasks"]:
        state_path = root / "full" / hashlib.sha256(task["id"].encode()).hexdigest() / "STATE.json"
        state = read(state_path) if state_path.exists() else {"status": "pending"}
        counts[state["status"]] += 1
        if state["status"] not in ("running", "failed"):
            continue
        row = {"fit_id": task["id"], "status": state["status"], "attempt": state.get("attempt")}
        directory = (Path(state["directory"]) if "directory" in state else
                     Path(state["attempt_result"]["path"]).parent)
        child = directory / "CHILD.json"
        if child.exists():
            row["process"] = process(read(child).get("identity"))
        output = Path(state["output"])
        progress = output / "PROGRESS.json"
        if progress.exists():
            value = read(progress)
            row.update(epoch=value.get("train", {}).get("epoch"),
                       optimizer_calls=value.get("optimizer_calls"),
                       progress_mtime_utc=datetime.datetime.fromtimestamp(
                           progress.stat().st_mtime, datetime.timezone.utc).isoformat())
        if state["status"] == "failed":
            row["failure"] = state.get("failure")
            native = state.get("native_result")
            if native and Path(native["path"]).is_file():
                value = read(native["path"])
                failure = value.get("failure")
                row["native_failure"] = failure.get("message") if isinstance(failure, dict) else failure
        rows.append(row)
    # Account only for our queue outputs; do not scan other users' files.
    size = subprocess.check_output(["du", "-s", "-B1", str(root)], text=True)
    retained = int(size.split()[0])
    session = "gnnm-allfit-" + worker.replace("_", "-") + "-20261016"
    panes = subprocess.run(["tmux", "list-panes", "-t", session, "-F",
        "#{pane_id} #{pane_dead} #{pane_pid} #{pane_current_command}"], capture_output=True, text=True)
    return {"worker": worker, "phase": status["phase"], "reported_counts": status["counts"],
            "observed_counts": dict(counts), "updated": status["updated"],
            "owner": process(status.get("owner")), "active_or_failed": rows,
            "queue_retained_bytes": retained, "tmux": panes.stdout.strip(),
            "error": status.get("error"), "source": status.get("actual_scheduler_producer", {}).get(
                "actual_outer_source_manifest"),
            "counts_are_nonatomic_snapshot": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=HOSTS, required=True)
    args = parser.parse_args()
    hostname, uuids, project, workers = HOSTS[args.host]
    assert socket.gethostname() == hostname
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == uuids
    expansion = Path(project) / "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "host": args.host,
              "free_disk_bytes": shutil.disk_usage(project).free,
              "gpu": subprocess.check_output(["nvidia-smi",
                  "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                  "--format=csv,noheader"], text=True).splitlines(),
              "workers": [inspect_worker(expansion, worker) for worker in workers],
              "TEST_accessed": False, "training_or_process_mutations": False}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
