"""Resume an explicitly audited stopped queue in its existing dead tmux pane."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time

from repair_native import guard, pin, require, source
from audit_recovery import PLAN_SHAS

HERE = Path(__file__).resolve().parent


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    repair_source = source(args.source_sha256)
    config_path = Path(args.config).resolve()
    require(pin(config_path)["sha256"] == args.config_sha256, "Audited recovery config changed")
    config = json.loads(config_path.read_text())
    worker = config["worker"]
    require(worker.split("/")[0] == ("allocation" if socket.gethostname() == "anogena-2-0" else "server51"), "Wrong host for worker")
    project = Path(config["project_root"]).resolve()
    require(config_path.is_relative_to(project), "Recovery authority escapes project")
    plan_path = Path(config["plan"]["path"])
    require(pin(plan_path) == config["plan"] and pin(plan_path)["sha256"] == PLAN_SHAS[worker], "Frozen plan changed")
    plan = json.loads(plan_path.read_text())
    expansion = project / "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
    root = Path(plan["state_root"])
    require(root == expansion / "campaign_execution_20261016_v7/runs" / worker.replace("/", "_"), "Unexpected queue root")
    status = json.loads((root / "STATUS.json").read_text())
    require(status["phase"] == "full_failed" and status["active_fit_ids"] == [], "Queue is no longer fully stopped")
    require(set(status["failed_fit_ids"]) == set(config["repairs"]), "Failed job set changed after audit")
    require(pin(Path(plan["global_smoke_pass"]))["sha256"] ==
            "065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4", "Global smoke release changed")
    with (root / "SUPERVISOR.lock").open("r+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    session = "gnnm-allfit-" + worker.replace("/", "-") + "-20261016"
    panes = subprocess.check_output(["tmux", "list-panes", "-t", session,
                                     "-F", "#{pane_id}\t#{pane_dead}"], text=True).splitlines()
    require(len(panes) == 1 and panes[0].split("\t")[1] == "1", "Only an existing dead tmux pane may resume")
    pane_id = panes[0].split("\t")[0]
    journal = config_path.parent / ("activation-" + str(time.time_ns()))
    journal.mkdir()
    for name in ("STATUS.json", "OWNER.json", "TMUX_LAUNCH.json"):
        path = root / name
        if path.exists():
            shutil.copy2(path, journal / name)
    # Use the same interpreter as the frozen execution worker.
    python = str(project / ".venv/bin/python")
    argv = [python, "-B", "-u", str(HERE / "recovery_worker.py"),
            "--worker", worker, "--plan", str(plan_path), "--plan-sha256", PLAN_SHAS[worker],
            "--source-sha256", repair_source["sha256"],
            "--process-title-dependency-dir", str(expansion / "campaign_process_title_dependency_20261021_v1"),
            "--recovery-config", str(config_path), "--recovery-config-sha256", args.config_sha256]
    command = "exec " + shlex.join(argv) + " >> " + shlex.quote(str(journal / "owner.log")) + " 2>&1"
    receipt = {"schema": "reporting-recovery-activation-v2", "argv": argv, "shell_command": command,
               "source": repair_source, "recovery_config": pin(config_path),
               "previous_status": pin(journal / "STATUS.json"), "pane": pane_id,
               "no_processes_signalled": True, "existing_global_release_preserved": True}
    (journal / "ACTIVATION_REQUEST.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # No -k: tmux refuses to replace a pane that became live meanwhile.
    subprocess.check_call(["tmux", "respawn-pane", "-t", pane_id, "-c", str(project), command])
    (journal / "ACTIVATION_DISPATCHED.json").write_text(json.dumps({"utc_epoch": time.time(), "pane": pane_id}) + "\n")
    print(json.dumps({"status": "DISPATCHED", "journal": str(journal), "pane": pane_id,
                      "maximum_jobs_per_gpu": 2, "explicit_reporting_recoveries": len(config["repairs"])}))


if __name__ == "__main__":
    main()
