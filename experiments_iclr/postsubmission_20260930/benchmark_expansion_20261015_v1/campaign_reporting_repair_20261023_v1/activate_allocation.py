"""Resume only the stopped allocation tmux pane with the audited repair."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from repair_native import guard, pin, require, source

HERE = Path(__file__).resolve().parent
PROJECT = Path("/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs")
EXPANSION = PROJECT / "experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1"
CONFIG = EXPANSION / "planning/allocation_recovery_20261023_v1/RECOVERY_CONFIG.json"
CONFIG_SHA = "f0ed9d1f5c5f5ad4092610f0e480f4b81be5d08063b9c6242fc0de8211b7999f"
PLAN = EXPANSION / "campaign_execution_20261016_v7/PLAN-allocation_gpu0.json"
PLAN_SHA = "19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad"
SESSION = "gnnm-allfit-allocation-gpu0-20261016"


def main():
    guard()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    repair_source = source(args.source_sha256)
    require(pin(CONFIG)["sha256"] == CONFIG_SHA, "Audited recovery config changed")
    require(pin(PLAN)["sha256"] == PLAN_SHA, "Frozen allocation plan changed")
    plan = json.loads(PLAN.read_text())
    root = Path(plan["state_root"])
    require(root == EXPANSION / "campaign_execution_20261016_v7/runs/allocation_gpu0", "Unexpected queue root")
    status = json.loads((root / "STATUS.json").read_text())
    require(status["phase"] == "full_failed" and status["active_fit_ids"] == [], "Queue is no longer the stopped failed queue")
    require(pin(Path(plan["global_smoke_pass"]))["sha256"] ==
            "065b2574147b257ab7aa47163eaaa475f7b8a0f6df1f4a6378b4a43f30d8c4f4", "Global smoke release changed")
    # The owner has exited; never signal it or displace another live owner.
    with (root / "SUPERVISOR.lock").open("r+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    panes = subprocess.check_output([
        "tmux", "list-panes", "-t", SESSION, "-F", "#{pane_id}\t#{pane_dead}"
    ], text=True).splitlines()
    require(len(panes) == 1 and panes[0].split("\t")[1] == "1", "Only the existing dead tmux pane may resume")
    pane_id = panes[0].split("\t")[0]
    journal = EXPANSION / "planning/allocation_recovery_20261023_v1" / ("activation-" + str(time.time_ns()))
    journal.mkdir()
    for name in ("STATUS.json", "OWNER.json", "TMUX_LAUNCH.json"):
        path = root / name
        if path.exists():
            shutil.copy2(path, journal / name)
    argv = [str(PROJECT / ".venv/bin/python"), "-B", "-u", str(HERE / "recovery_worker.py"),
            "--worker", "allocation/gpu0", "--plan", str(PLAN), "--plan-sha256", PLAN_SHA,
            "--source-sha256", repair_source["sha256"],
            "--process-title-dependency-dir", str(EXPANSION / "campaign_process_title_dependency_20261021_v1"),
            "--recovery-config", str(CONFIG), "--recovery-config-sha256", CONFIG_SHA]
    command = "exec " + shlex.join(argv) + " >> " + shlex.quote(str(journal / "owner.log")) + " 2>&1"
    receipt = {"schema": "allocation-reporting-recovery-activation-v1", "argv": argv,
               "shell_command": command, "source": repair_source, "recovery_config": pin(CONFIG),
               "previous_status": pin(journal / "STATUS.json"), "pane": pane_id,
               "no_processes_signalled": True, "existing_global_release_preserved": True}
    (journal / "ACTIVATION_REQUEST.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # Without -k tmux refuses to replace a pane that became live meanwhile.
    subprocess.check_call(["tmux", "respawn-pane", "-t", pane_id, "-c", str(PROJECT), command])
    (journal / "ACTIVATION_DISPATCHED.json").write_text(json.dumps({"utc_epoch": time.time(), "pane": pane_id}) + "\n")
    print(json.dumps({"status": "DISPATCHED", "journal": str(journal), "pane": pane_id,
                      "maximum_jobs_per_gpu": 2, "explicit_reporting_recoveries": 2}))


if __name__ == "__main__":
    main()
