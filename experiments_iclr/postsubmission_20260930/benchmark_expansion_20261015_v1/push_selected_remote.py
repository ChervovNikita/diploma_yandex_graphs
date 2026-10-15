"""Interactive named-key push using an explicit existing own commit receipt."""
import argparse
import shlex
import subprocess
from remote_control import SSH, GUARD, REMOTE_ROOT, REMOTE_PHASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-receipt", required=True)
    parser.add_argument("--push-receipt", required=True)
    args = parser.parse_args()
    prefix = "benchmark_expansion_20261015_v1/"
    if any("/" in name or not name.endswith(".json") for name in (args.commit_receipt, args.push_receipt)):
        raise ValueError("Use confined new receipt names")
    script = REMOTE_PHASE+"/reference_wrapper_redesign_20261013_v1/smoke_20261014/push_smokes_allocation.py"
    code = GUARD+"import os,subprocess; os.chdir("+repr(REMOTE_ROOT)+"); " \
        +"os.environ['GNNM_SSH_DESTINATION']='anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'; " \
        +"subprocess.run("+repr([REMOTE_ROOT+"/.venv/bin/python", "-B", script,
                               "--commit-receipt", prefix+args.commit_receipt,
                               "--push-receipt", prefix+args.push_receipt])+",check=True)"
    return subprocess.call(SSH+["python3 -B -c "+shlex.quote(code)])


if __name__ == "__main__":
    raise SystemExit(main())
