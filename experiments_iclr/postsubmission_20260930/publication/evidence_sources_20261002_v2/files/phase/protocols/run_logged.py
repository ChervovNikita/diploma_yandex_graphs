"""Record a repository-confined research subprocess, its environment and exit.

This supervisor does not select a model or change its recipe. Commands are
passed as argument lists, never through a shell. Only explicitly listed cache
and temporary environment settings are recorded; credentials are excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parents[1]
REPOSITORY = PHASE.parents[1] if PHASE.parent.name == "experiments_iclr" else PHASE.parent
NAMES = ["GNNM_PHASE_ROOT", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS",
         "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
         "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE", "NUMBA_CACHE_DIR", "WANDB_DIR"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if len(command) < 2:
        raise ValueError("Provide a repository interpreter and phase script")
    # A repository venv interpreter may be a symlink to the installed Python.
    # Authorize its lexical venv entry point; no interpreter is installed here.
    executable, script = Path(os.path.abspath(command[0])), Path(command[1]).resolve()
    if REPOSITORY not in executable.parents or PHASE not in script.parents:
        raise ValueError("Interpreter/script must belong to the authorized repository")
    output = args.output.resolve()
    if PHASE not in output.parents or output.exists():
        raise ValueError("Supervisor output must be new and inside phase")
    values = {name: os.environ.get(name) for name in NAMES}
    if any(not value or not Path(value).resolve().is_relative_to(PHASE) for value in values.values()):
        raise ValueError("Temporary/cache roots must all stay inside the research phase")
    if Path(values["GNNM_PHASE_ROOT"]).resolve() != PHASE or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("Require the verified repository environment wrapper")
    output.mkdir(parents=True)
    env = {"repository": str(REPOSITORY), "phase": str(PHASE), "cache_and_temp": values,
           "python_bytecode_disabled": True, "python_supervisor": sys.version,
           "platform": platform.platform(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "supervisor_sha256": sha(Path(__file__)), "script_sha256": sha(script),
           "wrapper_sha256": sha(PHASE / "protocols/repo_env.sh"),
           "capture_UTC": datetime.now(timezone.utc).isoformat()}
    write(output / "environment.json", env)
    write(output / "command.json", {"argv": command, "cwd": str(Path.cwd()), "shell": False})
    start = time.perf_counter()
    with (output / "stdout_stderr.log").open("x") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
    result = {"exit_code": completed.returncode, "seconds": time.perf_counter() - start,
              "environment_sha256": sha(output / "environment.json"),
              "command_sha256": sha(output / "command.json"), "log_sha256": sha(output / "stdout_stderr.log"),
              "completion_UTC": datetime.now(timezone.utc).isoformat()}
    write(output / "completion.json", result)
    print(json.dumps({"supervisor_output": str(output), **result}), flush=True)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
