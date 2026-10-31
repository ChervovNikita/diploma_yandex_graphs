"""Confinement, fixed recipe and explicit engineering phase authorization."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SCHEMA = "gnnm-complete-graph-propagation-engineering-v1"
ALLOWED_SSH_ACCOUNT = "anogena-2.ai0001053-01174"
IMPL_MANIFEST_SHA256 = "b9f76667aaf2479ee5cd8816877ebc650c99380189c4869545519e753fdf63b9"
TASKS = {"Reddit2": {"nodes": 232965, "features": 602, "classes": 41, "index": 0},
         "ogbn-products": {"nodes": 2449029, "features": 100, "classes": 47, "index": 1}}
MEMORY_BYTES = 60 * 2**30
DISK_BYTES = 200 * 2**30
TOTAL_GPU_SECONDS = 8 * 3600
FORECAST_GPU_SECONDS = 6.5 * 3600
CPU_SECONDS = 16 * 3600
AMENDMENT_SHA256 = "d30f2d05c1b66dc621d5cb10f86689f1365e49b5f84380472b928e17c04ec055"
CACHE_NAMES = (
    "GNNM_PHASE_ROOT", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
    "CONDA_PKGS_DIRS", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "MPLCONFIGDIR", "HF_HOME", "HF_DATASETS_CACHE",
    "NUMBA_CACHE_DIR", "WANDB_DIR",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(path, root, *, existing=False):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Path must stay strictly inside the declared project folder")
    if existing and not path.is_file():
        raise FileNotFoundError(path)
    return path


def output_path(path):
    return within(path, ROOT)


def verify_own_seal():
    path = ROOT / "SOURCE_MANIFEST_local_run01.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "source_only_sealed":
        raise RuntimeError("Engineering source seal missing")
    for row in manifest["files"]:
        file = output_path(ROOT / row["relative_path"])
        if file.stat().st_size != row["bytes"] or sha256(file) != row["sha256"]:
            raise RuntimeError("Engineering sealed source changed")
    return sha256(path)


def verify_impl():
    path = PROJECT / "propagation_cost_impl_v1/SOURCE_MANIFEST_local_run01.json"
    if sha256(path) != IMPL_MANIFEST_SHA256:
        raise RuntimeError("Sealed mechanics packet manifest mismatch")
    from propagation_cost_impl_v1.contract import check_seal
    if check_seal() != IMPL_MANIFEST_SHA256:
        raise RuntimeError("Sealed mechanics packet changed")
    return IMPL_MANIFEST_SHA256


class WallBudget:
    def __init__(self, seconds, deadline_utc=None):
        self.start = time.monotonic()
        self.seconds = float(seconds)
        self.deadline = None if deadline_utc is None else datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if self.deadline is not None and self.deadline.tzinfo is None:
            raise ValueError("Deadline must have an explicit timezone")

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise RuntimeError("Engineering budget exhausted; incomplete gate")
        if self.deadline is not None and datetime.now(timezone.utc) >= self.deadline:
            raise RuntimeError("Engineering phase deadline reached")

    def elapsed(self):
        return time.monotonic() - self.start


def live_context(supervisor, entrypoint):
    """Repository supervisor/route/cache checks before any third-party import."""
    supervisor = within(supervisor, PROJECT)
    environment_path = within(supervisor / "environment.json", PROJECT, existing=True)
    command_path = within(supervisor / "command.json", PROJECT, existing=True)
    environment, command = json.loads(environment_path.read_text()), json.loads(command_path.read_text())
    route_path = PROJECT / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json"
    route = json.loads(route_path.read_text())
    destination = ALLOWED_SSH_ACCOUNT + "@ssh-sr003-jupyter.ai.cloud.ru"
    if (route.get("ssh_destination") != destination or route.get("visible_gpu_count") != 1
            or route.get("forbidden_destination_reconnect_allowed") is not False
            or os.environ.get("GNNM_SSH_DESTINATION") != destination
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"):
        raise RuntimeError("Incorrect authorized route or cache/GPU environment")
    repo = Path(route["repository"]).resolve()
    entrypoint = Path(entrypoint).resolve()
    if (Path.cwd().resolve() != repo or environment.get("repository") != str(repo)
            or command.get("cwd") != str(repo) or environment.get("phase") != str(PROJECT)
            or environment.get("script_sha256") != sha256(entrypoint)
            or environment.get("supervisor_sha256") != sha256(PROJECT / "protocols/run_logged.py")
            or environment.get("wrapper_sha256") != sha256(PROJECT / "protocols/repo_env.sh")):
        raise RuntimeError("Live supervisor/source identity differs")
    argv = command.get("argv", [])
    if len(argv) < 2 or Path(argv[1]).resolve() != entrypoint:
        raise RuntimeError("Supervisor did not start this entrypoint")
    within(argv[0], repo)
    cached = environment.get("cache_and_temp", {})
    if set(cached) != set(CACHE_NAMES):
        raise RuntimeError("Cache/temp supervisor schema differs")
    for name in CACHE_NAMES:
        current = os.environ.get(name)
        if not current or cached[name] != current:
            raise RuntimeError("Cache/temp value differs: " + name)
        if name == "GNNM_PHASE_ROOT":
            if Path(current).resolve() != PROJECT:
                raise RuntimeError("Phase root differs")
        else:
            within(current, PROJECT)
    return {"command_sha256": sha256(command_path), "environment_sha256": sha256(environment_path),
            "route_sha256": sha256(route_path), "supervisor_relative_to_project": str(supervisor.relative_to(PROJECT)),
            "entrypoint_sha256": sha256(entrypoint), "completed_chain_status": "pending_until_child_exit"}


def load_phase(path, expected_phase):
    path = output_path(path)
    phase = json.loads(path.read_text())
    if phase.get("schema") != SCHEMA or phase.get("phase") != expected_phase:
        raise ValueError("Explicit engineering phase file required")
    if phase.get("engineering_only") is not True or phase.get("scientific_training_authorized") is not False:
        raise ValueError("This source can perform engineering only")
    repo = Path(phase["repo_root"]).resolve()
    if repo != Path.cwd().resolve() or repo not in ROOT.parents:
        raise ValueError("Run from the exact declared repository root")
    if phase.get("authorized_ssh_account") != ALLOWED_SSH_ACCOUNT:
        raise ValueError("Only the explicitly authorized original one-GPU account is allowed")
    output = output_path(repo / phase["output_relative_to_repo"])
    if output.exists():
        raise FileExistsError("Do not overwrite engineering evidence")
    for name, maximum in (("prior_gpu_seconds", TOTAL_GPU_SECONDS),
                          ("profile_gpu_seconds_cap", 3600), ("prior_cpu_seconds", CPU_SECONDS)):
        value = float(phase.get(name, 0))
        if not 0 <= value <= maximum:
            raise ValueError(f"Invalid or excessive {name}")
    if expected_phase == "profile" and (not phase.get("authorized_gpu_uuid")
                                         or phase["profile_gpu_seconds_cap"] <= 0):
        raise ValueError("Authorized UUID and bounded engineering time required")
    if phase["task"] not in TASKS and expected_phase != "forecast":
        raise ValueError("Only the two fixed complete tasks may enter engineering")
    if not 0 <= float(phase["prior_disk_bytes"]) <= DISK_BYTES:
        raise ValueError("Explicit acquisition/failed-artifact prior disk accounting required")
    amendment = PROJECT / "propagation_cost_scout_v1/PILOT_AMENDMENT_ROOT_before_fits_v2.json"
    if sha256(amendment) != AMENDMENT_SHA256:
        raise RuntimeError("Prospective fifth-arm amendment binding differs")
    verify_own_seal()
    verify_impl()
    return phase, output, sha256(path)


def check_gpu(phase):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one authorized CUDA device must be visible")
    torch.cuda.set_device(0)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=index,uuid,name,memory.total",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True,
                           text=True, timeout=10)
    rows = [[field.strip() for field in line.split(",")]
            for line in query.stdout.splitlines() if line.strip()]
    if (len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != "0"
            or rows[0][1] != phase["authorized_gpu_uuid"]
            or rows[0][2] != "NVIDIA A100-SXM4-80GB" or int(rows[0][3]) != 81920):
        raise RuntimeError("Bounded live GPU query differs from the phase authorization")
    if "A100" not in properties.name or properties.total_memory < 70 * 2**30:
        raise RuntimeError("Frozen engineering nomination requires one A100 80GB")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"name": properties.name, "uuid": rows[0][1],
            "total_memory_bytes": properties.total_memory, "torch": torch.__version__,
            "TF32": False, "AMP": False}


def check_peak():
    import torch
    reserved = torch.cuda.max_memory_reserved(0)
    allocated = torch.cuda.max_memory_allocated(0)
    if reserved > MEMORY_BYTES:
        raise RuntimeError("Complete-graph reserved-memory ceiling exceeded")
    return {"peak_reserved_bytes": reserved, "peak_allocated_bytes": allocated}
