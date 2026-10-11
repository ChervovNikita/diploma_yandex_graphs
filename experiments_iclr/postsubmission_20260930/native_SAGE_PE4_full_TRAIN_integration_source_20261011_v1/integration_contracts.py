"""Root metadata admission and finite budgets before numerical imports."""
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HOST = "anogena-2-0"
GPU_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
SOURCE_KEYS = {"runner", "contracts", "qualifier", "F_owner", "PE_adapter", "PE_packed",
               "PE_protocol", "common_routes", "factors", "models", "run_base", "run_common",
               "sage_conv", "mean_aggregation", "scatter"}
FIXED = {"stage": "F", "K": None, "k_selection_receipt": None, "depth": 2,
         "hidden": 128, "dropout": .2, "learning_rate": .001, "max_updates": 1000,
         "patience_updates": 300, "member_seed_stride": 1000003,
         "factor_seed_offset": 2000003, "dropout_seed_offset": 3000007,
         "block_seed_offset": 4000037, "seeds": [7301, 7403, 7507]}


def require_route():
    if socket.gethostname() != HOST:
        raise ValueError("Excluded route: root must verify the literal authorized allocation")
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True, timeout=5)
    if query.stdout.strip().splitlines() != [GPU_UUID]:
        raise ValueError("Excluded route: sole authorized GPU identity differs")


def receipt(item):
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
        raise ValueError("Exact path/bytes/sha256 receipt required")
    p = Path(item["path"]).resolve()
    if type(item["bytes"]) is not int or not p.is_file() or p.stat().st_size != item["bytes"]:
        raise ValueError("Bound source/metadata/role size differs")
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("Bound source/metadata/role hash differs")
    return p


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def owner_alive(owner):
    if (owner.get("finite_pending_call_watchdog") is not True
            or owner.get("child_pid") != os.getpid() or owner.get("supervisor_pid") != os.getppid()
            or owner.get("child_pgid") != os.getpgid(0)
            or type(owner.get("watchdog_pid")) is not int or owner["watchdog_pid"] <= 0
            or owner["watchdog_pid"] == os.getpid()):
        raise ValueError("Current child and live external finite owner/watchdog binding required")
    os.kill(owner["supervisor_pid"], 0)
    os.kill(owner["watchdog_pid"], 0)


def admit(release_path):
    require_route()  # No project metadata is read on an excluded route.
    release = json.loads(Path(release_path).read_text())
    if release.get("enabled") is not True or release.get("mode") not in ("qualify", "fit"):
        raise ValueError("Source integration is disabled without root action admission")
    mode = release["mode"]
    cfg_path = receipt(release.get("config")); cfg = json.loads(cfg_path.read_text())
    if any(cfg.get(k) != v for k, v in FIXED.items()):
        raise ValueError("One unchanged F recipe and fixed three-seed roster required")
    if not isinstance(cfg.get("device"), str) or not cfg["device"].startswith("cuda"):
        raise ValueError("Actual authorized CUDA runtime required")
    limits = release.get("limits")
    if (not isinstance(limits, dict) or set(limits) != {"wall_seconds", "cuda_bytes", "rss_bytes"}
            or any(type(v) is not int or v <= 0 for v in limits.values())
            or limits["wall_seconds"] > (300 if mode == "qualify" else 86400)):
        raise ValueError("Finite root wall/CUDA/RSS limits required")
    source = release.get("sources")
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS:
        raise ValueError("Exact PE/F/native/current PyG source roster required")
    paths = {k: receipt(v) for k, v in source.items()}
    here = Path(__file__).resolve().parent
    expected = {"runner": here / "run.py", "contracts": here / "integration_contracts.py",
                "qualifier": here / "qualify_full_train.py",
                "common_routes": Path(cfg["common_routes"]).resolve(), "factors": Path(cfg["factors"]).resolve()}
    expected.update({k: (Path(cfg["native_repo"]) / (k + ".py")).resolve() for k in ("models", "run_base", "run_common")})
    if any(paths[k] != p for k, p in expected.items()):
        raise ValueError("Source/config mapping differs")
    if paths["PE_adapter"].parent != paths["PE_packed"].parent or paths["PE_protocol"] != paths["PE_adapter"].with_name("PROTOCOL.json"):
        raise ValueError("Use the existing sealed PE source packet together")
    roles = release.get("roles")
    keys = {"train"} if mode == "qualify" else {"train", "valid"}
    if not isinstance(roles, dict) or set(roles) != keys:
        raise ValueError("Qualifier accepts TRAIN only; fit accepts exact TRAIN/VALID roles")
    role_paths = {k: receipt(v) for k, v in roles.items()}
    if any(role_paths[k] != Path(cfg[k + "_npz"]).resolve() for k in roles):
        raise ValueError("Safe role path differs")
    owner = json.loads(receipt(release.get("external_owner")).read_text())
    owner_alive(owner)
    if owner.get("limits") != limits or owner.get("mode") != mode:
        raise ValueError("External owner must enforce these exact finite action limits")
    admission = json.loads(receipt(release.get("host_runtime_admission")).read_text())
    wanted = {"hostname": HOST, "gpu_uuid": GPU_UUID, "mode": mode,
              "config_sha256": release["config"]["sha256"],
              "source_sha256": {k: v["sha256"] for k, v in source.items()},
              "safe_role_sha256": {k: v["sha256"] for k, v in roles.items()}}
    if any(admission.get(k) != v for k, v in wanted.items()) or admission.get("status") != "HOST_GPU_SOURCE_ROLES_RUNTIME_ADMITTED":
        raise ValueError("Root host/GPU/runtime/source/role admission differs")
    if (not isinstance(admission.get("runtime_fingerprint_sha256"), str)
            or len(admission["runtime_fingerprint_sha256"]) != 64
            or not all(isinstance(admission.get(k), str) and admission[k] for k in ("torch_version", "pyg_version"))):
        raise ValueError("Explicit actual runtime fingerprint and versions required")
    if mode == "fit":
        qualification = json.loads(receipt(release.get("qualification")).read_text())
        if (qualification.get("status") != "PASS" or qualification.get("source_sha256") != wanted["source_sha256"]
                or qualification.get("safe_TRAIN_sha256") != roles["train"]["sha256"]
                or qualification.get("runtime_fingerprint_sha256") != admission["runtime_fingerprint_sha256"]
                or qualification.get("VALID_payload_deserializations") != 0):
            raise ValueError("Exact-source/same-runtime/full-TRAIN qualification PASS required")
        closure = json.loads(receipt(release.get("endpoint_native_closure")).read_text())
        if (closure.get("scope") != "endpoint_VALUE_native_closure" or closure.get("native_fits_complete") is not True
                or closure.get("banks") != 15 or closure.get("optimizer_bundles") != 24
                or closure.get("native_body_records") != 24):
            raise ValueError("Root schedules acquisition only after current endpoint native closure")
        if release.get("historical_cross_runtime_score_reuse") is not False:
            raise ValueError("Fresh admitted same-role anchors; no historical cross-runtime score reuse")
    return release, cfg, paths, admission, owner


class Budget:
    def __init__(self, limits, owner):
        self.limits, self.owner, self.started = limits, owner, time.perf_counter()
        self.peak_allocated, self.peak_reserved = 0, 0

    def check(self, torch=None, device=None):
        owner_alive(self.owner)
        if time.perf_counter() - self.started > self.limits["wall_seconds"]:
            raise RuntimeError("Root wall budget exceeded")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > self.limits["rss_bytes"]:
            raise RuntimeError("Process-lifetime RSS budget exceeded")
        if torch is not None and device.type == "cuda":
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(device))
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(device))
            if self.peak_allocated > self.limits["cuda_bytes"]:
                raise RuntimeError("CUDA allocated budget exceeded")

    def costs(self, torch, device):
        self.check(torch, device)
        return {"elapsed_seconds": time.perf_counter() - self.started,
                "process_peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                "cuda_peak_allocated_bytes": self.peak_allocated,
                "cuda_peak_reserved_bytes": self.peak_reserved}
