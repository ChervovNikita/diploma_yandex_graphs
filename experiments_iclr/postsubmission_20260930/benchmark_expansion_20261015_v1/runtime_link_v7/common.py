"""Source loading, project custody and atomic receipts for bounded pilots."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ALLOCATION_UUID = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
EXCLUDED_DATASETS = {"roman-empire", "amazon-ratings", "minesweeper", "questions", "tolokers"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def receipt(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def inside(path, root):
    path = Path(path)
    path = path if path.is_absolute() else Path(root) / path
    path = path.resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Deliberate project operation outside repository: " + str(path))
    return path


def require_allocation():
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Scientific pilots require the dedicated authorized allocation")
    values = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                     text=True, timeout=10).splitlines()
    if values != [ALLOCATION_UUID]:
        raise ValueError("Wrong allocation GPU route; no pilot source/data may be loaded")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def event(path, value):
    with Path(path).open("a") as handle:
        handle.write(canonical({"UTC": utc(), **value}) + "\n")
        handle.flush()


def load_module(path, root, expected_sha256=None):
    path = inside(path, root)
    pin = receipt(path)
    if expected_sha256 is not None and pin["sha256"] != expected_sha256:
        raise ValueError("Frozen source receipt changed: " + str(path))
    name = "pilot_source_" + pin["sha256"]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name], pin


def seed_all(torch, seed, device="cuda:0"):
    import numpy as np
    if type(seed) is not int or seed < 0:
        raise ValueError("Explicit nonnegative independent initialization seed required")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    if device == "cpu":
        # torch.manual_seed also schedules CUDA seeds; CPU engineering uses
        # only the native CPU generator and never initializes a CUDA context.
        torch.random.default_generator.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def global_rng(torch, device="cuda:0"):
    import numpy as np
    return {"python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [] if device == "cpu" else torch.cuda.get_rng_state_all()}


def restore_rng(torch, value, device="cuda:0"):
    import numpy as np
    if device == "cpu" and value["torch_cuda"]:
        raise ValueError("CPU engineering cannot restore a CUDA RNG trajectory")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    if device != "cpu":
        torch.cuda.set_rng_state_all([v.cpu() for v in value["torch_cuda"]])


def cpu_clone(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_clone(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_clone(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(torch, v) for v in value)
    return copy.deepcopy(value)


def atomic_torch(torch, path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    torch.save(value, temporary)
    temporary.replace(path)
    return receipt(path)


def metadata(session):
    value = getattr(session, "source_metadata", None)
    if value is None:
        value = getattr(session, "metadata", None)
    return value() if callable(value) else value
