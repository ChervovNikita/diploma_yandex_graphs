"""Bounded own-process resource observations during the necessary native gate."""
from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time


GIB = 1024 ** 3


def scalar_file(path):
    try:
        value = Path(path).read_text().strip()
        return int(value) if value != "max" else None
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def memory_state():
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemAvailable", "Cached", "MemFree"):
            meminfo[key + "_bytes"] = int(value.split()[0]) * 1024
    cg = Path("/sys/fs/cgroup")
    current, maximum = scalar_file(cg / "memory.current"), scalar_file(cg / "memory.max")
    available = meminfo["MemAvailable_bytes"]
    if current is not None and maximum is not None:
        available = min(available, max(0, maximum - current))
    result = {**meminfo, "cgroup_memory_current_bytes": current, "cgroup_memory_max_bytes": maximum,
              "cgroup_memory_peak_bytes": scalar_file(cg / "memory.peak"),
              "conservative_available_bytes": available, "cgroup_scope": "allocation-wide; not attributable to this process"}
    for file in ("memory.events", "memory.stat"):
        try:
            rows = dict(line.split() for line in (cg / file).read_text().splitlines())
            wanted = rows if file == "memory.events" else {k: rows.get(k) for k in ("anon", "file", "file_mapped", "kernel")}
            result[file.replace(".", "_")] = {k: None if v is None else int(v) for k, v in wanted.items()}
        except (FileNotFoundError, PermissionError):
            result[file.replace(".", "_")] = None
    return result


def gpu_processes():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                                  text=True, timeout=5)
    rows = []
    for line in raw.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append({"pid": int(parts[0]), "used_gpu_memory_bytes": int(parts[1]) * 1024 ** 2})
    return rows


def preflight(device, output, limits):
    memory = memory_state()
    if memory["conservative_available_bytes"] < limits["minimum_conservative_available_bytes"]:
        raise ValueError("Insufficient conservative CPU memory headroom for the frozen gate")
    disk = shutil.disk_usage(Path(output).parent)
    if disk.free < limits["disk_reservation_bytes"] or len(os.sched_getaffinity(0)) < 2:
        raise ValueError("Insufficient disk or two-core CPU availability")
    result = {"memory": memory, "disk_free_bytes": disk.free, "CPU_affinity_count": len(os.sched_getaffinity(0))}
    if device != "cpu":
        peers = gpu_processes()
        if len({r["pid"] for r in peers if r["pid"] != os.getpid()}) >= 2:
            raise ValueError("Two other CUDA processes exist; this gate must not become a third worker")
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()) * 1024 ** 2
        if free < limits["minimum_GPU_free_bytes"]:
            raise ValueError("Insufficient GPU free memory for the root-reviewed conservative reservation")
        result.update(GPU_free_bytes=free, GPU_peer_processes=peers,
                      GPU_reservation_note="Declared root-review envelope, not a measured conditional-J bound; root must account for peer growth")
    return result


class Sampler:
    def __init__(self, output, torch, device):
        self.output, self.torch, self.device = Path(output), torch, device
        self.started, self.phase_name = time.perf_counter(), "construction_preparation"
        self.rows, self.errors = [], []
        self.lock, self.stop_event = threading.RLock(), threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="molecular-gate-resource-sampler")

    def start(self):
        self.sample()
        self.thread.start()

    def phase(self, name):
        self.phase_name = name

    def sample(self):
        with self.lock:
            row = {"elapsed_seconds": time.perf_counter() - self.started, "phase": self.phase_name,
                   "own_peak_RSS_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                   "memory": memory_state()}
            try:
                status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
                row["own_current_RSS_bytes"] = int(status["VmRSS"].split()[0]) * 1024
                if self.device != "cpu":
                    rows = gpu_processes()
                    row["own_GPU_process_bytes"] = max((r["used_gpu_memory_bytes"] for r in rows if r["pid"] == os.getpid()), default=0)
                if self.device != "cpu" and self.torch.cuda.is_initialized():
                    row.update(cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                               cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
            except Exception as error:
                self.errors.append({"elapsed_seconds": row["elapsed_seconds"], "phase": self.phase_name, "error": str(error)})
            self.rows.append(row)
            with (self.output / "RESOURCE_SAMPLES.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _loop(self):
        while not self.stop_event.wait(1):
            try:
                self.sample()
            except Exception as error:
                self.errors.append({"sampler_error": str(error)})

    def summary(self):
        with self.lock:
            keys = ("own_peak_RSS_bytes", "own_current_RSS_bytes", "own_GPU_process_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
            return {"samples": len(self.rows), "sample_interval_seconds": 1,
                    "observed_maxima": {key: max((r[key] for r in self.rows if key in r), default=None) for key in keys},
                    "errors": list(self.errors), "cgroup_before": self.rows[0]["memory"] if self.rows else None,
                    "cgroup_latest": self.rows[-1]["memory"] if self.rows else None,
                    "limits": "Sampled process VRAM/current RSS may miss a peak. cgroup/cache include peers. Allocator peaks are recorded separately across invocation resets. This one method/device is not a reciprocal-pair or all-preset resource envelope."}

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        self.sample()
        if self.thread.is_alive():
            raise RuntimeError("Resource sampler failed to join within its declared subprocess timeout")
