"""Post-audit matched WikiCS resource profile using the primary timing recipe."""
from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric

import tuning as primary
import wikics_matched54 as study


ROOT = Path(__file__).resolve().parent
OUT = study.OUT
ROUNDS = 3
WARMUPS = 20
REPETITIONS = 100
SEED = 0


def gpu_context():
    try:
        return subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    output = OUT / "PROFILE_SELECTED_SEED0.json"
    study.require(not output.exists(), "Profile overwrite refused")
    study.require((OUT / "finalize.exit").is_file() and
                  (OUT / "finalize.exit").read_text().strip() == "0",
                  "Complete training and independent audit required")
    freeze_sha = study.check_freeze()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    lock_path = OUT / "VALIDATION_SELECTION_LOCK.json"
    lock = json.loads(lock_path.read_text())
    study.require(lock["freeze_sha256"] == freeze_sha and len(lock["cells"]) == 54,
                  "Complete validation lock required")
    bundle, descriptor = primary.load_graph("wikics", device, include_test=False)
    records = {arm: {"blocks": []} for arm in study.ARMS}
    report = {
        "protocol": "wikics_matched54_selected_seed0_profile_v1",
        "posthoc": True, "source_sha256": study.sha(Path(__file__)),
        "primary_profile_source_sha256": study.sha(ROOT / "profile_selected_inference.py"),
        "freeze_sha256": freeze_sha, "selection_lock_sha256": study.sha(lock_path),
        "python": platform.python_version(), "torch": torch.__version__,
        "pyg": torch_geometric.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_properties": str(torch.cuda.get_device_properties(device)),
        "cpu_threads": torch.get_num_threads(), "optimization_seed": SEED,
        "rounds": ROUNDS, "warmups_per_block": WARMUPS,
        "repetitions_per_block": REPETITIONS,
        "scope": "full resident graph, eval mode, float32, sequential member forwards, stack and mean raw logits; excludes loading and host transfers",
        "timing": "host wall clock bounded by CUDA synchronization for each forward; median and quartiles across 300 repeats",
        "memory": "peak torch CUDA allocated bytes including resident graph/model, plus increment above idle graph/model; excludes reserved allocator and external process memory",
        "gpu_before": gpu_context(), "graph_descriptor": descriptor,
        "arms": records,
    }
    saved_width = primary.WIDTH
    try:
        for rnd in range(ROUNDS):
            order = list(study.ARMS)
            if rnd == 1:
                order.reverse()
            elif rnd == 2:
                order = order[1:] + order[:1]
            for arm in order:
                study.with_width(arm)
                lr, wd = lock["selections"][arm]["selected_candidate"]
                identity = study.key(arm, lr, wd, SEED)
                cell = study.cell_dir(arm, lr, wd, SEED)
                checkpoint = cell / "checkpoint.pt"
                study.require(study.sha(checkpoint) == lock["cells"][identity]["checkpoint_sha256"],
                              f"Selected checkpoint differs: {identity}")
                primary.seed_all(SEED)
                model, _ = primary.make_model(arm, bundle, device)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                model.load_state_dict(payload["state_dict"], strict=True)
                model.eval()
                del payload
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)

                def predict():
                    return torch.stack(primary.member_logits(model, arm, bundle)).mean(0)

                with torch.inference_mode():
                    for _ in range(WARMUPS):
                        prediction = predict()
                        study.require(bool(torch.isfinite(prediction).all()), "Nonfinite inference")
                        del prediction
                    torch.cuda.synchronize(device)
                    gc.collect()
                    idle = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    prediction = predict()
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_allocated(device)
                    del prediction
                    durations = []
                    for _ in range(REPETITIONS):
                        torch.cuda.synchronize(device)
                        start = time.perf_counter_ns()
                        prediction = predict()
                        torch.cuda.synchronize(device)
                        durations.append((time.perf_counter_ns() - start) / 1e6)
                        del prediction
                rec = records[arm]
                rec.update({"selected_candidate": [lr, wd],
                            "checkpoint_sha256": study.sha(checkpoint),
                            "parameter_count": sum(p.numel() for p in model.parameters()),
                            "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters())})
                study.require(rec["parameter_count"] == study.PARAMETER_COUNTS[arm],
                              f"Parameter count differs: {arm}")
                rec["blocks"].append({"round": rnd, "position": order.index(arm),
                                      "latency_ms": durations, "idle_allocated_bytes": idle,
                                      "peak_allocated_bytes": peak,
                                      "peak_increment_bytes": peak-idle})
                del model
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        primary.WIDTH = saved_width
    for rec in records.values():
        values = [v for block in rec["blocks"] for v in block["latency_ms"]]
        rec["latency_ms_median"] = statistics.median(values)
        rec["latency_ms_q25"], rec["latency_ms_q75"] = np.quantile(values,[.25,.75]).tolist()
        rec["block_latency_medians_ms"] = [statistics.median(b["latency_ms"]) for b in rec["blocks"]]
        rec["peak_allocated_bytes_max"] = max(b["peak_allocated_bytes"] for b in rec["blocks"])
        rec["peak_increment_bytes_max"] = max(b["peak_increment_bytes"] for b in rec["blocks"])
    report["gpu_after"] = gpu_context()
    primary.write_json(output, report)
    print(json.dumps({"profile_file": output.name, "sha256": study.sha(output),
                      "latency_ms_median": {a: r["latency_ms_median"] for a,r in records.items()},
                      "peak_allocated_bytes_max": {a: r["peak_allocated_bytes_max"] for a,r in records.items()}}), flush=True)


if __name__ == "__main__":
    main()
