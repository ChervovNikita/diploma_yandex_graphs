#!/usr/bin/env python3
"""Root-gated, bounded engineering CLI. No fits or stream generation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from qualification_common import (HERE, STAGES, require, utc, file_sha, atomic_json,
                                  preflight, assemble, final_custody, load_module)


def worker(context):
    output = context["output"]
    receipt = {"schema": "pooled-native-batch-qualification-v1", "UTC": utc(),
               "identity": context["identity"], "release_sha256": context["release_sha256"],
               "invocation": context["invocation"], "stage": context["stage"], "status": "IN_PROGRESS",
               "fits": 0, "VALID_evaluations": 0, "TEST_file_opened": False, "state_donor": False,
               "data_files_opened": [], "old_checkpoint_opened": False}
    atomic_json(output / "RECEIPT.json", receipt)
    started = time.monotonic()
    rt = resources = None
    try:
        atomic_json(output / "PROGRESS.json", {"UTC": utc(), "arm": None, "step": "runtime_source_admission"})
        preimport = {"Torch_absent": "torch" not in sys.modules,
                     "CUBLAS_WORKSPACE_CONFIG_before": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        rt, mods = assemble(context)
        receipt.update(runtime_identity=rt["identity"]["runtime"],
                       runtime_profile_transition=rt["runtime_profile_transition"], preimport=preimport,
                       native_source={"repository": "GraphPKU/NeuralCommonNeighbor",
                                      "commit": "11d597013750da17ce7468e344bec756a7af39a4",
                                      "private_pins_sha256": file_sha(context["paths"]["native_private_pins"])})
        checks = load_module("_pooled_native_checks", HERE / "native_checks.py")
        resources = checks.Resources(context)
        if context["stage"] == "native":
            result = checks.native(context, rt, mods, resources)
        else:
            native_prior = json.loads(Path(context["release"]["native_qualification"]["path"]).read_text())
            require(native_prior["runtime_identity"] == rt["identity"]["runtime"], "Native qualification runtime differs")
            # Enforce the separately passed provider's actual complete epoch
            # equality receipts before reading any TRAIN arrays.
            sys.modules["replay_common"].qualification_pins(
                {"stage": "generate_epoch", "release": context["release"],
                 "provider_manifest_sha256": file_sha(context["paths"]["provider_manifest"])}, rt["identity"])
            atomic_json(output / "PROGRESS.json", {"UTC": utc(), "arm": None, "step": "admitted_TRAIN_raw_reads"})
            receipt["data_files_opened"] = ["split/time/train.pt", "raw/edge.csv.gz", "raw/node-feat.csv.gz"]
            data = resources.call("complete_TRAIN_integer_load", lambda: sys.modules["replay_runtime"].load_train_only(rt))
            x = resources.call("complete_raw_feature_load", lambda: load_features(context, rt))
            release = context["release"]
            replay = resources.call("authenticated_actual_epoch_arrays", lambda: mods["replay_provider"].EpochReplay(
                            rt, data, release["epoch_root"], release["epoch_receipt"]["sha256"]))
            require(replay.receipt["master_seed"] == 2026100401 and replay.receipt["epoch"] == 1, "Different replay choice")
            receipt.update(epoch_receipt_sha256=release["epoch_receipt"]["sha256"], TRAIN_identity=data["identity"]["TRAIN"],
                           raw_feature_sha256=rt["tensor_sha"](x))
            result = checks.batch(context, rt, mods, resources, data, x, replay)
            resources.call("final_complete_input_immutability", lambda: verify_inputs(rt, data, x, replay))
        final_custody(context)
        actual_profile = sys.modules["pilot_model"].runtime_settings()
        require(actual_profile == rt["identity"]["runtime"]["profile"], "Final runtime profile changed")
        receipt.update(status="PASS", result=result, measurements=resources.times,
                       final_resources=resources.sample(), final_runtime_profile=actual_profile)
    except BaseException as exc:
        receipt.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc))
        (output / "TRACEBACK.txt").write_text(traceback.format_exc())
        if resources is not None:
            receipt["measurements"] = resources.times
            try:
                receipt["failed_resources"] = resources.sample(synchronize=False, enforce=False)
            except BaseException as capture_error:
                receipt["failed_resource_capture_error"] = str(capture_error)
        if (output / "PROGRESS.json").exists():
            receipt["failure_stage"] = json.loads((output / "PROGRESS.json").read_text())
        # Failed engineering states remain explicitly non-donors in this output.
        receipt["retained_engineering_state_files"] = [p.name for p in output.glob("ENGINEERING_*.pt")]
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["UTC_finished"] = utc()
        atomic_json(output / "RECEIPT.json", receipt)


def load_features(context, rt):
    import numpy as np
    import pandas as pd
    torch = rt["torch"]
    authority = context["authority"]
    root = Path(authority["dataset_root"]).resolve()
    path = (root / "raw/node-feat.csv.gz").resolve()
    require(path.is_relative_to(root), "Feature path escaped authority")
    pin = authority["files"]["raw/node-feat.csv.gz"]
    require(path.stat().st_size == pin["bytes"] and file_sha(path) == pin["sha256"], "Raw feature file differs")
    value = torch.from_numpy(pd.read_csv(path, compression="gzip", header=None).values.astype(np.float32))
    require(value.shape == (235868,128) and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
            and rt["tensor_sha"](value) == authority["train_raw_tensor_digests"]["raw_features"], "Full raw feature identity differs")
    return value.to(rt["device"])


def verify_inputs(rt, data, x, replay):
    sys.modules["replay_provider"].verify_TRAIN_identity(rt, data)
    require(rt["tensor_sha"](x) == rt["authority"]["train_raw_tensor_digests"]["raw_features"]
            and rt["tensor_sha"](replay.permutation) == replay.receipt["permutation_sha256"]
            and rt["tensor_sha"](replay.negatives.T) == replay.receipt["native_negative_draw_sha256"],
            "Complete admitted input arrays changed")


def owned_worker_rss(pid):
    # Read only this child's Linux status, never enumerate/signal other jobs.
    path = Path("/proc") / str(pid) / "status"
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def supervisor(context):
    import fcntl
    output = context["output"]
    output.mkdir(parents=True, mode=0o700)
    lock = os.open(output / ".RUN_LOCK", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    limits = context["plan"]["resource_limits"]
    wall_limit = limits[context["stage"] + "_wall_seconds"]
    command = [sys.executable, "-B", str(HERE / "qualify.py"), "--stage", context["stage"],
               "--release", str(context["release_path"]), "--output", str(output), "--worker"]
    receipt = {"schema": "pooled-native-batch-bounded-supervisor-v1", "UTC": utc(),
               "identity": context["identity"], "release_sha256": context["release_sha256"],
               "command": command, "resource_limits": limits, "wall_limit_seconds": wall_limit,
               "status": "IN_PROGRESS", "state_donor": False, "fits": 0}
    atomic_json(output / "SUPERVISOR_STARTED.json", receipt)
    started = time.monotonic()
    proc = None
    reason = None
    peak_rss = 0
    try:
        with (output / "STDOUT.txt").open("wb") as stdout, (output / "STDERR.txt").open("wb") as stderr:
            proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=context["research"])
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                peak_rss = max(peak_rss, owned_worker_rss(proc.pid))
                disk = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                progress = json.loads((output / "PROGRESS.json").read_text()) if (output / "PROGRESS.json").exists() else {}
                arm = progress.get("arm")
                if arm != receipt.get("current_arm"):
                    receipt["current_arm"], receipt["arm_started_monotonic"] = arm, time.monotonic()
                arm_elapsed = time.monotonic() - receipt.get("arm_started_monotonic", started)
                if elapsed > wall_limit:
                    reason = "TOTAL_WALL_BOUND"
                elif context["stage"] == "batch" and arm and arm_elapsed > limits["batch_arm_wall_seconds"]:
                    reason = "PER_ARM_WALL_BOUND"
                elif peak_rss > limits["host_RSS_bytes"]:
                    reason = "HOST_RSS_BOUND"
                elif disk > limits["output_bytes"]:
                    reason = "OUTPUT_BYTES_BOUND"
                if reason:
                    receipt["failure_stage"] = progress
                    proc.kill(); break
                time.sleep(.5)
            exit_code = proc.wait()
        result = json.loads((output / "RECEIPT.json").read_text()) if (output / "RECEIPT.json").exists() else {}
        require(reason is None and exit_code == 0 and result.get("status") == "PASS", "Worker did not pass: " + str(reason or exit_code))
        receipt["status"] = "PASS"
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.kill(); proc.wait()
        receipt.update(status="FAIL", exception_type=type(exc).__name__, exception=str(exc), bound_failure=reason)
        (output / "SUPERVISOR_TRACEBACK.txt").write_text(traceback.format_exc())
        raise
    finally:
        receipt.update(UTC_finished=utc(), wall_seconds=time.monotonic()-started,
                       observed_child_peak_RSS_bytes=peak_rss, child_exit_code=None if proc is None else proc.returncode)
        receipt.pop("arm_started_monotonic", None)
        atomic_json(output / "SUPERVISOR_RECEIPT.json", receipt)
        os.close(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    context = preflight(args.release, args.stage, args.output, worker=args.worker)
    if args.worker:
        worker(context)
    else:
        supervisor(context)


if __name__ == "__main__":
    main()
