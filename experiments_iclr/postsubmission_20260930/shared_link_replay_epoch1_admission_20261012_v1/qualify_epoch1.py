"""Prepared root-gated model-free source epoch1 shared-replay proof.

Not executed by preparation. Uses the reviewed provider/full_batch source,
exact GRAPH/TRAIN pack and factory helper bytes. No models or optimizer exist.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    tmp.write_bytes(canonical(value)+b"\n"); tmp.replace(path)


def bound(phase, row):
    path = (phase/row["path"]).resolve()
    if not path.is_relative_to(phase) or pin(path) != {k: row[k] for k in ("bytes", "sha256")}:
        raise ValueError("Prepared source/data authority changed")
    return path


@contextmanager
def deny_targets(paths):
    forbidden = {Path(path).resolve() for path in paths}
    opened = []
    old_builtin, old_io = builtins.open, io.open
    def wrapped(original):
        def open_checked(path, *args, **kwargs):
            if not isinstance(path, int):
                resolved = Path(path).resolve()
                if resolved in forbidden:
                    raise ValueError("SELECT/CAL/EVALUATE target IO is forbidden")
                opened.append(str(resolved))
            return original(path, *args, **kwargs)
        return open_checked
    builtins.open, io.open = wrapped(old_builtin), wrapped(old_io)
    try:
        yield opened
    finally:
        builtins.open, io.open = old_builtin, old_io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    args = parser.parse_args()
    release_path = Path(args.release).resolve()
    release = json.loads(release_path.read_text())
    if (release.get("schema") != "shared-link-replay-epoch1-root-release-v1"
            or release.get("execution_enabled") is not True
            or not release.get("root_authorization_reference")
            or release.get("epoch") != 1
            or release.get("master_seed") != 2026100401
            or release.get("source_provider_draws") != 2
            or release.get("scientific_fits_authorized") is not False
            or release.get("model_constructors_authorized") is not False
            or release.get("optimizer_updates_authorized") is not False):
        raise ValueError("Exactly one separate root-admitted epoch1 source proof required")
    if "torch" in sys.modules:
        raise ValueError("Gate must precede numerical imports")
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Wrong scientific allocation hostname")
    if subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Wrong scientific allocation GPU route")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Factory CUDA device0 must be explicit before imports")
    phase = Path(release["research_root"]).resolve()
    repo = phase.parents[1]
    if str(repo) != "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs":
        raise ValueError("Wrong project root")
    if not HERE.is_relative_to(phase) or pin(__file__) != release["program_pin"]:
        raise ValueError("Another source program cannot inherit this admission")
    if Path(sys.executable).resolve() != Path(release["runtime"]["interpreter"]).resolve():
        raise ValueError("Root-bound factory interpreter required")
    if pin(sys.executable) != release["runtime"]["interpreter_pin"]:
        raise ValueError("Root-bound interpreter bytes differ")
    versions = {name: importlib.metadata.version(name) for name in release["runtime"]["versions"]}
    if versions != release["runtime"]["versions"]:
        raise ValueError("Root-bound source runtime versions differ")
    if not release["runtime"].get("binary_file_pins"):
        raise ValueError("Captured source numerical binary identities required")
    for row in release["runtime"]["binary_file_pins"]:
        if pin(row["path"]) != {k: row[k] for k in ("bytes", "sha256")}:
            raise ValueError("Root-bound numerical dependency bytes differ")
    # Root supervises exact owner pid/start_ticks and protects Physics alongside
    # this process. This child independently refuses insufficient free VRAM.
    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free_mib < release["reservation_mib"]+release["headroom_mib"]:
        raise ValueError("Insufficient free VRAM for the one admitted source proof")
    output = (phase/release["output"]).resolve()
    if not output.is_relative_to(phase) or output.exists():
        raise ValueError("Fresh prospective output within project required")
    output.mkdir()
    started = time.monotonic()
    attempt = {"schema": "shared-source-epoch1-attempt-v1", "release_pin": pin(release_path),
               "source_epoch": 1, "status": "RUNNING", "scientific_fits": 0,
               "source_arrays_are_model_donors": False}
    atomic(output/"ATTEMPT.json", attempt)
    cfg = release["source_config"]
    forbidden = [(phase/cfg["inputs"]["select"]["path"]).resolve()]
    forbidden += [(phase/p).resolve() for p in release["forbidden_target_paths"]]
    try:
        with deny_targets(forbidden) as io_reads:
            # Authenticate source, GRAPH and TRAIN only; SELECT receipt belongs
            # to the identity but its physical file is never read here.
            graph_path = bound(phase, cfg["inputs"]["graph"])
            train_path = bound(phase, cfg["inputs"]["train"])
            for row in cfg["sources"].values():
                bound(phase, row)
            for row in release["additional_source_files"].values():
                bound(phase, row)
            sys.path.insert(0, str(phase/release["factory_runtime"]))
            from adapters import link_prototypes as port
            from adapters.common_native_wrapper import load_module
            import torch
            from torch_geometric.utils import negative_sampling
            torch.cuda.set_device(0)
            profile = {"deterministic": torch.are_deterministic_algorithms_enabled(),
                       "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                       "TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
                       "TF32_cudnn": torch.backends.cudnn.allow_tf32}
            if profile != release["runtime"]["profile"]:
                raise ValueError("Source factory numerical flags differ")
            counters = {"model_constructors": 0, "optimizer_constructors": 0}
            def denied_model(*_, **__):
                counters["model_constructors"] += 1
                raise ValueError("Model construction is forbidden in source replay proof")
            def denied_optimizer(*_, **__):
                counters["optimizer_constructors"] += 1
                raise ValueError("Optimizer construction is forbidden in source replay proof")
            torch.nn.Module.__init__ = denied_model
            torch.optim.Optimizer.__init__ = denied_optimizer
            modules = port.source_modules(phase, cfg["sources"])
            _, utils = port.bridge.native_modules(phase, {"model": cfg["sources"]["native"], "utils": cfg["sources"]["utils"]})
            rt, provider, transformed = port.runtime_replay(torch, modules, utils, negative_sampling)
            if transformed is not None:
                raise ValueError("Actual CUDA provider bytes must remain unchanged")
            rt.update(device=torch.device("cuda:0"), utils=utils,
                      frozen_epoch_stream=modules["pilot_data"].epoch_stream)
            graph = torch.load(graph_path, map_location="cpu", weights_only=True)
            train = torch.load(train_path, map_location="cpu", weights_only=True)
            if set(graph) != {"x", "raw_edge_index"} or set(train) != {"positive"}:
                raise ValueError("Exact factory GRAPH/TRAIN schema required")
            pairs, raw = train["positive"].to("cuda:0"), graph["raw_edge_index"].to("cuda:0")
            if tuple(pairs.shape) != (1179052, 2) or tuple(raw.shape) != (2, 2358104):
                raise ValueError("Complete native TRAIN geometry required")
            identity = {"nodes": 235868, "records": len(pairs), "ordered_raw_entries": raw.shape[1],
                        "batch_size": 65536, "pairs_sha256": rt["tensor_sha"](pairs), "raw_sha256": rt["tensor_sha"](raw)}
            data = {"pairs": pairs, "raw": raw, "nodes": 235868, "batch_size": 65536,
                    "identity": {"TRAIN": identity, "source_replay_schema": "unchanged_source_integer_context_v1",
                                 "input_receipts": cfg["inputs"], "source_receipts": cfg["sources"]}}
            with port.aliases({"replay_common": modules["replay_common"], "replay_provider": provider}):
                qualification = load_module(phase, release["additional_source_files"]["qualification"])
            sys.path.insert(0, str(phase/release["shared_store_source"]))
            from shared_replay_store import SharedReplayStore, stream_key
            key = stream_key(input_receipts=cfg["inputs"], source_receipts=cfg["sources"],
                             geometry=identity, master_seed=2026100401, runtime=release["runtime"])
            store = SharedReplayStore(output/"shared_fixture", key)
            rng_before = rt["state"].rng_digest(rt["state"].rng_state())
            source_result = {}
            def producer(stage):
                parent = output/("source_attempt_"+uuid.uuid4().hex); parent.mkdir()
                # Reviewed helper: source draw once, independently frozen source
                # draw once, all17 complete context comparisons, one file roster.
                with provider.isolated_rng(rt, 2026100401):
                    source_result.update(qualification.full_batch(rt, data, 2026100401, 1, parent))
                trace = parent/"complete_TRAIN_trace"
                trace.rename(stage)
            def verifier(stage):
                before = rt["state"].rng_digest(rt["state"].rng_state())
                reader = provider.EpochReplay(rt, data, stage, pin(stage/"EPOCH.json")["sha256"])
                if reader.receipt["full_batches"] != 17 or reader.receipt["master_seed"] != 2026100401 or reader.receipt["epoch"] != 1:
                    raise ValueError("Another source epoch cannot inherit shared authority")
                if rt["state"].rng_digest(rt["state"].rng_state()) != before:
                    raise ValueError("Shared reader altered caller RNG")
            directory, manifest = store.ensure_epoch(1, producer, verifier)
            source_result["source_helper_temporary_trace_path"] = source_result.pop("trace")
            source_result["trace"] = str(directory)
            source_result["same_bytes_atomically_relocated_to_shared_fixture"] = True
            # Second prospective consumer reads the same authority; zero redraw.
            again, second = store.ensure_epoch(1, lambda _: (_ for _ in ()).throw(ValueError("Unexpected redraw")), verifier)
            if directory != again or manifest != second:
                raise ValueError("Shared consumers did not receive one immutable authority")
            leases = [store.bind_consumer(name, 1, directory, manifest) for name in release["lease_fixture_consumers"]]
            rng_after = rt["state"].rng_digest(rt["state"].rng_state())
            if rng_after != rng_before or counters != {"model_constructors": 0, "optimizer_constructors": 0}:
                raise ValueError("Source-only proof changed RNG or constructed a learner")
            torch.cuda.synchronize(0)
            result = {"schema": "shared-link-replay-epoch1-qualification-v1", "status": "PASS",
                      "source_result": source_result, "shared_epoch_manifest": manifest,
                      "stream_key": key, "stream_key_hash": store.key_hash,
                      "lease_receipts": [pin(p) for p in leases], "lease_identical_epoch": True,
                      "RNG_before": rng_before, "RNG_after": rng_after,
                      "source_provider_draws": 2, "independent_source_contexts": 17,
                      "source_epoch_rosters_written": 1, "SELECT_CAL_EVALUATE_reads": 0,
                      "io_paths": sorted(set(io_reads)), "constructors": counters,
                      "scientific_fits": 0, "optimizer_updates": 0,
                      "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                      "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                      "CPU_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      "wall_seconds": time.monotonic()-started,
                      "scientific_store_authority": False, "future_controller_integration_pending": True}
        atomic(output/"RESULT.json", result)
        attempt.update(status="PASS", result_pin=pin(output/"RESULT.json"))
    except BaseException as error:
        attempt.update(status="FAILED", error_type=type(error).__name__, error=str(error),
                       wall_seconds=time.monotonic()-started, failed_outputs_preserved=True)
        atomic(output/"ATTEMPT.json", attempt)
        raise
    atomic(output/"ATTEMPT.json", attempt)
    print(json.dumps({k: result[k] for k in ("status", "source_provider_draws", "scientific_fits", "optimizer_updates", "wall_seconds")}))


if __name__ == "__main__":
    main()
