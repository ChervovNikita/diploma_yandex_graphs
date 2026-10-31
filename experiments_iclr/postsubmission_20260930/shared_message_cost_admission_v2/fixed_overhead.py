"""Bounded stdlib metadata-cost measurement; no model, tensor or label access.

Every workload artifact is an explicitly marked surrogate. This entrypoint is
unexecuted until a distinct exact root request is authorized and supervised.
"""
import sys
import os
sys.path[:] = [p for p in sys.path if p and os.path.isabs(p)
               and any(os.path.commonpath([p, root]) == root for root in (sys.prefix, sys.base_prefix))]
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
BOUNDED_SHA = "b096966c1570da0a998dbb9fb464a9ad71490b74d4ba3406b751390637d19a23"
SCIENCE_MANIFEST_SHA = "b9a348bd41c456fcd974b257ef65a19655163022d5f18bf0dc1a4429f57ccec4"
SCIENCE_BINDINGS_SHA = "7e2a68e6b1d3840238ee92a0c6cee95afe7d42579af0bc7e529719d2741d4e11"
COUNTS = {"fits": 30, "epochs_per_fit": 300, "trace_writes": 9000,
          "FIT_END_writes": 30, "trace_reads_parses": 45000,
          "FIT_END_reads_parses": 150, "trace_and_FIT_END_full_hashes": 36120,
          "manifest_writes": 63,
          "manifest_reads_parses": 252, "supervisor_chain_resolutions": 5617,
          "measured_source_closure_hash_passes": 16,
          "source_closure_forecast_passes_bound": 2048,
          "quality_result_reads_parses": 448}
SIZES = {"nonvalidation_epoch_minimum_bytes": 4096,
         "validation_epoch_minimum_bytes": 16384,
         "FIT_END_extra_padding_bytes": 65536,
         "manifest_minimum_bytes": 524288,
         "supervisor_JSON_minimum_bytes": 65536,
         "supervisor_log_bytes_each": 1048576}
_WRITTEN_BYTES = 0


def within(path, root=PHASE):
    root, path = Path(root).resolve(), Path(os.path.abspath(path))
    if path == root or not path.is_relative_to(root):
        raise ValueError("Path must stay strictly inside its declared project folder")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink evidence/output forbidden")
    return path.resolve()


def sha(path):
    path = within(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def reference(path):
    path = within(path)
    return dict(path=str(path.relative_to(PHASE)), sha256=sha(path))


def checked(ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Exact path/SHA256 metadata reference required")
    path = within(PHASE / ref["path"])
    if str(path.relative_to(PHASE)) != ref["path"] or sha(path) != ref["sha256"]:
        raise ValueError("Changed or noncanonical metadata/source reference")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    global _WRITTEN_BYTES
    path = within(path, HERE)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    _WRITTEN_BYTES += path.stat().st_size


def padded(value, minimum):
    value = {**value, "metadata_only_surrogate_no_scientific_values": True,
             "metadata_padding": ""}
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    value["metadata_padding"] = "x" * max(0, minimum - len(raw.encode()))
    return value


def epoch_metadata(epoch):
    # Fixed invented scalar placeholders, solely to reproduce JSON structure.
    # They never represent losses, predictions or quality observations.
    scalar = 0.3141592653589793
    row = dict(epoch=epoch, training_own_CE_preupdate=scalar,
        train_update_seconds=scalar, validation=None, sufficient_statistics=None,
        improved_checkpoint=None)
    if epoch % 5 == 0:
        pair = {f"{a}_{b}": scalar for a in range(4) for b in range(a)}
        row.update(validation_forward_and_CPU_transfer_seconds=scalar,
            CPU_metric_seconds=scalar, checkpoint_transfer_write_hash_seconds=scalar,
            validation=dict(nodes=2449029, members=4, classes=47, NLL=scalar,
                accuracy_percent=scalar, member_NLL=[scalar] * 4,
                member_accuracy_percent=[scalar] * 4,
                pairwise_disagreement=pair, pairwise_error_intersection_fraction=pair,
                pairwise_error_Jaccard=pair, pooled_gain_over_mean_member_accuracy_pp=scalar,
                scoring_precision="metadata-only surrogate"),
            sufficient_statistics=dict(nodes=2449029, members=4, classes=47,
                pooled_CE_sum=scalar, pooled_correct=0, member_CE_sum=[scalar] * 4,
                member_correct=[0] * 4,
                pair_counts={key: dict(disagree=0, both_wrong=0, either_wrong=0) for key in pair}),
            improved_checkpoint=dict(path="metadata-surrogate/" + "x" * 512, sha256="0" * 64))
    return padded(row, SIZES["validation_epoch_minimum_bytes" if epoch % 5 == 0
                             else "nonvalidation_epoch_minimum_bytes"])


def admission(path, supervisor):
    path = within(path, HERE)
    value = json.loads(path.read_text())
    if (value.get("schema") != "gnnm-shared-message-fixed-overhead-request-v1"
            or value.get("root_authorized_metadata_measurement") is not True
            or value.get("Torch_models_provider_labels_or_scientific_scores_authorized") is not False
            or value.get("work_counts") != COUNTS or value.get("surrogate_sizes") != SIZES
            or value.get("helper_sha256") != sha(HERE / "fixed_overhead.py")
            or value.get("bounded_source_sha256") != BOUNDED_SHA
            or type(value.get("wall_cap_seconds")) not in (int, float)
            or not math.isfinite(value["wall_cap_seconds"])
            or not 0 < value["wall_cap_seconds"] <= 900
            or value.get("disk_cap_bytes") != 256 * 2**20):
        raise ValueError("Exact root-authorized metadata-only bounded request required")
    manifest = checked(value["source_manifest"])
    seal = json.loads(manifest.read_text())
    if seal.get("schema") != "gnnm-shared-message-cost-admission-source-manifest-v1":
        raise ValueError("Exact source-only cost helper seal required")
    for row in seal["files"]:
        file = within(HERE / row["relative_path"], HERE)
        if file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
            raise ValueError("Cost helper sealed source changed")
    science = checked(value["scientific_source_manifest"])
    bindings = checked(value["scientific_source_bindings"])
    if sha(science) != SCIENCE_MANIFEST_SHA or sha(bindings) != SCIENCE_BINDINGS_SHA:
        raise ValueError("Exact preserved scientific source closure required")
    quality = checked(value["completed_quality_result"])
    record = json.loads(quality.read_text())
    if record.get("schema") != "gnnm-shared-message-complete-training-qualification-v3" or record.get("quality_scores_computed") is not False:
        raise ValueError("Only the exact training-engineering metadata result is readable")
    request_ref = reference(path)
    outer_path = within(os.environ["GNNM_BOUND_START_JSON"])
    outer = json.loads(outer_path.read_text())
    supervisor = within(supervisor)
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    argv = [str(Path(route["repository"]) / ".venv/bin/python"), str(HERE / "fixed_overhead.py"),
            "--request", str(path), "--supervisor", str(supervisor)]
    command = json.loads((supervisor / "command.json").read_text())
    environment = json.loads((supervisor / "environment.json").read_text())
    expected_outer = [argv[0], str(PHASE / "protocols/run_authorized_v2.py"), "--output", str(supervisor),
                      "--require-idle-gpu", "--", *argv]
    if (outer_path.name != "START.json" or outer.get("schema") != "gnnm-whole-process-bound-start-v1"
            or outer.get("root_request") != request_ref or outer.get("root_request_child_flag") != "--request"
            or outer.get("child_argv") != argv or command.get("argv") != argv or outer.get("argv") != expected_outer
            or outer.get("inner_supervisor_directory") != str(supervisor)
            or outer.get("whole_cap_seconds") != value["wall_cap_seconds"]
            or outer.get("shell") is not False or outer.get("start_new_session") is not True
            or outer.get("cwd") != route["repository"] or command.get("cwd") != route["repository"]
            or environment.get("repository") != route["repository"] or environment.get("phase") != str(PHASE)
            or outer.get("phase") != str(PHASE) or outer.get("ssh_destination") != route["ssh_destination"]
            or os.environ.get("GNNM_SSH_DESTINATION") != route["ssh_destination"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
            or not 0 < outer.get("termination_grace_seconds", 0) < outer["whole_cap_seconds"]
            or outer.get("normal_child_budget_seconds") != outer["whole_cap_seconds"] - outer["termination_grace_seconds"]):
        raise ValueError("Exact sole-allocation source-bound external supervisor required")
    for key, file in (("source_sha256", PHASE / "protocols/bounded_run_v1.py"),
            ("authorization_source_sha256", PHASE / "protocols/run_authorized_v2.py"),
            ("logging_source_sha256", PHASE / "protocols/run_logged.py"),
            ("wrapper_sha256", PHASE / "protocols/repo_env.sh"), ("research_script_sha256", HERE / "fixed_overhead.py")):
        if outer.get(key) != sha(file):
            raise ValueError("External supervisor/source pin differs")
    if outer["source_sha256"] != BOUNDED_SHA or environment.get("script_sha256") != sha(HERE / "fixed_overhead.py"):
        raise ValueError("Helper/envelope source generation changed")
    started = datetime.fromisoformat(outer["start_UTC"])
    if started.tzinfo is None:
        raise ValueError("Whole-process time must be timezone-aware")
    return value, request_ref, dict(outer_start=reference(outer_path),
        supervisor_directory=str(supervisor.relative_to(PHASE)), started=started.isoformat()), bindings, quality


def run(value, request_ref, live, source_bindings, quality_result):
    global _WRITTEN_BYTES
    _WRITTEN_BYTES = 0
    output = within(PHASE / value["output_relative_to_phase"], HERE)
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.fromisoformat(live["started"])
    def cap():
        if (datetime.now(timezone.utc) - started).total_seconds() >= value["wall_cap_seconds"]:
            raise RuntimeError("Fixed-overhead whole-process cap exhausted")
        if _WRITTEN_BYTES > value["disk_cap_bytes"]:
            raise RuntimeError("Fixed-overhead disk cap exhausted")
    write(output / "START.json", dict(request=request_ref, live=live,
        scope="metadata-only surrogate I/O and exact sealed-source hashes", scientific_scores_exist=False))
    rows, fits, manifests, timing = [], [], [], {}
    try:
        begin = time.perf_counter()
        for cell in range(COUNTS["fits"]):
            directory = output / f"surrogate_fit{cell:02d}"
            directory.mkdir()
            trace = []
            for epoch in range(1, 301):
                row = epoch_metadata(epoch)
                file = directory / f"epoch{epoch:03d}.json"
                write(file, row)
                rows.append(file); trace.append(row)
            file = directory / "FIT_END.json"
            write(file, dict(schema="metadata-only-surrogate-FIT_END", trace=trace,
                metadata_only_surrogate_no_scientific_values=True,
                metadata_padding="x" * SIZES["FIT_END_extra_padding_bytes"]))
            fits.append(file)
            cap()
        timing["trace_and_FIT_END_generation_and_writes_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        for _ in range(5):
            for index, file in enumerate(rows):
                json.loads(file.read_text())
                if index % 1000 == 0:
                    cap()
        timing["45000_trace_reads_parses_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        for _ in range(5):
            for file in fits:
                parsed = json.loads(file.read_text())
                if len(parsed["trace"]) != 300:
                    raise ValueError("Surrogate trace shape changed")
            cap()
        timing["150_FIT_END_reads_parses_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        for _ in range(4):
            for index, file in enumerate(rows + fits):
                sha(file)
                if index % 1000 == 0:
                    cap()
        timing["36120_trace_and_FIT_END_full_hashes_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        for _ in range(2):
            for index, file in enumerate(rows + fits):
                first = within(file)
                second = within(first)
                second.stat()
                if index % 1000 == 0:
                    cap()
        timing["18060_extra_cached_reference_path_stat_checks_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        for index in range(COUNTS["manifest_writes"]):
            file = output / f"surrogate_manifest{index:02d}.json"
            write(file, padded(dict(schema="metadata-only-surrogate-manifest", files=[
                dict(path="metadata-surrogate/" + "x" * 256, sha256="0" * 64) for _ in range(364)]),
                SIZES["manifest_minimum_bytes"]))
            manifests.append(file)
        for _ in range(4):
            for file in manifests:
                json.loads(file.read_text()); sha(file)
            cap()
        timing["63_manifest_writes_252_reads_parses_hashes_seconds"] = time.perf_counter() - begin
        supervisor = output / "surrogate_supervisor"
        supervisor.mkdir()
        json_files = []
        for name in ("authorization", "completion", "environment", "command", "outer_START", "outer_TERMINAL"):
            file = supervisor / (name + ".json")
            write(file, padded(dict(schema="metadata-only-surrogate-supervisor"), SIZES["supervisor_JSON_minimum_bytes"]))
            json_files.append(file)
        log_files = []
        for name in ("inner", "outer"):
            file = supervisor / (name + ".log")
            with file.open("xb") as stream:
                stream.write(b"x" * SIZES["supervisor_log_bytes_each"])
            _WRITTEN_BYTES += file.stat().st_size
            log_files.append(file)
        begin = time.perf_counter()
        for index in range(COUNTS["supervisor_chain_resolutions"]):
            for file in json_files:
                json.loads(file.read_text()); sha(file)
            for file in log_files:
                sha(file)
            if index % 200 == 0:
                cap()
        timing["5617_padded_supervisor_chain_metadata_parses_hashes_seconds"] = time.perf_counter() - begin
        bindings = json.loads(source_bindings.read_text())["sha256"]
        if any(Path(name).suffix not in (".py", ".md", ".json", ".sh", ".jinja", ".log") for name in bindings):
            raise ValueError("Source closure may not contain tensor/provider data or checkpoints")
        source_samples = []
        for _ in range(COUNTS["measured_source_closure_hash_passes"]):
            begin = time.perf_counter()
            for name, wanted in bindings.items():
                if sha(within(PHASE / name)) != wanted:
                    raise ValueError("Preserved scientific source changed during overhead benchmark")
            source_samples.append(time.perf_counter() - begin)
            cap()
        timing["16_exact_source_closure_full_hash_pass_samples_seconds"] = source_samples
        timing["2048_source_closure_passes_worst_sample_bound_seconds"] = max(source_samples) * COUNTS["source_closure_forecast_passes_bound"]
        begin = time.perf_counter()
        for index in range(COUNTS["quality_result_reads_parses"]):
            json.loads(quality_result.read_text()); sha(quality_result)
            if index % 32 == 0:
                cap()
        timing["448_exact_quality_engineering_result_reads_parses_hashes_seconds"] = time.perf_counter() - begin
        cap()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(schema="gnnm-shared-message-fixed-overhead-measurement-v1", status="completed_metadata_only",
            request=request_ref, live=live, work_counts=COUNTS, surrogate_sizes=SIZES, measured_seconds=timing,
            input_source_bindings=reference(source_bindings), input_engineering_result=reference(quality_result),
            logical_regular_file_bytes_before_report=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
            peak_host_RSS_bytes=int(rss * (1024 if sys.platform.startswith("linux") else 1)),
            tensors_labels_models_checkpoints_or_scientific_scores_accessed=False,
            quality_novelty_feasibility_or_scientific_admission_established=False,
            fsync_or_crash_durability_measured=False, surrogate_size_bounds_require_root_qualification=True,
            full_wall_cost_requires_successful_outer_terminal=True)
        write(output / "RESULT.json", result)
        files = [p for p in sorted(output.rglob("*")) if p.is_file()]
        write(output / "ARTIFACT_MANIFEST.json", dict(schema="gnnm-shared-message-fixed-overhead-artifacts-v1",
            files=[reference(p) for p in files]))
        cap()
        print(json.dumps(dict(status=result["status"], scientific_scores_exist=False)), flush=True)
    except BaseException as error:
        write(output / "FAILURE.json", dict(type=type(error).__name__, message=str(error), partial_outputs_preserved=True,
            retries_or_scientific_promotion_authorized=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    args = parser.parse_args()
    value, request_ref, live, bindings, quality = admission(args.request, args.supervisor)
    run(value, request_ref, live, bindings, quality)


if __name__ == "__main__":
    main()
