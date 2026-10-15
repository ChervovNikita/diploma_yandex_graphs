"""Reconstruct one own run_fit's selected complete VALID, never train or TEST.

Only explicit frozen runtime/callback/extension profiles are supported. Source
imports are scoped so runtime v1/v2 cannot silently borrow each other's common,
bookkeeping, controllers or adapters modules. No current/base fallback exists.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-4
LOSS_ATOL = 1e-4
LOSS_RTOL = 1e-4
METRIC_ATOL = 0.002


class FidelityRefusal(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def inside(path, root):
    root, path = Path(root).resolve(), Path(path)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise FidelityRefusal("Path escapes the own repository: " + str(path))
    return path


def receipt(path):
    path = Path(path).resolve()
    size, hasher = 0, hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": hasher.hexdigest()}


def verified(row, root):
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} \
            or type(row["bytes"]) is not int or row["bytes"] <= 0 \
            or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
        raise FidelityRefusal("Explicit path/bytes/SHA256 descriptors required")
    path = inside(row["path"], root)
    actual = receipt(path)
    if any(actual[key] != row[key] for key in ("bytes", "sha256")):
        raise FidelityRefusal("Recorded input bytes changed: " + str(path))
    return path


def admit_cpu():
    """Route admission only; no new human approval or GO inference."""
    if socket.gethostname() != "anogena-2-0" or subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines() != [
                "GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise FidelityRefusal("Wrong allocation route; no project model/data read allowed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise FidelityRefusal("CPU replay v1 requires CUDA_VISIBLE_DEVICES empty")
    return {"hostname": "anogena-2-0", "GPU_UUID_route_check": "GPU-44039938-fd82-41d2-fefd-de71514e2fac",
            "device": "cpu", "CUDA_VISIBLE_DEVICES": "", "threads": 2, "new_approval_flow": False}


def verify_package(folder, expected, root):
    folder = inside(folder, root)
    manifest = receipt(folder / "MANIFEST.json")
    if manifest["sha256"] != expected:
        raise FidelityRefusal("Recorded package manifest changed: " + str(folder))
    result = [manifest]
    for row in json.loads((folder / "MANIFEST.json").read_text())["files"]:
        path = inside(folder / row["path"], folder)
        pin = receipt(path)
        if any(pin[key] != row[key] for key in ("bytes", "sha256")):
            raise FidelityRefusal("Frozen package member changed: " + str(path))
        result.append(pin)
    return result


def extension_declaration(job):
    fields = [key for key in ("session_transform", "extension_factory", "session_extension") if job.get(key)]
    if len(fields) > 1:
        raise FidelityRefusal("Ambiguous extension aliases are unsupported")
    if not fields:
        return None
    value = copy.deepcopy(job[fields[0]])
    if isinstance(value, str):
        path, separator, function = value.partition(":")
        value = {"path": path, "function": function if separator else "extend_session"}
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise FidelityRefusal("Unsupported extension descriptor")
    value.setdefault("function", "extend_session")
    return value


def match_runtime(binding, profiles):
    candidates = [row for row in profiles if binding.get("runtime_sha256") == row["runtime_sha256"]]
    if len(candidates) != 1:
        raise FidelityRefusal("Unsupported recorded runtime; never substitute current/base runtime")
    profile = candidates[0]
    keys = {"runtime_sha256", "callback_sha256", "extension_sha256", "dependency_sha256"}
    if profile["id"] == "runtime_v2":
        keys.add("policy_sha256")
    if set(binding) != keys or binding.get("policy_sha256") is not None:
        raise FidelityRefusal("Unknown runtime binding/policy for node/OGB replay")
    return profile


def source_preflight(job, binding, root, expected_replayer_manifest):
    pins = verify_package(HERE, expected_replayer_manifest, root)
    registry = json.loads((HERE / "REGISTRY.json").read_text())
    for row in registry["native_sources"]:
        path = inside(PHASE / row["path_relative_to_phase"], root)
        if receipt(path)["sha256"] != row["sha256"]:
            raise FidelityRefusal("Pinned native source changed: " + str(path))
        pins.append(receipt(path))
    profile = match_runtime(binding, registry["runtimes"])
    runtime_path = inside(PHASE / profile["path_relative_to_phase"], root)
    for name, sha in profile["runtime_sha256"].items():
        path = inside(runtime_path / name, root)
        if receipt(path)["sha256"] != sha:
            raise FidelityRefusal("Recorded runtime bytes unavailable: " + str(path))
        pins.append(receipt(path))
    if job.get("schema") != "native-full-epoch-fit-v1" or job.get("family") not in ("node", "ogb") \
            or job.get("TEST_access") is not False or type(job.get("seed")) is not int:
        raise FidelityRefusal("Explicit own node/OGB full-epoch VALID-only JOB required")
    if profile["id"] == "runtime_v1" and job.get("source_rng_emulation") is not None:
        raise FidelityRefusal("runtime_v1 has no source RNG emulation contract")
    callback = inside(job["callback_module"], root)
    declared = extension_declaration(job)
    matches = [row for row in registry["methods"] if row["family"] == job["family"]
        and job["procedure"] in row["procedures"]
        and callback == inside(PHASE / row["callback_relative_to_phase"], root)
        and ((row["extension_relative_to_phase"] is None and declared is None) or
             (declared is not None and row["extension_relative_to_phase"] is not None
              and inside(declared["path"], root) == inside(PHASE / row["extension_relative_to_phase"], root)
              and declared["function"] == row["extension_function"]))]
    if len(matches) != 1:
        raise FidelityRefusal("Unsupported callback/procedure/versioned extension combination")
    method = matches[0]
    if binding["callback_sha256"] != method["callback_sha256"] \
            or receipt(callback)["sha256"] != method["callback_sha256"] \
            or job.get("callback_sha256", method["callback_sha256"]) != method["callback_sha256"]:
        raise FidelityRefusal("Callback source binding differs")
    if binding["extension_sha256"] != method["extension_sha256"]:
        raise FidelityRefusal("Recorded extension source differs")
    pins.append(receipt(callback))
    if declared:
        path = inside(declared["path"], root)
        if receipt(path)["sha256"] != method["extension_sha256"] \
                or declared.get("sha256", method["extension_sha256"]) != method["extension_sha256"]:
            raise FidelityRefusal("Extension file/descriptor differs")
        if job.get("base_procedure") != "gnnm_boundary_native":
            raise FidelityRefusal("Recognized extension requires its exact native boundary base")
        if method.get("method_manifest_sha256") is not None \
                and job.get("method_settings", {}).get("manifest_sha256") != method["method_manifest_sha256"]:
            raise FidelityRefusal("Method settings use another package generation")
        pins.append(receipt(path))
    elif job.get("base_procedure", job["procedure"]) != job["procedure"]:
        raise FidelityRefusal("A base JOB cannot substitute another procedure")
    for row in method["packages"]:
        pins.extend(verify_package(PHASE / row["path_relative_to_phase"], row["sha256"], root))
    for row in method.get("extra_sources", []):
        path = inside(PHASE / row["path_relative_to_phase"], root)
        if receipt(path)["sha256"] != row["sha256"]:
            raise FidelityRefusal("Known method dependency changed")
        pins.append(receipt(path))
    for path, sha in binding["dependency_sha256"].items():
        native_path = inside(path, root)
        if receipt(native_path)["sha256"] != sha:
            raise FidelityRefusal("Recorded native dependency unavailable: " + str(native_path))
        pins.append(receipt(native_path))
    for field in ("data_root",):
        inside(job[field], root)
    if inside(job["recipe"]["_redesign_root"], root) != PHASE / "reference_wrapper_redesign_20261013_v1":
        raise FidelityRefusal("Another native redesign checkout is unsupported")
    return profile, method, pins


def data_preflight(job, declarations, root):
    """Pin exactly the source provider artifacts; no TEST split is requested."""
    data_root = inside(job["data_root"], root)
    dataset = job["recipe"]["dataset"]
    if job["family"] == "node":
        names = {"citeseer": "CiteSeer", "pubmed": "PubMed", "coauthor-cs": "CS", "coauthor-physics": "Physics"}
        if dataset not in names:
            raise FidelityRefusal("Unsupported source node data provider")
        expected = {"processed": inside(data_root / dataset / names[dataset] / "processed/data.pt", root),
            "basis": inside(PHASE / "reference_wrapper_redesign_20261013_v1/smoke_20261014/working_data/node_bases"
                            / (job["recipe"]["id"] + ".pt"), root)}
    else:
        if dataset not in ("ogbg-molhiv", "ogbg-molbace"):
            raise FidelityRefusal("Unsupported source molecular data provider")
        dirname = dataset.replace("-", "_")
        folder = data_root / dirname
        if not (folder / "processed/geometric_data_processed.pt").is_file() and data_root.name == dirname:
            folder = data_root
        expected = {"processed": inside(folder / "processed/geometric_data_processed.pt", root)}
        for role in ("train", "valid"):
            path = folder / "split/scaffold" / (role + ".pt")
            if not path.is_file():
                path = folder / "split/scaffold" / (role + ".csv.gz")
            expected[role + "_split"] = inside(path, root)
    if not isinstance(declarations, dict) or set(declarations) != set(expected):
        raise FidelityRefusal("Pin exactly the native processed/basis or processed/TRAIN/VALID split artifacts")
    result = {}
    for key, path in expected.items():
        if verified(declarations[key], root) != path:
            raise FidelityRefusal("Declared data path differs from the actual source provider: " + key)
        result[key] = receipt(path)
    return result


@contextmanager
def runtime_namespace(profile, root):
    names = ("run_fit", "adapters", "common", "bookkeeping", "controllers", "source_rng")
    previous = {name: sys.modules.get(name) for name in names}
    old_path = list(sys.path)
    folder = inside(PHASE / profile["path_relative_to_phase"], root)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(folder))
        modules = {}
        for name in ("common", "bookkeeping") + (("controllers", "source_rng") if profile["id"] == "runtime_v2" else ()) + ("adapters",):
            spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
        yield modules
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def restore_cpu_rng(torch, value):
    import numpy as np
    if set(value) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise FidelityRefusal("Unknown recorded global RNG schema")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    return {"restored": ["python", "numpy", "torch_cpu"], "CUDA_states_archived_not_applied_on_CPU": len(value["torch_cuda"])}


def binding_for(adapter, profile):
    value = {"runtime_sha256": dict(profile["runtime_sha256"]),
        "callback_sha256": adapter.callback_receipt["sha256"],
        "extension_sha256": None if adapter.extension_receipt is None else adapter.extension_receipt["sha256"],
        "dependency_sha256": {pin["path"]: pin["sha256"] for pin in adapter.dependency_receipts}}
    if profile["id"] == "runtime_v2":
        value["policy_sha256"] = None
    return value


def actual_validation(torch, adapter, require_mean):
    """Capture native diagnostics from the SAME actual adapter VALID invocation."""
    original = adapter.session.validate
    captured = {}
    def capture(self):
        value = original()
        captured["value"] = value
        return value
    adapter.session.validate = types.MethodType(capture, adapter.session)
    try:
        with torch.no_grad():
            value = adapter.validate()
    finally:
        adapter.session.validate = original
    if value.get("role") != "VALID" or value.get("complete_source_VALID") is not True or value.get("TEST_accessed") is not False:
        raise FidelityRefusal("Actual callback did not return complete native VALID")
    raw = captured.get("value", {})
    diagnostics = None
    if require_mean:
        keys = ("same_bank_mean_outputs", "same_bank_mean_loss", "same_bank_mean_metrics")
        if any(key not in raw for key in keys):
            raise FidelityRefusal("C6 native same-bank mean diagnostic is missing")
        mean = raw[keys[0]].detach().cpu()
        if mean.shape != value["outputs"].shape or not torch.isfinite(mean).all().item():
            raise FidelityRefusal("C6 mean diagnostic shape/values differ")
        diagnostics = {"same_bank_mean_raw_logits": mean, "same_bank_mean_loss": float(raw[keys[1]]),
            "same_bank_mean_metrics": {k: float(v) for k, v in raw[keys[2]].items()},
            "same_route_forwards_as_actual_readout": True, "used_for_selection": False}
        if not math.isfinite(diagnostics["same_bank_mean_loss"]) or any(
                not math.isfinite(v) for v in diagnostics["same_bank_mean_metrics"].values()):
            raise FidelityRefusal("Nonfinite C6 native mean diagnostics")
    return value, diagnostics


def compare_export(torch, actual, saved):
    ids, labels, outputs = [actual[key].detach().cpu() for key in ("ids", "labels", "outputs")]
    reference = saved["raw_logits"].detach().cpu()
    if ids.dtype != saved["ids"].dtype or not torch.equal(ids, saved["ids"].cpu()):
        raise FidelityRefusal("Selected VALID IDs/order changed")
    if labels.shape != saved["labels"].shape or labels.dtype != saved["labels"].dtype \
            or not torch.allclose(labels.to(torch.float64), saved["labels"].cpu().to(torch.float64), atol=0, rtol=0, equal_nan=True):
        raise FidelityRefusal("Selected VALID labels/observed support changed")
    if outputs.shape != reference.shape or outputs.dtype != reference.dtype \
            or not torch.isfinite(outputs).all().item() or not torch.isfinite(reference).all().item():
        raise FidelityRefusal("Selected VALID logit shape/dtype/finite support changed")
    difference = (outputs.to(torch.float64) - reference.to(torch.float64)).abs()
    torch.testing.assert_close(outputs, reference, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    expected = saved["validation"]
    if expected.get("role") != "VALID" or expected.get("complete_source_VALID") is not True \
            or expected.get("TEST_accessed") is not False or expected.get("records") != len(ids):
        raise FidelityRefusal("Saved export is not a complete native VALID authority")
    loss = float(actual["loss"])
    if not math.isfinite(loss) or not math.isfinite(float(expected["loss"])) \
            or not math.isclose(loss, float(expected["loss"]), abs_tol=LOSS_ATOL, rel_tol=LOSS_RTOL):
        raise FidelityRefusal("Selected VALID native loss parity failed")
    metrics = {key: float(value) for key, value in actual["metrics"].items()}
    if set(metrics) != set(expected["metrics"]) or any(not math.isfinite(value) or
            not math.isfinite(float(expected["metrics"][key])) or
            abs(value - float(expected["metrics"][key])) > METRIC_ATOL for key, value in metrics.items()):
        raise FidelityRefusal("Selected VALID native metric parity failed")
    return {"passed": True, "records": len(ids), "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()), "loss_abs_difference": abs(loss - float(expected["loss"])),
        "metric_abs_differences": {key: abs(value - float(expected["metrics"][key])) for key, value in metrics.items()},
        "fixed_tolerances": {"logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL,
            "loss_atol": LOSS_ATOL, "loss_rtol": LOSS_RTOL, "metric_atol": METRIC_ATOL}}


def selected_identity(job, checkpoint, export, checkpoint_pin):
    job_hash = digest(job)
    if checkpoint.get("schema") != "native-full-epoch-recovery-v1" or checkpoint.get("TEST_accessed") is not False \
            or checkpoint.get("job_hash") != job_hash or export.get("schema") != "own-selected-VALID-logits-v1" \
            or export.get("role") != "VALID" or export.get("TEST_accessed") is not False or export.get("job_hash") != job_hash:
        raise FidelityRefusal("Recorded JOB/checkpoint/export identity differs")
    mapping = {"job_id": "id", "cohort_id": "cohort_id", "family": "family", "procedure": "procedure", "seed": "seed"}
    if any(export.get(export_key) != job.get(job_key) for export_key, job_key in mapping.items()) \
            or export.get("dataset") != job["recipe"]["dataset"] or export.get("recipe_id") != job["recipe"]["id"] \
            or export.get("body") != job.get("body") or export.get("split") != job.get("split", 0) \
            or export.get("policy") != job["policy"]:
        raise FidelityRefusal("Selected export belongs to a different own fit")
    selected = export.get("selected_checkpoint", {})
    if any(selected.get(key) != checkpoint_pin[key] for key in ("bytes", "sha256")) \
            or Path(selected.get("path", "")).resolve() != Path(checkpoint_pin["path"]).resolve():
        raise FidelityRefusal("Export does not authenticate this exact selected checkpoint")
    epoch = export.get("selected_epoch")
    if type(epoch) is not int or epoch < 1 or checkpoint["selector"].get("selected_epoch") != epoch \
            or checkpoint["adapter"].get("completed_epochs") != epoch:
        raise FidelityRefusal("Selected checkpoint/adapter/export epoch differs")
    return job_hash, epoch


def replay_selected(request, root):
    admission = admit_cpu()  # before project metadata or model/data reads
    root = Path(root).resolve()
    if request.get("schema") != "own-selected-VALID-replay-v1" or request.get("role") != "VALID" \
            or request.get("TEST_access") is not False or request.get("device") != "cpu":
        raise FidelityRefusal("This bounded backend serves selected VALID on CPU only")
    paths = {key: verified(request[key], root) for key in ("job", "checkpoint", "selected_VALID")}
    if paths["checkpoint"].name != "selected.pt" or paths["selected_VALID"].name != "selected_VALID.pt" \
            or len({path.parent for path in paths.values()}) != 1 or paths["job"].name != "JOB.json":
        raise FidelityRefusal("Use the producer's own colocated JOB/selected.pt/selected_VALID.pt; no recovery/final substitution")
    import torch
    torch.set_num_threads(2)
    job = json.loads(paths["job"].read_text())
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    saved = torch.load(paths["selected_VALID"], map_location="cpu", weights_only=False)
    job_hash, epoch = selected_identity(job, checkpoint, saved, receipt(paths["checkpoint"]))
    profile, method, sources = source_preflight(job, checkpoint["runtime_binding"], root, request["replayer_manifest_sha256"])
    data_receipts = data_preflight(job, request.get("data_files"), root)
    started = time.perf_counter()
    adapter = None
    optimizer_attempts = [0]
    try:
        with runtime_namespace(profile, root) as runtime:
            runtime["common"].seed_all(torch, job["seed"])
            policy = runtime["bookkeeping"].Policy.from_dict(job["policy"], job["family"])
            construction_job = copy.deepcopy(job)
            construction_job["device"] = "cpu"  # identity remains bound to the original JOB above
            adapter = runtime["adapters"].create_adapter(torch, construction_job, policy, root)
            if binding_for(adapter, profile) != checkpoint["runtime_binding"]:
                raise FidelityRefusal("Reconstructed runtime/callback/extension/dependency binding differs")
            adapter.load_state(copy.deepcopy(checkpoint["adapter"]))
            def forbidden_step(*arguments, **keywords):
                optimizer_attempts[0] += 1
                raise FidelityRefusal("Optimizer transitions are forbidden during VALID replay")
            for optimizer in adapter.optimizers:
                optimizer.step = forbidden_step
            rng = restore_cpu_rng(torch, checkpoint["global_rng"])
            actual, diagnostics = actual_validation(torch, adapter, method["C6_mean_diagnostic"])
            parity = compare_export(torch, actual, saved)
            artifact = {"schema": "own-selected-VALID-reconstruction-v1", "role": "VALID", "TEST_accessed": False,
                "job_id": job["id"], "job_hash": job_hash, "family": job["family"], "procedure": job["procedure"],
                "selected_epoch": epoch, "runtime_profile": profile["id"], "runtime_binding": checkpoint["runtime_binding"],
                "admission": admission, "source_receipts": sources, "inputs": copy.deepcopy({k: request[k] for k in paths}),
                "data_receipts": data_receipts,
                "ids": actual["ids"].detach().cpu(), "labels": actual["labels"].detach().cpu(),
                "raw_logits": actual["outputs"].detach().cpu(), "loss": float(actual["loss"]),
                "metrics": {k: float(v) for k, v in actual["metrics"].items()}, "parity": parity,
                "C6_diagnostics": diagnostics, "selection_or_training_changed": False, "RNG_replay": rng,
                "optimizer_steps": 0, "optimizer_attempts": optimizer_attempts[0], "scientific_fits": 0,
                "elapsed_reconstruction_seconds": time.perf_counter() - started,
                "purpose": "actual selected-VALID source reconstruction; not final TEST serving authorization"}
            return artifact
    finally:
        if adapter is not None:
            adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    admit_cpu()
    root = Path(args.project_root).resolve()
    request_path, output = inside(args.request, root), inside(args.output, root)
    if output.exists():
        raise FidelityRefusal("Preserve prior replay output; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema": "selected-VALID-replay-result-v1", "request": receipt(request_path), "TEST_accessed": False,
              "optimizer_steps": 0, "scientific_fits": 0}
    try:
        artifact = replay_selected(json.loads(request_path.read_text()), root)
        import torch
        artifact_path = output / "reconstructed_VALID.pt"
        torch.save(artifact, artifact_path)
        report.update(status="passed", parity=artifact["parity"], artifact=receipt(artifact_path),
                      C6_mean_diagnostic_preserved=artifact["C6_diagnostics"] is not None)
    except Exception as error:
        import traceback
        report.update(status="refused_or_failed", failure={"type": type(error).__name__, "message": str(error),
                      "traceback": traceback.format_exc(limit=12)})
    report["elapsed_seconds"] = time.perf_counter() - started
    (output / "RESULT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
