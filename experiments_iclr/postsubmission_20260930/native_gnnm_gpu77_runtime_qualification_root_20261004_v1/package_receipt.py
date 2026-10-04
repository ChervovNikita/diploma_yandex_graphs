"""Package completed native qualification receipts; no training or network calls."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    with (ROOT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    local = read("native_reference_ordinary_native_gnnm_local.json")
    global_ = read("native_reference_ordinary_native_gnnm_global.json")
    qualification = read("QUALIFICATION.json")
    monitor = read("owned_monitor001/MONITOR.json")
    launch = read("DETACHED_LAUNCH.json")
    terminal = read("SUPERVISOR_TERMINAL.json")
    review = read("CHECKER_REVIEW.json")
    inventory = read("STAGE_INVENTORY.json")
    stage_receipt = read("STAGE_RECEIPT.json")
    assert len(inventory) == stage_receipt["files"] == 55
    assert sum(record["bytes"] for record in inventory) == stage_receipt["bytes"] == 440188
    assert len({record["path"] for record in inventory}) == len(inventory)
    for record in inventory:
        relative = record["path"]
        path = PHASE / relative.removeprefix("staged_phase/") if relative.startswith("staged_phase/") else ROOT / relative
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        assert path.stat().st_size == record["bytes"]
        assert digest(path) == record["sha256"], relative
    for record in monitor["files"]:
        path = ROOT / "owned_monitor001" / record["path"]
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
        if (ROOT / record["path"]).is_file():
            assert digest(ROOT / record["path"]) == record["sha256"]
    assert qualification["expected_cases"] == len(qualification["cases"]) == 2
    assert qualification["all_component_checks_complete"]
    assert terminal["completed"]
    assert terminal["identity"]["PID"] == launch["supervisor_PID"] == 3281320
    assert terminal["identity"]["start_ticks"] == launch["supervisor_start_ticks"] == 1729210361
    assert not monitor["supervisor_state"]["exists"]
    assert monitor["selected_GPU_inventory"] == "GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998, 66913 MiB\n"
    assert digest(ROOT / "check_one.py") == review["checker_sha256"]
    assert digest(ROOT / "supervisor.py") == review["supervisor_sha256"]
    bank = json.loads((PHASE / "graph_view_gpu77_runtime_qualification_root_20261004_v1/bank_tied_persistent_local.json").read_text())
    for case, result in zip(qualification["cases"], (local, global_)):
        stage = case["stage"]
        assert case["exit_code"] == 0 and case["child_closed_and_reaped"]
        assert case["result"] == result
        assert case["result_sha256"] == digest(ROOT / ("native_reference_ordinary_native_gnnm_" + stage + ".json"))
        started = read("owned_monitor001/" + stage + "_CHILD_STARTED.json")
        physical = read("owned_monitor001/" + stage + "_PHYSICAL_TERMINAL.json")
        assert physical == case
        assert started["identity"] == case["identity"]
        assert case["identity"]["parent"] == launch["supervisor_PID"]
        assert case["identity"]["PID"] == monitor["child_states"][stage]["PID"]
        assert case["identity"]["start_ticks"] == monitor["child_states"][stage]["expected_start_ticks"]
        assert not monitor["child_states"][stage]["exists"]
        shared_fields = set(result["runtime"]) & set(bank["runtime"])
        assert all(result["runtime"][key] == bank["runtime"][key] for key in shared_fields)
        assert result["runtime"]["physical_gpu_UUIDs"] == bank["runtime"]["physical_gpu_uuids"]
        assert result["runtime"]["CUDA_VISIBLE_DEVICES"] == bank["runtime"]["cuda_visible_devices"]
        assert result["runtime"]["deterministic_algorithms"] == bank["deterministic_algorithms"] is True
        assert set(result["runtime"]) - shared_fields == {"physical_gpu_UUIDs", "CUDA_VISIBLE_DEVICES", "deterministic_algorithms"}
        assert set(bank["runtime"]) - shared_fields == {"physical_gpu_uuids", "cuda_visible_devices"}
        assert result["source_manifest_sha256"] == "fbe1aec5eafb029fce9c9d42ebf9a5b94b98cfbc2ce9881502f0e3fc8e192ca6"
        assert result["protocol_sha256"] == "9da2f5ebed5660d09b77ef4a620caa50bda81e4dc8ee999444aefc8f87e58524"
        assert result["checker_sha256"] == digest(ROOT / "check_one.py")
        assert result["input_bindings_sha256"] == digest(ROOT / "INPUT_BINDINGS.json")
        assert result["coverage_sha256"] == "0e05618da1fe4916eebd77ccadd26b8aa476fb23bd872f20b6fa5391c48bf74c"
        assert result["GPU_UUID"] == launch["selected_GPU_UUID"]
        assert result["training_updates_checked"] == 3
        assert result["training_forwards_and_backwards"] == 12
        assert result["native_evaluation_forwards"] == 8
        assert result["parameters"] == 9121536
        assert result["component_stage_fresh_initialization"]
        for key in ("function_replay", "next_update_replay"):
            assert result[key]["exact"] and result[key]["maximum_absolute_error"] == 0.0
        assert result["scientific_training_updates"] == 0
        for key in ("VALIDATION_or_TEST_targets_read", "predictive_values_reported", "checkpoint_or_logits_saved", "eligible_as_donor", "exact_whole_schedule_or_competence_qualification"):
            assert result[key] is False
        for descriptor in result["source_provenance"]:
            original = PHASE / descriptor["declared_phase_path"]
            assert digest(original) == descriptor["sha256"] and original.stat().st_size == descriptor["bytes"]
    for object_ in (qualification, launch, monitor, terminal):
        assert not object_["job_signals_sent"]
    for object_ in (qualification, launch):
        assert not object_["automatic_retry"]
    fit_hours = (200 * (local["step_seconds"] + local["evaluation_seconds"]) + 2500 * (global_["step_seconds"] + global_["evaluation_seconds"])) / 3600
    snapshot_hours = (200 * local["snapshot_seconds"] + 2500 * global_["snapshot_seconds"]) / 3600
    verification = dict(
        UTC=datetime.now(timezone.utc).isoformat(),
        status="PASS_TWO_TRAIN_ONLY_NATIVE_GNNM_FULL_SHAPE_COMPONENTS",
        source_manifest_sha256=local["source_manifest_sha256"],
        protocol_sha256=local["protocol_sha256"],
        qualification_sha256=digest(ROOT / "QUALIFICATION.json"),
        runtime_packages_and_policy_equal_after_explicit_field_aliases=True,
        exact_function_replay=True,
        exact_next_update_replay=True,
        all_owned_children_closed_and_reaped=True,
        supervisor_and_child_identities_absent_at_final_monitor=True,
        final_monitor_reference="owned_monitor001/MONITOR.json",
        final_monitor_sha256=digest(ROOT / "owned_monitor001/MONITOR.json"),
        source_payloads_verified=55,
        TRAIN_only_engineering_updates=6,
        scientific_training_updates=0,
        predictive_values_or_VALTEST_targets_read=False,
        checkpoints_or_logits_saved=False,
        full_schedule_measured=False,
        cross_runtime_bitwise_initialization_equality_claimed=False,
        illustrative_native_GNNM_fit_GPU_hours=fit_hours,
        illustrative_snapshot_every_update_extra_hours=snapshot_hours,
        job_signals_sent=False,
        automatic_retry=False,
    )
    write("VERIFICATION.json", verification)
    report = f"""# 18.77 native GNNM component qualification

**Both native GNNM component cases pass exact CUDA function and next-update replay.** They use fresh local and global components, full Amazon Ratings graph/features and all 12,246 official TRAIN labels on split 0. They establish correct replay and a bounded resource estimate; predictive quality remains untested by this packet.

| Component | First training update (s) | Native committee evaluation (s) | Snapshot (s) | Peak allocated (GB, decimal) | Peak reserved (GB, decimal) |
|---|---:|---:|---:|---:|---:|
| Fresh local | {local['step_seconds']:.3f} | {local['evaluation_seconds']:.3f} | {local['snapshot_seconds']:.3f} | {local['peak_allocated_bytes']/1e9:.3f} | {local['peak_reserved_bytes']/1e9:.3f} |
| Fresh global | {global_['step_seconds']:.3f} | {global_['evaluation_seconds']:.3f} | {global_['snapshot_seconds']:.3f} | {global_['peak_allocated_bytes']/1e9:.3f} | {global_['peak_reserved_bytes']/1e9:.3f} |

The two-case supervisor wall time was {qualification['wall_seconds']:.3f} seconds. Each case calls the unchanged native v2 training kernel for three engineering updates: four streaming member cross-entropies/backwards per update, divided by four, followed by one Adam update. Two native committee evaluations perform eight forwards. Snapshot/restore and next-update comparison cover model parameters, Adam state, gradients and RNG. Function replay and next-update replay have zero maximum absolute error; the local and global next-update trees contain 524 and 576 tensors, respectively. The global component starts freshly initialized at update 201 with the global flag enabled; a local-to-global trained-state handoff is not measured.

## Custody and closure

The native v2 source manifest is `{local['source_manifest_sha256']}` and its protocol is `{local['protocol_sha256']}`. The checker invokes `native_driver.native_train_step` and never invokes the full scientific driver. All 55 source/context/checker payloads (440,188 bytes) were staged under this packet's owned remote `staged_phase`. The packaging verification rechecks each local source byte against the staged inventory and all returned monitor receipts. It makes no edit to source, loss or initialization recipes.

Existing remote input artifacts matched their frozen hashes before execution: public graph/features `19757299bcfd9e493e9ceae9e73248753ab1e773ccc1f6b57c6fadf8c843310f`, split 0 TRAIN labels `9d700d1897bbf73ec70a4c3158e07d6b42909603e28ac54716800f3d4883f749`, and data manifest `8d565c20c6d669c2577e9fb8c5405575a6ed97a9964f91818b052e43cd0379bf`. Coverage equals the byte-bound official TRAIN coverage receipt. VALIDATION/TEST memberships are role metadata only; no heldout targets or scores are opened. Engineering states and raw logits remain in memory only. No donor, scientific checkpoint or predictive result is produced.

The runtime descriptors equal the completed bank qualification on `peptide`: Python 3.12.11, Torch 2.7.1/CUDA 12.6, NumPy 1.26.4, PyG 2.4.0, strict deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 disabled. Package initializer file hashes and executed model source hashes are recorded. These descriptors do not fingerprint every linked binary or imply equality with another runtime's initialized weights/trajectories.

Supervisor PID 3281320/start ticks 1729210361 launched once. Children PID 3281326/start ticks 1729210372 and PID 3281366/start ticks 1729214206 each exited 0 and were closed/reaped. The final monitor at {monitor['UTC']} confirms the exact supervisor and child identities are absent and GPU0 free memory is 66,913 MiB. Only authorized physical GPU0 `GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998` executed these cases, co-resident with the root-reported DDI queue. No retry, timeout or signal was used.

## Resource implication for the frozen comparison

Using the measured fresh-component timings, 200 local + 2,500 global updates with a native committee evaluation after every update gives **an illustrative {fit_hours:.2f} GPU-hours per native GNNM fit**. Snapshotting after every update would add about {snapshot_hours:.2f} hours before preparation, checkpoint I/O, queue interference and other omissions. This is arithmetic from short engineering components, not a measured full schedule, trained-state throughput or upper bound.

The planned 30-fit accounting is 12 augmented-bank fits + 15 ordinary reference fits + 3 native GNNM fits. The earlier tied-bank estimate was 14.06 GPU-hours per augmented fit. Applying that rate to all 12 augmented fits plus this native rate to three fits gives about {12*14.062649837798542+3*fit_hours:.1f} illustrative GPU-hours **before the 15 ordinary references and omissions**; untied bank throughput remains unmeasured and can be slower. A total 30-fit completion ETA is unsupported until the remaining kernels are measured. The bank-only 12-fit arithmetic was about 168.8 GPU-hours, or 3.52 days on two fully available GPUs.

Use one owned fit per GPU, balance competing arms and split/seed blocks across the same 18.77 runtime, and complete frozen paired comparisons before inspecting their development outcomes. Source/data custody and comparator competence requirements remain the parent's responsibility before a scientific launch. Accuracy is the selection objective; time and memory are measured costs. This packet changes no canonical ledger, scientific queue, manuscript or original paper scores and launches no predictive training.
"""
    with (ROOT / "REPORT.md").open("x") as stream:
        stream.write(report)
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name in ("MANIFEST.json", "SEAL.json"):
            continue
        assert not path.is_symlink()
        files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path)))
    write("MANIFEST.json", dict(schema="native-gnnm-gpu77-engineering-receipt-manifest-v1", files=files, excluded="MANIFEST.json/SEAL.json self-links; incidental __pycache__"))
    write("SEAL.json", dict(schema="native-gnnm-gpu77-engineering-receipt-seal-v1", manifest_sha256=digest(ROOT / "MANIFEST.json"), immutable=True))
    seal = read("SEAL.json")
    assert digest(ROOT / "MANIFEST.json") == seal["manifest_sha256"]
    for record in read("MANIFEST.json")["files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
    print(json.dumps(dict(status="SEALED_AND_VERIFIED", qualification_sha256=digest(ROOT / "QUALIFICATION.json"), manifest_sha256=digest(ROOT / "MANIFEST.json"), seal_sha256=digest(ROOT / "SEAL.json"), files=len(files), illustrative_native_GNNM_fit_GPU_hours=fit_hours)))


if __name__ == "__main__":
    main()
