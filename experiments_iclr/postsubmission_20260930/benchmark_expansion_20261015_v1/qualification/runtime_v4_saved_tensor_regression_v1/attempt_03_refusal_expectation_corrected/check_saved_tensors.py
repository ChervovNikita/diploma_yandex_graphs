"""Actual saved VALID tensors through v4 normalization/sidecar metadata paths.

Session/provider views and selection counters are explicitly simulated. No
native constructor, TRAIN, optimizer, source inference or selector is run.
"""
import copy
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
PHASE = EXPANSION.parent
ROOT = PHASE.parent
RUNTIME = EXPANSION / "runtime_v4"
sys.path.insert(0, str(RUNTIME))
import adapters
from common import digest, receipt


class SimulatedStop(Exception):
    pass


def main():
    import torch
    torch.set_num_threads(2)
    started = time.perf_counter()
    fixture_path = HERE / "ACTUAL_SAVED_TENSOR_FIXTURES.pt"
    acquisition = json.loads((HERE / "ACQUISITION.json").read_text())["receipt"]
    assert fixture_path.stat().st_size == acquisition["bytes"] and receipt(fixture_path)["sha256"] == acquisition["sha256"]
    bundle = torch.load(fixture_path, map_location="cpu", weights_only=False)
    checkpoints = json.loads((HERE / "ORIGINAL_INPUT_RECEIPTS.json").read_text())
    remote_phase = checkpoints["native_export"]["path"].split("/benchmark_expansion_20261015_v1/")[0]
    producer_files = ("run_fit.py", "adapters.py", "bookkeeping.py", "common.py", "controllers.py", "source_rng.py")
    runtime_sources = {name: receipt(RUNTIME / name) for name in producer_files}
    result = {"schema": "runtime-v4-saved-tensor-regression-v1", "status": "running", "torch": str(torch.__version__),
        "actual_fixture": receipt(fixture_path), "runtime_sources": runtime_sources, "cases": [],
        "native_constructions": 0, "native_forward_calls": 0, "TRAIN_calls": 0, "optimizer_steps": 0,
        "TEST_accessed": False, "selector_executed": False,
        "provider_views_and_selected_epoch_context_simulated": True,
        "selected_checkpoint_context": "unit regression uses the saved fixture-file receipt as a byte-authority proxy; no native selected export is claimed",
        "native_v4_stop_resume_qualification": "pending"}
    total_recorded_calls = total_refusals = 0

    class EmptyModel:
        def parameters(self): return ()

    for fixture in bundle["fixtures"]:
        family = fixture["family"]
        adapter_type = adapters.NodeAdapter if family == "node" else adapters.OGBAdapter
        view = adapter_type.__new__(adapter_type)
        view.torch, view.root = torch, ROOT
        view.job = copy.deepcopy(fixture["job"])
        view._job_authority = copy.deepcopy(fixture["job"])
        view._job_authority.setdefault("id", "saved-tensor-unit-context/" + fixture["name"])
        view.completed_epochs = 11  # Explicit unit counter, never a native epoch claim.
        view._validation_generation = view._validation_serial = 0
        view._same_bank_mean_cache = None
        view._anchors, view.optimizers, view.model = {}, [], EmptyModel()
        view.dependency_receipts = []  # Unit context; real construction supplies all native receipts.
        raw = copy.deepcopy(fixture["raw"])
        actual_saved = fixture["actual"]
        calls = [0]

        def recorded_validate():
            calls[0] += 1
            return copy.deepcopy(raw)

        session = SimpleNamespace(validate=recorded_validate, load_state=lambda _: None)
        if family == "node":
            ids = actual_saved["ids"]
            mask = torch.zeros(int(ids.max()) + 1, dtype=torch.bool); mask[ids] = True
            y = torch.zeros(len(mask), dtype=actual_saved["labels"].dtype); y[ids] = actual_saved["labels"]
            session.data = SimpleNamespace(val_mask=mask, y=y)
        else:
            ids = actual_saved["ids"]
            placeholder = float("nan") if actual_saved["labels"].is_floating_point() else 0
            y = torch.full((int(ids.max()) + 1, 1), placeholder, dtype=actual_saved["labels"].dtype)
            y[ids] = actual_saved["labels"]
            session.valid_view = SimpleNamespace(dataset=SimpleNamespace(_data=SimpleNamespace(y=y)))
        native_callback = PHASE / "reference_wrapper_redesign_20261013_v1/smoke_20261014" / (
            "node_port.py" if family == "node" else "ogb_port/session.py")
        view.callback_receipt = receipt(native_callback)
        view.extension_receipt = None
        if fixture["gated"]:
            declaration = copy.deepcopy(fixture["method_metadata"])
            for item in declaration["source_receipts"]:
                if item["path"].startswith(remote_phase + "/"):
                    item["path"] = str(PHASE) + item["path"][len(remote_phase):]
            setattr(session, fixture["provider"], declaration)
            view.extension_receipt = receipt(PHASE / fixture["extension_relative"])
        view.session = session
        runtime_binding = {"runtime_sha256": {n: r["sha256"] for n, r in runtime_sources.items()},
            "policy_sha256": None, "callback_sha256": view.callback_receipt["sha256"],
            "extension_sha256": None if view.extension_receipt is None else view.extension_receipt["sha256"],
            "dependency_sha256": {}}
        context = {"job_hash": digest(view._job_authority), "selected_epoch": 11,
            "selected_checkpoint": receipt(fixture_path), "runtime_binding": runtime_binding}
        value = view.validate()
        for key in ("outputs", "ids", "labels"):
            torch.testing.assert_close(value[key], actual_saved[key], atol=0, rtol=0, equal_nan=True)
        assert value["loss"] == actual_saved["loss"] and value["metrics"] == actual_saved["metrics"]
        row = {"name": fixture["name"], "family": family, "gated": fixture["gated"],
            "source_provenance": fixture["provenance"], "recorded_validate_calls_for_first_result": calls[0],
            "actual_outputs_ids_labels_exact": True, "refused_mutations": []}
        assert calls[0] == 1
        if not fixture["gated"]:
            assert "same_bank_mean_VALID" not in value and view.selected_mean_export(value, **context) is None
            row["original_ungated_path_preserved"] = True
        else:
            exported = view.selected_mean_export(value, **context)
            assert calls[0] == 1 and exported["extra_forward_calls"] == 0
            torch.testing.assert_close(exported["raw_logits"], raw["same_bank_mean_outputs"], atol=0, rtol=0)
            assert exported["validation"]["metrics"] == raw["same_bank_mean_metrics"]
            assert exported["selected_checkpoint"] == context["selected_checkpoint"] and exported["job_hash"] == context["job_hash"]
            row["same_call_mean_preserved_exactly"] = True

            def refuses(name, function, exceptions=(ValueError,)):
                nonlocal total_refusals
                try:
                    function()
                except exceptions as error:
                    row["refused_mutations"].append({"name": name, "exception_type": type(error).__name__}); total_refusals += 1
                else:
                    raise AssertionError("Accepted " + name)

            refuses("different JOB hash", lambda: view.selected_mean_export(value, **{**context, "job_hash": "0" * 64}))
            refuses("different selected epoch", lambda: view.selected_mean_export(value, **{**context, "selected_epoch": 12}))
            refuses("changed checkpoint bytes receipt", lambda: view.selected_mean_export(value, **{**context,
                "selected_checkpoint": {**context["selected_checkpoint"], "sha256": "0" * 64}}))
            wrong = copy.deepcopy(context); wrong["runtime_binding"]["runtime_sha256"]["adapters.py"] = "0" * 64
            refuses("different runtime source", lambda: view.selected_mean_export(value, **wrong))
            wrong = copy.deepcopy(context); wrong["runtime_binding"]["extension_sha256"] = "0" * 64
            refuses("different extension source", lambda: view.selected_mean_export(value, **wrong))
            for field in ("ids", "labels", "outputs"):
                changed = copy.deepcopy(value); changed[field].view(-1)[0] += 1
                refuses("modified actual " + field, lambda v=changed: view.selected_mean_export(v, **context))
            changed = copy.deepcopy(value); changed["same_bank_mean_VALID"]["outputs"].view(-1)[0] += .1
            refuses("modified mean logits", lambda: view.selected_mean_export(changed, **context))
            changed = copy.deepcopy(value); metric = next(iter(changed["same_bank_mean_VALID"]["metrics"]))
            changed["same_bank_mean_VALID"]["metrics"][metric] += .1
            refuses("modified mean metric", lambda: view.selected_mean_export(changed, **context))
            # A new recorded VALID transaction invalidates an older identical tensor return.
            old = value; value = view.validate()
            refuses("older VALID transaction", lambda: view.selected_mean_export(old, **context))
            adapters.BaseAdapter.load_state(view, {"session": {}, "completed_epochs": 11})
            refuses("stale cache after adapter load", lambda: view.selected_mean_export(value, **context))
            value = view.validate()
            session.train_update = lambda: (_ for _ in ()).throw(SimulatedStop("no native TRAIN"))
            if family == "ogb":
                class EmptyTrainingEntry:
                    def __iter__(self): raise SimulatedStop("no native TRAIN iterator")
                view._iterator = None; view.train_loader = EmptyTrainingEntry()
            try:
                view.train_epoch()
            except SimulatedStop:
                pass
            else:
                raise AssertionError("Unit training entry did not stop before native work")
            refuses("stale cache after TRAIN entry", lambda: view.selected_mean_export(value, **context))
            value = view.validate()
            session.validate = lambda: (_ for _ in ()).throw(SimulatedStop("recorded VALID failure"))
            try:
                view.validate()
            except SimulatedStop:
                pass
            refuses("stale cache after failed VALID", lambda: view.selected_mean_export(value, **context))
            session.validate = recorded_validate
            for remove in ("same_bank_mean_loss", "same_bank_mean_outputs", "VALID_node_ids" if family == "node" else "VALID_graph_ids"):
                original = raw.pop(remove)
                refuses("missing native field " + remove, lambda: view.validate(),
                    exceptions=(ValueError, KeyError) if remove == "VALID_graph_ids" else (ValueError,))
                raw[remove] = original
            originals = {key: raw.pop(key) for key in ("same_bank_mean_outputs", "same_bank_mean_loss", "same_bank_mean_metrics")}
            refuses("declared provider silently omits all mean fields", lambda: view.validate())
            raw.update(originals)
            declaration = getattr(session, fixture["provider"])
            prior = declaration["method_id"]; declaration["method_id"] = "wrong"
            refuses("different provider method", lambda: view.validate()); declaration["method_id"] = prior
            # C6's original callbacks may omit raw labels; provider-normalized saved labels remain authoritative.
            raw_label = raw.pop("VALID_labels", None); value = view.validate()
            assert view.selected_mean_export(value, **context)["same_call_authority"]["labels"] == view._tensor_identity(actual_saved["labels"])
            if raw_label is not None: raw["VALID_labels"] = raw_label
            row["provider_labels_support_original_C6_interface"] = True
        row["status"] = "passed"; row["total_recorded_validate_calls"] = calls[0]
        total_recorded_calls += calls[0]; result["cases"].append(row)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(status="passed", mutation_refusals=total_refusals, recorded_tensor_callback_calls=total_recorded_calls,
        elapsed_seconds=time.perf_counter() - started, process_user_seconds=usage.ru_utime,
        process_system_seconds=usage.ru_stime, process_peak_RSS=usage.ru_maxrss,
        process_peak_RSS_units="bytes" if platform.system() == "Darwin" else "KiB",
        no_new_native_forward_or_training=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
