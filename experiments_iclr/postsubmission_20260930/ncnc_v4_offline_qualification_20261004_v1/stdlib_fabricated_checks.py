"""Four bounded fabricated stdlib helper checks; not the sealed22-case suite."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import sys

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / "ncnc_selected_checkpoint_metric_audit_v4_source_20261004"
PIN = "fd59e048a53695c945c6cfb613a77ce0e4e21c2631ce56085348c3941f5ec6d6"
assert (OUT / "BOUNDED_RATIONALE.json").exists()
assert sha256((SOURCE / "MANIFEST.json").read_bytes()).hexdigest() == PIN
manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
for row in manifest["files"]:
    value = (SOURCE / row["path"]).read_bytes()
    assert len(value) == row["bytes"] and sha256(value).hexdigest() == row["sha256"]
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
from audit_contract import SlotLedger, schedule, terminal
from replay_numeric import private_receipt
from replay_contract import public_cells

cases = []
try:
    rows = schedule()
    assert len(rows) == 120
    assert Counter(row["route"] for row in rows) == {"native64": 75, "native70": 15, "private": 15, "pooled_after_clamp": 15}
    cases.append({"case": "literal_slot_schedule", "status": "PASS", "slots_metadata": 120})
    tuples = {}
    keys = ("attempted", "entered_original_scorer", "returned", "completed_validated")
    for name, last_events, expected in (
        ("late_before_scorer", 1, (120,119,119,119)),
        ("late_original_scorer_error", 2, (120,120,119,119)),
        ("late_after_scorer", 3, (120,120,120,119))):
        ledger = SlotLedger()
        for slot in range(1, 120):
            for key in keys: ledger.event(slot, key)
        for key in keys[:last_events]: ledger.event(120, key)
        ledger.fail(120, RuntimeError("FABRICATED_LOCAL_HELPER_FAULT"))
        actual = tuple(ledger.counts()[key] for key in keys)
        assert actual == expected
        tuples[name] = list(actual)
    cases.append({"case": "late_ledger_tuples_with_fabricated_events", "status": "PASS", "tuples": tuples,
                  "events_are_scorer_invocations": False})
    malformed = {"wall_seconds": float("nan"), "object": object()}
    malformed["cycle"] = malformed
    safe = private_receipt(malformed)
    assert safe["wall_seconds"] == {"nonfinite_float": "nan"}
    assert safe["cycle"]["unavailable_node"] == "cyclic_receipt"
    assert safe["object"]["tensor_or_object_body_exported"] is False
    json.dumps(safe, allow_nan=False)
    cases.append({"case": "malformed_fabricated_receipt_safe_json", "status": "PASS"})
    assert terminal([{"category": "FAILED_STRICT_METRIC"}, {"category": "FAILED_I4_REFERENCE_METRIC"}]) == "FAILED_I4_REFERENCE_METRIC"
    assert terminal([{"category": "FAILED_I4_REFERENCE_METRIC"}, {"category": "FAILED_FINAL_CUSTODY"}]) == "FAILED_FINAL_CUSTODY"
    cells = [{"arm": arm, "base_seed": seed, "status": "PASS", "replayed_VALID_hits50": 0.5, "selected_hits50_equal": True}
             for arm in ("native_single_64", "independent_native_4", "factorized_private_4", "factorized_pooled_after_clamp_4", "native_single_70") for seed in range(5)]
    suppressed = public_cells(cells, full_pass=False)
    assert len(suppressed) == 25 and all(row["replayed_VALID_hits50"] is None and row["selected_hits50_equal"] is None for row in suppressed)
    cases.append({"case": "failure_precedence_and_all25_null_public_helper", "status": "PASS"})
    assert "torch" not in sys.modules
    status = "PASS_BOUNDED_STDLIB_HELPERS_ONLY"
except Exception as error:
    status = "FAILED_BOUNDED_STDLIB_HELPERS"
    cases.append({"case": "exception", "status": "FAILED", "type": type(error).__name__, "condition": str(error)})
result = {"schema": "ncnc-v4-local-fabricated-stdlib-helper-checks-v1", "status": status,
    "source_manifest_sha256": PIN, "cases": cases, "checks_count": 4,
    "sealed_qualification_cases_executed": 0, "sealed_qualification_case_count": 22,
    "sealed_qualification_suite_slots_planned": 1920, "actual_scorer_work": {"attempted": 0, "entered_original_scorer": 0, "returned": 0, "completed_validated": 0},
    "new_training_updates": 0, "fixture_creation_calls": 0, "torch_imported": "torch" in sys.modules,
    "CUDA_invocation": False, "fabricated_model_checkpoint_tensor_fixture_opened": False,
    "scientific_data_checkpoint_score_TEST_read": False, "remote_access": False,
    "ordinary_runtime_qualified": False, "sealed22_case_qualification_PASS": False,
    "actual_python_executable": sys.executable, "actual_python_version": sys.version,
    "imported_v4_modules": [{"module": name, "path": str(Path(module.__file__).resolve()), "sha256": sha256(Path(module.__file__).read_bytes()).hexdigest()}
        for name, module in sorted(sys.modules.items()) if name.startswith(("audit_contract", "replay_")) and getattr(module, "__file__", None)]}
target = OUT / "STDLIB_HELPER_RESULT.json"
if target.exists(): raise RuntimeError("Fresh result already exists")
target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(result))
raise SystemExit(0 if status == "PASS_BOUNDED_STDLIB_HELPERS_ONLY" else 1)
