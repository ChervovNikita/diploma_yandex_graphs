"""Own fabricated-case custody and injections; never mutate source/runtime/study."""
from pathlib import Path
import copy
import json
import shutil
from replay_gate import descriptor, require, ARMS
from replay_run import atomic_json
from replay_numeric import trusted_load
def local(path):
    return {**descriptor(path), "path": path.name}


def clone_fixture(fixture, target, api, torch):
    """Retain one separate authenticated input set per attempted QA case."""
    source = fixture["fixture_root"]
    shutil.copytree(source, target)
    context = copy.deepcopy(fixture["context"])
    identity = {**context["identity"], "family_lock_output_directory": str(target / "lock")}
    context["identity"] = identity
    state_api = api["pilot_state"]
    pins = []
    lock = json.loads((target / "lock" / "FAMILY_LOCK.json").read_text())
    lock["identity"] = identity
    old_pins = {(p["unit"], p["base_seed"]): p for p in context["release"]["unit_custody"]}
    for binding in lock["inputs"]:
        unit, seed = binding["unit"], binding["base_seed"]
        root = target / (unit + "_" + str(seed))
        prior = old_pins[(unit, seed)]
        journal = json.loads((root / "JOURNAL.json").read_text())
        payload = trusted_load(torch, root, journal["state_file"])
        payload["identity"] = identity
        journal["identity"] = identity
        journal["state_file"] = state_api.atomic_torch(root / journal["state_file"]["path"], payload)
        atomic_json(root / "JOURNAL.json", journal)
        complete = json.loads((root / "COMPLETE.json").read_text())
        complete["identity"] = identity
        for arm, receipt in complete["selected_checkpoints"].items():
            selected = trusted_load(torch, root, receipt)
            selected["identity"] = identity
            complete["selected_checkpoints"][arm] = state_api.atomic_torch(root / receipt["path"], selected)
            choice = json.loads((root / ("PRIVATE_SELECTION_" + arm + ".json")).read_text())
            choice["identity"] = identity
            choice["checkpoint"] = complete["selected_checkpoints"][arm]
            atomic_json(root / ("PRIVATE_SELECTION_" + arm + ".json"), choice)
            complete["selection_artifacts"][arm] = local(root / ("PRIVATE_SELECTION_" + arm + ".json"))
            cell = next(c for c in lock["cells"] if (c["arm"], c["base_seed"]) == (arm, seed))
            cell["checkpoint"] = {"output_directory": str(root), **complete["selected_checkpoints"][arm]}
        atomic_json(root / "COMPLETE.json", complete)
        ledger = json.loads((root / "ATTEMPTS.json").read_text())
        ledger["identity"] = identity
        atomic_json(root / "ATTEMPTS.json", ledger)
        physical = target / (unit + "_" + str(seed) + "_PHYSICAL.json")
        value = json.loads(physical.read_text())
        value["output_directory"] = str(root)
        atomic_json(physical, value)
        binding["output_directory"] = str(root)
        binding["complete_sha256"] = descriptor(root / "COMPLETE.json")["sha256"]
        pins.append({**prior, "output_directory": str(root), "terminal": local(root / "COMPLETE.json"),
                     "journal": local(root / "JOURNAL.json"), "attempts": local(root / "ATTEMPTS.json"), "physical_terminal": descriptor(physical)})
    context["release"]["unit_custody"] = pins
    save_lock(context, lock, target)
    context["synthetic_tensor_custody"] = descriptor(target / "FABRICATED_TENSORS.pt")
    return context


def save_lock(context, lock, target):
    path = target / "lock" / "FAMILY_LOCK.json"
    atomic_json(path, lock)
    context["release"]["family_lock"] = descriptor(path)
    for pin in context["release"]["unit_custody"]:
        root = Path(pin["output_directory"])
        atomic_json(root / "FAMILY_CLOSURE.json", {"schema": "ncnc-pilot-immutable-family-closure-v1", "identity": context["identity"],
                    "family_lock_path": str(path), "family_lock_sha256": context["release"]["family_lock"]["sha256"],
                    "status": lock["status"], "resume_or_seed_replacement_permitted": False})
        pin["closure"] = local(root / "FAMILY_CLOSURE.json")


def alter_own_payload(context, api, torch, kind):
    """Rebind deliberately invalid own fixture bytes to test numeric contracts."""
    pin = next(p for p in context["release"]["unit_custody"] if (p["unit"], p["base_seed"]) == ("native70", 4))
    root = Path(pin["output_directory"])
    state_api = api["pilot_state"]
    complete = json.loads((root / "COMPLETE.json").read_text())
    arm = ARMS[4]
    if kind == "selected_state":
        selected = trusted_load(torch, root, complete["selected_checkpoints"][arm])
        values = selected["states"][0]["models"][0]
        key = next(k for k, v in values.items() if torch.is_tensor(v) and v.is_floating_point())
        values[key] = values[key] + 1.
        complete["selected_checkpoints"][arm] = state_api.atomic_torch(root / complete["selected_checkpoints"][arm]["path"], selected)
    else:
        journal = json.loads((root / "JOURNAL.json").read_text())
        payload = trusted_load(torch, root, journal["state_file"])
        record = payload["state"]["fits"][0]
        if kind == "raw_score":
            record["epochs"][0]["VALID"]["score_digests"]["positive"] = "DELIBERATE_FABRICATED_WRONG_DIGEST"
        elif kind == "selected_metric":
            wrong = 1. if record["best"]["hits50"] != 1. else 0.
            record["best"]["hits50"] = wrong
            for epoch in record["epochs"]:
                epoch["private_hits50"] = wrong
            selected = trusted_load(torch, root, complete["selected_checkpoints"][arm])
            selected["selection"]["hits50"] = wrong
            complete["selected_checkpoints"][arm] = state_api.atomic_torch(root / complete["selected_checkpoints"][arm]["path"], selected)
        else:
            raise RuntimeError("Unknown admitted fabricated fault")
        journal["state_file"] = state_api.atomic_torch(root / journal["state_file"]["path"], payload)
        atomic_json(root / "JOURNAL.json", journal)
        pin["journal"] = local(root / "JOURNAL.json")
    choice = json.loads((root / ("PRIVATE_SELECTION_" + arm + ".json")).read_text())
    choice["checkpoint"] = complete["selected_checkpoints"][arm]
    if kind == "selected_metric":
        choice["selection"]["hits50"] = wrong
    atomic_json(root / ("PRIVATE_SELECTION_" + arm + ".json"), choice)
    complete["selection_artifacts"][arm] = local(root / ("PRIVATE_SELECTION_" + arm + ".json"))
    atomic_json(root / "COMPLETE.json", complete)
    pin["terminal"] = local(root / "COMPLETE.json")
    lock_path = Path(context["release"]["family_lock"]["path"])
    lock = json.loads(lock_path.read_text())
    binding = next(b for b in lock["inputs"] if (b["unit"], b["base_seed"]) == ("native70", 4))
    binding["complete_sha256"] = pin["terminal"]["sha256"]
    cell = next(c for c in lock["cells"] if (c["arm"], c["base_seed"]) == (arm, 4))
    cell["checkpoint"] = {"output_directory": str(root), **choice["checkpoint"]}
    if kind == "selected_metric":
        cell["selection"]["hits50"] = wrong
        cell["served_VALID_hits50"] = wrong
    save_lock(context, lock, lock_path.parent.parent)


def corrupt_after_metadata(context):
    pin = context["release"]["unit_custody"][0]
    root = Path(pin["output_directory"])
    complete = json.loads((root / "COMPLETE.json").read_text())
    receipt = next(iter(complete["selected_checkpoints"].values()))
    path = root / receipt["path"]
    path.write_bytes(path.read_bytes() + b"FABRICATED_CUSTODY_FAULT")


def corrupt_admitted_release(context):
    path = context["release_path"]
    path.write_bytes(path.read_bytes() + b"\n ")


def assert_failed_suppression(code, result):
    require(code != 0 and result["summary"] is None and len(result["cells"]) == 25, "Injected failure did not retain null full25 result")
    require(all(c["replayed_VALID_hits50"] is None and c["selected_hits50_equal"] is None for c in result["cells"]), "Injected failure disclosed a successful subset")


def make_historical_references_unavailable(context, api, torch):
    """Own fabricated copies only: valid immutable digests with no available arrays."""
    for pin in context["release"]["unit_custody"]:
        root = Path(pin["output_directory"])
        journal = json.loads((root / "JOURNAL.json").read_text())
        payload = trusted_load(torch, root, journal["state_file"])
        for record in payload["state"]["fits"]:
            for epoch in record["epochs"]:
                epoch["VALID"]["score_digests"] = {"positive": "0" * 64, "negative": "0" * 64}
        for candidate in payload["state"]["ensemble_candidates"]:
            for receipt in candidate.get("extra_VALID_evaluations", []):
                receipt["score_digests"] = {"positive": "0" * 64, "negative": "0" * 64}
        journal["state_file"] = api["pilot_state"].atomic_torch(root / journal["state_file"]["path"], payload)
        atomic_json(root / "JOURNAL.json", journal)
        pin["journal"] = local(root / "JOURNAL.json")


def inconsistent_I4_reference(context, api, torch):
    """Own fabricated copy: authenticate fixed anchor bytes whose pooled metric contradicts selection."""
    pin = next(p for p in context["release"]["unit_custody"] if (p["unit"], p["base_seed"]) == ("native_bank4", 0))
    root = Path(pin["output_directory"])
    complete = json.loads((root / "COMPLETE.json").read_text())
    selected = json.loads((root / complete["selection_artifacts"][ARMS[1]]["path"]).read_text())["selection"]["hits50"]
    positive_value = 0. if selected == 1. else 2.
    digests = {"positive": api["pilot_data"].tensor_sha(torch.full((60084,), positive_value, dtype=torch.float32)),
               "negative": api["pilot_data"].tensor_sha(torch.ones(100000, dtype=torch.float32))}
    journal = json.loads((root / "JOURNAL.json").read_text())
    payload = trusted_load(torch, root, journal["state_file"])
    for record in payload["state"]["fits"]:
        for epoch in record["epochs"]: epoch["VALID"]["score_digests"] = dict(digests)
    for candidate in payload["state"]["ensemble_candidates"]:
        for receipt in candidate.get("extra_VALID_evaluations", []): receipt["score_digests"] = dict(digests)
    journal["state_file"] = api["pilot_state"].atomic_torch(root / journal["state_file"]["path"], payload)
    atomic_json(root / "JOURNAL.json", journal); pin["journal"] = local(root / "JOURNAL.json")
    return {"positive_value": positive_value, "negative_value": 1., "saved_selected_hits50": selected, "typed_digests": digests}


def assert_provisional_receipts(context):
    for name in ("STATUS.json", "AUDIT_ATTEMPT.json", "AUDIT_RESULT.json"):
        value = json.loads((context["output"] / name).read_text())
        require(value["status"] == "FINALIZATION_PENDING", "Premature final terminal published before final custody")
    result = json.loads((context["output"] / "AUDIT_RESULT.json").read_text())
    require(result["summary"] is None and all(c["replayed_VALID_hits50"] is None for c in result["cells"]), "Provisional publication contains metrics")


def assert_terminal_receipts(context, result):
    """The result is authoritative; status mirrors it; attempt is distinctly provisional."""
    output = context["output"]
    attempt = json.loads((output / "AUDIT_ATTEMPT.json").read_text())
    status = json.loads((output / "STATUS.json").read_text())
    actual_result = json.loads((output / "AUDIT_RESULT.json").read_text())
    require(actual_result == result and actual_result["terminal_authority"] == "AUDIT_RESULT.json_only", "Authoritative result differs")
    require(status["status"] == result["status"] and status["record_kind"] == "terminal_mirror_of_authoritative_result" and status["final_terminal_authority"] is False, "Final status mirror contradicts authoritative terminal")
    require(status["authoritative_result_receipt"] == descriptor(output / "AUDIT_RESULT.json"), "Final status/result bytes binding differs")
    require(attempt["status"] == "FINALIZATION_PENDING" and attempt["record_kind"] == "provisional_accounting_before_final_custody" and attempt["final_terminal_authority"] is False and attempt["authoritative_terminal_file"] == "AUDIT_RESULT.json", "Attempt receipt falsely publishes a final terminal")
    require(result["audit_attempt_receipt"] == descriptor(output / "AUDIT_ATTEMPT.json") and attempt["work"] == result["work"] == status["work"], "Retained receipt custody/work differs")
    details_path = output / "PRIVATE_AUDIT_DETAILS.json"
    if details_path.exists():
        require(result["private_audit_details_receipt"] == descriptor(details_path), "Private return evidence/result binding differs")
        details = json.loads(details_path.read_text())
        require(sum(r["returned"] for r in details["slots"]) == result["work"]["returned"] and sum(r["validated"] for r in details["slots"]) == result["work"]["completed_validated"], "Private slot return/validation evidence counts differ")


def assert_unvalidated_return(context, *, slot, expected_returned_rows, expected_validated_rows):
    details = json.loads((context["output"] / "PRIVATE_AUDIT_DETAILS.json").read_text())
    row = next(r for r in details["slots"] if r["slot"] == slot)
    require(row["returned"] is True and row["validated"] is False and row["validation_status"] == "RETURNED_GUARD_FAILED" and row["guard_failure"]["phase"] == "post_return", "Failed returned slot was marked validated or omitted")
    original = row["returned_original_receipt"]
    require(type(original) is dict and original["positive_queries"] == 60084 and original["negative_queries"] == 100000 and original["encoder_calls"] == 1 and original["query_batches"] == [1, 1] and set(original["score_digests"]) == {"positive", "negative"}, "Original full scorer receipt was not retained before the failing guard")
    require(sum(r["returned"] for r in details["slots"]) == expected_returned_rows and sum(r["validated"] for r in details["slots"]) == expected_validated_rows, "Failed returned slot retention denominator differs")
