"""Stdlib-only synthetic metadata review; no real family inputs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
SOURCE = REVIEW.parent / "buddy_paired_analysis_v1" / "analyze.py"
source_bytes = SOURCE.read_bytes()
spec = importlib.util.spec_from_file_location("paired_analysis_review_subject", SOURCE)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return digest(path)


def prepare(case, mutation):
    phase = REVIEW / "synthetic_phase" / case
    subject.PHASE = phase
    subject.PACKET = phase / "buddy_gpu77_postfamily_eval_preparation_v4"
    subject.FAMILY = phase / "buddy_gpu77_resource_family_launcher_v3/root_family_v2"
    rows, results = [], []
    common_test = "b" * 64
    for ai, arm in enumerate(subject.ARMS):
        for seed in subject.SEEDS:
            folder = subject.FAMILY / "runs" / f"{arm}_seed{seed}"
            checkpoint = hashlib.sha256(f"synthetic:{arm}:{seed}".encode()).hexdigest()
            rows.append(dict(arm=arm, seed=seed,
                             run_directory=str(subject.REMOTE / folder.relative_to(phase)),
                             selected_checkpoint_sha256=checkpoint))
    lock_path = subject.PACKET / "root_lock_v1/FAMILY_LOCK.json"
    lock = dict(schema="buddy-family-lock-v2", status="family_locked", runs=rows,
                cache_manifest_sha256="c" * 64, config_sha256="d" * 64,
                implementation_hashes={"synthetic": "e" * 64})
    lock_hash = write(lock_path, lock)
    for ai, row in enumerate(rows):
        arm, seed = row["arm"], row["seed"]
        # Fabricated metadata values, unrelated to actual models or scores.
        value = .5 + .01 * subject.ARMS.index(arm) + .001 * seed
        result = dict(arm=arm, seed=seed, hits50=value, prediction_seconds=.1,
                      family_lock_sha256=lock_hash,
                      checkpoint_sha256=row["selected_checkpoint_sha256"],
                      test_manifest_sha256=common_test)
        result_path = subject.FAMILY / "runs" / f"{arm}_seed{seed}" / "final_test.json"
        result_hash = write(result_path, result)
        results.append(dict(arm=arm, seed=seed, result_sha256=result_hash,
                            prediction_seconds=.1))
    terminal_path = subject.FAMILY / "FAMILY_LAUNCH_RECEIPT.json"
    terminal = dict(schema="buddy77-family-launch-receipt-v2", exit_code=0,
                    family_cells=15, optimizer_fits=24, test_access=False)
    terminal_hash = write(terminal_path, terminal)
    audit_path = subject.PACKET / "root_lock_v1/LOCK_AUDIT.json"
    audit = dict(schema="buddy77-production-family-lock-audit-v1",
                 status="all15_production_lock_and_selected_checkpoints_audited",
                 all_selected_checkpoints_runtime_validated=True, exit_code=0,
                 family_lock_sha256=lock_hash, locked_runs=copy.deepcopy(rows))
    audit_hash = write(audit_path, audit)
    receipt_path = subject.PACKET / "root_eval_v1/EVALUATION_RECEIPT.json"
    receipt = dict(schema="buddy77-postfamily-evaluation-receipt-v2",
                   status="all15_locked_cells_scored_once", scoring_exit_code=0,
                   family_lock_sha256=lock_hash, lock_audit_sha256=audit_hash,
                   terminal_family_receipt_sha256=terminal_hash,
                   final_results=copy.deepcopy(results),
                   test_payload_opened=True, scoring_attempted=True,
                   evaluation_preparation_manifest_sha256="fd77b2f09418070e3e8d358231e4a98388d3e81796d513ba2165c9755ddeab22",
                   source_manifest_sha256="c479cedbd244ff645c7ee822625120b9f809d5082f4d9d4ace265237b5712e0f",
                   final_cache_qualification=dict(test_manifest_sha256=common_test,
                                                 full_candidate_order_and_shapes_qualified=True,
                                                 test_topology_added_to_graph=False))
    if mutation == "mixed_test_manifest":
        path = subject.FAMILY / "runs/native1024_seed0/final_test.json"
        payload = json.loads(path.read_text())
        payload["test_manifest_sha256"] = "f" * 64
        result_hash = write(path, payload)
        receipt["final_results"][0]["result_sha256"] = result_hash
    elif mutation == "wrong_audit_lock":
        audit["family_lock_sha256"] = "0" * 64
        receipt["lock_audit_sha256"] = write(audit_path, audit)
    elif mutation == "missing_audit_cohort":
        audit.pop("locked_runs")
        receipt["lock_audit_sha256"] = write(audit_path, audit)
    elif mutation == "false_scoring_exit":
        receipt["scoring_exit_code"] = False
    elif mutation == "missing_result":
        receipt["final_results"].pop()
    elif mutation == "wrong_checkpoint":
        path = subject.FAMILY / "runs/native1024_seed0/final_test.json"
        payload = json.loads(path.read_text())
        payload["checkpoint_sha256"] = "0" * 64
        result_hash = write(path, payload)
        receipt["final_results"][0]["result_sha256"] = result_hash
    elif mutation == "duplicate_result":
        receipt["final_results"][-1] = copy.deepcopy(receipt["final_results"][0])
    write(receipt_path, receipt)


outcomes = []
cases = [
    ("baseline", None, True),
    ("mixed_test_manifest", "mixed_test_manifest", False),
    ("wrong_audit_lock", "wrong_audit_lock", False),
    ("missing_audit_cohort", "missing_audit_cohort", False),
    ("false_scoring_exit", "false_scoring_exit", False),
    ("missing_result", "missing_result", False),
    ("wrong_checkpoint", "wrong_checkpoint", False),
    ("duplicate_result", "duplicate_result", False),
]
for case, mutation, expected_accept in cases:
    prepare(case, mutation)
    try:
        scores, inputs = subject.collect()
    except (ValueError, KeyError) as error:
        accepted, message = False, str(error)
    else:
        accepted, message = True, f"accepted {len(scores)} synthetic cells"
    outcomes.append(dict(case=case, accepted=accepted, expected_accept=expected_accept,
                         matches_expectation=accepted == expected_accept, message=message))
assert SOURCE.read_bytes() == source_bytes, "analyzer source changed during fixture review"
result = dict(schema="buddy-paired-synthetic-collector-review-v1",
              synthetic_metadata_only=True, actual_results_opened=False,
              source_sha256=hashlib.sha256(source_bytes).hexdigest(),
              cases=outcomes)
write(REVIEW / "SYNTHETIC_RESULTS.json", result)
print(json.dumps(result, indent=2))
