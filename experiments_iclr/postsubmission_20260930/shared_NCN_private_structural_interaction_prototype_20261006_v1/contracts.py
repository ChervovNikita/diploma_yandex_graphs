"""Narrow prospective recipe/admission contract, shared by existing-style drivers."""
import json
import custody

TRAIN_PURPOSE = "TRAIN_VALID_structural_interaction_fit"
COST_PURPOSE = "TRAIN_only_structural_discarded_complete_cycle_cost"
QUALIFY_PURPOSE = "TRAIN_only_structural_discarded_first_fixture_qualification"
ARMS = ("structural_private", "structural_tied", "informed_single",
        "informed_independent4", "structure_blind", "count_only")
IDENTITY = ("arm", "rule", "geometry", "seed", "factor_seed", "structural_seed",
            "message_width", "message_hidden", "chunk_size", "paired_seed_block", "outer_size", "inner_size")


def recipe():
    return json.loads((custody.ROOT/"FIXED_RECIPE.json").read_text())


def validate_recipe(job, *, qualification=False):
    fixed = recipe()
    if job.get("rule") != "ordinary_joint" or job.get("geometry") != "endpoint":
        raise ValueError("Fixed original ordinary BCE/endpoint recipe required")
    if job.get("loss") != fixed["loss"] or job.get("fixed_recipe_sha256") != custody.sha(custody.ROOT/"FIXED_RECIPE.json"):
        raise ValueError("Exact original BCE source recipe differs")
    if qualification:
        if job.get("seed") != 0 or job.get("fits_authorized") is not False or job.get("VALID_values_access") is not False:
            raise ValueError("Qualification is a discarded seed0 TRAIN fixture only")
        expected = next(cell for cell in fixed["cells"] if cell["seed"] == 0)
    else:
        expected = next((cell for cell in fixed["cells"] if cell["cell_id"] == job.get("cell_id")), None)
        if expected is None: raise ValueError("New fit/cost absent from six-control three-seed recipe")
    for key in IDENTITY:
        if qualification and key == "arm": continue
        if job.get(key) != expected[key]: raise ValueError("Prospective recipe identity differs: "+key)
    return fixed


def admissible(job):
    validate_recipe(job)
    gate = job.get("training_step_gate", {})
    if gate.get("approved") is not True: raise ValueError("New structural FP32 Adam/source gate not admitted")
    path = custody.phase_file(gate["path"])
    if custody.sha(path) != gate["sha256"]: raise ValueError("New structural gate bytes changed")
    receipt = json.loads(path.read_text())
    if receipt.get("passed") is not True or receipt.get("scope") != "discarded_FP32_structural_training_step_qualification":
        raise ValueError("Only the new structural first-fixture gate may admit this source")
    for key in ("source_manifest_sha256", "available_manifest_sha256", "fixed_recipe_sha256", "physical_gpu_uuid", "target"):
        if receipt.get(key) != job.get(key): raise ValueError("Structural gate binding differs: "+key)
    if set(receipt.get("architectures", {})) != set(ARMS) or not all(row.get("passed") is True for row in receipt["architectures"].values()):
        raise ValueError("All six structural architecture gates are required on this GPU")
    if receipt.get("fixture") != {"outer_size": job["outer_size"], "inner_size": job["inner_size"], "cycle": 1, "episode": 1}:
        raise ValueError("Structural first-fixture geometry differs")


def freeze_schedule(job, purpose):
    if purpose == COST_PURPOSE:
        if job.get("fits_authorized") is not False or job.get("VALID_values_access") is not False or job.get("cycles") != 1 or job.get("result_based_early_stop") is not False:
            raise ValueError("Cost gate is one complete discarded TRAIN cycle only")
        return {"max_cycles": 1, "eval_every_cycles": None, "validation_miss_limit": None}
    if purpose != TRAIN_PURPOSE or job.get("fits_authorized") is not True or job.get("VALID_values_access") is not True:
        raise ValueError("Exact prospective TRAIN/VALID fit scope required")
    plan_path = custody.phase_file(job["cohort_plan_relative"])
    if custody.sha(plan_path) != job["cohort_plan_sha256"]: raise ValueError("Adopted prospective cohort changed")
    plan = json.loads(plan_path.read_text()); fixed = recipe()
    if plan.get("root_adopted_after_TRAIN_cost") is not True or plan.get("TEST_closed") is not True or plan.get("fixed_recipe_sha256") != job["fixed_recipe_sha256"]:
        raise ValueError("Root must adopt the exact recipe after new measured TRAIN costs, before scores")
    if plan.get("execution_order") != fixed["execution_order"]:
        raise ValueError("Predetermined two-stage order changed")
    if [row.get("cell_id") for row in plan.get("cells", [])] != fixed["execution_order"]:
        raise ValueError("Adopted plan must include the complete ordered 18-cell cohort")
    for expected, actual in zip(fixed["cells"], plan["cells"]):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Adopted cohort recipe differs: "+expected["cell_id"])
    cell = next(row for row in plan["cells"] if row["cell_id"] == job["cell_id"])
    if plan.get("caps_granted") is not True or type(plan.get("phase_hard_seconds")) is not int or plan["phase_hard_seconds"] < job["hard_seconds"]:
        raise ValueError("Root-frozen cohort/owned-process phase budget is missing")
    if type(plan.get("max_owned_concurrent_jobs")) is not int or not 1 <= plan["max_owned_concurrent_jobs"] <= len(custody.TARGETS[job["target"]]["gpu_uuids"]):
        raise ValueError("Adopted concurrency exceeds the exact authorized target inventory")
    for key in IDENTITY + ("target", "physical_gpu_uuid", "soft_seconds", "hard_seconds"):
        if job.get(key) != cell.get(key): raise ValueError("Adopted cell/resource binding differs: "+key)
    evidence = plan.get("complete_cycle_cost_evidence", [])
    qualified_costs = set()
    for bound in evidence:
        path = custody.phase_file(bound["path"])
        if custody.sha(path) != bound["sha256"]: raise ValueError("Measured structural complete-cycle cost changed")
        row = json.loads(path.read_text())
        if row.get("scope") != "discarded_TRAIN_complete_cycle_cost" or row.get("states_discarded") is not True or row.get("TEST_access") is not False or row.get("complete_VALID_scoring") is not False or row.get("last_cycle") != 1:
            raise ValueError("Cost receipt is not one actual complete discarded TRAIN cycle")
        for key in ("source_manifest_sha256", "fixed_recipe_sha256", "available_manifest_sha256"):
            if row.get(key) != job.get(key): raise ValueError("Structural cost source/input binding differs: "+key)
        qualified_costs.add((row.get("arm"), row.get("target"), row.get("physical_gpu_uuid")))
    required = {(row["arm"], row["target"], row["physical_gpu_uuid"]) for row in plan["cells"]}
    if not required <= qualified_costs: raise ValueError("Every assigned architecture/GPU needs a new actual full TRAIN-cycle cost")
    if plan.get("selection") != fixed["selection"] or plan.get("selection_budget_fairness_approved") is not True or not plan.get("paid_budget_description"):
        raise ValueError("Exact VALID selection and paid comparator budgets require root adoption")
    return fixed["schedule"]
