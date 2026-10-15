"""Finite source-bound campaign contracts. Stdlib only; no scientific imports."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

WORKERS = ("allocation/gpu0", "server77/gpu0", "server77/gpu1")
BASE = "benchmark_expansion_20261015_v1"
RUNTIME = f"{BASE}/runtime_v2"
LAUNCHER = f"{BASE}/campaign_launcher_v1"
RUNTIME_FILES = ("run_fit.py", "common.py", "adapters.py", "bookkeeping.py", "controllers.py", "source_rng.py", "aggregate_selected.py")
GATES = ("reviewed_finite_roster", "source_and_data_binding", "native_runtime_and_controller",
         "checkpoint_resume_and_rng", "validation_only_selection", "resource_mature_and_two_slot",
         "complete_failure_and_cost_reporting", "campaign_scope_bridge")
RETIRED_V29 = "0a35daf95e7f1ad8e8df1037923a5a38638ed37e8c0a65c9dbcc3bf73f55488b"
EXCLUDED = {"roman-empire", "amazon-ratings", "minesweeper", "questions", "tolokers"}
MAX_JOBS = 20_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Project-relative path required")
    require(".." not in Path(relative).parts, "Parent path components are forbidden")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Path escapes project root")
    return path


def read(root, relative):
    return json.loads(inside(root, relative).read_text())


def pin(root, relative):
    return {"path": relative, "sha256": sha(inside(root, relative))}


def verify_pin(root, value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, "Exact path/SHA256 receipt required")
    require(pin(root, value["path"]) == value, "Receipt bytes changed: " + str(value["path"]))


def source_inventory(plan):
    require(plan.get("schema") == "reference-wrapper-campaign-v2", "Current redesign plan schema required; no v29")
    require(plan.get("workers") == list(WORKERS) and plan.get("slots_per_gpu") == 2, "Explicit three-worker/two-slot plan required")
    require(plan.get("automatic_retries") == 0 and plan.get("human_GO_required") is True, "No retries; explicit review GO required")
    require(digest(plan) != RETIRED_V29, "Retired v29 plan rejected")
    jobs = {item["id"]: item for item in plan["jobs"]}
    require(0 < len(jobs) == len(plan["jobs"]) <= MAX_JOBS, "Finite unique source jobs required")
    assignments = {}
    for group in plan["groups"]:
        require(group["worker"] in WORKERS, "Unknown source worker")
        for job_id in group["job_ids"]:
            require(job_id in jobs and job_id not in assignments, "Missing/duplicate source assignment")
            assignments[job_id] = group["worker"]
    require(set(assignments) == set(jobs), "Source groups must partition all source jobs")
    fits = {key: item for key, item in jobs.items() if item["kind"] == "fit"}
    require(fits and all(item["kind"] in ("fit", "evaluate_roster") for item in jobs.values()), "Unknown source job kind")
    for item in fits.values():
        require(item["contract_hash"] == digest(item["contract"]), "Source fit contract changed")
        dataset = item["contract"]["dataset"].lower().replace("_", "-").replace(" ", "-")
        require(dataset not in EXCLUDED, "Original paper dataset remains excluded")
    return fits, assignments


def resource_contract(root, resources, cost):
    require(set(resources) == {"vram_bytes", "ram_bytes", "disk_bytes", "measured", "mature_optimizer", "pair_qualified", "profile", "qualified_peer_profiles", "max_concurrent_owned_children", "evidence"}, "Explicit measured resource reservation required")
    for field in ("vram_bytes", "ram_bytes", "disk_bytes"):
        require(type(resources[field]) is int and resources[field] > 0, "Positive byte reservation required")
    require(all(resources[key] is True for key in ("measured", "mature_optimizer")), "Measured mature resource qualification required")
    require(resources["max_concurrent_owned_children"] in (1, 2) and type(resources["max_concurrent_owned_children"]) is int
        and isinstance(resources["profile"], str) and resources["profile"], "Explicit serial/two-slot resource profile required")
    require(isinstance(resources["qualified_peer_profiles"], list) and all(isinstance(value, str) and value for value in resources["qualified_peer_profiles"]), "Qualified peer profile list required")
    require(type(resources["pair_qualified"]) is bool and (resources["max_concurrent_owned_children"] == 1 or
        (resources["pair_qualified"] is True and resources["qualified_peer_profiles"])), "Explicit pair qualification required for two-slot work")
    verify_pin(root, resources["evidence"])
    require(set(cost) == {"measured", "observed_wall_seconds", "observed_optimizer_calls", "scope", "evidence"}, "Measured cost with scope and evidence required")
    require(cost["measured"] is True and type(cost["observed_wall_seconds"]) in (int, float)
            and math.isfinite(cost["observed_wall_seconds"]) and cost["observed_wall_seconds"] > 0, "Positive observed wall cost required")
    require(type(cost["observed_optimizer_calls"]) is int and cost["observed_optimizer_calls"] >= 0
            and isinstance(cost["scope"], str) and cost["scope"], "Observed work and measurement scope required")
    verify_pin(root, cost["evidence"])


def validate_native_job(job, source):
    contract = source["contract"]
    require(job.get("schema") == "native-full-epoch-fit-v1" and job.get("TEST_access") is False, "Only explicit no-TEST native fits")
    require(job.get("family") in ("node", "ogb", "link") and job.get("device") == "cuda:0", "Explicit native family/cuda:0 descriptor required")
    expected = {"id": source["id"], "procedure": contract["model_procedure"], "seed": contract["initialization_seed"],
                "split": contract["split"], "body": contract["body"]}
    require(all(job.get(key) == value for key, value in expected.items()), "Runtime job differs from source fit identity")
    require(job["recipe"]["id"] == contract["recipe_id"] and job["recipe"]["dataset"] == contract["dataset"], "Runtime recipe differs from source contract")
    require(job["recipe"]["native_fixed_config"] == contract["native_config"], "Native settings differ from source contract")
    require(job["policy"]["track"] == ("source_reference" if contract["track"] == "native_replica" else "matched_extended"), "Source and matched tracks must remain separate")
    require(job["policy"]["maximum_epochs"] == contract["training"]["maximum"]
            and contract["training"]["unit"] == "epochs", "Native source clock/horizon bridge missing")
    require("command" not in job and "argv" not in job, "No arbitrary command fields")


def validate_source_hooks(job, sources):
    require(sources.get(job["callback_module"]) == job.get("callback_sha256"), "Native callback must be explicitly source-pinned")
    require(not any(name in job for name in ("extension_factory", "session_extension")), "Compiled jobs use the explicit session_transform object only")
    if job.get("session_transform") is not None:
        hook = job["session_transform"]
        require(isinstance(hook, dict) and set(hook) == {"path", "function", "sha256"}
            and sources.get(hook["path"]) == hook["sha256"], "Private session transform must be explicitly source-pinned")
    if job.get("family") == "link":
        require(sources.get(job.get("link_policy_module")) == job.get("link_policy_sha256")
            and job.get("link_policy_sha256") is not None, "Link policy source pin missing")
    if job.get("source_rng_emulation") is not None:
        audit = job["source_rng_emulation"]
        require(sources.get(audit["audit"]) == audit["audit_sha256"], "Source RNG audit must be in the pinned bundle")


def compile_manifest(root, source_plan, bindings, release_id, output_root):
    """Exact complete fit mapping; never turn abstract factories into commands."""
    plan = read(root, source_plan)
    fits, workers = source_inventory(plan)
    require(bindings.get("schema") == "finite-campaign-execution-bindings-v1", "Explicit execution binding schema required")
    require(bindings.get("source_plan") == pin(root, source_plan), "Bindings refer to different source plan bytes")
    require(isinstance(release_id, str) and release_id and release_id.replace("-", "").replace("_", "").isalnum(), "Simple immutable release ID required")
    inside(root, output_root)
    fit_bindings = bindings["fits"]
    require(set(fit_bindings) == set(fits), "Every declared fit needs exactly one immutable runtime binding; no omissions")
    sources = dict(bindings["source_sha256"])
    python_pins = bindings["python_executable_sha256_by_worker"]
    require(set(python_pins) == set(WORKERS) and all(isinstance(value, str) and len(value) == 64
        and set(value) <= set("0123456789abcdef") for value in python_pins.values()), "All worker Python executable hashes required")
    for relative in [f"{RUNTIME}/{name}" for name in RUNTIME_FILES] + [f"{LAUNCHER}/contracts.py", f"{LAUNCHER}/supervisor.py"]:
        require(sources.get(relative) == sha(inside(root, relative)), "Complete pinned runtime/supervisor source bundle required")
    for relative, expected_sha in sources.items():
        require(sha(inside(root, relative)) == expected_sha, "Pinned source changed")
    jobs = []
    for job_id, source in fits.items():
        binding = fit_bindings[job_id]
        require(set(binding) == {"job", "resources", "measured_cost"}, "Unknown execution binding fields")
        verify_pin(root, binding["job"])
        job = read(root, binding["job"]["path"])
        validate_native_job(job, source)
        validate_source_hooks(job, sources)
        resource_contract(root, binding["resources"], binding["measured_cost"])
        jobs.append({"id": job_id, "kind": "fit", "handler": "native_train_v2", "worker": workers[job_id],
            "job": binding["job"], "job_canonical_sha256": digest(job), "source_contract_sha256": source["contract_hash"],
            "output": f"{output_root}/fits/{digest(job_id)}", "dependencies": [],
            "resources": binding["resources"], "measured_cost": binding["measured_cost"]})
    # Explicit native-I4 VALID aggregation only. It is conservatively charged
    # a slot, even though this current handler loads saved tensors on CPU.
    banks = {bank["id"]: bank for bank in plan["logical_banks"] if bank["id"].endswith("/native-I4")}
    for entry in bindings.get("VALID_analyses", []):
        require(set(entry) == {"bank_id", "resources", "measured_cost"}, "Unknown analysis binding fields")
        bank = banks[entry["bank_id"]]
        dependencies = bank["fit_ids"]
        require(len(dependencies) == 4 and len(set(dependencies)) == 4 and all(key in fits for key in dependencies), "Four own native I4 fits required")
        require(len({workers[key] for key in dependencies}) == 1, "Cross-worker analysis bridge not supplied")
        resource_contract(root, entry["resources"], entry["measured_cost"])
        jobs.append({"id": bank["id"] + "/selected-VALID-aggregate", "kind": "analysis", "handler": "aggregate_VALID_v2",
            "worker": workers[dependencies[0]], "output": f"{output_root}/analyses/{digest(bank['id'])}",
            "dependencies": dependencies, "resources": entry["resources"], "measured_cost": entry["measured_cost"]})
    require(len({job["id"] for job in jobs}) == len(jobs) <= MAX_JOBS, "Unique finite executable jobs required")
    return {"schema": "finite-two-slot-campaign-v1", "status": "compiled_requires_GO_and_readiness",
        "release_id": release_id, "source_plan": pin(root, source_plan), "source_plan_canonical_sha256": digest(plan),
        "source_sha256": sources, "source_bundle_sha256": digest(sources), "workers": list(WORKERS),
        "python_executable_sha256_by_worker": python_pins,
        "slots_per_worker": 2, "automatic_retries": 0, "TEST_access": False, "output_root": output_root,
        "jobs": jobs, "statistical_evaluate_roster_jobs_not_executed_here": sum(item["kind"] == "evaluate_roster" for item in plan["jobs"]),
        "planning_note": "Measured observations and reservations are carried unchanged; no balancing or ETA inferred"}


def validate_manifest(root, manifest):
    require(manifest.get("schema") == "finite-two-slot-campaign-v1", "Only compiled finite campaign manifests")
    require(manifest.get("status") == "compiled_requires_GO_and_readiness" and manifest.get("TEST_access") is False, "Prepared inventory is not executable")
    require(manifest["workers"] == list(WORKERS) and manifest["slots_per_worker"] == 2 and manifest["automatic_retries"] == 0, "Finite worker policy changed")
    verify_pin(root, manifest["source_plan"])
    plan = read(root, manifest["source_plan"]["path"])
    fits, assignments = source_inventory(plan)
    require(digest(plan) == manifest["source_plan_canonical_sha256"], "Source plan canonical hash changed")
    require(digest(manifest["source_sha256"]) == manifest["source_bundle_sha256"], "Source bundle hash changed")
    for relative in [f"{RUNTIME}/{name}" for name in RUNTIME_FILES] + [f"{LAUNCHER}/contracts.py", f"{LAUNCHER}/supervisor.py"]:
        require(relative in manifest["source_sha256"], "Required source pin missing")
    for relative, expected_sha in manifest["source_sha256"].items():
        require(sha(inside(root, relative)) == expected_sha, "Source changed: " + relative)
    jobs = manifest["jobs"]
    require(0 < len({job["id"] for job in jobs}) == len(jobs) <= MAX_JOBS, "Finite unique jobs required")
    outputs = []
    ids = {job["id"] for job in jobs}
    require({job["id"] for job in jobs if job["kind"] == "fit"} == set(fits), "Compiled fit set omits/adds source fits")
    bank_index = {bank["id"] + "/selected-VALID-aggregate": bank for bank in plan["logical_banks"] if bank["id"].endswith("/native-I4")}
    for job in jobs:
        require(job["worker"] in WORKERS and job["handler"] in ("native_train_v2", "aggregate_VALID_v2"), "Unknown worker/handler; arbitrary commands rejected")
        require("command" not in job and "argv" not in job and "entrypoint" not in job, "No arbitrary executable commands")
        require(set(job["dependencies"]) <= ids and job["id"] not in job["dependencies"], "Invalid dependencies")
        require(inside(root, job["output"]).is_relative_to(inside(root, manifest["output_root"])), "Fit output outside campaign")
        outputs.append(job["output"])
        resource_contract(root, job["resources"], job["measured_cost"])
        if job["handler"] == "native_train_v2":
            require(job["kind"] == "fit" and job["id"] in fits and job["worker"] == assignments[job["id"]]
                and not job["dependencies"], "Training fit source identity/worker/dependencies changed")
            verify_pin(root, job["job"])
            native = read(root, job["job"]["path"])
            require(digest(native) == job["job_canonical_sha256"], "Runtime job changed")
            validate_native_job(native, fits[job["id"]])
            require(job["source_contract_sha256"] == fits[job["id"]]["contract_hash"], "Source contract hash changed")
            validate_source_hooks(native, manifest["source_sha256"])
        else:
            require(job["kind"] == "analysis" and job["id"] in bank_index, "Only declared native-I4 VALID analysis")
            dependencies = bank_index[job["id"]]["fit_ids"]
            require(job["dependencies"] == dependencies and len(dependencies) == 4
                and len({assignments[key] for key in dependencies}) == 1
                and job["worker"] == assignments[dependencies[0]], "Analysis dependency/worker bridge changed")
    require(len(set(outputs)) == len(outputs), "Output paths must be unique")
    return jobs


def verify_admission_snapshot(root, manifest, job, authority_pins):
    """Recheck the frozen authority/source/selected job, without N² job scans."""
    for value in authority_pins:
        verify_pin(root, value)
    verify_pin(root, manifest["source_plan"])
    for relative, expected_sha in manifest["source_sha256"].items():
        require(sha(inside(root, relative)) == expected_sha, "Source changed before admission: " + relative)
    resource_contract(root, job["resources"], job["measured_cost"])
    if job["kind"] == "fit":
        verify_pin(root, job["job"])


def require_go(root, manifest_path, manifest, go_path, readiness_path, worker):
    require(go_path and readiness_path, "Actual --run requires explicit reviewed GO and readiness receipts")
    def snapshot(relative):
        raw = inside(root, relative).read_bytes()
        return json.loads(raw), {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}
    observed_manifest, manifest_pin = snapshot(manifest_path)
    go, go_pin = snapshot(go_path)
    readiness, readiness_pin = snapshot(readiness_path)
    require(digest(observed_manifest) == digest(manifest), "Loaded manifest differs from reviewed manifest bytes")
    require(go.get("schema") == "finite-campaign-human-GO-v1" and go.get("user_explicit_GO") is True, "Explicit human GO receipt required")
    require(go.get("manifest_sha256") == manifest_pin["sha256"] and go.get("source_bundle_sha256") == manifest["source_bundle_sha256"], "GO manifest/source hashes mismatch")
    require(go.get("readiness") == readiness_pin and worker in go.get("allowed_workers", []), "GO readiness/worker mismatch")
    require(isinstance(go.get("review_evidence"), str) and go["review_evidence"].strip(), "GO must cite the actual human review evidence")
    require(readiness.get("schema") == "finite-campaign-readiness-v1"
            and readiness.get("manifest_sha256") == manifest_pin["sha256"]
            and readiness.get("source_bundle_sha256") == manifest["source_bundle_sha256"], "Readiness binds different manifest/source bytes")
    record = readiness["workers"][worker]
    for name in GATES:
        gate = record["gates"][name]
        require(gate.get("status") == "PASS", "Readiness gate is not PASS: " + name)
        verify_pin(root, gate["evidence"])
    budget = record["budget"]
    require(set(budget) == {"gpu_headroom_bytes", "host_ram_headroom_bytes", "disk_headroom_bytes"}
            and all(type(value) is int and value >= 0 for value in budget.values()), "Explicit measured headroom budget required")
    return go, record, [manifest_pin, go_pin, readiness_pin]
