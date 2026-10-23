"""Read exact execution7 task custody without relocating or rewriting evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
PLAN_SHAS = {
    "allocation/gpu0": "19fa1a8b96bcf3383965c266c4fa900e0ac09c62ca040553e7669acb659ffaad",
    "server51/gpu0": "b23301b8674faf5dfad2000efeeb067d0c1a2fb97524277c2185d174a307a345",
    "server51/gpu1": "bc025b3209c0a7f5c2f10fd011fd54ca072c78269693b035ca3e53bbf243ad3c",
}
EXPECTED_COUNTS = {"allocation/gpu0": (2154, 90), "server51/gpu0": (2153, 90), "server51/gpu1": (2153, 90)}
REPAIR_SHAS = {1: "2797e55b3a0d996e44ad85f88aa54f30879ec19c32e27b5ec5635a7df3cb436c",
               2: "dafb24e45bab7316cd6e850e1d511e72058134609c6bfbb2f3e90c6c505f4d21"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def verify(receipt, root):
    path = Path(receipt["path"]).resolve()
    require(path.is_relative_to(root), "Evidence escapes project")
    require(pin(path) == receipt, "Evidence receipt differs: " + str(path))
    return path


def token(value):
    return hashlib.sha256(value.encode()).hexdigest()


def reporting_token(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class QueueEvidence:
    """Snapshot terminal task states; active work is explicitly omitted."""
    def __init__(self, root, plans, *, expected_pins=PLAN_SHAS):
        self.root = Path(root).resolve()
        self.plans, self.tasks, self.locations, self.omitted, self.attempt_history = [], {}, {}, [], []
        workers = set()
        for path in plans:
            source = pin(path)
            require(Path(source["path"]).is_relative_to(self.root), "Plan escapes project")
            plan = read(path)
            worker = plan["worker"]
            require(worker not in workers and source["sha256"] == expected_pins[worker], "Plan identity changed/duplicated")
            require(Path(plan["project_root"]).resolve() == self.root, "Plan belongs to another project")
            workers.add(worker)
            self.plans.append(source)
            state_root = Path(plan["state_root"]).resolve()
            require(state_root.is_relative_to(self.root), "Queue state root escapes project")
            for task in plan["tasks"]:
                fit_id = task["id"]
                require(fit_id not in self.tasks and task["worker"] == worker, "Duplicate/wrong task ownership")
                self.tasks[fit_id] = task
                folder = state_root / "full" / token(fit_id)
                for prior_path in sorted((folder / "attempts").glob("[0-9][0-9][0-9][0-9]/ATTEMPT_RESULT.json")):
                    prior = read(prior_path)
                    require(prior["id"] == fit_id and prior["worker"] == worker and prior["phase"] == "full",
                            "Historical attempt belongs to another task")
                    native = prior.get("native_result")
                    if native is not None:
                        native_path = verify(native, self.root)
                        require(native_path.parent == prior_path.parent, "Historical native snapshot escapes attempt")
                        cost = read(native_path)
                    else:
                        cost = {}
                    self.attempt_history.append({"id": fit_id, "attempt": prior["attempt"],
                        "status": prior["status"], "attempt_result": pin(prior_path), "native_result": native,
                        "invocation_wall_seconds": cost.get("invocation_wall_seconds"),
                        "optimizer_calls_this_invocation": cost.get("optimizer_calls_this_invocation"),
                        "cost_attribution": "retained attempt ledger; may overlap observer/history rows, do not add twice"})
                state_path = folder / "STATE.json"
                if not state_path.exists():
                    self.omitted.append({"id": fit_id, "status": "pending"})
                    continue
                state = read(state_path)
                if state["status"] not in ("complete", "failed", "no_selected_state"):
                    self.omitted.append({"id": fit_id, "status": state["status"]})
                    continue
                attempt_pin = state["attempt_result"]
                attempt_path = verify(attempt_pin, self.root)
                expected_attempt = folder / "attempts" / f'{state["attempt"]:04d}' / "ATTEMPT_RESULT.json"
                require(attempt_path == expected_attempt, "Attempt custody escapes exact task")
                terminal = read(attempt_path)
                require({k: v for k, v in state.items() if k != "attempt_result"} == terminal,
                        "Queue state differs from terminal attempt")
                require(terminal["id"] == fit_id and terminal["worker"] == worker and terminal["phase"] == "full",
                        "Wrong terminal task identity")
                output = Path(terminal["output"]).resolve()
                require(output.parent.parent == folder / "attempts" and output.name == "output"
                        and output.parent.name.isdigit(), "Result output escapes own task")
                native_pin = terminal.get("native_result")
                if native_pin is None:
                    self.omitted.append({"id": fit_id, "status": terminal["status"],
                        "attempt_result": attempt_pin, "reason": "No native RESULT; failure remains in queue ledger"})
                    continue
                native_copy = verify(native_pin, self.root)
                require(native_copy.parent == attempt_path.parent, "Native result snapshot escapes attempt")
                actual = output / "RESULT.json"
                require(actual.is_file() and read(actual) == read(native_copy), "Live output differs from terminal native snapshot")
                require(read(actual).get("TEST_accessed") is False, "TEST scope changed")
                self.locations[fit_id] = {"output": output, "result": actual,
                    "reporting": folder / "attempts" / "reporting" / reporting_token(fit_id),
                    "state": pin(state_path), "attempt_result": attempt_pin,
                    "native_result": native_pin, "status": terminal["status"]}

    def require_registry(self, ids):
        require(len(ids) == len(set(ids)) and set(ids) == set(self.tasks), "Registry/host queue task coverage differs")

    def locate(self, fit_id, spec):
        task = self.tasks[fit_id]
        if task["kind"] == "fit":
            require(task["descriptor"] == spec["descriptor"] and task["runtime_job_hash"] == spec["job_hash"],
                    "Registry descriptor differs from frozen queue")
        else:
            require(set(task["dependencies"]) == set(spec["component_fit_ids"]), "I4 registry dependencies differ")
        return self.locations.get(fit_id)

    def receipt(self):
        return {"schema": "execution7-queue-reporting-custody-v1", "plans": self.plans,
                "terminal_tasks": {key: {k: v for k, v in row.items() if k not in ("output", "result", "reporting")}
                                   for key, row in self.locations.items()},
                "omitted_tasks": self.omitted, "new_model_forwards": 0, "new_training_updates": 0,
                "all_retained_attempts": self.attempt_history,
                "evidence_moved_or_rewritten": False}


class WitnessVerifier:
    """Accept historical v1 and new v2 receipts under each producer's exact rules."""
    def __init__(self):
        self.modules = {}
        for version, folder in ((1, "campaign_reporting_repair_20261023_v1"),
                                (2, "campaign_reporting_repair_20261024_v2")):
            path = EXPANSION / folder / "reporting_patch.py"
            require(pin(path)["sha256"] == REPAIR_SHAS[version], "Reporting repair source changed")
            spec = importlib.util.spec_from_file_location("_queue_reporting_patch_v" + str(version), path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.modules[version] = module
        self.v2_receipt = self.modules[2].install()
        self.composed = sys.modules["composed_authority"]
        self.original = self.composed.runtime_witness.__wrapped__
        v1 = self.modules[1]
        self.v1_receipt = {
            "schema": v1.REPAIR_SCHEMA,
            "producer_receipt": v1._receipt(v1.__file__),
            "frozen_reporting_sources": v1._verify_frozen_sources(v1._reporting_directory()),
            "allowed_additional_dependency": {"path_relative_to_project": v1.README_PROJECT_RELATIVE,
                                               "sha256": v1.README_SHA256},
            "native_calls_added": 0,
        }

    def verify(self, job, spec, actual, recorded):
        require(isinstance(recorded, dict), "Observed composed witness is missing")
        observed = copy.deepcopy(recorded)
        repair = observed.pop("reporting_repair", None)
        expected = self.original(job, spec, actual, verify_dependency_bytes=False)
        require(observed == expected, "Recorded composed runtime identity differs")
        if repair is None:
            self.original(job, spec, actual)  # Preserve the original producer's narrow rule.
        else:
            template = self.v1_receipt if repair.get("schema") == self.v1_receipt["schema"] else self.v2_receipt
            require(repair == {**template, "dependency_bytes_verified": True}, "Unrecognized or altered reporting repair receipt")
            version = 1 if template is self.v1_receipt else 2
            self.modules[version]._verify_dependencies(self.composed, spec["composed_runtime_authority"], actual)
        return copy.deepcopy(recorded)
