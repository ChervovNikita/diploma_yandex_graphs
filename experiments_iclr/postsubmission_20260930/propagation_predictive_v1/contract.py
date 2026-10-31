"""Stdlib admission and completed-chain checks for a prospective fixed30 family."""
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
DESTINATION = "anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru"
AMENDMENT_SHA = "d30f2d05c1b66dc621d5cb10f86689f1365e49b5f84380472b928e17c04ec055"
TASKS = ("Reddit2", "ogbn-products")
ARMS = ("candidate", "detached_current_member_mean", "exact_member", "single_base", "naive_shared_sample")
RECIPE = dict(epochs=100, members=4, hidden_width=64, layers=2, dropout=0.2,
              ffn_multiplier=1, optimizer="AdamW", lr=0.001, weight_decay=0.0001,
              betas=[0.9, 0.999], eps=1e-8, prediction="mean_raw_logits",
              train_loss="mean_own_member_cross_entropy",
              checkpoint="lowest_complete_pooled_validation_CE_then_earliest_epoch",
              deployment_colors=[0, 2], AMP=False, TF32=False, dtype="float32")
CAP_GPU = 8 * 3600
CAP_DISK = 200 * 2**30
CAP_MEMORY = 60 * 2**30
_HASH_CACHE = {}
_HASH_STATS = {"fully_hashed_files": 0, "fully_hashed_bytes": 0, "unchanged_signature_reuses": 0}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def within(path, root=PHASE):
    root = Path(root).resolve()
    p = Path(os.path.abspath(path))
    if not p.is_relative_to(root):
        raise ValueError("Path outside declared project phase")
    current = root
    for component in p.relative_to(root).parts:
        current /= component
        if current.is_symlink():
            raise ValueError("Symlink acquisition/scientific evidence paths are forbidden")
    return p.resolve()


def sha(path):
    p = within(path)
    before = p.stat()
    signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    previous = _HASH_CACHE.get(str(p))
    if previous is not None:
        if previous[0] != signature:
            raise ValueError("Previously verified evidence file changed stat signature")
        _HASH_STATS["unchanged_signature_reuses"] += 1
        return previous[1]
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    after = p.stat()
    if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Evidence changed during hashing")
    result = h.hexdigest()
    _HASH_CACHE[str(p)] = (signature, result)
    _HASH_STATS["fully_hashed_files"] += 1
    _HASH_STATS["fully_hashed_bytes"] += before.st_size
    return result


def hash_cache_report():
    return {**_HASH_STATS, "scope": "one process; first complete byte hash then unchanged device/inode/size/mtime/ctime signature",
            "changed_signature_policy": "fail", "adversarial_integrity_guarantee": False}


def checked(ref):
    if set(ref) != {"path", "sha256"}:
        raise ValueError("Reference requires exact phase-relative path and SHA256")
    p = within(PHASE / ref["path"])
    if str(p.relative_to(PHASE)) != ref["path"]:
        raise ValueError("Evidence paths must be canonical phase-relative names")
    if not p.is_file() or sha(p) != ref["sha256"]:
        raise ValueError("Unqualified or changed reference: " + ref["path"])
    return p


def reference(path):
    p = within(path)
    return {"path": str(p.relative_to(PHASE)), "sha256": sha(p)}


def write(path, value):
    p = within(path, HERE)
    with p.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cells():
    return [dict(task=t, seed=s, arm=a) for t in TASKS for s in range(3) for a in ARMS]


def cell_id(cell):
    if cell not in cells():
        raise ValueError("Cell outside fixed complete30 family")
    return ("reddit2" if cell["task"] == "Reddit2" else "products") + "_seed%d_%s" % (cell["seed"], cell["arm"])


def target_directory(admitted, task, split):
    if task not in TASKS or split not in ("valid", "test"):
        raise ValueError("Only fixed complete-task validation/test admission outputs allowed")
    return within(PHASE / admitted["output_root"] / "targets" / split /
                  ("reddit2" if task == "Reddit2" else "products"), HERE)


def source_check():
    p = HERE / "SOURCE_BINDINGS.json"
    value = json.loads(p.read_text())
    if value.get("schema") != "gnnm-propagation-predictive-source-bindings-v1":
        raise ValueError("Source-only generation has no sealed approved dependency closure")
    for name, wanted in value["sha256"].items():
        if sha(PHASE / name) != wanted:
            raise ValueError("Prospective scientific source/dependency changed: " + name)
    for path in HERE.glob("*.py"):
        if str(path.relative_to(PHASE)) not in value["sha256"]:
            raise ValueError("Unsealed Python file could alter scientific import resolution")
    if "propagation_cost_engineering_v1/SOURCE_MANIFEST_local_run01.json" not in value["sha256"]:
        raise ValueError("Sealed author engineering packet required; no dynamic implementation substitution")
    return value


def runtime_before_import(admitted):
    policy = admitted["runtime_policy"]
    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != policy["CUBLAS_WORKSPACE_CONFIG"]
            or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"):
        raise ValueError("Frozen CUDA workspace and cache policy must precede third-party imports")
    if policy["deterministic_algorithms"] and policy["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("Deterministic CUDA runtime requires a frozen supported CUBLAS workspace")


def completed(directory):
    p = within(directory)
    a, c, e, command = [json.loads((p / n).read_text()) for n in
                        ("authorization.json", "completion.json", "environment.json", "command.json")]
    if (a.get("schema") != "gnnm-authorized-allocation-supervision-v2"
            or a.get("ssh_destination") != DESTINATION or a.get("visible_gpu_count") != 1
            or a.get("gpu_name") != "NVIDIA A100-SXM4-80GB"
            or a.get("allocation_guard_sha256") != sha(PHASE / "protocols/run_authorized_v2.py")
            or a.get("route_record_sha256") != sha(PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json")
            or a.get("child_exit_code") != 0 or c.get("exit_code") != 0
            or a.get("inner_completion_sha256") != sha(p / "completion.json")):
        raise ValueError("Wrong-route, failed or incomplete supervisor chain")
    for key, name in (("environment_sha256", "environment.json"),
                      ("command_sha256", "command.json"), ("log_sha256", "stdout_stderr.log")):
        if c[key] != sha(p / name):
            raise ValueError("Completed supervision bytes changed")
    route = json.loads((PHASE / "protocols/AUTHORIZED_ALLOCATION_ROUTE_20260930_v1.json").read_text())
    if (e.get("repository") != route["repository"] or command.get("cwd") != route["repository"]
            or e.get("supervisor_sha256") != sha(PHASE / "protocols/run_logged.py")
            or e.get("wrapper_sha256") != sha(PHASE / "protocols/repo_env.sh")):
        raise ValueError("Wrong completed repository or supervisor sources")
    argv = command.get("argv", [])
    if len(argv) < 2 or e.get("script_sha256") != sha(within(argv[1])):
        raise ValueError("Completed child entrypoint source differs from supervised bytes")
    return {"authorization": reference(p / "authorization.json"), "completion": reference(p / "completion.json"),
            "gpu_uuid": a["gpu_uuid"], "seconds": c["seconds"], "argv": argv}


def binding(path):
    """No Torch, provider storage, predictions or target tensor import here."""
    p = within(path, HERE)
    value = json.loads(p.read_text())
    if (value.get("schema") != "gnnm-propagation-fixed30-scientific-admission-v1"
            or value.get("scientific_training_authorized") is not True
            or value.get("recipe") != RECIPE or value.get("cells") != cells()
            or value.get("test_target_access_before_complete_family_lock") is not False):
        raise ValueError("Complete root-approved prospective30 scientific admission required")
    source_check()
    if value["source_bindings_sha256"] != sha(HERE / "SOURCE_BINDINGS.json"):
        raise ValueError("Root did not freeze this source generation")
    amendment_path = checked(value["amendment"])
    amendment = json.loads(amendment_path.read_text())
    if sha(amendment_path) != AMENDMENT_SHA or amendment.get("total_fits") != 30:
        raise ValueError("Exact prospective five-arm amendment required")
    approval = json.loads(checked(value["root_approval"]).read_text())
    payload = {k: v for k, v in value.items() if k != "root_approval"}
    if (approval.get("schema") != "gnnm-propagation-fixed30-root-approval-v1"
            or approval.get("approved_payload_sha256") != digest(payload)
            or approval.get("qualified") is not True
            or approval.get("all_complete_inputs_and_gates_qualified") is not True
            or approval.get("validation_label_admission_authorized") is not True
            or approval.get("test_before_family_lock_authorized") is not False):
        raise ValueError("Actual inputs/runtime/gates are not root-approved")
    mandatory = {"synthetic_cuda", "complete_engineering_Reddit2", "complete_engineering_ogbn-products",
                 "complete30_resources", "source_audit", "runtime_freeze", "execution_equivalence"}
    if set(value["qualified_gates"]) != mandatory:
        raise ValueError("Both complete graphs and every required source/resource gate must qualify")
    for name, row in value["qualified_gates"].items():
        report = checked(row["report"])
        if name in {"synthetic_cuda", "complete_engineering_Reddit2", "complete_engineering_ogbn-products", "execution_equivalence"}:
            receipt = completed(PHASE / row["supervisor_directory"])
            if receipt["gpu_uuid"] != value["gpu_uuid"]:
                raise ValueError("Gate GPU differs from frozen actual runtime")
        if row.get("qualified") is not True or row.get("report_sha256") != sha(report):
            raise ValueError("Root has not qualified gate: " + name)
    resource = json.loads(checked(value["resource_ledger"]).read_text())
    if (resource.get("qualified") is not True or resource.get("fit_count") != 30
            or resource.get("all_artifact_replay_component_timing_costs_included") is not True
            or resource.get("total_GPU_seconds_forecast", float("inf")) > 6.5 * 3600
            or resource.get("projected_additional_disk_bytes", float("inf")) > CAP_DISK
            or resource.get("peak_reserved_bytes", float("inf")) > CAP_MEMORY
            or resource.get("total_CPU_seconds_forecast", float("inf")) > 16 * 3600):
        raise ValueError("Expanded complete-family resource admission is closed")
    if set(value["tasks"]) != set(TASKS) or set(value["execution_policy"]) != set(TASKS):
        raise ValueError("No missing, substituted or sampled-down graph allowed")
    for task in TASKS:
        task_value = value["tasks"][task]
        phase = task_value["engineering_phase"]
        if phase["task"] != task:
            raise ValueError("Actual complete task identity differs")
        checked(task_value["prepared_manifest"])
        checked(task_value["acquisition"])
        checked(task_value["acquisition_completed_audit"])
        preparation = task_value["preparation_charged_seconds"]
        if set(preparation) != set(ARMS) or any(type(v) not in (int, float) or v < 0 for v in preparation.values()):
            raise ValueError("Every arm requires root-qualified attributable once-only preparation cost")
        checked(task_value["preparation_cost_accounting"])
        template = task_value["test_label_request_template"]
        if (template.get("schema") != "gnnm-propagation-root-split-admission-request-v1"
                or template.get("root_authorized") is not True or template.get("task") != task
                or template.get("split") != "test"):
            raise ValueError("Both complete test request templates must be method-frozen")
        checked(template["processed_public_file"])
        if within(PHASE / template["output_relative_to_phase"], HERE) != target_directory(value, task, "test"):
            raise ValueError("Method-frozen test export differs from prescribed budgeted location")
        if set(value["execution_policy"][task]) != set(ARMS):
            raise ValueError("All five execution choices must be frozen before outcomes")
        for arm, choices in value["execution_policy"][task].items():
            if set(choices) != {"training_validation", "deployment"}:
                raise ValueError("Training/checkpoint and independently qualified deployment policies required")
            for variant in choices.values():
                if set(variant) != {"arm", "execution", "naive_path", "checkpoint_blocks", "native_source"} or variant["arm"] != arm:
                    raise ValueError("Unfrozen model execution variant")
                if arm in ARMS[:2] and (variant["execution"] != "packed" or variant["native_source"]):
                    raise ValueError("Coupled current means require complete packed member states")
    within(PHASE / value["output_root"], HERE)
    policy = value["runtime_policy"]
    if (policy.get("device") != "cuda:0" or policy.get("TF32") is not False or policy.get("AMP") is not False
            or type(policy.get("CPU_threads")) is not int or policy["CPU_threads"] < 1
            or type(policy.get("deterministic_algorithms")) is not bool
            or not {"torch", "numpy", "torch-geometric"}.issubset(policy.get("package_versions", {}))):
        raise ValueError("Actual complete runtime and precision freeze required")
    resident = value["timing_policy"]["resident_family_memory_admission"]
    if (resident.get("qualified") is not True
            or resident.get("method") not in ("conservative_sum_of_selected_individual_peaks", "measured_complete_resident_family")
            or set(resident.get("maximum_reserved_bytes", {})) != set(TASKS)
            or any(v > CAP_MEMORY for v in resident["maximum_reserved_bytes"].values())):
        raise ValueError("Complete five-arm resident timing family must have a qualified memory bound")
    checked(resident["report"])
    if resident["method"] == "measured_complete_resident_family":
        resident_chain = completed(PHASE / resident["supervisor_directory"])
        if resident_chain["gpu_uuid"] != value["gpu_uuid"]:
            raise ValueError("Resident-family measurement used a different frozen allocation")
    if (value["timing_policy"].get("warmups") != 10 or value["timing_policy"].get("timed_forwards") != 50
            or value["timing_policy"].get("separate_component_warmups") != 10
            or value["timing_policy"].get("separate_component_forwards") != 50
            or value["timing_policy"].get("paired_order_seed_offset") != 50000
            or value["timing_policy"].get("component_order_seed_offset") != 60000):
        raise ValueError("Raw/component timing work and paired order must be frozen before fits")
    return value, reference(p)


def live(supervisor, entrypoint):
    # Import the engineering *stdlib* guard only after closure verification.
    import sys
    if str(PHASE) not in sys.path:
        sys.path.insert(0, str(PHASE))
    from propagation_cost_engineering_v1.contract import live_context
    context = live_context(supervisor, entrypoint)
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True)
    rows = [[part.strip() for part in line.split(",")] for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][0] != "0" or rows[0][2] != "NVIDIA A100-SXM4-80GB" or int(rows[0][3]) != 81920:
        raise ValueError("Wrong live one-A100 allocation")
    return {**context, "gpu_uuid": rows[0][1]}


class Budget:
    """Conservative supervised wall time is charged as GPU allocation time."""
    def __init__(self, admitted, prior_seconds, context=None):
        self.start = time.monotonic()
        if context is not None:
            env = json.loads((PHASE / context["supervisor_relative_to_project"] / "environment.json").read_text())
            capture = datetime.fromisoformat(env["capture_UTC"])
            self.start -= max(0, (datetime.now(timezone.utc) - capture).total_seconds())
        self.prior = float(prior_seconds)
        self.prior_cpu = float(admitted["remaining_resource_policy"]["prior_CPU_preparation_audit_wall_seconds"])
        self.previous_scientific_wall = self.prior - float(admitted["remaining_resource_policy"]["prior_GPU_seconds"])
        self.base_disk = admitted["remaining_resource_policy"]["prior_additional_disk_bytes"]
        self.root = within(PHASE / admitted["output_root"], HERE)
        self.stored = self._scan()
        self.last_scan = time.monotonic()
        # JSON/manifest overhead remains bounded conservatively between scans.
        self.metadata_reserve = 512 * 2**20
        if not 0 <= self.prior < CAP_GPU or not 0 <= self.base_disk <= CAP_DISK:
            raise ValueError("Invalid actual family resource ledger")
        if self.prior_cpu < 0 or self.previous_scientific_wall < 0:
            raise ValueError("Invalid CPU preparation/audit resource ledger")

    def _scan(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) if self.root.exists() else 0

    def committed(self, path):
        self.stored += within(path, HERE).stat().st_size

    def check(self, prospective_bytes=0):
        if self.prior + time.monotonic() - self.start >= CAP_GPU:
            raise RuntimeError("Eight GPU-hour complete-family budget exhausted")
        if self.prior_cpu + self.previous_scientific_wall + time.monotonic() - self.start >= 16 * 3600:
            raise RuntimeError("Conservatively charged CPU preparation/audit wall budget exhausted")
        if time.monotonic() - self.last_scan >= 60:
            self.stored = self._scan()
            self.last_scan = time.monotonic()
        if self.base_disk + self.stored + prospective_bytes + self.metadata_reserve > CAP_DISK:
            raise RuntimeError("200GiB additional-disk budget would be exceeded")


def manifest(output):
    output = within(output, HERE)
    write(output / "artifact_manifest.json", {"schema": "gnnm-propagation-scientific-artifacts-v1",
          "files": {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                    for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}})


def audit_manifest(directory):
    directory = within(directory, HERE)
    record = json.loads((directory / "artifact_manifest.json").read_text())
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
              if p.is_file() and p.name != "artifact_manifest.json"}
    if actual != set(record["files"]):
        raise ValueError("Scientific artifact inventory differs from sealed complete set")
    for name, expected in record["files"].items():
        p = within(directory / name, directory)
        if p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
            raise ValueError("Scientific artifact changed: " + name)
    return reference(directory / "artifact_manifest.json")
