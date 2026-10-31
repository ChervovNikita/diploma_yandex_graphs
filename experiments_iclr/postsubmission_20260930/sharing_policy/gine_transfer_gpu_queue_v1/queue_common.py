"""Confined source/route qualification and GPU snapshots; no model imports."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
REPO = PHASE.parents[1] if PHASE.parent.name == "experiments_iclr" else PHASE.parent
PARENT = PHASE / "sharing_policy/gine_transfer_grouping_fix_v6"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confined(path, root=HERE):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Path escapes authorized source packet/research phase")
    return path


def phase_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("A newly qualified phase-relative path is required")
    path = confined(PHASE / value, PHASE)
    if "_authorized1gpu_" not in str(path.relative_to(PHASE)):
        raise ValueError("Only new corrected-allocation output paths can qualify")
    return path


def write_json(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plan():
    return json.loads((HERE / "PLAN.json").read_text())


def source_hashes():
    science = json.loads((HERE / "SCIENTIFIC_SOURCE_BINDING.json").read_text())["source_sha256"]
    values = {}
    for name, expected in science.items():
        reference = HERE / "scientific_reference" / name
        if sha(reference) != expected:
            raise ValueError("Exact scientific reference changed:" + name)
        values["scientific_reference/" + name] = expected
    own = ("queue_common.py", "qualify_inputs.py", "gpu_queue.py", "PLAN.json",
           "SCIENTIFIC_SOURCE_BINDING.json", "AUTHORIZED_ROUTE_REFERENCE.json",
           "QUALIFICATION_REQUEST_TEMPLATE.json", "QUALIFICATION_REQUEST_authorized1gpu_run01.json",
           "README.md", "engineering_reference_cuda_gate.py")
    values.update({name: sha(HERE / name) for name in own})
    return values


def check_parent_sources(protocol):
    science = json.loads((HERE / "SCIENTIFIC_SOURCE_BINDING.json").read_text())["source_sha256"]
    for name, expected in science.items():
        if sha(confined(PARENT / name, PARENT)) != expected:
            raise ValueError("Parent scientific source changed:" + name)
    for name, expected in protocol["source_sha256"].items():
        if sha(confined(PARENT / name, PARENT)) != expected:
            raise ValueError("Qualified parent runtime source differs:" + name)


def runtime_scope():
    value = plan()
    if os.environ.get("GNNM_SSH_DESTINATION") != value["ssh_destination"]:
        raise ValueError("Require coordinator's exact corrected SSH destination")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Expose only physical GPU0 as cuda:0")
    if Path.cwd().resolve() != REPO:
        raise ValueError("Require exact authorized repository cwd")
    route = PHASE / value["route_phase_relative"]
    if sha(route) != value["route_sha256"]:
        raise ValueError("Corrected route evidence differs")
    record = json.loads(route.read_text())
    if (record["ssh_destination"] != value["ssh_destination"]
            or record["repository"] != str(REPO) or record["visible_gpu_count"] != 1
            or not record["repository_pwd_matches"] or record["verification_exit_code"] != 0):
        raise ValueError("Route does not qualify the original one-GPU allocation")
    for path_field, sha_field in (("allocation_guard_phase_relative", "allocation_guard_sha256"),
                                ("inner_supervisor_phase_relative", "inner_supervisor_sha256"),
                                ("CUDA_gate_phase_relative", "CUDA_gate_source_sha256")):
        if sha(PHASE / value[path_field]) != value[sha_field]:
            raise ValueError("Required allocation guard/supervisor changed")
    return value


def parse_gpu_rows(text):
    rows = [line for line in text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("Exactly one GPU required; seven-GPU allocation is forbidden")
    row = [item.strip() for item in rows[0].split(",")]
    if (len(row) != 6 or int(row[0]) != 0 or row[2] != "NVIDIA A100-SXM4-80GB"
            or int(row[4]) != 81920):
        raise RuntimeError("GPU differs from corrected allocation")
    return {"index": 0, "uuid": row[1], "name": row[2], "memory_MiB": int(row[3]),
            "total_MiB": int(row[4]), "utilization_percent": int(row[5])}


def idle_guard(expected_uuid=None, timeout_seconds=30):
    value = runtime_scope()
    if expected_uuid is not None and expected_uuid != value["authorized_gpu_uuid"]:
        raise RuntimeError("Requested GPU UUID differs from the exact authorized allocation")
    expected_uuid = value["authorized_gpu_uuid"]
    start, observations = time.perf_counter(), []
    while True:
        query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
        gpu = parse_gpu_rows(query.stdout)
        if gpu["uuid"] != expected_uuid:
            raise RuntimeError("GPU UUID changed since qualified prerequisite jobs")
        apps = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                              check=True, capture_output=True, text=True)
        gpu["compute_context_count"] = len([line for line in apps.stdout.splitlines() if line.strip()])
        observations.append(gpu)
        if gpu["memory_MiB"] <= 100 and gpu["utilization_percent"] == 0 and gpu["compute_context_count"] == 0:
            return {"status": "idle_no_compute_context", "GPU": gpu, "observations": observations,
                    "wait_seconds": time.perf_counter() - start, "snapshot_not_exclusive_reservation": True}
        if time.perf_counter() - start >= timeout_seconds:
            raise RuntimeError("Authorized GPU remains occupied; no next child started")
        time.sleep(2)


def authorization(supervisor, expected_uuid=None, output_flag=None, expected_output=None):
    value = plan()
    supervisor = confined(supervisor, PHASE)
    auth = json.loads((supervisor / "authorization.json").read_text())
    complete = json.loads((supervisor / "completion.json").read_text())
    if (auth.get("schema") != "gnnm-authorized-allocation-supervision-v2"
            or auth.get("ssh_destination") != value["ssh_destination"]
            or auth.get("route_record_sha256") != value["route_sha256"]
            or auth.get("allocation_guard_sha256") != value["allocation_guard_sha256"]
            or auth.get("visible_gpu_count") != 1 or auth.get("gpu_name") != "NVIDIA A100-SXM4-80GB"
            or auth.get("child_exit_code") != 0 or complete.get("exit_code") != 0
            or auth.get("inner_completion_sha256") != sha(supervisor / "completion.json")):
        raise ValueError("Missing/failed corrected-allocation job authorization")
    environment = json.loads((supervisor / "environment.json").read_text())
    if (complete["environment_sha256"] != sha(supervisor / "environment.json")
            or complete["command_sha256"] != sha(supervisor / "command.json")
            or complete["log_sha256"] != sha(supervisor / "stdout_stderr.log")):
        raise ValueError("Prerequisite supervisor artifact changed")
    if environment["supervisor_sha256"] != value["inner_supervisor_sha256"]:
        raise ValueError("Prerequisite uses another inner supervisor")
    if expected_uuid is not None and auth["gpu_uuid"] != expected_uuid:
        raise ValueError("Prerequisite jobs do not share the qualified GPU identity")
    if auth["gpu_uuid"] != value["authorized_gpu_uuid"]:
        raise ValueError("Prerequisite does not match coordinator's corrected GPU UUID")
    command = json.loads((supervisor / "command.json").read_text())
    if command["cwd"] != str(REPO):
        raise ValueError("Prerequisite was launched outside exact repository cwd")
    script = confined(command["argv"][1], PHASE)
    if environment["script_sha256"] != sha(script):
        raise ValueError("Prerequisite script differs from its executed source")
    if output_flag is not None:
        argv = command["argv"]
        if argv.count(output_flag) != 1:
            raise ValueError("Prerequisite output flag not unambiguous")
        target = Path(argv[argv.index(output_flag) + 1])
        target = target if target.is_absolute() else REPO / target
        if target.resolve() != Path(expected_output).resolve():
            raise ValueError("Prerequisite supervisor does not bind supplied output")
    return {"gpu_uuid": auth["gpu_uuid"], "seconds": complete["seconds"],
            "authorization_sha256": sha(supervisor / "authorization.json"),
            "completion_sha256": sha(supervisor / "completion.json"),
            "command_sha256": sha(supervisor / "command.json"),
            "environment_sha256": sha(supervisor / "environment.json")}
