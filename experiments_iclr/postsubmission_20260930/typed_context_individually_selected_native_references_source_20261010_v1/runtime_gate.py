"""Closed reference authority; literal route check before deliberate file reads."""
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typed_label_context_factor_source_prototype_20261010_v3.caps import CLOSED
from typed_label_context_factor_source_prototype_20261010_v3 import runtime_gate as v3_gate
from .model import FAMILIES

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
GPU = "GPU-44039938-fd82-41d2-fefd-de71514e2fac"
SERVER_PHASE = "/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930"
METHOD_SEAL = "8f7222aafe57743ab6c444c80beeffc3497a5feaea7ce9926268db8a4f980681"
METHOD_PROTOCOL = "67e00f8080320c7fd85b5dfcedac660e6a0884649eed561617c879c8444fac3c"
require, read, sha, bound, inside, binding = (getattr(v3_gate, n) for n in ("require", "read", "sha", "bound", "inside", "binding"))


def source_gate(caps=CLOSED):
    caps.require("source_bound")
    seal, manifest = read(HERE/"SEAL.json"), read(HERE/"MANIFEST.json")
    require(seal["source_only"] and seal["execution_enabled"] is False
            and sha(HERE/"MANIFEST.json") == seal["manifest_sha256"], "Exact inactive reference source")
    for row in manifest["files"]:
        path = (HERE/row["path"]).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Unchanged reference payload")
    for row in read(HERE/"SOURCE_BINDINGS.json")["files"]:
        path = (PROJECT/row["path"]).resolve(strict=True)
        require(path.is_relative_to(PROJECT) and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], "Unchanged pinned method source")
    sources = v3_gate.source_gate()
    method = PROJECT/"typed_label_context_factor_source_prototype_20261010_v3"
    require(sha(method/"SEAL.json") == METHOD_SEAL and sha(method/"PROTOCOL.json") == METHOD_PROTOCOL,
            "Unchanged typed V3 method and candidate protocol")
    return sources


def release_gate(path, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    require(socket.gethostname() == "anogena-2-0" and str(PROJECT) == SERVER_PHASE,
            "Literal authorized allocation before project reads")
    require(subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            text=True, timeout=10).splitlines() == [GPU], "Sole exact allocation GPU")
    sources = source_gate(caps)
    path = inside(path)
    release = read(path)
    scientific = release["action"] == "fit_independently_selected_references"
    require(release["action"] in ("qualify_independently_selected_references", "fit_independently_selected_references"), "Fixed reference action")
    if scientific:
        caps.require("scientific")
    require(release["enabled"] is True and release["root_source_review_approved"] is True
            and release["scientific_execution_approved"] is scientific
            and release["data_scope_approved"] is True and release["provider_runtime_approved"] is True
            and release["root_review"]["sha256"] == caps.root_review_sha256,
            "Separate explicit root reference approval")
    review = read(bound(release["root_review"]))
    require(review["approved"] is True and review["scientific_reference_fits_approved"] is scientific
            and review["method_source_modified"] is False
            and review["reference_source_seal_sha256"] == sha(HERE/"SEAL.json")
            and review["reference_protocol_sha256"] == sha(HERE/"PROTOCOL.json")
            and release["reference_source_seal_sha256"] == sha(HERE/"SEAL.json")
            and release["reference_protocol_sha256"] == sha(HERE/"PROTOCOL.json"), "Root binds this exact reference source/protocol")
    require(release["source_seal_sha256"] == METHOD_SEAL and release["protocol_sha256"] == METHOD_PROTOCOL
            and release["families"] == list(FAMILIES)
            and release["seed_specs"] == list(v3_gate.SEEDS if scientific else v3_gate.SEEDS[:1]), "Same candidate source/roles/contexts; complete reference roster")
    require(release["capabilities"] == {n: getattr(caps, n) for n in ("source_bound", "model", "data", "runtime", "scientific")}, "Root release/capability agreement")
    require(set(release["runtime_environment"]) == {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
            and release["runtime_environment"]["CUDA_VISIBLE_DEVICES"] == GPU
            and all(os.environ.get(k) == v for k, v in release["runtime_environment"].items())
            and str(Path(sys.executable).absolute()) == release["python_executable"]
            and platform.python_version() == release["python_version"]
            and Path(sys.prefix).resolve().is_relative_to(PROJECT), "Exact qualified interpreter and owner environment before numerical imports")
    require(set(release["development_files"]) == {"node.dat", "link.dat", "label.dat"}
            and release["cuda_device_uuid"] == GPU and release["device"] == "cuda:0", "Exact existing development-only task/device")
    require(set(release["expected_runtime_versions"]) == set(v3_gate.PROVIDERS)
            and all(isinstance(v, str) and v for v in release["expected_runtime_versions"].values()), "Exact qualified provider version roster")
    require(set(release["roles"]) == {"1", "2", "3"}
            and all(Path(row["path"]).name == "seed"+seed+".json" for seed, row in release["roles"].items()),
            "Three named frozen role descriptors")
    roles = {int(seed): read(bound(row)) for seed, row in release["roles"].items()}
    require(set(roles) == {1, 2, 3} and all(r["seed"] == seed and r["input_files"] == release["development_files"]
            for seed, r in roles.items()), "Three unchanged frozen role descriptors")
    schema = read(bound(release["schema_receipt"]))
    require(schema["status"] == "complete" and schema["schema_qualified"] is True and schema["all_ten_roles_qualified"] is True
            and schema["TEST_member_open_stat_hash_or_parse"] is False
            and schema["TEST_membership_known"] is False
            and schema["actual_input_schema"]["input_files"] == release["development_files"], "Reuse existing qualified full schema without TEST")
    for key, expected_seal in (("native_qualification_receipt", sha(PROJECT/sources["runtime_sources"]["native_seal"])),
                               ("candidate_qualification_receipt", METHOD_SEAL)):
        receipt = read(bound(release[key]))
        require(receipt["status"] == "complete" and receipt["complete"] is True and receipt["mode"] == "qualification"
                and receipt["source_seal_sha256"] == expected_seal
                and receipt["input_files"] == release["development_files"]
                and receipt["runtime"]["versions"] == release["expected_runtime_versions"]
                and receipt["runtime"]["math_flags"] == release["expected_math_flags"]
                and receipt["runtime"]["cuda_device_uuid"] == GPU
                and receipt["runtime"]["python_executable"] == release["python_executable"]
                and receipt["runtime"]["environment"] == release["runtime_environment"], "Exact already qualified full-input runtime/task")
        require(receipt["schema_receipt"]["sha256"] == release["schema_receipt"]["sha256"] if key == "native_qualification_receipt"
                else receipt["roles"] == release["roles"], "Same qualified schema and frozen candidate roles")
        if key == "candidate_qualification_receipt":
            require(receipt["qualification_passed"] is True and receipt["protocol_sha256"] == METHOD_PROTOCOL
                    and receipt["conditions"] == list(v3_gate.CONDITIONS) and len(receipt["runs"]) == 6
                    and all(r["complete"] and r["qualification_passed"] for r in receipt["runs"]), "Reuse complete six-path V3 pairing/serving qualification")
        else:
            require(receipt["native_backbone_numerically_qualified"] is True, "Ordinary native backbone qualification")
    require(release["resource_budget"]["root_owns_external_resource_monitor"] is True
            and release["resource_budget"]["host_RSS_bytes"] >= 32*1024**3
            and release["resource_budget"]["device_bytes"] >= 24*1024**3, "Complete-task resource budget; no shrink fallback")
    output = inside(release["output_directory"], exists=False)
    require(not output.exists() and not output.is_relative_to(HERE)
            and not output.is_relative_to(PROJECT/"typed_label_context_factor_source_prototype_20261010_v3")
            and inside(release["input_root"]).is_dir(), "Fresh separate output; unchanged source/input directory")
    return release, roles, sources, binding(path)
