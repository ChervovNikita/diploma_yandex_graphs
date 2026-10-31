"""Root-only metadata staging for reviewed, finite zero-update source checks.

No CUDA/model/provider construction, optimizer, fit, outcome decoding or GO.
Only copies the explicitly reviewed helper package to a fresh project directory,
binds current frozen source studies/helpers and prepares engineering request JSON.
Run on allocation only after root source review; no network operations.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

STAGED_FILES = ("README.md", "PROPOSAL.json", "RISKS.json", "build_proposal.py", "qualify_exact_sources.py", "prepare_native_TRAIN_packs.py", "stage_admission.py")


def verify_route():
    if socket.gethostname() != "anogena-2-0":
        raise ValueError("Exact authorized allocation hostname required")
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    if result.returncode or result.stdout.split() != ["GPU-44039938-fd82-41d2-fefd-de71514e2fac"]:
        raise ValueError("Exact sole authorized allocation UUID required")


def local(root, relative):
    value = Path(relative); path = root / value
    if value.is_absolute() or ".." in value.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Confined project path required")
    return path


def receipt(root, path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve().relative_to(root)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def bound(root, row):
    path = local(root, row["path"])
    if receipt(root, path) != row:
        raise ValueError("Exact binding changed")
    return path


def allocation_study_relative(row):
    path = row["study_receipt"]["path"]
    for family in ("massive_private_bootstrap_packet_20261012_v1", "ogb_batchnorm_author_cli_packet_20261012_v1"):
        marker = "/allocation_artifacts/" + family + "/"
        if marker in path:
            return family + "/" + path.split(marker, 1)[1]
    return path


def stage(root, source, output, cap_helper, tensor_helper, runtime, *, reviewed, cases, prepared_packs=None):
    if output.exists():
        raise ValueError("Fresh stage required; never overwrite earlier failures/source receipts")
    proposal = json.loads((source / "PROPOSAL.json").read_text())
    if proposal["launch_enabled"] is not False or proposal["scientific_GO"] is not False or len(proposal["requested_cases"]) != 32:
        raise ValueError("Exact disabled finite32 proposal required")
    all_ids = {row["case_id"] for row in proposal["requested_cases"]}
    if cases and not set(cases) <= all_ids:
        raise ValueError("Requested case is outside finite proposal")
    selected_ids = list(cases) if cases else [row["case_id"] for row in proposal["requested_cases"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("No duplicate admission case")
    case_studies = {}; source_rows = {}; bank_shapes = {}
    author_study = None
    for case in proposal["requested_cases"]:
        path = local(root, allocation_study_relative(case))
        current = receipt(root, path)
        original = case["study_receipt"]
        if (current["bytes"], current["sha256"]) != (original["bytes"], original["sha256"]):
            raise ValueError("Current allocation source study differs from local reviewed mirror")
        value = json.loads(gzip.decompress(path.read_bytes()))
        case_studies[case["case_id"]] = current
        source_rows[current["path"]] = current
        if case["family"] == "ogb_author":
            author_study = current
        elif case["dataset"] not in bank_shapes:
            custodies = [row for row in value["source"]["files"] if Path(row["path"]).name == "CUSTODY.json"]
            if len(custodies) != 1:
                raise ValueError("One exact prepared node custody metadata receipt required")
            custody = json.loads(bound(root, custodies[0]).read_text())
            bank_shapes[case["dataset"]] = {role: max(row["role_counts"][role] for row in custody["splits"]) for role in ("cal", "evaluate")}
            bank_shapes[case["dataset"]]["custody_receipt"] = custodies[0]
    cap_receipt = receipt(root, local(root, cap_helper))
    tensor_receipt = receipt(root, local(root, tensor_helper))
    runtime_dir = local(root, runtime)
    runtime_receipts = [receipt(root, runtime_dir / name) for name in ("engine.py", "run_study.py", "spec.py", "registry.py")]
    molecular = {}
    for dataset in ("ogbg-molhiv", "ogbg-molbace"):
        molecular_study = local(root, "massive_molecular_packet_20261012_v3/matrix/studies/molecular-v2-" + dataset + "-e100.json.gz")
        value = json.loads(gzip.decompress(molecular_study.read_bytes()))
        # MolHIV v3 deliberately reuses closed v1 data custody. Do not assume
        # every current study owns a newly regenerated v3 dataset directory.
        manifests = [row for row in value["source"]["files"] if Path(row["path"]).name == "MANIFEST.json"]
        if len(manifests) != 1:
            raise ValueError("One existing molecular manifest custody required")
        bound(root, manifests[0]);molecular[dataset] = manifests[0]
    output.mkdir(parents=True)
    source_out = output / "source"; source_out.mkdir()
    for name in STAGED_FILES:
        shutil.copyfile(source / name, source_out / name)
    admission = {"schema": "root-approved-zero-update-resource-admission-v1",
                 "root_source_review_confirmed": bool(reviewed), "helper_sha256": receipt(root, source_out / "qualify_exact_sources.py")["sha256"],
                 "proposal_receipt": receipt(root, source_out / "PROPOSAL.json"), "scientific_GO": False, "max_actual_optimizer_updates": 0,
                 "approved_case_ids": selected_ids if reviewed else [], "reviewed_proposed_case_ids": selected_ids,
                 "case_study_receipts": case_studies, "role_optimizer_cap_helper": cap_receipt,
                 "tensor_commit_tree_helper": tensor_receipt, "runtime_package": str(runtime_dir.relative_to(root)),
                 "runtime_dependency_receipts": runtime_receipts, "node_bank_role_shapes": bank_shapes,
                 "TRAIN_packs": {}, "TRAIN_pack_source_bindings": {}, "TRAIN_pack_custodian_verified": {},
                 "two_process_pairs_approved": False, "no_automatic_retry": True,
                 "scope": "finite random-state source/restore/memory/CPU-bank-engineering only"}
    if prepared_packs:
        pack_path = local(root, prepared_packs)
        packs = json.loads(pack_path.read_text())
        if packs.get("schema") != "ogb-native-TRAIN-engineering-packs-preparation-v1" or packs.get("heldout_targets_decoded") is not False:
            raise ValueError("Reviewed CPU-only TRAIN pack closure required")
        for dataset, item in packs["datasets"].items():
            admission["TRAIN_pack_source_bindings"][dataset] = item["source_bindings"]
            admission["TRAIN_pack_custodian_verified"][dataset] = item["TRAIN_pack_custodian_verified"]
            for row in item["packs"].values():
                bound(root, row)
            for case in proposal["requested_cases"]:
                if case["family"] == "ogb_author" and case["dataset"] == dataset:
                    admission["TRAIN_packs"][case["case_id"]] = item["packs"]
        admission["TRAIN_pack_preparation_closure"] = receipt(root, pack_path)
    request = {"schema": "root-reviewed-native-TRAIN-pack-preparation-v1", "scientific_GO": False, "source_preparation_only": True,
               "root_source_review_confirmed": bool(reviewed),
               "preparer_sha256": receipt(root, source_out / "prepare_native_TRAIN_packs.py")["sha256"],
               "author_study": author_study, "molecular_manifests": molecular}
    (output / "ENGINEERING_ADMISSION.json").write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    (output / "TRAIN_PACK_PREPARATION_REQUEST.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    inventory = [receipt(root, source_out / name) for name in STAGED_FILES]
    inventory += [receipt(root, output / name) for name in ("ENGINEERING_ADMISSION.json", "TRAIN_PACK_PREPARATION_REQUEST.json")]
    (output / "STAGE_MANIFEST.json").write_text(json.dumps({"schema": "exact-source-resource-stage-v1", "files": inventory,
                                                          "inherited_helpers": [cap_receipt, tensor_receipt], "runtime": runtime_receipts,
                                                          "source_studies": list(source_rows.values()), "scientific_GO": False,
                                                          "CUDA_executions": 0, "scientific_fits": 0, "models_created": 0,
                                                          "root_source_review_confirmed": bool(reviewed)}, indent=2, sort_keys=True) + "\n")
    return {"stage": str(output.relative_to(root)), "approved_cases": len(admission["approved_case_ids"]),
            "author_TRAIN_packs_supplied": bool(prepared_packs), "scientific_GO": False, "CUDA_executions": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "source-directory", "output", "cap-helper", "tensor-helper", "runtime-package"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--root-reviewed", action="store_true", help="Root reviewed the exact helper/source package; engineering-only admission")
    parser.add_argument("--case-id", action="append", help="Admit only these exact proposed cases; omission keeps the proposed32 roster")
    parser.add_argument("--prepared-TRAIN-packs", dest="prepared_packs", help="Completed CPU-only PACKS.json, if available")
    args = parser.parse_args()
    verify_route()
    root = Path(args.project_root).resolve()
    result = stage(root, local(root, args.source_directory), local(root, args.output), args.cap_helper, args.tensor_helper,
                   args.runtime_package, reviewed=args.root_reviewed, cases=args.case_id, prepared_packs=args.prepared_packs)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
