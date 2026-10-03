"""Stdlib-only custody binder/sealer. Never imports Torch or numerical code."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
PROPOSAL = PROJECT / "postsubmission_research_20260930/graph_contrastive_private_paths_quality_gap_20261003_v1"
PINNED_PROPOSAL_MANIFEST = "c89df89038ec9548a93d647af0e5a5655116acc6951fe33d88516d7dad638ff6"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def binding(path, role):
    return {"path": path.relative_to(PROJECT).as_posix(), "sha256": digest(path),
            "size": path.stat().st_size, "role": role, "read_scope": "custody_hash_only"}


def write_json(name, content):
    (ROOT / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def bind_inputs():
    if digest(PROPOSAL / "MANIFEST.json") != PINNED_PROPOSAL_MANIFEST:
        raise RuntimeError("Immutable proposal manifest changed")
    original = json.loads((PROPOSAL / "MANIFEST.json").read_text())
    inputs = [binding(PROPOSAL / "MANIFEST.json", "immutable_contrastive_manifest")]
    for row in original["files"]:
        path = PROPOSAL / row["path"]
        if digest(path) != row["sha256"]:
            raise RuntimeError(f"Immutable proposal payload changed: {row['path']}")
        inputs.append(binding(path, "immutable_contrastive_payload"))
    protected = {
        "postsubmission_research_20260930/genn_primary_access_resolution_20261003_v1/MANIFEST.json": "8049b1915b0883159e14ec8113b368611ae4ac8f9c8bebfc42f52b26f814b703",
        "postsubmission_research_20260930/genn_primary_access_resolution_20261003_v1/REPORT.md": "d173a529db1dc5870a91076d732757cc5c23827bcccc9721ea6093bd7d225afb",
        "postsubmission_research_20260930/literature_memory/index_v31/LITERATURE_INDEX.json": "109cb9b981db5320d509df9d5b4e1bd44336690a954f2c0621c7ca8673365c0c",
    }
    for relative, expected in protected.items():
        path = PROJECT / relative
        if digest(path) != expected:
            raise RuntimeError(f"Protected prior changed: {relative}")
        inputs.append(binding(path, "protected_prior_custody"))
    index = PROJECT / "postsubmission_research_20260930/literature_memory/index_v31"
    for path in sorted(index.iterdir()):
        if path.is_file() and path.name != "LITERATURE_INDEX.json":
            inputs.append(binding(path, "protected_index_v31_support_file"))
    prior_receipt = ROOT / "PARENT_FUNCTIONAL_KERNEL_ATTRIBUTION.json"
    if prior_receipt.exists():
        receipt = json.loads(prior_receipt.read_text())
        for row in receipt.get("external_bindings", []):
            path = (PROJECT / row["path"]).resolve()
            path.relative_to(PROJECT)
            if digest(path) != row["sha256"]:
                raise RuntimeError("Parent functional-kernel attribution binding changed")
            inputs.append(binding(path, "parent_supplied_functional_kernel_overlap_attribution"))
    write_json("PROVENANCE.json", {
        "schema": "sealed-qualification-preparation-provenance-v1",
        "UTC": datetime.now(timezone.utc).isoformat(),
        "packet": ROOT.name,
        "source_proposal": PROPOSAL.relative_to(PROJECT).as_posix(),
        "proposal_manifest_sha256": PINNED_PROPOSAL_MANIFEST,
        "parent_scope": "source_and_numerical_qualification_preparation_only",
        "numerical_execution_authorized_in_this_packet": False,
        "launch_authorized": False,
        "new_primary_source_scopes": 0,
        "new_full_paper_read_certifications": 0,
        "literature_attribution": "Reused immutable proposal scopes; functional-kernel overlap supplied by parent/literature agent, no new primary read here",
        "external_bindings": inputs,
        "prohibited_actions_performed": [],
        "immutable_proposal_modified": False,
        "canonical_ledgers_or_status_modified": False,
        "complete_data_resource_measurements": None,
        "numerical_witnesses_status": "prepared_not_executed",
    })


def seal():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    if any("__pycache__" in p.parts or p.suffix == ".pyc" for p in files):
        raise RuntimeError("Bytecode/cached runtime artifacts are forbidden in the sealed preparation")
    if not (ROOT / "SOURCE_VERIFICATION.json").exists():
        raise RuntimeError("Record permitted source-only verification before sealing")
    write_json("MANIFEST.json", {
        "schema": "sha256-qualification-preparation-manifest-v1",
        "UTC": datetime.now(timezone.utc).isoformat(), "packet": ROOT.name,
        "excludes": ["MANIFEST.json (self)"], "payload_file_count": len(files),
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "size": p.stat().st_size}
                  for p in files],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-inputs-only", action="store_true")
    args = parser.parse_args()
    bind_inputs()
    if not args.bind_inputs_only:
        seal()
    print(json.dumps({"status": "inputs_bound" if args.bind_inputs_only else "sealed",
                      "numerical_execution": False, "launch_authorized": False}))
