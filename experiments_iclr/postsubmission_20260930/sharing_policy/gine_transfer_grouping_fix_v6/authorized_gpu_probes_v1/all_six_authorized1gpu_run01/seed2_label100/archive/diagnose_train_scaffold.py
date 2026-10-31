"""Unscored first v5 scaffold failure from exact TRAIN IDs and mapping only.

No torch, target cache, model, graph-label selection or prediction. CSV
parsing can internally materialize public fields; only TRAIN-row SMILES
are selected. Report v5 failure, v6 route and primary standard route.
"""
import argparse
import csv
import gzip
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from common import HERE, PHASE, confined, sha, write_json
from chemical_groups import molecular_keys
from mapping_identity import iter_train_smiles, MAPPING_README_SHA256


def failure(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def state(mol):
    if mol is None:
        return None
    return {"atoms": [{"index": a.GetIdx(), "atomic_number": a.GetAtomicNum(),
                       "aromatic": a.GetIsAromatic(), "formal_charge": a.GetFormalCharge(),
                       "explicit_H": a.GetNumExplicitHs(), "noImplicit": a.GetNoImplicit(),
                       "radicals": a.GetNumRadicalElectrons()}
                      for a in mol.GetAtoms()],
            "bonds": [{"begin": b.GetBeginAtomIdx(), "end": b.GetEndAtomIdx(),
                       "type": str(b.GetBondType()), "aromatic": b.GetIsAromatic()}
                      for b in mol.GetBonds()]}


def diagnose(args):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if importlib.metadata.version("rdkit") != "2025.3.6":
        raise ValueError("Require exact pinned RDKit2025.3.6")
    manifest_path = confined(args.data_manifest, PHASE)
    manifest = json.loads(manifest_path.read_text())
    split, mapping = (confined(p, PHASE) for p in (args.split_train, args.molecule_map))
    expected = {str((manifest_path.parent / row["relative_path"]).resolve()): row["sha256"]
                for row in manifest["raw_and_processed_files"]}
    for path in (split, mapping):
        if expected.get(str(path)) != sha(path):
            raise ValueError("TRAIN-ID/mapping input differs from existing frozen manifest")
    readme = mapping.with_name("README.md")
    if expected.get(str(readme)) != MAPPING_README_SHA256 or sha(readme) != MAPPING_README_SHA256:
        raise ValueError("Exact official row-ID README binding differs")
    with gzip.open(split, "rt", newline="") as stream:
        train_ids = [int(row[0]) for row in csv.reader(stream)]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate official TRAIN ID")
    old_folder = HERE.parent / "gine_transfer_grouping_fix_v5"
    old_manifest = json.loads((old_folder / "DELIVERY_MANIFEST_v5.json").read_text())
    if (sha(old_folder / "DELIVERY_MANIFEST_v5.json")
            != "0f6c98e256f0a1c516e40b365552d7ad04e95d8b42a398040e6d8bb14d1e5eb2"
            or sha(old_folder / "chemical_groups.py") != old_manifest["files_sha256"]["chemical_groups.py"]):
        raise ValueError("Failed sealed v5 grouping source changed")
    spec = importlib.util.spec_from_file_location("failed_v5_chemistry", old_folder / "chemical_groups.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    report = {"schema": "gnnm-unscored-TRAIN-scaffold-failure-diagnostic-v1",
              "diagnostic_source_sha256": sha(Path(__file__)),
              "v5_helper_sha256": sha(old_folder / "chemical_groups.py"),
              "v6_helper_sha256": sha(HERE / "chemical_groups.py"),
              "rdkit_version": importlib.metadata.version("rdkit"),
              "data_manifest_sha256": sha(manifest_path),
              "input_sha256": {str(p.relative_to(PHASE)): sha(p) for p in (split, mapping, readme)},
              "target_cache_or_torch_loaded": False, "labels_or_scores_selected": False,
              "only_official_TRAIN_row_SMILES_selected": True,
              "CSV_internal_public_heldout_field_materialization_possible": True,
              "graph_topology_gate_or_complete_export_established": False,
              "first_failure": None}
    visited = 0
    with gzip.open(mapping, "rt", newline="") as stream:
        for official, smiles in iter_train_smiles(stream, train_ids):
            visited += 1
            try:
                old.molecular_keys(smiles)
            except Exception as exc:
                row = {"official_train_row_index": official, "training_smiles": smiles,
                       "v5_failure": failure(exc)}
                parser = Chem.SmilesParserParams()
                parser.sanitize, parser.removeHs = False, True
                mol = Chem.MolFromSmiles(smiles, parser)
                row["v5_raw_parsed_molecule"] = state(mol)
                try:
                    normalized = old.normalize(mol)
                    row["v5_normalized_molecule"] = state(normalized)
                    raw_scaffold = Chem.MurckoDecompose(normalized)
                    row["post_Murcko_scaffold"] = state(raw_scaffold)
                    raw_scaffold.ClearComputedProps()
                    raw_scaffold.UpdatePropertyCache(strict=False)
                    Chem.GetSymmSSSR(raw_scaffold)
                    row["published_permissive_cache_ring_scaffold"] = {
                        "state": state(raw_scaffold),
                        "canonical": Chem.MolToSmiles(raw_scaffold, canonical=True, isomericSmiles=False)}
                except Exception as inner:
                    row["molecule_or_published_scaffold_stage_failure"] = failure(inner)
                try:
                    candidate, canonical, scaffold = molecular_keys(smiles)
                    row["v6"] = {"status": "completed", "molecule": canonical,
                                 "scaffold": scaffold, "molecule_state": state(candidate)}
                except Exception as inner:
                    row["v6"] = {"status": "failed", "failure": failure(inner)}
                try:
                    strict = Chem.MolFromSmiles(smiles)
                    if strict is None:
                        row["standard"] = {"status": "strict_parse_rejected"}
                    else:
                        standard_scaffold = MurckoScaffold.GetScaffoldForMol(strict)
                        row["standard"] = {
                            "status": "completed",
                            "molecule": Chem.MolToSmiles(strict, canonical=True, isomericSmiles=True),
                            "scaffold": Chem.MolToSmiles(standard_scaffold, canonical=True, isomericSmiles=False),
                            "scaffold_state": state(standard_scaffold)}
                except Exception as inner:
                    row["standard"] = {"status": "failed", "failure": failure(inner)}
                row["v6_keys_match_standard"] = (
                    row["v6"].get("molecule") == row["standard"].get("molecule")
                    and row["v6"].get("scaffold") == row["standard"].get("scaffold")) if (
                    row["v6"]["status"] == row["standard"]["status"] == "completed") else None
                report["first_failure"] = row
                break
    report["TRAIN_mapping_rows_visited_until_first_failure"] = visited
    report["status"] = "first_v5_TRAIN_failure_diagnosed" if report["first_failure"] else "no_v5_failure_found"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--split-train", type=Path, required=True)
    parser.add_argument("--molecule-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args)
    write_json(args.report, result)
    row = result["first_failure"]
    print(json.dumps({"status": result["status"], "report_sha256": sha(args.report),
                      "v6_status": row["v6"]["status"] if row else None,
                      "v6_keys_match_standard": row["v6_keys_match_standard"] if row else None,
                      "labels_or_scores_selected": False}, indent=2))


if __name__ == "__main__":
    main()
