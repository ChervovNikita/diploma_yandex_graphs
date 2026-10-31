"""All exact v4 synthetic fixtures: actual grouping vs standard RDKit stages.

Engineering diagnostic only. No dataset, graph targets, model, score,
fixture removal or grouping repair. Continue across failures so root can
see every mismatch before proposing another immutable source amendment.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "gine_transfer_grouping_fix_v4"
PARENT_MANIFEST_SHA256 = "50fa76a126cdbc269184a3a3857c93558eb4a1c5b3ad071c44e17260caa59861"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(value):
    return {"type": type(value).__name__, "message": str(value)}


def exact_fixtures():
    # Extract the sealed fixture dictionary with literal AST evaluation.
    tree = ast.parse((PARENT / "verify_chemical_groups.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    constants = {}
    fixture_node = None
    for node in function.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "metal_aryl":
                constants[target.id] = ast.literal_eval(node.value)
            if isinstance(target, ast.Name) and target.id == "fixtures":
                fixture_node = node.value
    class ResolveConstants(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in constants:
                raise ValueError("Unexpected nonliteral fixture variable:" + node.id)
            return ast.Constant(value=constants[node.id])
    return ast.literal_eval(ResolveConstants().visit(fixture_node))


def atom_state(mol):
    if mol is None:
        return None
    values = []
    for atom in mol.GetAtoms():
        value = {"index": atom.GetIdx(), "atomic_number": atom.GetAtomicNum(),
                 "isotope": atom.GetIsotope(), "aromatic": atom.GetIsAromatic(),
                 "degree": atom.GetDegree(), "formal_charge": atom.GetFormalCharge(),
                 "explicit_H": atom.GetNumExplicitHs(), "noImplicit": atom.GetNoImplicit(),
                 "radicals": atom.GetNumRadicalElectrons()}
        # Unsanitized raw molecules need not have an implicit-valence cache.
        if not atom.NeedsUpdatePropertyCache():
            try:
                value["implicit_H"] = atom.GetNumImplicitHs()
            except Exception as exc:
                value["implicit_H_error"] = error(exc)
        else:
            value["implicit_H"] = "uncomputed_before_property_cache"
        values.append(value)
    return {"atoms": values, "bonds": [
        {"begin": bond.GetBeginAtomIdx(), "end": bond.GetEndAtomIdx(),
         "type": str(bond.GetBondType()), "aromatic": bond.GetIsAromatic()}
        for bond in mol.GetBonds()]}


def stages(smiles, module, Chem):
    result = {}
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError("Raw parse returned None")
        result["raw_molecule"] = atom_state(mol)
        mol.UpdatePropertyCache(strict=False)
        result["permissive_cache_molecule"] = atom_state(mol)
        mol = module.normalize(mol)
        result["v4_normalized_molecule"] = atom_state(mol)
        result["v4_normalized_molecule"]["canonical"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold = Chem.MurckoDecompose(mol)
        result["after_MurckoDecompose"] = atom_state(scaffold)
        scaffold.ClearComputedProps()
        result["after_scaffold_ClearComputedProps"] = atom_state(scaffold)
        scaffold.UpdatePropertyCache(strict=False)
        result["permissive_cache_scaffold"] = atom_state(scaffold)
        scaffold = module.normalize(scaffold)
        result["v4_normalized_scaffold"] = atom_state(scaffold)
        result["v4_normalized_scaffold"]["canonical"] = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    except Exception as exc:
        result["failure"] = error(exc)
    return result


def run():
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if importlib.metadata.version("rdkit") != "2025.3.6":
        raise ValueError("Require exact separately pinned RDKit2025.3.6")
    manifest_path = PARENT / "DELIVERY_MANIFEST_v4.json"
    if sha(manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("Sealed parent delivery manifest changed")
    parent = json.loads(manifest_path.read_text())
    for name in ("chemical_groups.py", "verify_chemical_groups.py"):
        if sha(PARENT / name) != parent["files_sha256"][name]:
            raise ValueError("Sealed parent source changed:" + name)
    spec = importlib.util.spec_from_file_location("sealed_v4_chemistry", PARENT / "chemical_groups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for name, smiles in exact_fixtures().items():
        row = {"fixture": name, "synthetic_smiles": smiles}
        try:
            mol, canonical, scaffold = module.molecular_keys(smiles)
            row["actual_v4"] = {"status": "completed", "molecule": canonical,
                                "scaffold": scaffold, "molecule_state": atom_state(mol)}
            _, canonical2, scaffold2 = module.molecular_keys(canonical)
            row["actual_v4"]["canonical_round_trip"] = {
                "molecule": canonical2, "scaffold": scaffold2,
                "matches": (canonical, scaffold) == (canonical2, scaffold2)}
        except Exception as exc:
            row["actual_v4"] = {"status": "failed", "failure": error(exc)}
        row["v4_stage_states"] = stages(smiles, module, Chem)
        try:
            strict = Chem.MolFromSmiles(smiles)
            if strict is None:
                row["standard"] = {"status": "strict_parse_rejected"}
            else:
                strict_scaffold = MurckoScaffold.GetScaffoldForMol(strict)
                row["standard"] = {
                    "status": "completed", "molecule": Chem.MolToSmiles(strict, canonical=True, isomericSmiles=True),
                    "scaffold": Chem.MolToSmiles(strict_scaffold, canonical=True, isomericSmiles=False),
                    "molecule_state": atom_state(strict), "scaffold_state": atom_state(strict_scaffold)}
        except Exception as exc:
            row["standard"] = {"status": "failed", "failure": error(exc)}
        comparable = row["actual_v4"]["status"] == row["standard"]["status"] == "completed"
        row["strict_standard_comparable"] = comparable
        row["keys_match_standard"] = (row["actual_v4"]["molecule"] == row["standard"]["molecule"]
                                      and row["actual_v4"]["scaffold"] == row["standard"]["scaffold"]) if comparable else None
        rows.append(row)
    return {"schema": "gnnm-all-v4-synthetic-chemical-fixture-diagnostic-v1",
            "rdkit_distribution_version": importlib.metadata.version("rdkit"),
            "diagnostic_source_sha256": sha(Path(__file__)),
            "sealed_v4_delivery_manifest_sha256": PARENT_MANIFEST_SHA256,
            "sealed_v4_fixture_source_sha256": sha(PARENT / "verify_chemical_groups.py"),
            "sealed_v4_helper_source_sha256": sha(PARENT / "chemical_groups.py"),
            "procedure": module.PROCEDURE, "fixtures": rows,
            "mismatch_fixtures": [r["fixture"] for r in rows if r["keys_match_standard"] is False],
            "v4_failure_fixtures": [r["fixture"] for r in rows if r["actual_v4"]["status"] == "failed"],
            "all_exact_parent_fixtures_exercised": True,
            "synthetic_only": True, "dataset_or_labels_accessed": False,
            "model_or_predictive_score_accessed": False, "source_repair_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.report.resolve()
    if HERE not in output.parents:
        raise ValueError("Diagnostic report must stay in this new v5 sibling")
    if output.exists():
        raise ValueError("Refuse to overwrite diagnostic evidence")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "all_synthetic_fixtures_diagnosed", "fixtures": len(result["fixtures"]),
                      "mismatch_fixtures": result["mismatch_fixtures"],
                      "v4_failure_fixtures": result["v4_failure_fixtures"],
                      "report_sha256": sha(output), "dataset_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
