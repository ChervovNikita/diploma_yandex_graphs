"""Confined three-fixture aromatic-N diagnosis; no dataset or labels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def states(mol):
    return [{"index": atom.GetIdx(), "atomic_number": atom.GetAtomicNum(),
             "aromatic": atom.GetIsAromatic(), "degree": atom.GetDegree(),
             "formal_charge": atom.GetFormalCharge(), "explicit_H": atom.GetNumExplicitHs(),
             "noImplicit": atom.GetNoImplicit(), "radicals": atom.GetNumRadicalElectrons()}
            for atom in mol.GetAtoms()]


def run():
    import rdkit
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    source = HERE.parent / "gine_transfer_grouping_fix_v3/chemical_groups.py"
    spec = importlib.util.spec_from_file_location("gnnm_failed_grouping_v3_diagnostic", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    details = []
    for name, smiles in (("pyrrole_H", "[nH]1cccc1"), ("pyrrole_N_methyl", "Cn1cccc1"), ("pyridine", "n1ccccc1")):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        row = {"fixture": name, "raw_atoms": states(mol)}
        try:
            normalized = module.normalize(Chem.Mol(mol))
            row["normalized_molecule_atoms"] = states(normalized)
            row["normalized_molecule_canonical"] = Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)
            scaffold = Chem.MurckoDecompose(normalized)
            row["MurckoDecompose_atoms"] = states(scaffold)
            scaffold.ClearComputedProps()
            scaffold.UpdatePropertyCache(strict=False)
            row["permissive_cache_atoms"] = states(scaffold)
            try:
                result = module.normalize(Chem.Mol(scaffold))
                row["v3_scaffold_normalize"] = {"status": "passed", "atoms": states(result),
                                                  "canonical": Chem.MolToSmiles(result, canonical=True, isomericSmiles=False)}
            except Exception as error:
                row["v3_scaffold_normalize"] = {"status": "failed", "type": type(error).__name__, "message": str(error)}
            # Match installed helper, changing only strict property caching.
            reference_permissive = Chem.Mol(scaffold)
            Chem.GetSymmSSSR(reference_permissive)
            row["installed_helper_permissive_cache_only"] = {
                "atoms": states(reference_permissive),
                "canonical": Chem.MolToSmiles(reference_permissive, canonical=True, isomericSmiles=False)}
            stage = Chem.Mol(scaffold)
            operations = [("FINDRADICALS", Chem.SanitizeFlags.SANITIZE_FINDRADICALS),
                          ("KEKULIZE", Chem.SanitizeFlags.SANITIZE_KEKULIZE)]
            row["scaffold_single_operation_steps"] = []
            for step, operation in operations:
                try:
                    failed = Chem.SanitizeMol(stage, sanitizeOps=operation, catchErrors=True)
                    row["scaffold_single_operation_steps"].append({"step": step, "failed_operation": int(failed), "atoms": states(stage)})
                    if int(failed):
                        break
                except Exception as error:
                    row["scaffold_single_operation_steps"].append({"step": step, "type": type(error).__name__, "message": str(error)})
                    break
        except Exception as error:
            row["normalized_molecule_error"] = {"type": type(error).__name__, "message": str(error)}
        standard = Chem.MolFromSmiles(smiles)
        standard_scaffold = MurckoScaffold.GetScaffoldForMol(standard)
        row["published_standard_fixture_route"] = {"molecule_atoms": states(standard),
                "scaffold_atoms": states(standard_scaffold),
                "scaffold_canonical": Chem.MolToSmiles(standard_scaffold, canonical=True, isomericSmiles=False)}
        details.append(row)
    return {"schema": "gnnm-aromatic-N-synthetic-stage-diagnostic-v1", "rdkit_version": rdkit.__version__,
            "diagnostic_source_sha256": sha(Path(__file__)), "failed_v3_source_sha256": sha(source),
            "pinned_Murcko_module_sha256": sha(Path(MurckoScaffold.__file__)),
            "fixtures": details, "dataset_or_labels_accessed": False,
            "predictive_claim": False, "synthetic_fixture_SMILES_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if HERE not in output.parents:
        raise ValueError("Output must remain inside prospective v4 folder")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(run(), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha(output), "dataset_accessed": False}))


if __name__ == "__main__":
    main()
