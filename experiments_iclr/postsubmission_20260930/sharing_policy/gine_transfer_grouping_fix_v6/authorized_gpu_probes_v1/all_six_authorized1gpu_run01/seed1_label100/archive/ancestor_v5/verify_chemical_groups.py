"""Metal/organic partial-sanitization fixtures; no dataset or training."""
import argparse
import hashlib
import json
from pathlib import Path

from chemical_groups import PROCEDURE, molecular_keys, topology


def run():
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    metal_aryl = "[Al](c1ccccc1)(c1ccccc1)(c1ccccc1)(c1ccccc1)(c1ccccc1)(c1ccccc1)c1ccccc1"
    fixtures = {
        "benzene_aromatic": "c1ccccc1", "benzene_kekule": "C1=CC=CC=C1",
        "toluene": "Cc1ccccc1", "toluene_reordered": "c1ccccc1C",
        "ethylbenzene": "CCc1ccccc1", "pyrrole": "[nH]1cccc1",
        "N_methyl_pyrrole": "Cn1cccc1", "pyridine": "n1ccccc1",
        "alanine_left": "N[C@@H](C)C(=O)O", "alanine_right": "N[C@H](C)C(=O)O",
        "disconnected_salt": "[Na+].[Cl-]", "explicit_hydrogen": "[H]OC",
        "isotopic_hydrogen": "[2H]OC", "hydrogen_molecule": "[H][H]",
        "isolated_hydrogen": "[H]",
        "aluminium_seven_ligands": "[Al](F)(F)(F)(F)(F)(F)F",
        "aluminium_seven_ligands_reordered": "F[Al](F)(F)(F)(F)(F)F",
        "aluminium_seven_aryl_ligands": metal_aryl,
        "cookbook_hypervalent_P": "F[P-](F)(F)(F)(F)F.CN(C)C(F)=[N+](C)C",
    }
    values, details, nitrogen_states, hydrogen_topologies = {}, [], {}, {}
    for name, smiles in fixtures.items():
        mol, canonical, scaffold = molecular_keys(smiles)
        if name in ("explicit_hydrogen", "isotopic_hydrogen", "hydrogen_molecule", "isolated_hydrogen"):
            hydrogen_topologies[name] = {"atomic_numbers": [a.GetAtomicNum() for a in mol.GetAtoms()],
                                         "isotopes": [a.GetIsotope() for a in mol.GetAtoms()],
                                         "bond_pairs": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]}
        # A uniform canonical round-trip is checked without strict valence parsing.
        replay, second, second_scaffold = molecular_keys(canonical)
        if second != canonical or second_scaffold != scaffold:
            raise AssertionError("Version-pinned permissive canonicalization does not round-trip:" + name)
        values[name] = (canonical, scaffold)
        if name in ("pyrrole", "N_methyl_pyrrole", "pyridine"):
            scaffold_mol = Chem.MolFromSmiles(scaffold)
            assert scaffold_mol is not None
            nitrogen_states[name] = {
                "molecule_aromatic_N_explicit_H": [a.GetNumExplicitHs() for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetIsAromatic()],
                "scaffold_aromatic_N_explicit_H": [a.GetNumExplicitHs() for a in scaffold_mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetIsAromatic()],
                "canonical": canonical, "scaffold": scaffold}
        details.append({"fixture": name, "atoms": mol.GetNumAtoms(), "bonds": mol.GetNumBonds(),
                        "molecule_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                        "scaffold_sha256": hashlib.sha256(scaffold.encode()).hexdigest(),
                        "empty_scaffold": scaffold == "", "permissive_round_trip": True})
        if not name.startswith("aluminium") and name != "cookbook_hypervalent_P":
            strict = Chem.MolFromSmiles(smiles)
            assert strict is not None
            strict_molecule = Chem.MolToSmiles(strict, canonical=True, isomericSmiles=True)
            strict_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(strict), canonical=True, isomericSmiles=False)
            if (canonical, scaffold) != (strict_molecule, strict_scaffold):
                raise AssertionError("Organic fixture grouping differs from documented standard RDKit route:" + name)
    assert hydrogen_topologies["explicit_hydrogen"]["atomic_numbers"] == [8, 6]
    assert hydrogen_topologies["explicit_hydrogen"]["bond_pairs"] == [[0, 1]]
    assert hydrogen_topologies["isotopic_hydrogen"]["atomic_numbers"] == [1, 8, 6]
    assert hydrogen_topologies["isotopic_hydrogen"]["isotopes"][0] == 2
    assert hydrogen_topologies["hydrogen_molecule"]["atomic_numbers"] == [1, 1]
    assert hydrogen_topologies["isolated_hydrogen"]["atomic_numbers"] == [1]
    assert values["benzene_aromatic"] == values["benzene_kekule"]
    assert values["toluene"] == values["toluene_reordered"]
    assert values["toluene"][1] == values["ethylbenzene"][1] == values["benzene_aromatic"][1]
    assert values["pyrrole"][1] == values["N_methyl_pyrrole"][1] == "c1cc[nH]c1"
    assert values["pyridine"][1] == "c1ccncc1"
    assert values["pyridine"][1] != values["pyrrole"][1]
    assert nitrogen_states["pyrrole"]["molecule_aromatic_N_explicit_H"] == [1]
    assert nitrogen_states["N_methyl_pyrrole"]["molecule_aromatic_N_explicit_H"] == [0]
    assert nitrogen_states["pyrrole"]["scaffold_aromatic_N_explicit_H"] == [1]
    assert nitrogen_states["N_methyl_pyrrole"]["scaffold_aromatic_N_explicit_H"] == [1]
    assert nitrogen_states["pyridine"]["molecule_aromatic_N_explicit_H"] == [0]
    assert nitrogen_states["pyridine"]["scaffold_aromatic_N_explicit_H"] == [0]
    assert values["alanine_left"][0] != values["alanine_right"][0]
    assert values["aluminium_seven_ligands"] == values["aluminium_seven_ligands_reordered"]
    assert values["aluminium_seven_aryl_ligands"][1] != ""
    return {"status": "passed", "scope": "version-pinned metal/organic grouping fixtures only",
            "procedure": PROCEDURE, "fixtures": details, "nitrogen_H_states": nitrogen_states, "hydrogen_topologies": hydrogen_topologies, "molecules_dropped": 0,
            "singleton_exception_fallbacks": 0, "features_recomputed_or_changed": False,
            "labels_or_dataset_accessed": False, "predictive_claim": False,
            "chemical_validity_claim": False, "all_actual_training_topology_checks_pending": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    here, output = Path(__file__).resolve().parent, args.report.resolve()
    if here not in output.parents:
        raise ValueError("Report must remain in grouping-v5 packet")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = run()
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
