"""Uniform, version-pinned partial sanitization for grouping only.

Uses the RDKit Cookbook partial-valence recipe for EVERY molecule and
scaffold. No strict-parse branch, graph deletion, singleton fallback,
feature conversion, label, score or chemical-validity assertion exists.
"""
import importlib.metadata


PROCEDURE = {
    "name": "rdkit2025.03.6-uniform-partial-sanitization-murcko-v1",
    "parse": "MolFromSmiles(sanitize=False)",
    "property_cache": "UpdatePropertyCache(strict=False)",
    "partial_flags": ["SANITIZE_FINDRADICALS", "SANITIZE_KEKULIZE", "SANITIZE_SETAROMATICITY",
                      "SANITIZE_SETCONJUGATION", "SANITIZE_SETHYBRIDIZATION", "SANITIZE_SYMMRINGS"],
    "stereo": "AssignStereochemistry(cleanIt=True,force=True)",
    "molecule_identity": "canonical MolToSmiles(isomericSmiles=True)",
    "scaffold": "MurckoDecompose; ClearComputedProps; same permissive cache+partial normalization; stereo-free canonical SMILES",
    "failures": "stop whole export; no deletion, strict/permissive case selection, or exception singleton",
    "limits": "grouping-only graph identity; not proof of chemical validity, tautomer/pH normalization or provider-era exact chemistry",
}


def topology(mol):
    atoms = tuple(atom.GetAtomicNum() for atom in mol.GetAtoms())
    bonds = sorted(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds())
    return atoms, tuple(bonds)


def normalize(mol):
    from rdkit import Chem
    before = topology(mol)
    mol.UpdatePropertyCache(strict=False)
    operations = Chem.SanitizeFlags.SANITIZE_NONE
    for name in PROCEDURE["partial_flags"]:
        operations |= getattr(Chem.SanitizeFlags, name)
    failed = Chem.SanitizeMol(mol, sanitizeOps=operations, catchErrors=True)
    if int(failed) != 0:
        raise ValueError("Uniform partial sanitization failed operation:" + str(int(failed)))
    mol.UpdatePropertyCache(strict=False)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if topology(mol) != before:
        raise ValueError("Grouping normalization changed atom/connectivity topology")
    return mol


def molecular_keys(smiles):
    from rdkit import Chem
    if importlib.metadata.version("rdkit") != "2025.3.6":
        raise ValueError("Use the separately pinned RDKit2025.3.6 runtime")
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Official training SMILES cannot be parsed; export must stop")
    mol = normalize(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    # This is the installed GetScaffoldForMol topology operation, without its
    # strict UpdatePropertyCache() call that rejects some released graph inputs.
    scaffold = Chem.MurckoDecompose(mol)
    scaffold.ClearComputedProps()
    scaffold = normalize(scaffold)
    scaffold_string = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    return mol, canonical, scaffold_string


def require_frozen_topology(mol, graph):
    """Compare input atoms/connectivity only; never modify OGB features."""
    atoms, bonds = topology(mol)
    atom_categories = [number - 1 if 1 <= number <= 118 else 118 for number in atoms]
    if graph["x"][:, 0].tolist() != atom_categories:
        raise ValueError("Parsed atom order/atomic numbers differ from frozen OGB graph")
    expected_edges = sorted(edge for a, b in bonds for edge in ((a, b), (b, a)))
    actual_edges = sorted(tuple(pair) for pair in graph["edge_index"].T.tolist())
    if actual_edges != expected_edges:
        raise ValueError("Parsed connectivity differs from frozen inverse-edge OGB graph")
    return True
