"""Version-pinned molecule preparation and published Murcko scaffold route.

Uses explicit standard input-H removal and partial+ADJUSTHS normalization
for every molecule. Every Murcko scaffold follows the published cache/rings
route, with the same permissive property cache for documented metal valence. No strict-parse branch, graph deletion, singleton fallback,
feature conversion, label, score or chemical-validity assertion exists.
"""
import importlib.metadata


PROCEDURE = {
    "name": "rdkit2025.03.6-uniform-molecule-partial-adjustHs-published-murcko-v4",
    "parse": "MolFromSmiles with explicit SmilesParserParams(sanitize=False,removeHs=True)",
    "input_hydrogens": "standard parser RemoveHs(updateExplicitCount=True,sanitize=False); isotope/disconnected/default exceptions retained by RDKit, never blanket-delete H atoms",
    "property_cache": "UpdatePropertyCache(strict=False)",
    "partial_flags": ["SANITIZE_FINDRADICALS", "SANITIZE_KEKULIZE", "SANITIZE_SETAROMATICITY",
                      "SANITIZE_SETCONJUGATION", "SANITIZE_SETHYBRIDIZATION", "SANITIZE_SYMMRINGS",
                      "SANITIZE_ADJUSTHS"],
    "hydrogen_bookkeeping": "standard MolOps.adjustHs stage restores disappearing aromatic H; never blanket-sets aromatic nitrogen hydrogens",
    "stereo": "AssignStereochemistry(cleanIt=True,force=True)",
    "molecule_identity": "canonical MolToSmiles(isomericSmiles=True)",
    "scaffold": "published GetScaffoldForMol route: MurckoDecompose; ClearComputedProps; UpdatePropertyCache(strict=False); GetSymmSSSR; stereo-free canonical SMILES; no second molecule sanitization",
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


def prepare_scaffold(mol):
    """Published scaffold route; permissive cache is the sole helper change."""
    from rdkit import Chem
    scaffold = Chem.MurckoDecompose(mol)
    before = topology(scaffold)
    scaffold.ClearComputedProps()
    scaffold.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(scaffold)
    if topology(scaffold) != before:
        raise ValueError("Scaffold cache/ring preparation changed Murcko topology")
    return scaffold


def molecular_keys(smiles):
    from rdkit import Chem
    if importlib.metadata.version("rdkit") != "2025.3.6":
        raise ValueError("Use the separately pinned RDKit2025.3.6 runtime")
    parser = Chem.SmilesParserParams()
    parser.sanitize = False
    parser.removeHs = True
    mol = Chem.MolFromSmiles(smiles, parser)
    if mol is None:
        raise ValueError("Official training SMILES cannot be parsed; export must stop")
    mol = normalize(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = prepare_scaffold(mol)
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
