"""Capture installed phase-local grouping APIs without accessing a dataset."""
import hashlib
import inspect
import json
import os
from pathlib import Path

from rdkit import Chem, __version__
from rdkit.Chem.Scaffolds import MurckoScaffold

PHASE = Path(__file__).resolve().parents[1]
if os.environ.get("GNNM_PHASE_ROOT") != str(PHASE):
    raise ValueError("Require explicit repository wrapper")
module = Path(MurckoScaffold.__file__).resolve()
if PHASE not in module.parents:
    raise ValueError("Only capture the installed phase-local RDKit dependency")
output = PHASE / "protocols/rdkit_api_evidence_v1.json"
value = {"rdkit_version": __version__, "dataset_accessed": False,
         "module_path": str(module.relative_to(PHASE)),
         "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
         "MurckoScaffold_GetScaffoldForMol_source": inspect.getsource(MurckoScaffold.GetScaffoldForMol),
         "API_docstrings": {name: getattr(Chem, name).__doc__ for name in
                            ["MolFromSmiles", "MolToSmiles", "GetSymmSSSR", "SetAromaticity",
                             "AssignStereochemistry", "SanitizeMol", "MurckoDecompose"]},
         "UpdatePropertyCache_doc": Chem.Mol.UpdatePropertyCache.__doc__,
         "SanitizeFlags": {name: int(getattr(Chem.SanitizeFlags, name)) for name in
                           dir(Chem.SanitizeFlags) if name.startswith("SANITIZE_")},
         "scientific_or_predictive_claim": False}
with output.open("x") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
print(json.dumps({"rdkit_version": __version__, "module_sha256": value["module_sha256"],
                  "output": str(output.relative_to(PHASE)), "dataset_accessed": False}), flush=True)
