"""Standard-library mapping-ID guard; fixture only, no dataset access."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from mapping_identity import IDENTITY_RULE, MAPPING_README_SHA256, iter_train_smiles


def run():
    here = Path(__file__).resolve().parent
    readme = here / "provider_evidence/mapping_README.md"
    fixture = here / "provider_evidence/ID_FIXTURE.csv"
    assert hashlib.sha256(readme.read_bytes()).hexdigest() == MAPPING_README_SHA256
    with fixture.open(newline="") as stream:
        values = list(iter_train_smiles(stream, [0, 2, 4]))
    assert values == [(0, "C"), (2, "N"), (4, "O")]
    accesses = []
    class GuardedRow(dict):
        def __init__(self, index):
            self.index = index
        def __getitem__(self, field):
            accesses.append([self.index, field])
            if self.index not in (0, 2, 4) or field != "smiles":
                raise AssertionError("Unexpected heldout row or mol_id/target access")
            return {0: "C", 2: "N", 4: "O"}[self.index]
    class GuardedReader:
        fieldnames = ["task_1", "smiles", "mol_id"]
        def __iter__(self):
            return iter([GuardedRow(index) for index in range(5)])
    with patch("mapping_identity.csv.DictReader", return_value=GuardedReader()):
        assert list(iter_train_smiles(None, [0, 2, 4])) == values
    assert accesses == [[0, "smiles"], [2, "smiles"], [4, "smiles"]]
    return {"status": "passed", "scope": "stdlib synthetic row-ID/column-access guard",
            "identity_rule": IDENTITY_RULE, "selected_row_ids": [0, 2, 4],
            "blank_and_non_numeric_mol_id_accepted": True, "mol_id_or_label_column_selected": False,
            "nontraining_smiles_selected": False,
            "CSV_internal_public_heldout_field_materialization_possible": True,
            "mapping_README_sha256": MAPPING_README_SHA256,
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "dataset_or_predictive_scores_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    here, output = Path(__file__).resolve().parent, args.report.resolve()
    if here not in output.parents:
        raise ValueError("Report must remain in v2 packet")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = run()
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
