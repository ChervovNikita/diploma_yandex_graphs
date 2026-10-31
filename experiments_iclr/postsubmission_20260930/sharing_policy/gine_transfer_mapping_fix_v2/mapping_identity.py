"""Official OGB mapping identity is zero-based CSV data-row position.

mol_id may be blank/non-numeric and is never interpreted as a graph ID.
DictReader may internally materialize public label/SMILES fields for all
rows. The exporter selects only training-row SMILES, never label columns.
"""
import csv


MAPPING_README_SHA256 = "e6b64bdc76f68f18bb3c4c9c9826eefa64be48748848670f6c5a11835d618c3f"
IDENTITY_RULE = "zero-based mol.csv.gz data row index; mol_id is ignored"


def iter_train_smiles(stream, wanted):
    wanted = set(wanted)
    if any(type(index) is not int or index < 0 for index in wanted):
        raise ValueError("Official graph IDs must be nonnegative integer row positions")
    reader = csv.DictReader(stream)
    if not reader.fieldnames or "smiles" not in reader.fieldnames:
        raise ValueError("Frozen molecule mapping lacks a smiles column")
    for row_index, row in enumerate(reader):
        if row_index not in wanted:
            continue
        yield row_index, row["smiles"]
