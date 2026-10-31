"""Export only official TRAIN graphs from an existing frozen MolHIV cache.

No provider constructor/download, validation graph or predictive score
exists. The all-graph processed torch file DOES materialize its stored
target tensor. CSV parsing can internally materialize public heldout
label columns and SMILES strings; only training-row SMILES and training
tensor targets are selected or exported. These limitations are recorded.
The resulting bundle contains training tensors only. Run separately from
the selector, before prospectively freezing folds. No model is fitted.
"""
import argparse
import csv
import gzip
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from common import HERE, PHASE, confined, sha, source_hashes, write_json
import torch
from torch_geometric.data.separate import separate
from recurrence_collector import tensor_hash
from shareable_guard import require_shareable_manifest
from mapping_identity import IDENTITY_RULE, MAPPING_README_SHA256, iter_train_smiles
from chemical_groups import PROCEDURE, molecular_keys, require_frozen_topology


def simple_hash(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def grouping(train_ids, mapping_path, frozen_graphs):
    from rdkit.Chem.Scaffolds import MurckoScaffold
    wanted, molecules, scaffolds = set(train_ids), {}, {}
    graph_by_id = dict(zip(train_ids, frozen_graphs))
    if len(graph_by_id) != len(train_ids):
        raise ValueError("Frozen train graph coverage differs")
    with gzip.open(mapping_path, "rt", newline="") as stream:
        for official, training_smiles in iter_train_smiles(stream, wanted):
            if official in molecules:
                raise ValueError("Duplicate mapping row ID")
            molecule, canonical, scaffold_string = molecular_keys(training_smiles)
            require_frozen_topology(molecule, graph_by_id[official])
            molecules[official], scaffolds[official] = simple_hash(canonical), simple_hash(scaffold_string)
    if set(molecules) != wanted:
        raise ValueError("Official training IDs not exactly covered by existing mapping")
    # Union enforces both relationships, including the single empty-scaffold group.
    parent = {key: key for key in train_ids}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for identities in (molecules, scaffolds):
        seen = {}
        for key in sorted(train_ids):
            digest = identities[key]
            if digest in seen:
                a, b = find(key), find(seen[digest])
                parent[max(a, b)] = min(a, b)
            else:
                seen[digest] = key
    groups = {str(key): "molhiv:train:group:" + str(find(key)) for key in train_ids}
    sizes = {}
    for group in groups.values():
        sizes[group] = sizes.get(group, 0) + 1
    return groups, {
        "rule": "exact canonical isomeric molecule OR stereo-free Bemis-Murcko scaffold union; no thresholds",
        "empty_scaffold_policy": "all empty scaffolds in the same group; no favorable balancing exception",
        "rdkit_version": importlib.metadata.version("rdkit"),
        "MurckoScaffold_source_sha256": simple_hash(inspect.getsource(MurckoScaffold)),
        "molecule_digest_by_official_train_id": {str(k): v for k, v in molecules.items()},
        "scaffold_digest_by_official_train_id": {str(k): v for k, v in scaffolds.items()},
        "groups": len(sizes), "largest_group_graphs": max(sizes.values()),
        "groups_chosen_using_labels_or_metrics": False,
        "mapping_file_sha256": sha(mapping_path),
        "official_graph_identity": IDENTITY_RULE,
        "mapping_identity_README_sha256": MAPPING_README_SHA256,
        "uniform_grouping_procedure": PROCEDURE,
        "all_train_graphs_retained": True,
        "all_train_input_atom_connectivity_checks_passed": True,
        "original_x_edge_attr_and_edge_index_unchanged": True,
        "molecules_dropped_or_exception_singletons": 0,
    }


def export(args):
    manifest_path = confined(args.data_manifest, PHASE)
    data_manifest = json.loads(manifest_path.read_text())
    require_shareable_manifest(data_manifest, "molhiv")
    paths = [confined(p, PHASE) for p in (args.processed, args.split_train, args.molecule_map)]
    expected = {str((manifest_path.parent / row["relative_path"]).resolve()): row["sha256"]
                for row in data_manifest["raw_and_processed_files"]}
    for path in paths:
        if expected.get(str(path)) != sha(path):
            raise ValueError("Input file not exactly bound by the frozen provider manifest")
    mapping_readme = confined(paths[2].with_name("README.md"), PHASE)
    if (expected.get(str(mapping_readme)) != MAPPING_README_SHA256
            or sha(mapping_readme) != MAPPING_README_SHA256
            or sha(HERE / "provider_evidence/mapping_README.md") != MAPPING_README_SHA256):
        raise ValueError("Official row-identity README differs from exact bound provider evidence")
    with gzip.open(paths[1], "rt", newline="") as stream:
        train_ids = [int(row[0]) for row in csv.reader(stream)]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate official training index")
    fingerprints = data_manifest["split_summaries"]["train"]["graph_tensor_fingerprints"]
    if [row["official_index"] for row in fingerprints] != train_ids:
        raise ValueError("Train split order differs from immutable data freeze")
    # This existing all-target tensor materialization is recorded, never disguised.
    # No target outside train_ids is selected, hashed, summarized or exported.
    # Public target columns can additionally materialize inside csv.DictReader.
    cached = torch.load(paths[0], map_location="cpu", weights_only=False)
    if not isinstance(cached, tuple) or len(cached) != 2:
        raise ValueError("Expected frozen OGB (PyG Data, slices) cache")
    storage, slices = cached
    graphs = []
    for official, expected_graph in zip(train_ids, fingerprints):
        graph = separate(cls=storage.__class__, batch=storage, idx=official,
                         slice_dict=slices, decrement=False)
        for field in ("x", "edge_index", "edge_attr", "y"):
            if tensor_hash(getattr(graph, field)) != expected_graph[field]:
                raise ValueError("Official train graph tensor differs: " + field)
        if int(graph.num_nodes) != expected_graph["num_nodes"]:
            raise ValueError("Official train graph node count differs")
        graphs.append({field: getattr(graph, field).clone() for field in ("x", "edge_index", "edge_attr", "y")})
    groups, group_metadata = grouping(train_ids, paths[2], graphs)
    payload = {"schema": "gnnm-molhiv-official-train-only-bundle-v1", "dataset": "ogbg-molhiv",
               "official_train_indices": train_ids, "graphs": graphs,
               "group_by_official_train_id": groups, "grouping": group_metadata,
               "data_manifest_sha256": sha(manifest_path),
               "source_sha256": source_hashes(),
               "input_files_sha256": {str(p.relative_to(PHASE)): sha(p) for p in paths + [mapping_readme]},
               "export_scope": {"graph_targets_exported": "official training only",
                                "test_target_indexing_or_summary": False,
                                "all_target_storage_materialized_by_torch_load": True,
                                "public_heldout_label_columns_may_materialize_during_CSV_parse": True,
                                "heldout_label_columns_selected_by_exporter": False,
                                "only_training_row_SMILES_selected_for_RDKit": True,
                                "heldout_graph_object_constructed": False,
                                "scores_read": False, "training_performed": False}}
    output = confined(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(payload, stream)
    write_json(output.with_suffix(".provenance.json"), {
        "status": "training_only_bundle_exported", "bundle_sha256": sha(output),
        "training_graphs": len(graphs), "grouping": group_metadata,
        "export_scope": payload["export_scope"], "source_sha256": source_hashes()})
    print(json.dumps({"bundle_sha256": sha(output), "training_graphs": len(graphs),
                      "groups": group_metadata["groups"], "not_predictive_evidence": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-manifest", "processed", "split-train", "molecule-map", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
