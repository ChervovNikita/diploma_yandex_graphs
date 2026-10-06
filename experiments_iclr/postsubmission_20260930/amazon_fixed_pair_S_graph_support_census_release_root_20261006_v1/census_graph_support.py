"""Disabled CPU census of fixed ten S class-pair graph supports.

The custodian step decodes the joint B label member, including W/S/R. Only
fixed S IDs and S labels pass to the structural census. No A/VALID/TEST label
artifact, model, checkpoint, prediction, fitting or scoring callable is used.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

SOURCE_RELEASED = True
HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
N = 24492
COUNTS = [631, 906, 578, 231, 103]
PAIRS = [(left, right) for left in range(5) for right in range(left + 1, 5)]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(base, row, *, frozen=True):
    relative = Path(row['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative exact binding required')
    path = base / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base)
            and sha(path) == row['sha256']
            and ('bytes' not in row or path.stat().st_size == row['bytes']), 'Binding changed: ' + str(relative))
    require(not frozen or path.stat().st_mode & 0o222 == 0, 'Immutable artifact required')
    return path


def components(active, edges):
    """Undirected components; self-loops excluded before entry, isolates kept."""
    parent = list(range(N))
    size = [1] * N

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for first, second in edges:
        first, second = int(first), int(second)
        if not (active[first] and active[second]):
            continue
        left, right = find(first), find(second)
        if left != right:
            if size[left] < size[right]:
                left, right = right, left
            parent[right] = left
            size[left] += size[right]
    return [find(node) if active[node] else -1 for node in range(N)]


def support_summary(active, roots, edges, pair_ids, pair_labels, left, right):
    """Only pair S labels enter class counts; all other vertices are unlabeled."""
    total = Counter(root for root in roots if root >= 0)
    anchors = defaultdict(lambda: [0, 0])
    for node, label in zip(pair_ids, pair_labels):
        require(roots[int(node)] >= 0 and int(label) in (left, right), 'Pair support identity differs')
        anchors[roots[int(node)]][int(label) == right] += 1
    patterns = Counter()
    reachable = cross_reachable = anchored_components = multi_anchor_nodes = 0
    one_label_components = unanchored_components = unanchored_vertices = 0
    single_anchor_components = multi_anchor_components = cross_class_anchor_components = 0
    largest_anchor_component = 0
    for root, vertices in total.items():
        first, second = anchors[root]
        count = first + second
        patterns[(vertices, count, first, second)] += 1
        if count:
            anchored_components += 1
            single_anchor_components += int(count == 1)
            multi_anchor_components += int(count >= 2)
            cross_class_anchor_components += int(first > 0 and second > 0)
            reachable += count * (count - 1) // 2
            cross_reachable += first * second
            largest_anchor_component = max(largest_anchor_component, count)
            one_label_components += int(not (first and second))
            multi_anchor_nodes += count if count >= 2 else 0
        else:
            unanchored_components += 1
            unanchored_vertices += vertices
    require(sum(sum(counts) for counts in anchors.values()) == len(pair_ids)
            and anchored_components + unanchored_components == len(total), 'Complete component counts required')
    degree = [0] * N
    edge_count = 0
    for first, second in edges:
        first, second = int(first), int(second)
        if active[first] and active[second]:
            edge_count += 1
            degree[first] += 1
            degree[second] += 1
    return {'active_vertices': sum(bool(value) for value in active),
            'undirected_distinct_nonloop_edges': edge_count,
            'active_nonisolated_vertices': sum(value > 0 for value in degree),
            'active_components_including_isolates': len(total),
            'pair_nonisolated_nodes_in_this_graph': sum(degree[int(node)] > 0 for node in pair_ids),
            'pair_isolated_nodes_in_this_graph': sum(degree[int(node)] == 0 for node in pair_ids),
            'components_represented_by_pair_nodes': anchored_components,
            'components_with_exactly_one_pair_anchor': single_anchor_components,
            'components_with_at_least_two_pair_anchors': multi_anchor_components,
            'components_with_both_pair_classes_as_anchors': cross_class_anchor_components,
            'pair_nodes_with_another_pair_node_in_component': multi_anchor_nodes,
            'largest_component_pair_node_count': largest_anchor_component,
            'reachable_unordered_pair_node_pairs': reachable,
            'reachable_cross_class_pair_node_pairs': cross_reachable,
            'anchored_components_containing_only_one_pair_class': one_label_components,
            'unanchored_components': unanchored_components,
            'unanchored_vertices': unanchored_vertices,
            'complete_component_pattern_counts': [
                {'vertices': vertices, 'pair_nodes': count,
                 'pair_class_counts': {str(left): first, str(right): second}, 'components': frequency}
                for (vertices, count, first, second), frequency in sorted(patterns.items())]}


def census(np, native_edges, s_ids, s_labels):
    """Public topology plus S-only labels; no accessor or other label object."""
    require(s_ids.dtype == s_labels.dtype == np.int64 and s_ids.shape == s_labels.shape == (2449,)
            and s_ids.tolist() == sorted(set(s_ids.tolist()))
            and np.bincount(s_labels, minlength=5).tolist() == COUNTS, 'Fixed S counts differ')
    encoded = native_edges[0] * N + native_edges[1]
    reversed_encoded = native_edges[1] * N + native_edges[0]
    require(np.unique(encoded).size == encoded.size
            and np.array_equal(np.sort(encoded), np.sort(reversed_encoded)), 'Unique reciprocal native edges required')
    require(int((native_edges[0] == native_edges[1]).sum()) == N, 'Native self-loop identity differs')
    undirected = native_edges[:, native_edges[0] < native_edges[1]].T
    all_active = np.ones(N, dtype=np.bool_)
    all_roots = components(all_active, undirected)
    non_s = np.ones(N, dtype=np.bool_)
    non_s[s_ids] = False
    rows = []
    for left, right in PAIRS:
        keep = (s_labels == left) | (s_labels == right)
        pair_ids, pair_labels = s_ids[keep], s_labels[keep]
        active = np.zeros(N, dtype=np.bool_)
        active[pair_ids] = True
        augmented = non_s.copy()
        augmented[pair_ids] = True
        induced = support_summary(active, components(active, undirected), undirected,
                                  pair_ids, pair_labels, left, right)
        via_non_s = support_summary(augmented, components(augmented, undirected), undirected,
                                    pair_ids, pair_labels, left, right)
        public = support_summary(all_active, all_roots, undirected, pair_ids, pair_labels, left, right)
        same_edges = cross_edges = 0
        label_by_node = {int(node): int(label) for node, label in zip(pair_ids, pair_labels)}
        for first, second in undirected:
            first, second = int(first), int(second)
            if active[first] and active[second]:
                if label_by_node[first] == label_by_node[second]:
                    same_edges += 1
                else:
                    cross_edges += 1
        require(same_edges + cross_edges == induced['undirected_distinct_nonloop_edges'], 'Induced edge partition differs')
        a = induced['reachable_unordered_pair_node_pairs']
        b = via_non_s['reachable_unordered_pair_node_pairs']
        c = public['reachable_unordered_pair_node_pairs']
        require(0 <= a <= b <= c, 'Nested topology reachability violated')
        rows.append({'classes': [left, right], 'pair_nodes': len(pair_ids),
                     'pair_class_counts': {str(left): COUNTS[left], str(right): COUNTS[right]},
                     'public_node_fraction': {'numerator': len(pair_ids), 'denominator': N},
                     'all_unordered_pair_node_pairs': len(pair_ids) * (len(pair_ids) - 1) // 2,
                     'induced_exact_K': {**induced, 'same_class_edges': same_edges, 'cross_class_edges': cross_edges},
                     'pair_plus_nonS_paths': via_non_s, 'full_public_paths': public,
                     'additional_reachable_pairs_with_nonS_vertices': b - a,
                     'additional_reachable_pairs_after_other_S_classes_allowed': c - b,
                     'public_reachable_pairs_disconnected_in_induced_K': c - a,
                     'induced_reachability_fraction_of_public_reachability':
                         {'numerator': a, 'denominator': c, 'defined': c > 0}})
    require(len(rows) == 10 and sum(row['pair_nodes'] for row in rows) == 9796, 'All ten pairs and four appearances per S node required')
    return {'public_nodes': N, 'native_directed_edge_entries_including_loops': int(native_edges.shape[1]),
            'native_self_loops': N, 'public_undirected_distinct_nonloop_edges': len(undirected),
            'S_nodes': len(s_ids), 'S_class_counts': {str(i): count for i, count in enumerate(COUNTS)},
            'S_fraction': {'numerator': len(s_ids), 'denominator': N}, 'pair_count': 10,
            'primary_connectivity_view': 'full_public_paths',
            'restricted_bridge_descriptive_view': 'pair_plus_nonS_paths',
            'sum_pair_items': 9796, 'pairs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(SOURCE_RELEASED and args.execute_authorized, 'Structural census remains disabled')
    require(sys.platform.startswith('linux') and socket.gethostname() == 'anogena-2-0'
            and Path.cwd().resolve() == REPO and PHASE.resolve() == PHASE, 'Exact allocation repository required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('PYTHONPATH') == str(REPO / '.venv/lib/python3.11/site-packages')
            and sys.version_info[:2] == (3, 11)
            and Path(sys.executable).absolute() == PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python',
            'Existing explicit CPU runtime and Torch/PyG path required')
    release_path = args.release.absolute()
    require(release_path.is_relative_to(PHASE) and not release_path.is_symlink()
            and release_path.stat().st_mode & 0o222 == 0 and sha(release_path) == args.release_sha256,
            'Exact immutable root census release required')
    authority = read(release_path)
    require(authority['schema'] == 'root_fixed_pair_S_graph_support_CPU_census_release_v1'
            and authority['root_structural_census_approved'] is True
            and authority['B_joint_W_S_R_label_decode_acknowledged_and_approved'] is True
            and authority['S_labels_only_in_structural_analysis'] is True
            and authority['cpu_only'] is True
            and authority['source_sha256'] == sha(__file__)
            and authority['source_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
            and authority['inputs_sha256'] == sha(HERE / 'INPUTS.json'), 'Exact structural and custodian authority required')
    for key in ('fits_authorized', 'A_labels_access', 'VALID_TEST_labels_access', 'R_labels_used_in_analysis',
                'model_checkpoint_prediction_access', 'scientific_held_scoring', 'protocol_or_source_grid_change',
                'causal_accuracy_or_Q_equality_claim'):
        require(authority[key] is False, 'Census admits no ' + key)
    for row in read(HERE / 'MANIFEST.json')['files']:
        verify(HERE, row)
    require(authority['source_review_evidence'], 'Actual source review required')
    for row in authority['source_review_evidence']:
        verify(PHASE, row)
    inputs = read(HERE / 'INPUTS.json')
    accessor_path = verify(PHASE, inputs['accessor_source'])
    accessor_manifest = read(verify(PHASE, inputs['accessor_manifest']))
    for row in accessor_manifest['files']:
        verify(accessor_path.parent, row)
    protocol = read(verify(PHASE, inputs['protocol']))
    certificate = read(verify(HERE, inputs['qualified_identity_certificate']))
    require(protocol['data']['nodes'] == N and protocol['data']['classes'] == 5
            and protocol['affinity']['K'] == inputs['exact_K_definition']
            and certificate['public_b_manifest_sha256'] == inputs['public_b_manifest']['sha256']
            and certificate['roles_sha256'] == inputs['roles']['sha256']
            and certificate['public_graph_sha256'] == inputs['public_graph']['sha256']
            and certificate['native_edge_logical_sha256'] == inputs['native_edge_logical_sha256'],
            'Frozen assignment and actual qualified graph identities differ')
    public_b_dir = PHASE / inputs['public_b_directory']
    manifest_path = verify(PHASE, inputs['public_b_manifest'])
    require(manifest_path.parent == public_b_dir, 'Exact frozen public+B projection required')
    spec = importlib.util.spec_from_file_location('_fixed_census_pinned_accessor', accessor_path)
    accessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accessor)
    # This helper verifies membership and metadata only; it opens no labels.
    _, manifest, roles = accessor._projection(public_b_dir)
    require(manifest['B_labels'] == inputs['B_labels_member']
            and manifest['roles'] == inputs['roles_member'] and manifest['public_graph'] == inputs['public_graph'],
            'Custodian/public graph bindings differ')
    import numpy as np
    import torch
    require(torch.__version__ == '2.1.2+cu118'
            and Path(torch.__file__).resolve() == REPO / '.venv/lib/python3.11/site-packages/torch/__init__.py',
            'Previously qualified Torch source path required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    # Honest custodian projection: decoding labels opens all B/W/S/R values.
    b_path = accessor._verify(public_b_dir, manifest['B_labels'], frozen=True)
    b_labels = accessor._read_compact(np, b_path, roles['B'])
    s_ids = np.asarray(roles['innerS'], dtype=np.int64)
    s_labels = b_labels[np.searchsorted(roles['B'], roles['innerS'])].copy()
    del b_labels
    # Decode only edge_index from the public NPZ; other members remain closed.
    graph_path = accessor._verify(PHASE, accessor.PUBLIC_GRAPH)
    with np.load(graph_path, allow_pickle=False) as archive:
        require(set(archive.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'},
                'Exact public archive schema required')
        raw_edges = archive['edge_index']
    require(raw_edges.dtype == np.int64 and raw_edges.ndim == 2 and raw_edges.shape[0] == 2
            and bool(((raw_edges >= 0) & (raw_edges < N)).all()), 'Public edge schema differs')
    native_edges, preprocessing = accessor._native_edges(raw_edges, PHASE)
    require(preprocessing['edge_logical_sha256'] == inputs['native_edge_logical_sha256'], 'Actual qualified topology differs')
    result = census(np, native_edges, s_ids, s_labels)
    result.update({'schema': 'fixed_ten_pair_S_native_public_graph_support_census_v1',
                   'UTC': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
                   'source_sha256': sha(__file__), 'inputs_sha256': sha(HERE / 'INPUTS.json'),
                   'provenance': {'public_b_manifest': inputs['public_b_manifest'], 'roles': inputs['roles'],
                                  'B_labels_member': inputs['B_labels_member'], 'accessor_source': inputs['accessor_source'],
                                  'protocol': inputs['protocol'],
                                  'qualified_identity_certificate_original_metadata': inputs['qualified_identity_original_metadata'],
                                  'exact_packet_certificate_metadata': inputs['qualified_identity_certificate'],
                                  'native_preprocessing': preprocessing},
                   'visibility': {'joint_B_W_S_R_labels_decoded_by_custodian': True,
                                  'only_S_IDs_labels_passed_to_census': True, 'R_labels_used_in_analysis': False,
                                  'A_VALID_TEST_label_artifacts_opened': False,
                                  'public_NPZ_members_decoded': ['edge_index'],
                                  'features_or_masks_decoded': False, 'raw_node_IDs_or_labels_exported': False,
                                  'model_checkpoint_prediction_access': False, 'scientific_held_scoring': False},
                   'interpretation': 'Descriptive support and undirected reachability only. Reachability is not an implemented alternate K, a causal error explanation, a quality gain or a Q-equality claim. Exactly-one-pair-anchor components are counted separately from one-label components, which may have many anchors and meaningful graph coupling. Full native-public connectivity is primary; pair+nonS is only a restricted-bridge descriptive variant. Other S classes have labels ignored when they act as full-public structural intermediates. Unanchored means zero pair S nodes, not zero model supervision.'})
    output = args.output.absolute()
    require(output.is_relative_to(PHASE) and output.parent.is_dir() and not output.exists()
            and output.parent.resolve().is_relative_to(PHASE), 'Fresh in-repository census output required')
    output.mkdir()
    path = output / 'CENSUS.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    path.chmod(0o444)
    print(json.dumps({'path': str(path.relative_to(PHASE)), 'bytes': path.stat().st_size,
                      'sha256': sha(path), 'all_ten_pairs': True, 'new_fits_or_scoring': False}, sort_keys=True))


if __name__ == '__main__':
    main()
