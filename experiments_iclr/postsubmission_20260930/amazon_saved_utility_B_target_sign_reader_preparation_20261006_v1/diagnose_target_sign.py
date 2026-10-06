"""Disabled thin CPU extension of the existing frozen-B reader: utility F/U only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

SOURCE_RELEASED = False
HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
N, S, M, H, EPSILON = 24492, 2449, 4, 16, 0.001
COUNTS = [631, 906, 578, 231, 103]
PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def frozen(base, ref):
    relative = Path(ref['path'])
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts,
            'Exact relative metadata/source path required')
    path = base / relative
    require(path.resolve(strict=True).is_relative_to(base.resolve()) and path.is_file(),
            'Metadata/source leaves its root')
    for part in (path, *path.parents):
        if part == base.parent:
            break
        require(not part.is_symlink(), 'Symlink custody refused')
    require(sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes']
            and path.stat().st_mode & 0o222 == 0, 'Exact readonly metadata/source required')
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    require(Path(result.__file__).resolve() == path.resolve(), 'Wrong pinned helper')
    return result


def energy_row(torch, y, signs, first, second, cross, weight, n, scale2):
    signed = signs[:, None] * y
    delta = y[first] - y[second]
    signed_delta = signed[first] - signed[second]
    raw = float(delta.square().sum()) * weight
    oriented = float(signed_delta.square().sum()) * weight
    cross_dot = float((y[first[cross]] * y[second[cross]]).sum()) * weight
    node_denominator = n * M * scale2
    edge_denominator = len(first) * weight * M * scale2
    total = raw + oriented
    return {
        'quadratic_unsigned_without_half': raw, 'quadratic_target_signed_without_half': oriented,
        'node_member_scale_denominator': node_denominator,
        'edge_weight_member_scale_denominator': edge_denominator,
        'node_normalized_unsigned': raw / node_denominator,
        'node_normalized_target_signed': oriented / node_denominator,
        'node_normalized_signed_minus_unsigned': (oriented - raw) / node_denominator,
        'edge_normalized_unsigned': raw / edge_denominator if edge_denominator > 0 else None,
        'edge_normalized_target_signed': oriented / edge_denominator if edge_denominator > 0 else None,
        'relative_orientation_contrast': (oriented - raw) / total if total > 0 else None,
        'relative_orientation_contrast_defined': total > 0,
        'cross_edge_weighted_inner_product': cross_dot,
        'gauge_identity_residual': oriented - raw - 4.0 * cross_dot,
        'field_frobenius_squared': float(y.square().sum()),
        'signed_field_norm_residual': float(signed.square().sum() - y.square().sum()),
        'row_member_center_residual_max_abs': float(y.mean(1).abs().max()),
        'column_mean_max_abs_before_orientation': float(y.mean(0).abs().max()),
        'target_weighted_signed_column_mean_max_abs': float((signs[:, None] * signed).mean(0).abs().max())}


def inspect_banks(torch, reader, payload, graphs):
    require(isinstance(payload, dict)
            and set(payload) == {'episodes', 'endpoint_vs_common', 'interpretation'}
            and len(payload['episodes']) == H, 'Exact complete utility diagnostic schema required')
    cells = []
    for episode, info in enumerate(payload['episodes'], 1):
        require(info['arm'] == 'first_order_utility_live'
                and len(info['pairs']) == len(info['utility_raw_costs'])
                == len(info['observed_finite_response_costs']) == 10, 'All utility banks required')
        for number, (pair, graph) in enumerate(zip(PAIRS, graphs)):
            row = info['pairs'][number]
            n = graph['pair_nodes']
            require(tuple(row['classes']) == pair
                    and reader.count(torch, row['item_count']) == n
                    and reader.count(torch, row['left_count']) == COUNTS[pair[0]]
                    and reader.count(torch, row['right_count']) == COUNTS[pair[1]]
                    and row['cost_kind'] == 'first_order_private_gradient_utility'
                    and row['observed_response_kind'] == 'paid_finite_softplus_margin_response',
                    'Exact all-pair utility row identity required')
            banks = {'finite_response': reader.tensor(torch, info['observed_finite_response_costs'][number], (n, M)),
                     'first_order_utility': reader.tensor(torch, info['utility_raw_costs'][number], (n, M))}
            centered = {key: value - value.mean(1, keepdim=True) for key, value in banks.items()}
            finite_norm2 = float(centered['finite_response'].square().sum())
            scale2 = EPSILON * EPSILON + finite_norm2 / (n * M)
            utility_scale = math.sqrt(EPSILON * EPSILON
                                      + float(centered['first_order_utility'].square().sum()) / (n * M))
            readouts = {}
            for bank, x in centered.items():
                offset = x.mean(0, keepdim=True)
                projected = x - offset
                bank_rows = {}
                for mode, y in (('member_only', x), ('row_column_tangent', projected)):
                    bank_rows[mode] = energy_row(torch, y, graph['signs'], graph['first'], graph['second'],
                                                 graph['cross'], graph['weight'], n, scale2)
                bank_rows['member_column_offset_squared'] = float(offset.square().sum()) * n
                bank_rows['projection_pythagorean_residual'] = float(x.square().sum() - projected.square().sum()) - bank_rows['member_column_offset_squared']
                readouts[bank] = bank_rows
            cells.append({
                'episode': episode, 'classes': list(pair), 'pair_nodes': n,
                'pair_class_counts': {str(pair[0]): COUNTS[pair[0]], str(pair[1]): COUNTS[pair[1]]},
                'graph': graph['metadata'], 'shared_finite_reference_scale_squared': scale2,
                'finite_member_centered_norm_squared': finite_norm2,
                'source_recorded_utility_scale': reader.scalar(torch, row['smooth_cost_scale']),
                'recomputed_utility_scale': utility_scale,
                'source_utility_scale_minus_recomputed': reader.scalar(torch, row['smooth_cost_scale']) - utility_scale,
                'readouts': readouts})
    require(len(cells) == H * 10 and [(r['episode'], tuple(r['classes'])) for r in cells]
            == [(e, p) for e in range(1, H + 1) for p in PAIRS], 'Every fixed utility cell required')
    summaries = {}
    for bank in ('finite_response', 'first_order_utility'):
        summaries[bank] = {}
        for mode in ('member_only', 'row_column_tangent'):
            values = [cell['readouts'][bank][mode] for cell in cells]
            fields = ('node_normalized_unsigned', 'node_normalized_target_signed',
                      'node_normalized_signed_minus_unsigned', 'relative_orientation_contrast')
            summaries[bank][mode] = {
                field: {'complete_cells': len(values), 'defined_cells': len(v),
                        'undefined_cells': len(values) - len(v),
                        'equal_cell_mean': sum(v) / len(v) if v else None,
                        'min': min(v) if v else None, 'max': max(v) if v else None}
                for field in fields for v in [[r[field] for r in values if r[field] is not None]]}
    return cells, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(SOURCE_RELEASED and args.execute_authorized, 'Target-sign reader remains disabled')
    require(sys.platform.startswith('linux') and socket.gethostname() == 'anogena-2-0'
            and Path.cwd().resolve() == REPO and PHASE.resolve() == PHASE, 'Exact allocation repository required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('PYTHONPATH') == str(REPO / '.venv/lib/python3.11/site-packages')
            and sys.version_info[:2] == (3, 11)
            and Path(sys.executable).absolute() == PHASE / 'native_ncn_runtime_20261005_v1/.venv/bin/python',
            'Existing explicit CPU runtime required')
    release_path = args.release.absolute()
    require(release_path.is_relative_to(PHASE), 'Release leaves phase')
    release_path = frozen(PHASE, {'path': str(release_path.relative_to(PHASE)),
                                 'bytes': release_path.stat().st_size, 'sha256': args.release_sha256})
    authority = read(release_path)
    require(authority['schema'] == 'root_saved_utility_target_sign_CPU_diagnostic_release_v1'
            and all(authority[key] is True for key in ('root_diagnostic_approved', 'cpu_only',
                'joint_B_label_decode_acknowledged_and_approved',
                'whole_utility_B_payload_deserialization_acknowledged_and_approved',
                'only_utility_F_U_banks_analyzed'))
            and authority['source_sha256'] == sha(__file__)
            and authority['packet_manifest_sha256'] == sha(HERE / 'MANIFEST.json')
            and authority['inputs_sha256'] == sha(HERE / 'INPUTS.json'), 'Exact reviewed authority required')
    for key in ('fits_authorized', 'A_VALID_TEST_label_access', 'R_labels_used_in_analysis',
                'Q_or_query_loss_values_analyzed', 'model_checkpoint_prediction_access',
                'held_scoring', 'source_protocol_or_gate_change', 'tuning_or_graph_selection'):
        require(authority[key] is False, 'Reader admits no ' + key)
    for row in read(HERE / 'MANIFEST.json')['files']:
        frozen(HERE, row)
    require(authority['source_review_evidence'], 'Actual source review required')
    for row in authority['source_review_evidence']:
        frozen(PHASE, row)
    inputs = read(HERE / 'INPUTS.json')
    reader_path = frozen(PHASE, inputs['existing_reader'])
    for row in read(frozen(PHASE, inputs['existing_reader_manifest']))['files']:
        frozen(reader_path.parent, row)
    reader = module('_pinned_existing_B_reader', reader_path)
    context_path = frozen(PHASE, inputs['admitted_context_inputs'])
    context = read(context_path)
    accessor_path = frozen(PHASE, context['accessor_source'])
    for row in read(frozen(PHASE, context['accessor_manifest']))['files']:
        frozen(accessor_path.parent, row)
    protocol = read(frozen(PHASE, context['protocol']))
    certificate = read(frozen(context_path.parent, context['qualified_identity_certificate']))
    prior_census = read(frozen(PHASE, inputs['actual_census']))
    require(protocol['affinity']['K'] == context['exact_K_definition']
            and certificate['native_edge_logical_sha256'] == context['native_edge_logical_sha256']
            and prior_census['inputs_sha256'] == inputs['admitted_context_inputs']['sha256']
            and prior_census['S_nodes'] == S and prior_census['pair_count'] == 10
            and [tuple(r['classes']) for r in prior_census['pairs']] == PAIRS,
            'Exact admitted graph/context/census required')
    b_dir = PHASE / context['public_b_directory']
    manifest_path = frozen(PHASE, context['public_b_manifest'])
    require(manifest_path.parent == b_dir, 'Exact public+B directory required')
    accessor = module('_pinned_existing_S_custodian', accessor_path)
    _, manifest, roles = accessor._projection(b_dir)
    require(manifest['B_labels'] == context['B_labels_member']
            and manifest['roles'] == context['roles_member']
            and manifest['public_graph'] == context['public_graph'], 'Frozen S/public bindings differ')
    import numpy as np
    import torch
    require(torch.__version__ == '2.1.2+cu118'
            and Path(torch.__file__).resolve() == REPO / '.venv/lib/python3.11/site-packages/torch/__init__.py'
            and 'weights_only' in inspect.signature(torch.load).parameters, 'Existing qualified Torch reader required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    b_labels = accessor._read_compact(np, accessor._verify(b_dir, manifest['B_labels'], frozen=True), roles['B'])
    s_ids = np.asarray(roles['innerS'], dtype=np.int64)
    s_labels = b_labels[np.searchsorted(roles['B'], roles['innerS'])].copy()
    del b_labels
    require(s_ids.shape == s_labels.shape == (S,)
            and s_ids.tolist() == sorted(set(s_ids.tolist()))
            and np.bincount(s_labels, minlength=5).tolist() == COUNTS, 'Original exact S population required')
    with np.load(accessor._verify(PHASE, accessor.PUBLIC_GRAPH), allow_pickle=False) as archive:
        require(set(archive.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'},
                'Exact public graph archive schema required')
        raw_edges = archive['edge_index']
    require(raw_edges.dtype == np.int64 and raw_edges.ndim == 2 and raw_edges.shape[0] == 2
            and bool(((raw_edges >= 0) & (raw_edges < N)).all()), 'Native raw edge schema differs')
    native, preprocessing = accessor._native_edges(raw_edges, PHASE)
    require(preprocessing['edge_logical_sha256'] == context['native_edge_logical_sha256']
            and native.shape == (2, prior_census['native_directed_edge_entries_including_loops']),
            'Exact qualified native topology required')
    undirected = native[:, native[0] < native[1]]
    graphs = []
    for pair, saved in zip(PAIRS, prior_census['pairs']):
        keep = (s_labels == pair[0]) | (s_labels == pair[1])
        ids, labels = s_ids[keep], s_labels[keep]
        to_pair = np.full(N, -1, dtype=np.int64)
        to_pair[ids] = np.arange(len(ids))
        first, second = to_pair[undirected[0]], to_pair[undirected[1]]
        edge_keep = (first >= 0) & (second >= 0)
        first, second = first[edge_keep], second[edge_keep]
        degree = np.bincount(np.concatenate((first, second)), minlength=len(ids))
        cross = labels[first] != labels[second]
        require(len(ids) == saved['pair_nodes']
                and len(first) == saved['induced_exact_K']['undirected_distinct_nonloop_edges']
                and int(cross.sum()) == saved['induced_exact_K']['cross_class_edges']
                and int((degree == 0).sum()) == saved['induced_exact_K']['pair_isolated_nodes_in_this_graph'],
                'Current induced K must match the actual census')
        weight = 1.0 / (1 + int(degree.max()))
        graphs.append({'pair_nodes': len(ids), 'first': torch.from_numpy(first), 'second': torch.from_numpy(second),
                       'cross': torch.from_numpy(cross), 'signs': torch.from_numpy(np.where(labels == pair[0], 1.0, -1.0)),
                       'weight': weight, 'metadata': {'undirected_nonloop_edges': len(first),
                           'same_target_edges': int((~cross).sum()), 'cross_target_edges': int(cross.sum()),
                           'uniform_edge_weight': weight, 'edge_weight_mass': len(first) * weight,
                           'same_target_edge_weight_mass': int((~cross).sum()) * weight,
                           'cross_target_edge_weight_mass': int(cross.sum()) * weight,
                           'isolated_pair_nodes': int((degree == 0).sum()),
                           'nonisolated_pair_nodes': int((degree > 0).sum())}})
    # Full utility payload decode; only F/U banks and count/kind metadata are used.
    bank_path = reader.bound(PHASE, inputs['utility_bank'])
    with torch.no_grad():
        payload = torch.load(bank_path, map_location='cpu', weights_only=True)
        cells, summaries = inspect_banks(torch, reader, payload, graphs)
    result = {'schema': 'saved_utility_target_sign_graph_energy_diagnostic_v1',
              'UTC': datetime.now(timezone.utc).isoformat(), 'internal_seconds': time.monotonic() - started,
              'source_sha256': sha(__file__), 'inputs_sha256': sha(HERE / 'INPUTS.json'),
              'specification_sha256': inputs['specification_sha256'], 'utility_bank': inputs['utility_bank'],
              'admitted_context_inputs': inputs['admitted_context_inputs'], 'actual_census': inputs['actual_census'],
              'native_preprocessing': preprocessing, 'pair_episode_cells': len(cells),
              'cost_banks': ['finite_response', 'first_order_utility'],
              'centering_readouts': ['member_only', 'row_column_tangent'], 'cells': cells,
              'equal_cell_summaries': summaries, 'epsilon': EPSILON,
              'scope': {'whole_utility_B_payload_deserialized': True, 'joint_B_W_S_R_labels_decoded_by_custodian': True,
                        'only_S_labels_and_utility_F_U_used': True, 'Q_or_query_loss_values_analyzed': False,
                        'A_VALID_TEST_label_artifacts_opened': False, 'R_labels_used_in_analysis': False,
                        'features_or_masks_decoded': False, 'model_checkpoint_prediction_calls': False,
                        'raw_node_IDs_or_per_node_costs_or_labels_exported': False,
                        'fits_scores_tuning_or_current_gate_changes': False},
              'interpretation': 'Utility-trajectory signed loss-response/first-order cost proxies only. No canonical-logit or susceptibility identification; no six-arm raw-response field, predictive/causal benefit, alternate prior choice, Q equality or novelty claim. Pair items and episodes overlap. No result selects a graph, centering, coefficient or gate.'}
    output = args.output.absolute()
    require(output.is_relative_to(PHASE) and output.parent.is_dir() and not output.exists()
            and output.parent.resolve().is_relative_to(PHASE), 'One fresh phase output required')
    output.mkdir()
    target = output / 'TARGET_SIGN_DIAGNOSTIC.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    target.chmod(0o444)
    print(json.dumps({'path': str(target.relative_to(PHASE)), 'bytes': target.stat().st_size,
                      'sha256': sha(target), 'all160_cells': True, 'new_fits_or_scoring': False}, sort_keys=True))


if __name__ == '__main__':
    main()
