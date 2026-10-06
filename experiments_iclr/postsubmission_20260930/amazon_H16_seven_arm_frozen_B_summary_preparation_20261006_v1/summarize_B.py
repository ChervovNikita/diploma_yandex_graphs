"""Disabled fixed-seven-arm CPU reader of frozen training diagnostics only."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sys

SOURCE_RELEASED = False
HERE = Path(__file__).resolve().parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
ARMS = ('live', 'uniform', 'margins', 'graph_free', 'permuted', 'stop_q', 'first_order_utility_live')
PAIRS = tuple(itertools.combinations(range(5), 2))
H, M, EPSILON = 16, 4, 0.001
PAIR_FIELDS = ('centered_response_rms', 'centered_cost_rms', 'smooth_cost_scale',
               'epsilon_over_scale', 'normalized_cost_rms', 'assignment_relative_rms',
               'assignment_max_deviation', 'mean_entropy', 'min_assignment',
               'row_residual', 'column_residual')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def bound(root, descriptor):
    relative = Path(descriptor['path'])
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
            'Exact relative diagnostic/metadata path required')
    p = root / relative
    require(p.resolve(strict=True).is_relative_to(root.resolve()) and p.is_file(), 'Bound file leaves its root')
    for item in (p, *p.parents):
        if item == root.parent:
            break
        require(not item.is_symlink(), 'No symlink custody')
    require(p.stat().st_size == descriptor['bytes'] and sha(p) == descriptor['sha256'], 'Bound bytes changed')
    return p


def admission(args):
    require(SOURCE_RELEASED and args.execute_authorized, 'Source preparation remains disabled')
    require(sys.platform.startswith('linux') and socket.gethostname() == 'anogena-2-0'
            and Path.cwd().resolve() == REPO and PHASE.resolve() == PHASE,
            'Allocation repository CPU process only')
    require(not any(n == 'torch' or n.startswith('torch.') for n in sys.modules), 'Fresh reader before Torch')
    release_path = args.release.absolute()
    require(release_path.is_relative_to(PHASE) and not release_path.is_symlink()
            and release_path.stat().st_mode & 0o222 == 0 and sha(release_path) == args.release_sha256,
            'Exact immutable root release required')
    release = read_json(release_path)
    require(release['schema'] == 'root_frozen_seven_arm_H16_B_CPU_summary_release_v1'
            and release['root_training_diagnostic_summary_approved'] is True
            and release['cpu_only'] is True and release['all_seven_arms_all16_all10_pairs'] is True
            and release['arms'] == list(ARMS) and release['H'] == H and release['M'] == M
            and release['response_epsilon'] == EPSILON and release['source_sha256'] == sha(__file__),
            'Exact frozen training-only summary scope required')
    for key in ('A_access', 'VALID_TEST_access', 'model_execution', 'fits_authorized',
                'checkpoint_or_prediction_access', 'data_or_label_file_access', 'selection_change',
                'training_or_horizon_change', 'accuracy_or_generalization_verdict'):
        require(release[key] is False, 'Reader admits no ' + key)
    require(release['source_review_evidence'], 'Independent source review required')
    for ref in release['source_review_evidence']:
        bound(PHASE, ref)
    manifest = read_json(HERE / 'MANIFEST.json')
    require(sha(HERE / 'MANIFEST.json') == release['source_manifest_sha256'], 'Reader packet changed')
    for row in manifest['files']:
        bound(HERE, row)
    inputs = read_json(HERE / 'DIAGNOSTIC_INPUTS.json')
    require(release['diagnostic_inputs_sha256'] == sha(HERE / 'DIAGNOSTIC_INPUTS.json')
            and inputs['arms'] == list(ARMS) and inputs['H'] == H and inputs['M'] == M
            and inputs['response_epsilon'] == EPSILON, 'Diagnostic population changed')
    # Authenticate all seven files before the first tensor deserialization.
    files = {arm: bound(PHASE, inputs['files'][arm]) for arm in ARMS}
    output = args.output.absolute()
    require(output.is_relative_to(PHASE) and not output.exists() and output.parent.is_dir()
            and output.parent.resolve().is_relative_to(PHASE), 'Fresh in-phase compact output required')
    return release, inputs, files, output


def scalar(torch, value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == 'cpu' and value.numel() == 1 and not value.requires_grad,
                'Detached CPU diagnostic scalar required')
        value = value.item()
    require(type(value) in (int, float) and math.isfinite(value), 'Finite diagnostic scalar required')
    return float(value)


def count(torch, value):
    v = scalar(torch, value)
    require(v >= 0 and v.is_integer(), 'Nonnegative integer denominator required')
    return int(v)


def tensor(torch, value, shape):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.layout == torch.strided and tuple(value.shape) == shape and not value.requires_grad
            and bool(torch.isfinite(value).all()), 'Detached finite CPU training array required')
    return value.double()


def stats(values):
    require(values and all(math.isfinite(v) for v in values), 'Nonempty finite complete summary required')
    return {'count': len(values), 'mean': math.fsum(values) / len(values),
            'min': min(values), 'max': max(values)}


def summarize(torch, arm, value):
    require(isinstance(value, dict) and set(value) == {'episodes', 'endpoint_vs_common', 'interpretation'}
            and len(value['episodes']) == H, 'Exact frozen sixteen-episode training diagnostic payload required')
    pair_columns = {str(a) + ':' + str(b): {'classes': [a, b], 'episode_indices': list(range(1, H + 1))}
                    for a, b in PAIRS}
    losses, probe_changes = [], []
    all_metrics = {key: [] for key in PAIR_FIELDS}
    all_metrics.update(observed_response_rms_over_epsilon=[], normalized_observed_response_rms=[],
                       q_cpu_relative_rms=[], q_cpu_mean_entropy=[], q_cpu_max_deviation=[],
                       q_cpu_row_residual=[], q_cpu_column_residual=[])
    discrepancy = []
    total_rows = total_entries = changed_rows = changed_entries = 0
    for index, info in enumerate(value['episodes'], 1):
        require(len(info['pairs']) == len(info['assignments']) == len(PAIRS), 'All ten pair banks required')
        losses.append(scalar(torch, info['virtual_query_loss']))
        probe_changes.append(tensor(torch, info['probe_own_ce_change'], (M,)).tolist())
        if arm == 'first_order_utility_live':
            require(len(info['utility_raw_costs']) == len(info['observed_finite_response_costs']) == len(PAIRS),
                    'All same-state utility and paid-finite banks required')
        for number, (classes, row, qraw) in enumerate(zip(PAIRS, info['pairs'], info['assignments'])):
            require(tuple(row['classes']) == classes, 'Class-pair order changed')
            n = count(torch, row['item_count'])
            left, right = count(torch, row['left_count']), count(torch, row['right_count'])
            require(n > 0 and left + right == n, 'Complete raw pair denominator differs')
            q = tensor(torch, qraw, (n, M))
            require(bool((q > 0).all()), 'Positive recorded Q required')
            expected_kind = ('current_margins' if arm == 'margins' else
                             'first_order_private_gradient_utility' if arm == 'first_order_utility_live' else 'response')
            require(row['cost_kind'] == expected_kind, 'Recorded cost kind changed')
            column = pair_columns[str(classes[0]) + ':' + str(classes[1])]
            for key, value_ in (('item_count', n), ('left_count', left), ('right_count', right),
                                ('q_entry_count', n * M), ('cost_kind', row['cost_kind'])):
                column.setdefault(key, []).append(value_)
            metrics = {key: scalar(torch, row[key]) for key in PAIR_FIELDS}
            r = metrics['centered_response_rms']
            require(r >= 0 and metrics['smooth_cost_scale'] > 0, 'Recorded scales invalid')
            metrics['observed_response_rms_over_epsilon'] = r / EPSILON
            metrics['normalized_observed_response_rms'] = r / math.sqrt(r * r + EPSILON * EPSILON)
            delta = q - 1.0 / M
            metrics.update(q_cpu_relative_rms=M * float(delta.square().mean().sqrt()),
                           q_cpu_mean_entropy=float(-(q * q.log()).sum(1).mean()),
                           q_cpu_max_deviation=float(delta.abs().max()),
                           q_cpu_row_residual=float((q.sum(1) - 1).abs().max()),
                           q_cpu_column_residual=float((q.sum(0) - n / M).abs().max()))
            different = q != 1.0 / M
            nr, ne = int(different.any(1).sum()), int(different.sum())
            column.setdefault('q_exact_nonuniform_rows', []).append(nr)
            column.setdefault('q_exact_nonuniform_entries', []).append(ne)
            total_rows += n; total_entries += n * M; changed_rows += nr; changed_entries += ne
            for key, metric in metrics.items():
                require(math.isfinite(metric), 'Nonfinite derived diagnostic')
                column.setdefault(key, []).append(metric)
                all_metrics[key].append(metric)
            if arm == 'first_order_utility_live':
                require(row['observed_response_kind'] == 'paid_finite_softplus_margin_response',
                        'Utility paid finite response kind changed')
                utility = tensor(torch, info['utility_raw_costs'][number], (n, M))
                finite = tensor(torch, info['observed_finite_response_costs'][number], (n, M))
                uc, fc = utility - utility.mean(1, keepdim=True), finite - finite.mean(1, keepdim=True)
                diff = fc - uc
                d = {'raw_difference_rms': float((finite - utility).square().mean().sqrt()),
                     'centered_difference_rms': float(diff.square().mean().sqrt()),
                     'centered_difference_max_abs': float(diff.abs().max()),
                     'centered_difference_rms_over_epsilon': float(diff.square().mean().sqrt()) / EPSILON,
                     'utility_centered_rms_from_array': float(uc.square().mean().sqrt()),
                     'paid_finite_centered_rms_from_array': float(fc.square().mean().sqrt())}
                for key, metric in d.items():
                    column.setdefault('same_utility_state_' + key, []).append(metric)
                discrepancy.append(d)
    endpoint = value['endpoint_vs_common']
    require(set(endpoint) == {'shared_l2', 'private_member_l2'} and len(endpoint['private_member_l2']) == M,
            'Original endpoint displacement schema required')
    displacement = {'shared_l2': scalar(torch, endpoint['shared_l2']),
                    'private_member_l2': [scalar(torch, x) for x in endpoint['private_member_l2']],
                    'scope': 'Frozen H16 endpoint versus common400; not a per-episode update norm',
                    'per_episode_displacement': {'defined': False, 'value': None, 'reason': 'not_recorded_in_B_diagnostics'}}
    result = {'episodes': H, 'pairs_per_episode': len(PAIRS), 'pair_episode_cells': H * len(PAIRS),
              'raw_pair_denominators_and_all_episode_series': pair_columns,
              'pair_metric_summaries': {key: stats(values) for key, values in all_metrics.items()},
              'pair_summary_weighting': 'Equal weight to each of the160 pair-episode cells; raw item denominators retained. Pairs overlap in S items.',
              'virtual_query_loss': {'episode_values': losses, 'summary': stats(losses)},
              'probe_own_ce_change': {'episode_member_values': probe_changes,
                                     'member_summaries': [stats([r[i] for r in probe_changes]) for i in range(M)]},
              'q_exact_nonuniform': {'changed_rows': changed_rows, 'rows_denominator': total_rows,
                                     'changed_entries': changed_entries, 'entries_denominator': total_entries,
                                     'rule': 'Exact stored Q !=0.25, no tolerance/selection threshold'},
              'endpoint_vs_common': displacement}
    if discrepancy:
        result['same_utility_state_first_order_vs_paid_finite'] = {
            'pair_episode_cells': len(discrepancy),
            'summaries': {key: stats([r[key] for r in discrepancy]) for key in discrepancy[0]},
            'scope': 'Only paired arrays inside one utility episode/state; different fitted-arm trajectories are not approximation-error comparisons'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized', action='store_true')
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release, inputs, files, output = admission(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    require('weights_only' in inspect.signature(torch.load).parameters and torch.__version__ == '2.1.2+cu118',
            'Existing allocation safe-load runtime required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summaries = {}
    with torch.no_grad():
        for arm in ARMS:
            payload = torch.load(files[arm], map_location='cpu', weights_only=True)
            summaries[arm] = summarize(torch, arm, payload)
            require(sha(files[arm]) == inputs['files'][arm]['sha256'], 'Frozen training diagnostics changed during read')
            del payload
    require(not torch.cuda.is_initialized(), 'CPU summary must not initialize CUDA')
    result = {'schema': 'frozen_seven_arm_H16_training_B_CPU_summary_v1',
              'UTC': datetime.now(timezone.utc).isoformat(), 'arms': list(ARMS), 'H': H, 'M': M,
              'response_epsilon': EPSILON, 'input_bindings': inputs['files'],
              'release_sha256': sha(args.release), 'source_sha256': sha(__file__),
              'by_arm': summaries, 'diagnostic_scope': 'Already frozen training B/S/R arrays only',
              'normalization': 'Observed finite-response RMS/sqrt(RMS²+epsilon²) is distinct from recorded cost normalization in margins/first-order arms.',
              'numeric_scope': 'Author scalars retained; additional Q and utility/finite array descriptions computed in CPU float64.',
              'A_VALID_TEST_checkpoint_prediction_dataset_label_access': False,
              'models_forwards_autograd_training_or_selection_changes': False,
              'accuracy_generalization_specialization_or_novelty_verdict': None}
    output.mkdir()
    target = output / 'SUMMARY.json'
    with target.open('x') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    target.chmod(0o444)
    compact = {arm: {'mean_response_RMS_over_epsilon': r['pair_metric_summaries']['observed_response_rms_over_epsilon']['mean'],
                     'mean_Q_relative_RMS': r['pair_metric_summaries']['q_cpu_relative_rms']['mean'],
                     'mean_Q_entropy': r['pair_metric_summaries']['q_cpu_mean_entropy']['mean'],
                     'virtual_query_loss_mean': r['virtual_query_loss']['summary']['mean'],
                     'endpoint_vs_common': r['endpoint_vs_common']} for arm, r in summaries.items()}
    print(json.dumps({'status': 'COMPLETE_FROZEN_TRAINING_DIAGNOSTICS_ONLY',
                      'summary': {'path': str(target.relative_to(PHASE)), 'bytes': target.stat().st_size, 'sha256': sha(target)},
                      'compact_by_arm': compact}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
