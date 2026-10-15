"""One guarded read of 36 completed MolBACE scalar curves; stdout only, no native work."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
INPUTS_SHA256 = '16f59b758d65b402ee59e6a78071ebbf18013bbff802789ec51f98d59e183925'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.suffix in ('.json', '.jsonl', '.py'), 'Only confined scalar metadata/source')
    raw = path.read_bytes()
    pin = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if expected is not None:
        require(pin == expected, 'Pinned metadata changed: ' + str(path))
    return raw, pin


def trend(values):
    require(len(values) >= 100 and all(math.isfinite(v) for v in values), 'At least 100 finite scalar epochs required')
    tail, prior = values[-50:], values[-100:-50]
    first, last = statistics.mean(tail[:25]), statistics.mean(tail[25:])
    mean = statistics.mean(tail)
    slope = sum((i - 24.5) * (v - mean) for i, v in enumerate(tail)) / sum((i - 24.5) ** 2 for i in range(50))
    return {'last': values[-1], 'last50_mean': mean, 'preceding50_mean': statistics.mean(prior),
            'last50_minus_preceding50_mean': mean - statistics.mean(prior),
            'first25_of_last50_mean': first, 'last25_mean': last,
            'last25_minus_first25_of_last50': last - first,
            'last25_relative_change_percent': None if first == 0 else 100 * (last / first - 1),
            'last50_OLS_slope_per_epoch': slope, 'last50_min': min(tail), 'last50_max': max(tail)}


def analyze_fit(expected, result, events, policy, controller, result_pin, curve_pin):
    require(result['job_id'] == expected['job_id'] and result['status'] == 'complete'
            and result['TEST_accessed'] is False and result['dataset'] == 'ogbg-molbace', 'Wrong terminal fit/scope')
    require(result['policy'] == policy and len(events) == result['completed_epochs'], 'Policy or complete-curve length differs')
    selector = controller.Selector(controller.Policy.from_dict(policy, result['family']))
    auc, bce, train_loss, selections = [], [], [], []
    for epoch, event in enumerate(events, 1):
        train, valid = event['train'], event['validation']
        require(event['event'] == 'completed_epoch' and train['epoch'] == epoch
                and train['complete_native_TRAIN'] is True and train['TRAIN_records_traversed'] == 1210,
                'Not the complete ordered TRAIN curve')
        require(valid['role'] == 'VALID' and valid['complete_source_VALID'] is True
                and valid['records'] == 151 and valid['TEST_accessed'] is False, 'Incomplete/wrong validation scope')
        require(event['TRAIN_diagnostic'] is None and train['mean_update_loss_is_not_graph_weighted'] is True,
                'Unexpected TRAIN loss measurement semantics')
        decision = selector.observe(epoch, valid['metrics'])
        require(decision == event['selection'], 'Frozen selector replay differs at epoch ' + str(epoch))
        require(not decision['stop'] or epoch == len(events), 'Curve continued beyond the frozen stopping decision')
        auc.append(float(valid['metrics']['rocauc']))
        bce.append(float(valid['loss']))
        train_loss.append(float(train['mean_update_loss']))
        selections.append(decision)
    require(selections[-1]['stop'] and selections[-1]['stop_reason'] == result['stop_reason']
            and selector.selected_epoch == result['selected_epoch']
            and selector.best_metric == result['best_validation_metric'], 'Terminal selection does not match the curve')
    n = len(events); selected = result['selected_epoch']
    last50_improvements = [n - 49 + i for i, v in enumerate(selections[-50:]) if v['save_selected']]
    records_bce, best_bce = [], math.inf
    for epoch, value in enumerate(bce, 1):
        if value < best_bce:
            best_bce = value
            if epoch > n - 50:
                records_bce.append(epoch)
    return {'job_id': result['job_id'], 'group': expected['group'], 'procedure': result['procedure'],
            'seed': result['seed'], 'body': result.get('body'), 'completed_epochs': n,
            'selected_epoch': selected, 'epochs_after_selected': n - selected,
            'stop_reason': result['stop_reason'], 'runtime_convergence_established_flag': result.get('convergence_established'),
            'cap_hit': n >= policy['maximum_epochs'], 'final_stale_validation_checks': selector.stale,
            'selector_replay': 'exact', 'selected_VALID_AUC_percent': 100 * auc[selected - 1],
            'selected_VALID_BCE': bce[selected - 1], 'last_VALID_minus_selected_AUC_pp': 100 * (auc[-1] - auc[selected - 1]),
            'last_VALID_minus_selected_BCE': bce[-1] - bce[selected - 1],
            'last50_epoch_range': [n - 49, n], 'last50_frozen_VALID_record_improvement_epochs': last50_improvements,
            'last50_ties_with_selected_AUC': sum(v == selector.best_metric for v in auc[-50:]),
            'last50_best_minus_preceding_best_AUC_pp': 100 * (max(auc[-50:]) - max(auc[:-50])),
            'last50_new_BCE_minimum_epochs_diagnostic_only': records_bce,
            'TRAIN_update_loss': trend(train_loss), 'VALID_AUC_percent': trend([100 * v for v in auc]),
            'VALID_native_BCE': trend(bce), 'result_receipt': result_pin, 'curve_receipt': curve_pin,
            'curve_was_previously_pinned': expected['existing_curve_receipt'] is not None,
            'TRAIN_metric_scope': 'Mean minibatch update objective; not graph-weighted or fixed-model full TRAIN evaluation; compare only within this fit.'}


def main():
    # No project metadata, other study state or data are read before this identity guard.
    require(socket.gethostname() == 'anogena-2-0', 'Wrong allocation; stop before project access')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
            == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong sole allocation GPU')
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', default=str(Path(__file__).with_name('INPUTS.json')))
    args = parser.parse_args()
    raw, inputs_pin = read(args.inputs)
    require(inputs_pin['sha256'] == INPUTS_SHA256, 'Exact frozen 36-fit input authority required')
    inputs = json.loads(raw)
    authority_raw, authority_pin = read(inputs['completed30bank_result']['path'], inputs['completed30bank_result'])
    authority = json.loads(authority_raw)
    require(authority['status'] in ('complete', 'complete_with_warnings') and authority['TEST_accessed'] is False
            and len(authority['rows']) == 30 and len(authority['training_fit_outcomes']) == 36, 'Completed 30-bank/36-fit authority required')
    owners = {r['job_id']: r for r in authority['training_fit_outcomes']}
    require(len(owners) == 36 and {r['job_id'] for r in inputs['fits']} == set(owners)
            and len({r['terminal_receipt']['path'] for r in inputs['fits']}) == 36, 'Each genuine completed fit must occur exactly once')
    manifest_raw, manifest_pin = read(inputs['completed30bank_evidence_manifest']['path'], inputs['completed30bank_evidence_manifest'])
    require(any(r['path'] == 'RESULT.json' and r['sha256'] == authority_pin['sha256'] and r['bytes'] == authority_pin['bytes']
                for r in json.loads(manifest_raw)['files']), 'Completed evidence manifest does not bind the authority')
    sources = []
    for pin in inputs['controller_and_loss_source_receipts']:
        _, actual = read(pin['path'], pin); sources.append(actual)
    controller_pin = next(p for p in sources if p['path'].endswith('/bookkeeping.py'))
    spec = importlib.util.spec_from_file_location('completed_molbace_original_selector', controller_pin['path'])
    controller = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = controller
    spec.loader.exec_module(controller)  # Pinned pure stdlib bookkeeping only.
    rows = []
    for expected in inputs['fits']:
        require(expected['terminal_receipt'] == owners[expected['job_id']]['receipt'], 'Exact completed terminal pin differs')
        raw, terminal_pin = read(expected['terminal_receipt']['path'], expected['terminal_receipt'])
        result = json.loads(raw)
        for pin in sources:
            require(result['runtime_sources'][Path(pin['path']).name] == pin, 'Historical runtime source differs')
        curve_path = str(Path(terminal_pin['path']).with_name('CURVE.jsonl'))
        require(curve_path == expected['curve_path'], 'Only the completed terminal sibling curve is allowed')
        curve_raw, curve_pin = read(curve_path, expected['existing_curve_receipt'])
        events = [json.loads(line) for line in curve_raw.splitlines() if line.strip()]
        rows.append(analyze_fit(expected, result, events, inputs['policy'], controller, terminal_pin, curve_pin))
    summary = {'unique_fits': len(rows), 'group_counts': dict(collections.Counter(r['group'] for r in rows)),
               'total_completed_epochs': sum(r['completed_epochs'] for r in rows),
               'completed_epoch_histogram': dict(collections.Counter(r['completed_epochs'] for r in rows)),
               'stop_reasons': dict(collections.Counter(r['stop_reason'] for r in rows)), 'cap_hits': sum(r['cap_hit'] for r in rows),
               'selected_epoch_min_median_max': [min(r['selected_epoch'] for r in rows), statistics.median(r['selected_epoch'] for r in rows), max(r['selected_epoch'] for r in rows)],
               'last50_fits_with_frozen_VALID_record_improvement': sum(bool(r['last50_frozen_VALID_record_improvement_epochs']) for r in rows),
               'last25_TRAIN_mean_lower_than_first25_of_last50': sum(r['TRAIN_update_loss']['last25_minus_first25_of_last50'] < 0 for r in rows),
               'last50_TRAIN_mean_lower_than_preceding50': sum(r['TRAIN_update_loss']['last50_minus_preceding50_mean'] < 0 for r in rows),
               'last50_VALID_AUC_mean_higher_than_preceding50': sum(r['VALID_AUC_percent']['last50_minus_preceding50_mean'] > 0 for r in rows),
               'last50_VALID_BCE_mean_lower_than_preceding50': sum(r['VALID_native_BCE']['last50_minus_preceding50_mean'] < 0 for r in rows)}
    output = {'schema': 'completed36-molbace-scalar-convergence-extraction-v1', 'status': 'complete',
              'inputs': inputs_pin, 'completed30bank_authority': authority_pin, 'evidence_manifest': manifest_pin,
              'controller_and_loss_sources': sources, 'policy': inputs['policy'], 'summary': summary, 'fits': rows,
              'interpretation_limits': ['Validation patience is not optimization stationarity; the runtime convergence flag is assigned solely from stop_reason == patience.',
                  'TRAIN traces are within-fit minibatch update objectives with dropout/training BN; no full TRAIN evaluation was scheduled.',
                  'A decreasing TRAIN objective alone does not establish benefit from more epochs on VALID. No counterfactual continuation is observed.',
                  'VALID record improvement uses the frozen strict-greater AUC selector and first ties, with patience counted from epoch100; no new threshold or epoch selection is introduced.',
                  'BCE trends are diagnostics, not a replacement selection rule. Twelve native fits supply both M1/body0 and I4 and are counted once.'],
              'new_optimizer_steps': 0, 'new_native_forwards': 0, 'tensor_or_dataset_files_opened': 0,
              'TEST_accessed': False, 'partial_or_live_fit_state_read': False, 'elapsed_seconds': time.monotonic() - started,
              'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    print(json.dumps(output, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
