"""Require real two-update PASS receipts for all 6,460 canonical fit IDs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from support import CANONICAL_SHA, COUNTS, WORKERS, digest, own_source, pin, read, require, verify, write


def serialized_pin(value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def validate_result(row, result, inventory_pin, source_sha):
    require(result.get('schema') == 'all-fit-two-update-smoke-result-v1' and result.get('status') == 'PASS', 'Non-PASS native smoke')
    for field in ('id', 'worker', 'runtime_job_hash', 'canonical_runtime_job_hash', 'science_contract_hash', 'descriptor', 'runtime_entrypoint'):
        require(result.get(field) == row[field], 'Smoke identity differs: ' + field)
    require(all(result['inventory'][k] == inventory_pin[k] for k in ('bytes', 'sha256')), 'Smoke used another inventory')
    require(result['source_manifest']['sha256'] == source_sha, 'Smoke source differs')
    require(result.get('optimizer_calls') == {'attempts': [2], 'completed': [2]} and result.get('TRAIN_updates') == 2, 'Exactly two completed optimizer transitions required')
    require(result.get('TRAIN_finite') is True and result.get('VALID_finite') is True, 'Finite real TRAIN and VALID required')
    losses = result.get('TRAIN_losses', [])
    require(len(losses) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in losses), 'Two finite native TRAIN loss receipts required')
    validation = result.get('VALID', {})
    require(result.get('VALID_calls') == 1 and validation.get('role') == 'VALID' and validation.get('complete_source_VALID') is True and validation.get('records', 0) > 0, 'Complete native VALID receipt required')
    require(math.isfinite(validation['loss']) and all(math.isfinite(x) for x in validation['metrics'].values()), 'Nonfinite VALID summary')
    reload = result.get('checkpoint_reload', {})
    require(reload.get('status') == 'PASS' and reload.get('exact_adapter_state_restored') is True
            and reload.get('parameter_perturbation_restored') is True and reload.get('optimizer_state_reloaded') is True
            and reload.get('optimizer_moment_perturbation_restored') is True
            and reload.get('TRAIN_updates_after_reload') == 0, 'Actual state save/reload required')
    require(result.get('TEST_accessed') is False and result.get('scientific_checkpoint_reused') is False
            and result.get('scientific_fit_policy_changed') is False and result.get('automatic_retries') == 0, 'Smoke scope differs')
    require(not any(result.get(k) for k in ('failure', 'close_error', 'observer_close_error', 'resource_record_error')), 'Failure cannot become PASS')
    return True


def validate_results(items, source_sha):
    """Pure metadata validation; items embed locally verified inventory/summary."""
    require(len(items) == 3 and {item['worker'] for item in items} == set(WORKERS), 'Exactly three distinct workers required')
    seen = set()
    universe = None
    coverage = []
    inventory_shas = {}
    summary_shas = {}
    for item in items:
        worker = item['worker']
        inventory = item['inventory_value']
        inventory_pin = item['inventory']
        summary = item['summary_value']
        require(inventory['schema'] == 'all-fit-smoke-inventory-v1' and inventory['source_manifest']['sha256'] == source_sha, 'Inventory source/schema differs')
        require(inventory['canonical_plan']['sha256'] == CANONICAL_SHA and inventory['class_substitution'] is False, 'Exact canonical inventory required')
        ids = inventory['canonical_fit_ids']
        require(len(ids) == len(set(ids)) == 6460 and ids == sorted(ids) and inventory['canonical_fit_ids_sha256'] == digest(ids), 'Canonical universe malformed')
        if universe is None:
            universe = set(ids)
        require(universe == set(ids), 'Workers disagree on canonical fit universe')
        rows = {row['id']: row for row in inventory['fits'] if row['worker'] == worker}
        require(len(rows) == COUNTS[worker] and len(rows) == sum(row['worker'] == worker for row in inventory['fits']), 'Missing/duplicate owned fit')
        require(summary.get('schema') in ('allfit-worker-smoke-summary-v1', 'all-fit-worker-smoke-summary-v1')
                and summary.get('status') == 'PASS' and summary.get('worker') == worker, 'Worker summary not PASS')
        require(all(summary['inventory'][k] == inventory_pin[k] for k in ('bytes', 'sha256')), 'Worker inventory differs')
        require(len(summary['expected_fit_ids']) == len(rows) and set(summary['expected_fit_ids']) == set(rows), 'Worker expected IDs differ')
        if 'counts' in summary:
            require(summary['counts'] == {'expected': len(rows), 'passed': len(rows)}, 'Worker PASS counts differ')
        else:
            require(summary.get('fit_count') == len(rows) and summary.get('TEST_accessed') is False, 'Worker PASS count/scope differs')
        results = summary['results']
        require(len(results) == len(rows) and {r['id'] for r in results} == set(rows), 'Worker has missing/extra/duplicate smoke result')
        for entry in results:
            fid = entry['id']
            require(fid not in seen and fid in universe, 'Duplicate/foreign global fit')
            seen.add(fid)
            row = rows[fid]
            result = entry['result']
            require(entry['runtime_job_hash'] == row['runtime_job_hash'], 'Summary runtime job differs')
            require(type(entry['attempt']) is int and entry['attempt'] >= 1, 'Explicit retained attempt required')
            expected = serialized_pin(result)
            require(all(entry['result_receipt'][k] == expected[k] for k in ('bytes', 'sha256')), 'Embedded RESULT differs from actual serialized receipt')
            validate_result(row, result, inventory_pin, source_sha)
            coverage.append({'id': fid, 'worker': worker, 'runtime_job_hash': row['runtime_job_hash'],
                             'attempt': entry['attempt'], 'result_receipt': entry['result_receipt']})
        inventory_shas[worker] = inventory_pin['sha256']
        summary_shas[worker] = item['summary']['sha256']
    require(seen == universe and len(seen) == 6460, 'Every one of 6,460 real fits must pass')
    return {'schema': 'all-fit-global-smoke-pass-v1', 'status': 'PASS', 'fit_count': len(seen),
        'TRAIN_updates_completed_total': 2 * len(seen), 'canonical_plan_sha256': CANONICAL_SHA,
        'canonical_fit_ids': sorted(seen), 'canonical_fit_ids_sha256': digest(sorted(seen)), 'inventory_sha256s': inventory_shas,
        'worker_summary_sha256s': summary_shas, 'coverage': sorted(coverage, key=lambda r: r['id']),
        'class_substitution': False, 'TEST_accessed': False,
        'condition_satisfied': 'all 6460 exact fits have real two-update TRAIN/VALID/reload PASS',
        'full_training_launch_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = own_source()
    bundle = read(args.bundle)
    items = []
    for row in bundle['workers']:
        inventory_path = verify(row['inventory'])
        summary_path = verify(row['summary'])
        items.append({**row, 'inventory_value': read(inventory_path), 'summary_value': read(summary_path)})
    result = validate_results(items, source['sha256'])
    result['source_manifest'] = source
    result['bundle'] = pin(args.bundle)
    result['summaries'] = [item['summary'] for item in items]
    write(args.output, result, exclusive=True)
    print(json.dumps({'output': pin(args.output), 'status': result['status'], 'fit_count': result['fit_count']}))


if __name__ == '__main__':
    main()
