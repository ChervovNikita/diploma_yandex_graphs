"""Inspect completed campaign scalar results without training or test access."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--workers', nargs='+', required=True)
args = parser.parse_args()
root = Path(args.root)
assert root.resolve() == root
expansion = root / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
support = expansion / 'planning/future_queue_audit_20261024_v1'
sys.path.insert(0, str(support))
from queue_support import QueueEvidence, pin

plans = [expansion / 'campaign_execution_20261016_v7' / ('PLAN-' + worker + '.json')
         for worker in args.workers]
queue = QueueEvidence(root, plans)
rows, issues = [], []
counts = Counter((task['kind'], 'pending') for task in [])
for task_id, task in queue.tasks.items():
    location = queue.locations.get(task_id)
    if location is None:
        continue
    counts[(task['kind'], location['status'])] += 1
    if location['status'] != 'complete':
        continue
    result = json.loads(location['result'].read_text())
    assert result['TEST_accessed'] is False
    fit = task['kind'] == 'fit'
    selected = result.get('selected_VALID', {}) if fit else result
    metrics = selected.get('metrics', {})
    row = {key: task.get(key) for key in ('id', 'kind', 'cell_id', 'group_id',
            'protocol_track', 'scientific_method_id', 'cohort_seed', 'body', 'split', 'worker')}
    row.update(metrics=metrics, loss=selected.get('loss'),
               records=selected.get('records'), complete_source_VALID=selected.get('complete_source_VALID'),
               result_receipt=pin(location['result']))
    for key in ('completed_epochs', 'selected_epoch', 'best_validation_metric', 'stop_reason',
                'policy', 'convergence_established', 'procedure'):
        if key in result:
            row[key] = result[key]
    if not metrics:
        issues.append({'id': task_id, 'issue': 'No selected metrics'})
    for metric, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            issues.append({'id': task_id, 'issue': 'Nonfinite or nonnumeric metric', 'metric': metric})
        elif any(token in metric.lower() for token in ('accuracy', 'acc', 'auc', 'hits', 'mrr', 'ap')) and not 0 <= value <= 1:
            issues.append({'id': task_id, 'issue': 'Metric outside expected [0,1]', 'metric': metric, 'value': value})
    loss = row['loss']
    if loss is not None and (not isinstance(loss, (int, float)) or not math.isfinite(loss)):
        issues.append({'id': task_id, 'issue': 'Nonfinite or nonnumeric loss'})
    if fit:
        policy = result.get('policy', {})
        epochs, selected_epoch = result.get('completed_epochs'), result.get('selected_epoch')
        if not isinstance(epochs, int) or not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= epochs:
            issues.append({'id': task_id, 'issue': 'Invalid selected/completed epoch'})
        row['hit_epoch_cap'] = isinstance(epochs, int) and epochs >= policy.get('maximum_epochs', float('inf'))
        row['below_minimum_epochs'] = isinstance(epochs, int) and epochs < policy.get('minimum_epochs', 0)
        if selected.get('role') != 'VALID' or selected.get('complete_source_VALID') is not True:
            issues.append({'id': task_id, 'issue': 'Selected evaluation is not complete source VALID'})
    rows.append(row)

summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'root': str(root),
           'plans': queue.plans, 'terminal_counts': [{'kind': k[0], 'status': k[1], 'count': v}
                                                   for k, v in sorted(counts.items())],
           'completed_rows': rows, 'scalar_integrity_issues': issues,
           'omitted_counts': dict(Counter(row['status'] for row in queue.omitted)),
           'model_or_tensor_or_TEST_work': False, 'training_changes': False,
           'limits': 'Metadata sanity check. Validation is used for model selection. Finite scores and patience stopping do not prove convergence, reproduction, or superiority.'}
print(json.dumps(summary, indent=2, allow_nan=False))
