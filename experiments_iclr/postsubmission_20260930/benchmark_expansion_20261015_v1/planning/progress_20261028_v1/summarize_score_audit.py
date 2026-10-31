"""Save a descriptive snapshot of all currently collected completed scalars."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
inputs = [HERE / (host + '-completed-scores.json') for host in ('allocation', 'server51')]
documents = [json.loads(path.read_text()) for path in inputs]
rows = [row for document in documents for row in document['completed_rows']]
assert len({row['id'] for row in rows}) == len(rows)
issues = [issue for document in documents for issue in document['scalar_integrity_issues']]
groups = defaultdict(list)
for row in rows:
    for metric, score in row['metrics'].items():
        groups[row['cell_id'], metric].append(score)
ranges = [{'cell': key[0], 'metric': key[1], 'rows': len(values),
           'minimum': min(values), 'maximum': max(values)} for key, values in sorted(groups.items())]

buddy = [row for row in rows if row['cell_id'] == 'collab-buddy'
         and row['protocol_track'] == 'long_convergence']
banks = {row['group_id']: row for row in buddy if row['kind'] == 'aggregate_I4_exports'}
methods = sorted({row['scientific_method_id'] for row in buddy
                  if row['kind'] == 'fit' and row['scientific_method_id'] != 'native_single'})
comparisons = []
for method in methods:
    pairs = []
    for row in buddy:
        if row['scientific_method_id'] != method or row['group_id'] not in banks:
            continue
        control = banks[row['group_id']]
        score, base = row['metrics']['hits@50'], control['metrics']['hits@50']
        pairs.append({'cohort_id': row['group_id'], 'method_id': row['id'],
                      'control_id': control['id'], 'method_metric': score,
                      'control_metric': base, 'difference': score - base,
                      'method_loss': row['loss'], 'control_loss': control['loss'],
                      'method_result': row['result_receipt'], 'control_result': control['result_receipt']})
    n = len(pairs)
    differences = [pair['difference'] for pair in pairs]
    comparisons.append({'method': method, 'n_score_pairs': n, 'n_planned': 10,
                        'missing_pairs': 10 - n,
                        'full_reporter_resolution_status': 'not assessed by this scalar audit',
                        'incomplete': True,
                        'method_mean': statistics.mean(pair['method_metric'] for pair in pairs),
                        'I4_mean_on_same_pairs': statistics.mean(pair['control_metric'] for pair in pairs),
                        'mean_paired_difference': statistics.mean(differences),
                        'wins': sum(value > 0 for value in differences),
                        'ties': sum(value == 0 for value in differences),
                        'losses': sum(value < 0 for value in differences), 'pairs': pairs,
                        'interval_withheld': 'Fewer than ten planned pairs; full reporter binding/convergence resolution not assessed here.'})

fits = [row for row in rows if row['kind'] == 'fit']
long = [row for row in fits if row['protocol_track'] == 'long_convergence']
summary = {'source_files': [{'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in inputs],
           'training_fits': len(fits), 'ensemble_exports': len(rows) - len(fits),
           'scalar_integrity_issues': issues, 'ranges': ranges, 'BUDDY_incomplete_comparisons': comparisons,
           'long_training_fits': len(long),
           'long_stop_reasons': dict(Counter(row.get('stop_reason') for row in long)),
           'long_epoch_caps': [row['id'] for row in long if row.get('hit_epoch_cap')],
           'below_declared_minimum': [row['id'] for row in fits if row.get('below_minimum_epochs')],
           'test_metrics_read': False, 'training_or_scientific_settings_changed': False,
           'limitations': ['Validation outcomes, not test reproduction.',
                          'No inference from an incomplete subset to the full preset.',
                          'Method/control registry and complete reporting witness resolution remain the frozen full reporter responsibility.',
                          'Finite metrics and patience stopping do not prove optimization convergence.',
                          'No statistical significance, novelty, or general superiority claim.']}
(HERE / 'SCORE_AUDIT_SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
with (HERE / 'COMPLETED_SCALARS.csv').open('w', newline='') as stream:
    fields = ['id', 'cell_id', 'protocol_track', 'kind', 'scientific_method_id',
              'group_id', 'body', 'metric', 'score', 'loss', 'completed_epochs', 'selected_epoch', 'stop_reason']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        for metric, score in row['metrics'].items():
            writer.writerow({**{key: row.get(key) for key in fields if key not in ('metric', 'score')},
                             'metric': metric, 'score': score})
lines = ['# Completed-score audit — 28 October 2026', '',
         'Server51 access is restored. The supplied password was correct. A temporary private-terminal reader lost one byte; whitespace framing drains the competing read, and a private integrity handshake now verifies exact delivery before SSH. No password was saved to project files or written to the other Mac. The authentication terminals were closed.', '',
         '## Scope and numerical checks', '',
         f'Inspected all {len(fits)} completed fits and {len(rows)-len(fits)} independent-ensemble exports found in the three frozen worker plans. No selected metric/loss was missing, nonfinite, or outside its expected metric range. Every fit reports complete source VALID evaluation. This is a metadata sanity audit, not an independent checkpoint replay or proof of paper reproduction.', '',
         '| Model/dataset cell | Metric | Completed result rows | Observed validation range |',
         '| --- | --- | ---: | ---: |']
for row in ranges:
    lines.append(f"| {row['cell']} | {row['metric']} | {row['rows']} | {100*row['minimum']:.2f}–{100*row['maximum']:.2f}% |")
lines += ['', 'Ranges include different completed methods, cohorts, and protocol tracks. They are not mean performance estimates and are not interchangeable across cells.', '',
          '## Training duration', '',
          f'All {len(long)} longer-protocol fits stopped by validation patience; none hit its maximum epoch cap or ended below its declared minimum. Two BUDDY native-replication runs ended at the prescribed 100-epoch source horizon. These observations support execution of the intended stopping policy, not a proof of globally converged optimization.', '',
          '## Incomplete BUDDY comparisons', '',
          'All available shared-method comparisons are retained below. Each mean uses only the stated matched cohorts out of ten planned. These are incomplete descriptive observations. Full reporter binding/convergence resolution is not assessed by this scalar audit; no interval or superiority conclusion is reported. No method was promoted, removed, or retuned in response.', '',
          '| Method | Pairs / 10 | Shared mean | I4 on same pairs | Difference, pp | Wins / ties / losses |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
for row in comparisons:
    lines.append(f"| {row['method']} | {row['n_score_pairs']}/10 | {100*row['method_mean']:.3f} | {100*row['I4_mean_on_same_pairs']:.3f} | {100*row['mean_paired_difference']:+.3f} | {row['wins']}/{row['ties']}/{row['losses']} |")
lines += ['', '## Issues requiring interpretation', '',
          '- BUDDY ranking metrics are close to the matched native/ensemble baseline range. Several shared variants improve available paired scores; others lose or are essentially tied. The base GNNM wrapper is slightly behind I4 on its four currently available pairs.',
          '- Recorded BUDDY validation loss ranges up to 19.35, including high values for some native runs. Ranking performance alone cannot establish good probability quality. This deserves a separate loss/logit analysis; finite values alone do not resolve it.',
          '- Only two MolBACE/GCN shared-method fits have completed, without their matched native controls in this snapshot. The PubMed shared-method observations also lack the matching current-campaign control cohort. Their scores cannot yet establish a method advantage.',
          '- Current outcomes are VALID results. A source-paper TEST score cannot be treated as a like-for-like reproduction comparator here.', '',
          '## Evidence', '',
          '`allocation-completed-scores.json` and `server51-completed-scores.json` preserve source result receipts and all observed scores. `COMPLETED_SCALARS.csv` is the compact complete table. `SCORE_AUDIT_SUMMARY.json` retains exact matched IDs, all partial differences and limitations. Queue owners remain alive with six active fits and no current failed tasks in the host snapshots. Training and scientific settings were unchanged.']
(HERE / 'ACCURACY_AUDIT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'fits': len(fits), 'ensemble_exports': len(rows)-len(fits),
                  'scalar_issues': len(issues), 'partial_BUDDY_methods': len(comparisons)}))
