"""Descriptive audit of every completed scalar and all available matched pairs.

No model selection, test evaluation, training changes or significance claims.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
paths = [HERE / (host + '-completed-scores.json') for host in ('allocation', 'server51')]
documents = [json.loads(path.read_text()) for path in paths]
rows = [row for document in documents for row in document['completed_rows']]
assert len({row['id'] for row in rows}) == len(rows)
fits = [row for row in rows if row['kind'] == 'fit']
long = [row for row in fits if row['protocol_track'] == 'long_convergence']
issues = [issue for document in documents for issue in document['scalar_integrity_issues']]
groups = defaultdict(list)
for row in rows:
    for metric, value in row['metrics'].items():
        groups[(row['cell_id'], metric)].append(value)
ranges = [dict(cell=cell, metric=metric, n=len(values), minimum=min(values), maximum=max(values))
          for (cell, metric), values in sorted(groups.items())]
ensembles = {row['group_id']: row for row in rows if row['kind'] == 'aggregate_I4_exports'}
singles = {row['group_id']: row for row in fits if row['scientific_method_id'] == 'native_single'
           and row['protocol_track'] == 'long_convergence' and row['body'] == 0}
originals = {row['group_id']: row for row in fits if row['scientific_method_id'] == 'gnnm_boundary_native'}
matched = defaultdict(list)
for row in fits:
    if row['scientific_method_id'] == 'native_single' or row['protocol_track'] != 'long_convergence':
        continue
    for control_name, bank in [('I4', ensembles), ('M1', singles), ('original_GNNM', originals)]:
        control = bank.get(row['group_id'])
        if control is None or control['id'] == row['id']:
            continue
        for metric, value in row['metrics'].items():
            assert metric in control['metrics']
            matched[(row['cell_id'], row['scientific_method_id'], control_name, metric)].append({
                'group_id': row['group_id'], 'seed': row['cohort_seed'], 'split': row['split'],
                'method_id': row['id'], 'control_id': control['id'],
                'method_score': value, 'control_score': control['metrics'][metric],
                'difference': value - control['metrics'][metric]})
comparisons = []
for (cell, method, control, metric), pairs in sorted(matched.items()):
    differences = [pair['difference'] for pair in pairs]
    comparisons.append(dict(cell=cell, method=method, control=control, metric=metric,
        n=len(pairs), planned_n=10, complete=len(pairs) == 10,
        method_mean=statistics.mean(pair['method_score'] for pair in pairs),
        control_mean=statistics.mean(pair['control_score'] for pair in pairs),
        difference_mean=statistics.mean(differences),
        difference_sample_sd=statistics.stdev(differences) if len(pairs) > 1 else None,
        wins=sum(value > 0 for value in differences), ties=sum(value == 0 for value in differences),
        losses=sum(value < 0 for value in differences), pairs=pairs))
native_pairs = defaultdict(list)
for group, single in singles.items():
    if group in ensembles:
        ensemble = ensembles[group]
        for metric, value in single['metrics'].items():
            native_pairs[(single['cell_id'], metric)].append(dict(
                group_id=group, single=value, ensemble=ensemble['metrics'][metric]))
native_controls = [dict(cell=cell, metric=metric, n=len(pairs),
                       single_mean=statistics.mean(pair['single'] for pair in pairs),
                       ensemble_mean=statistics.mean(pair['ensemble'] for pair in pairs), pairs=pairs)
                   for (cell, metric), pairs in sorted(native_pairs.items())]
summary = dict(snapshot_utc=[document['utc'] for document in documents],
    sources=[dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
    completed_tasks=len(rows), completed_fits=len(fits), completed_ensemble_exports=len(ensembles),
    scalar_integrity_issues=issues, ranges=ranges,
    long_fits=len(long), long_stop_reasons=dict(Counter(row['stop_reason'] for row in long)),
    long_cap_hits=[row['id'] for row in long if row['hit_epoch_cap']],
    below_minimum=[row['id'] for row in fits if row['below_minimum_epochs']],
    missing_fit_losses=[row['id'] for row in fits if row['loss'] is None],
    comparisons=comparisons, native_controls=native_controls, TEST_accessed=False, training_modified=False,
    limitations=['VALID selection scores, not final TEST or source-paper reproduction.',
                 'Uneven incomplete method coverage; all available comparisons retained.',
                 'Repeated seeds on one molecular/link split do not measure split uncertainty.',
                 'No significance or method superiority declaration from interim comparisons.',
                 'Finite outputs and patience stopping do not prove convergence or calibration.'])
(HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
lines = ['# Completed metric audit — 30 October 2026', '',
         'Snapshot: 19:02 Moscow. The original queue assignments and scientific settings were unchanged. '
         'All evaluations here are saved validation outputs; no test split was evaluated.', '',
         '## Execution and stopping', '',
         f'- {len(fits)} completed fits and {len(ensembles)} ensemble exports audited across both servers.',
         f'- {len(issues)} missing/nonfinite/out-of-range metric or invalid selected-epoch issues.',
         f'- All {len(long)} completed longer-protocol fits stopped by validation patience. None hit its epoch cap or ended below the declared minimum.',
         '- Native source-reproduction schedules are reported separately; reaching their fixed horizon is expected.',
         '- This verifies saved result metadata and receipts, not checkpoint replay or optimization convergence.', '',
         '## Observed validation ranges', '',
         'These ranges combine different methods and seeds and are not comparative means.', '',
         '| Model/dataset setting | Metric | Completed outputs | Range (%) |',
         '|---|---|---:|---:|']
for row in ranges:
    lines.append(f"| {row['cell']} | {row['metric']} | {row['n']} | {row['minimum']*100:.2f}–{row['maximum']*100:.2f} |")
lines += ['', '## Matched native controls', '',
          'The single is the preassigned body 0, not the best member selected after evaluation. '
          'Both columns use the longer protocol and the same completed cohorts.', '',
          '| Setting | Matched cohorts | Single (%) | Independent ensemble (%) |',
          '|---|---:|---:|---:|']
for row in native_controls:
    lines.append(f"| {row['cell']} | {row['n']} | {100*row['single_mean']:.3f} | {100*row['ensemble_mean']:.3f} |")
lines += ['', '## All available method–I4 comparisons', '',
          'Each row uses exactly the same completed seed/split cohorts for the method and its independent ensemble. '
          'Coverage is out of ten planned trials. Different rows can cover different cohorts. '
          'The complete JSON also preserves comparisons with the matched single and original GNNM. '
          'No candidate was selected, removed or retuned from this interim audit.', '',
          '| Setting | Method | Pairs / 10 | Method (%) | I4 (%) | Difference (pp) | Wins / ties / losses |',
          '|---|---|---:|---:|---:|---:|---:|']
for row in comparisons:
    if row['control'] == 'I4':
        lines.append(f"| {row['cell']} | {row['method']} | {row['n']}/10 | {100*row['method_mean']:.3f} | {100*row['control_mean']:.3f} | {100*row['difference_mean']:+.3f} | {row['wins']}/{row['ties']}/{row['losses']} |")
lines += ['', '## Interpretation', '',
    '- BUDDY same-query contrast is promising in the available four cohorts: the global variant wins all four versus I4. This is an exploratory result within a large campaign, with six planned trials still missing.',
    '- MolBACE/GCN without dropout shows a substantial weakness in original GNNM. Separate running statistics for each route recover much of the deficit while keeping learned backbone parameters shared, but the available newer variants still trail I4 on mean ROC-AUC.',
    '- The smaller MolBACE/GCN dropout-0.5 sample has gains for several extensions. It is too incomplete to counter the no-dropout result or establish robustness.',
    '- Large BUDDY validation losses and a MolHIV loss near 195 require interpretation of the loss computation and confidence of predictions. Finite ranking scores do not establish calibrated probabilities. See the separate loss audit.',
    '- Source-paper test scores are not comparable to these selected validation scores. The frozen full reporting and heldout evaluation remain necessary.', '']
(HERE / 'AUDIT.md').write_text('\n'.join(lines))
print(json.dumps({key: summary[key] for key in ('completed_tasks', 'completed_fits', 'completed_ensemble_exports', 'long_fits', 'long_cap_hits', 'below_minimum')}))
