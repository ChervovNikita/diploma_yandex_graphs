"""Summarize three complete mixture diagnostics and their pinned prior results."""
import csv
import json
from pathlib import Path
import statistics
import sys


def main():
    # Compact RESULT.json files only; no tensor/model import or scoring here.
    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    rows = [json.loads((source / f'seed_{i}' / 'RESULT.json').read_text()) for i in range(3)]
    assert [r['seed'] for r in rows] == [101, 202, 303]
    assert all(r['status'] == 'complete' and r['native_forwards'] == 40 for r in rows)
    assert all(r['parameters_unchanged'] and r['banks_restored'] and not r['TEST_accessed'] for r in rows)
    assert all([c['condition'] for c in r['conditions']] == ['own', 'mixture'] for r in rows)
    assert all([c['condition'] for c in r['prior_conditions']] == ['own','tied','shift1','shift2','shift3'] for r in rows)
    assert all(c['VALID_graphs'] == 151 for r in rows for c in r['conditions'] + r['prior_conditions'])
    end = json.loads((source / 'END.json').read_text())
    assert end['status'] == 'complete'
    output.mkdir(parents=True, exist_ok=False)
    metrics = ['rocauc','BCE','Brier']
    pooled = {}
    pairs, flat = [], []
    for row in rows:
        current = {c['condition']:c for c in row['conditions']}
        prior = {c['condition']:c for c in row['prior_conditions']}
        pair = {'seed':row['seed'], 'selected_epoch':row['selected_epoch']}
        for metric in metrics:
            pair[metric+'_own_minus_mixture'] = current['own'][metric] - current['mixture'][metric]
            pair[metric+'_mixture_minus_arithmetic_tied'] = current['mixture'][metric] - prior['tied'][metric]
            pair[metric+'_own_minus_mean_shift_score'] = current['own'][metric] - statistics.mean(prior[s][metric] for s in ['shift1','shift2','shift3'])
            pair[metric+'_own_minus_prior_own'] = current['own'][metric] - prior['own'][metric]
        pairs.append(pair)
        for origin, conditions in [('current',row['conditions']),('prior',row['prior_conditions'])]:
            for condition in conditions:
                key = origin + ':' + condition['condition']
                pooled.setdefault(key, []).append(condition)
                flat.append({'seed':row['seed'], 'selected_epoch':row['selected_epoch'], 'origin':origin, **condition})
    summary = {condition:{metric:{'mean':statistics.mean(r[metric] for r in values),
                                   'SD':statistics.stdev(r[metric] for r in values)}
                          for metric in metrics} for condition,values in pooled.items()}
    differences = {key:{'mean':statistics.mean(row[key] for row in pairs),
                      'SD':statistics.stdev(row[key] for row in pairs)}
                   for key in pairs[0] if key not in ['seed','selected_epoch']}
    value = {'status':'complete', 'conditions':summary, 'paired_differences':pairs,
             'difference_summaries':differences,
             'difference_sign':'literal first minus second; higher AUC and lower BCE/Brier are favorable',
             'new_native_forwards':sum(row['native_forwards'] for row in rows),
             'new_full_VALID_calls':6, 'prior_conditions_reused_without_reforward':15,
             'wall_seconds':end['wall_seconds'],
             'max_peak_RSS_KiB':max(row['peak_RSS_KiB'] for row in rows),
             'checkpoint_receipts':[row['checkpoint'] for row in rows],
             'prior_assignment_receipts':[row['prior_assignment'] for row in rows],
             'bank_moment_summaries':[{k:row[k] for k in ['seed','bank_moment_summaries']} for row in rows],
             'optimizer_updates':0, 'TRAIN_calls':0, 'TEST_accessed':False,
             'new_serving_rule_selected':False, 'interpretation':'development diagnostic at fixed selected weights'}
    (output/'SUMMARY.json').write_text(json.dumps(value,indent=2)+'\n')
    with (output/'ALL_SCORES.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','selected_epoch','origin','condition','rocauc','BCE','Brier','VALID_graphs','seconds'])
        writer.writeheader();writer.writerows(flat)
    print(json.dumps({'status':'complete','new_conditions':6,'prior_conditions_reused':15,'new_training':0}))


if __name__ == '__main__':
    main()
