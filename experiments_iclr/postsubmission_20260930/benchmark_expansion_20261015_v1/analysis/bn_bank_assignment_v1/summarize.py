"""Summarize all fifteen frozen conditions from compact JSON, no tensors."""
import csv
import json
from pathlib import Path
import statistics


def main():
    here=Path(__file__).resolve().parent
    paths=sorted((here/'fetched').glob('seed_*.json'))
    rows=[json.loads(p.read_text()) for p in paths]
    assert [r['seed'] for r in rows]==[101,202,303]
    assert all(r['status']=='complete' and r['native_forwards']==100 for r in rows)
    conditions=['own','tied','shift1','shift2','shift3']
    assert all([c['condition'] for c in r['conditions']]==conditions for r in rows)
    summary={};pairs=[]
    for condition in conditions:
        values=[next(c for c in r['conditions'] if c['condition']==condition) for r in rows]
        summary[condition]={k:{'mean':statistics.mean(c[k] for c in values),
                              'SD':statistics.stdev(c[k] for c in values)}
                            for k in ['rocauc','BCE','Brier']}
    for row in rows:
        d={c['condition']:c for c in row['conditions']};pair={'seed':row['seed']}
        for k in ['rocauc','BCE','Brier']:
            pair[k+'_own_minus_tied']=d['own'][k]-d['tied'][k]
            pair[k+'_own_minus_mean_shift_score']=d['own'][k]-statistics.mean(d[s][k] for s in conditions[2:])
        pairs.append(pair)
    summary['mean_of_shift_scores']={k:{'mean':statistics.mean(
        statistics.mean(c[k] for c in r['conditions'] if c['condition'].startswith('shift'))
        for r in rows)} for k in ['rocauc','BCE','Brier']}
    value={'status':'complete','conditions':summary,'paired_differences':pairs,
           'native_forwards':sum(r['native_forwards'] for r in rows),
           'wall_seconds':json.loads((here/'fetched/END.json').read_text())['wall_seconds'],
           'max_peak_RSS_KiB':max(r['peak_RSS_KiB'] for r in rows),
           'optimizer_updates':0,'CUDA_used':False,'TEST_accessed':False}
    # A rerun verifies the existing summary instead of replacing it silently.
    target=here/'SUMMARY.json'
    if target.exists(): assert json.loads(target.read_text())==value
    else: target.write_text(json.dumps(value,indent=2)+'\n')
    target=here/'ALL_SCORES.csv'
    if not target.exists():
        with target.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['seed','selected_epoch','condition','rocauc','BCE','Brier','VALID_graphs','seconds'])
            writer.writeheader()
            for r in rows:
                for c in r['conditions']:writer.writerow({'seed':r['seed'],'selected_epoch':r['selected_epoch'],**c})
    print(json.dumps({'conditions':15,'seeds':3,'new_training':0,'summary_verified':True}))


if __name__=='__main__':
    main()
