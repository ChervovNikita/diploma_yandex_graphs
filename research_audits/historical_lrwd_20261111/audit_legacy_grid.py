"""Inspect saved historical scores using validation-only candidate selection."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT=Path.cwd()
OUT=ROOT/'.campaign_test_tmp/historical_lrwd_audit'
COMMIT='c33ce043f4a0d79622701c1c5cbed492dc272d01'
MODELS=['ResNet','GCN','SAGE','GAT','GAT-sep','GT','GT-sep']
paths=subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT,'results_grid'],text=True).splitlines()
blobs={name:subprocess.check_output(['git','show',f'{COMMIT}:{name}']) for name in paths if name.endswith('.csv')}
headers={}
for name,blob in blobs.items():
    for row in csv.reader(io.StringIO(blob.decode())):
        if row and row[0]=='dataset':
            key=Path(name).name
            if key in headers:
                assert headers[key]==row
            headers[key]=row
records=[]
manifest=[]
for name,blob in blobs.items():
    header=headers[Path(name).name]
    method={'base.csv':'BASE','ensemble.csv':'ENS','tabm.csv':'GNNM'}[Path(name).name]
    count=0
    for line,row in enumerate(csv.reader(io.StringIO(blob.decode())),1):
        if not row or row[0]=='dataset':
            continue
        assert len(row)==len(header),(name,line,len(row),len(header))
        values=dict(zip(header,row))
        record={key:values[key] for key in ['dataset','model','num_layers','hidden_dim','lr','split','val_metric','test_metric']}
        record.update(method=method,source_file=name,source_line=line)
        records.append(record)
        count+=1
    manifest.append(dict(path=name,sha256=hashlib.sha256(blob).hexdigest(),rows=count))
frame=pd.DataFrame(records)
for name in ['num_layers','hidden_dim','lr','split','val_metric','test_metric']:
    frame[name]=pd.to_numeric(frame[name],errors='raise')
keys=['dataset','model','num_layers','hidden_dim','lr','split','method']
conflicts=frame.groupby(keys)[['val_metric','test_metric']].nunique().max(axis=1)
assert not (conflicts>1).any(),conflicts[conflicts>1]
exact_duplicate_rows=int(frame.duplicated(keys).sum())
frame=frame.drop_duplicates(keys)
coverage=frame.groupby(['dataset','method']).agg(rows=('lr','size'),models=('model','nunique'),splits=('split','nunique'))
coverage.to_csv(OUT/'legacy_coverage.csv')
frame.to_csv(OUT/'legacy_saved_scores.csv',index=False)
# Every reported candidate has all ten original splits. Depth is selected by VALID only.
candidates=frame[(frame.hidden_dim==512)&frame.model.isin(MODELS)].groupby(
    ['dataset','model','method','lr','num_layers']).agg(
    n=('split','nunique'),valid=('val_metric','mean'),test=('test_metric','mean')).reset_index()
candidates=candidates[candidates.n==10]
winners=candidates.sort_values(['valid','num_layers'],ascending=[False,True]).groupby(
    ['dataset','model','method','lr'],sort=False).head(1)
winners.to_csv(OUT/'legacy_per_lr_validation_selected_depth.csv',index=False)
print('COVERAGE')
print(coverage.to_string())
print('LRS',sorted(frame.lr.unique()),'WIDTHS',sorted(frame.hidden_dim.unique()),'DEPTHS',sorted(frame.num_layers.unique()))
print('GAT width512, best existing depth independently selected by validation per LR')
print(winners[winners.model=='GAT'].sort_values(['dataset','method','lr']).round(6).to_string(index=False))
# Conservative common-cell comparison wherever all three methods exist.
paired=candidates.pivot(index=['dataset','model','lr','num_layers'],columns='method',values=['valid','test'])
all_methods=all(name in paired['valid'].columns for name in ['BASE','ENS','GNNM'])
if all_methods:
    complete=paired.dropna(subset=[('valid',name) for name in ['BASE','ENS','GNNM']]).reset_index()
    complete.columns=['_'.join(str(part) for part in column if part) if isinstance(column,tuple) else column for column in complete.columns]
    complete.to_csv(OUT/'legacy_complete_three_method_cells.csv',index=False)
    print('COMPLETE THREE-METHOD CANDIDATE CELLS',len(complete))
    print(complete[complete.model=='GAT'].round(6).to_string(index=False))
(OUT/'legacy_source_manifest.json').write_text(json.dumps(dict(commit=COMMIT,
    source_files=manifest,exact_duplicate_rows=exact_duplicate_rows,
    metric_handling='Summaries of existing saved scores only. No model inference or score replacement.',
    depth_selection='mean VALID over ten existing official splits; TEST never used for ranking',
    source_weight_decay='0, verified in historical run_base.py and run_tabm.py'),indent=2)+'\n')
