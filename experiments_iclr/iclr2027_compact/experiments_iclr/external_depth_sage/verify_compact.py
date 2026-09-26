"""Check all 24 decision records and validation-selection traces in the compact archive."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def need(cond,msg):
    if not cond: raise RuntimeError(msg)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

frozen=json.loads((ROOT/'PRETRAIN_FREEZE.json').read_text())['data_and_source_sha256']
for name in ('external_depth_grid.py','external_depth_protocol.md','models.py'):
    need(sha(ROOT/name)==frozen[name],f'Frozen source changed: {name}')
count=0
for dataset in ('wikics','actor'):
    for depth in (2,5):
        base=ROOT/'results'/dataset/f'depth{depth}'
        manifest=json.loads((base/'source_manifest.json').read_text())
        need(manifest['protocol']=='external_sage_depth_grid_v1' and manifest['configuration']['layers']==depth,'manifest protocol/depth')
        need(manifest['arms']==['tied','untied_propagation'] and manifest['optimization_seeds']==[0,1,2],'manifest grid')
        for name,digest in manifest['source_sha256'].items():
            if not name.startswith('data/'):
                need(sha(ROOT/name)==digest,f'manifest source changed: {name}')
        replay=json.loads((base/'completion_audit.json').read_text())
        need(replay['status']=='COMPLETE_CUDA_REPLAY_PASS' and len(replay['records'])==6,'frozen CUDA replay status')
        need({(r['seed'],r['arm']) for r in replay['records']}=={(s,a) for s in (0,1,2) for a in ('tied','untied_propagation')},'CUDA replay cell list')
        for seed in (0,1,2):
            pair=base/f'seed{seed}'
            ti=json.loads((pair/'tied'/'initialization.json').read_text())
            for arm in ('tied','untied_propagation'):
                run=pair/arm
                row=json.loads((run/'result.json').read_text())
                init=json.loads((run/'initialization.json').read_text())
                need(row['protocol']=='external_sage_depth_grid_v1' and row['dataset']==dataset and row['depth']==depth and row['optimization_seed']==seed and row['arm']==arm,'run identity')
                need(row['source_manifest_sha256']==sha(base/'source_manifest.json'),'manifest link')
                need(row['epochs_run']==300,'budget')
                for key in ('canonical_tied_state_sha256','cpu_rng_sha256','cuda_rng_sha256'):
                    need(ti[key]==init[key],'paired initialization hash')
                need(init['paired_initial_logits_max_abs_diff']<=1e-5,'paired initial logit tolerance')
                with (run/'validation_trace.csv').open(newline='') as f: rows=list(csv.DictReader(f))
                need(len(rows)==300,'trace length')
                best=(-1.0,float('inf'),0)
                for epoch,t in enumerate(rows,1):
                    acc,ce=float(t['valid_accuracy']),float(t['valid_ce'])
                    better=acc>best[0]+1e-12 or (abs(acc-best[0])<=1e-12 and ce<best[1]-1e-12)
                    need(int(t['epoch'])==epoch and int(t['improved'])==int(better),'trace order/selection')
                    if better: best=(acc,ce,epoch)
                need(best[2]==row['selected_epoch'] and abs(best[0]-row['valid_accuracy'])<1e-7 and abs(best[1]-row['valid_ce'])<1e-6,'trace winner')
                with np.load(run/'selected_decisions.npz',allow_pickle=False) as f: data={k:f[k] for k in f.files}
                for part in ('valid','test'):
                    y=data[f'{part}_labels']
                    p=data[f'{part}_pooled_predictions']
                    m=data[f'{part}_member_predictions']
                    need(p.shape==y.shape and m.shape==(4,len(y)) and len(data[f'{part}_indices'])==len(y),'decision shape')
                    acc=float(np.mean(p==y))
                    need(abs(acc-row[f'{part}_accuracy'])<1e-7,f'{part} score')
                count+=1
need(count==24,'incomplete compact grid')
print('PASS_COMPACT_24_EXTERNAL_DEPTH_CELLS')
