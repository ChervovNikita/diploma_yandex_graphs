"""CPU-only numerical checks for the anonymous new-study compact artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GRID=ROOT/'experiments_iclr'/'roman_depth_grid'
COLLAB=ROOT/'experiments_iclr'/'ogbl_collab'

def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def close(x,y,tol=1e-8):
    if abs(float(x)-float(y))>tol: raise AssertionError((x,y))

audit=load(GRID/'ROMAN_DEPTH_GRID_40_CELL_AUDIT.json')
dec=load(GRID/'ROMAN_DEPTH_GRID_DECISION_ACCOUNTING.json')
index=load(GRID/'grid_index.json')['cells']
assert audit['status']=='COMPLETE_40_CELL_READ_ONLY_AUDIT_PASS' and len(index)==40
assert dec['status'].startswith('COMPLETE')
rec={(x['depth'],x['seed'],x['arm']):x for x in audit['records']}
acct={(x['depth'],x['seed'],x['arm'],x['part']):x for x in dec['arms']}
assert len(rec)==40 and len(acct)==80
scores={}
for item in index:
    depth,seed,arm=item['depth'],item['seed'],item['arm']
    p=GRID/item['run_dir']
    result=load(p/'result.json')
    assert sha(p/'result.json')==rec[depth,seed,arm]['result_sha256']
    assert result['selected_epoch']==rec[depth,seed,arm]['selected_epoch']
    with np.load(p/'selected_decisions.npz',allow_pickle=False) as z:
        for part in ('valid','test'):
            y=z[f'{part}_labels']
            mem=z[f'{part}_member_classes']
            pool=z[f'{part}_pooled_classes']
            ids=z[f'{part}_indices']
            assert mem.shape==(4,len(y)) and pool.shape==y.shape and len(np.unique(ids))==len(ids)
            score=float(np.mean(pool==y))
            close(score,result[f'{part}_accuracy'],1e-6)
            row=acct[depth,seed,arm,part]
            close(score,row['pooled_accuracy'])
            k=(mem==y[None,:]).sum(axis=0)
            covered=k>=1
            c=float(np.mean(covered))
            u=float(np.mean(pool[covered]==y[covered]))
            r=float(np.mean(pool[~covered]==y[~covered])) if (~covered).any() else 0.0
            close(c,row['C']); close(u,row['U']); close(r,row['R'])
            close(float(np.mean(mem==y[None,:])),row['mean_member_accuracy'])
            close(score,row['mean_member_accuracy']+row['pooling_gain'])
            if part=='test':
                close(score,rec[depth,seed,arm]['test_accuracy'])
                scores[depth,seed,arm]=score
assert len(scores)==40
for summary in audit['depth_summary']:
    depth=summary['depth']
    diffs=[scores[depth,s,'tied']-scores[depth,s,'untied_propagation'] for s in range(5)]
    for a,b in zip(diffs,summary['paired_differences']): close(a,b)
    close(np.mean(diffs),summary['mean_difference'])

ca=load(COLLAB/'audit.json')
assert ca['required_pairs']==12 and ca['cpu_replay'] is True
runs={(r['variant'],r['seed']):r for r in ca['runs']}
assert len(runs)==12
edge_reference=None
for variant in ('tied','untied','ens','base'):
    for seed in range(3):
        p=ROOT/'experiments_iclr'/'ogbl_collab_results'/variant/f'seed_{seed}'
        meta=load(p/'selected.json')
        config=load(p/'run_config.json')
        assert config['data_fingerprints']==ca['data_fingerprints']
        assert config['protocol']==ca['protocol']
        assert meta['selected_step']==runs[variant,seed]['selected_step']
        with np.load(p/'selected_decisions.npz',allow_pickle=False) as z:
            assert set(z.files)=={'valid_member_hits','valid_pooled_hits','test_member_hits','test_pooled_hits','test_pos_edges'}
            if edge_reference is None: edge_reference=z['test_pos_edges'].copy()
            else: assert np.array_equal(edge_reference,z['test_pos_edges'])
            for part in ('valid','test'):
                member=z[f'{part}_member_hits']
                pooled=z[f'{part}_pooled_hits']
                assert member.shape[0]==(1 if variant=='base' else 4)
                score=float(np.mean(pooled))
                close(score,meta[f'{part}_hits50'])
                close(score,runs[variant,seed][part]['hits50'])
                for i,value in enumerate(meta[f'{part}_member_hits50']):
                    close(float(np.mean(member[i])),value)
for seed,reported in enumerate(ca['paired_tied_minus_untied_test_hits50']):
    a=load(ROOT/'experiments_iclr'/'ogbl_collab_results'/'tied'/f'seed_{seed}'/'selected.json')['test_hits50']
    b=load(ROOT/'experiments_iclr'/'ogbl_collab_results'/'untied'/f'seed_{seed}'/'selected.json')['test_hits50']
    close(a-b,reported)
print('PASS: 40 Roman grid cells, 80 decision parts, 12 link runs')
