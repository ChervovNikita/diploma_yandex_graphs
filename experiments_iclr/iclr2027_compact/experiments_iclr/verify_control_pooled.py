"""Recalculate the retained 14 earlier control-prediction files without a GPU."""
from pathlib import Path
import csv, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'experiments_iclr/CONTROL_POOLED_MANIFEST.json').read_text())
assert manifest['status']=='ALL14_CORE_PREDICTION_FILES_UNIFORM_POOLED_PROJECTION'
assert len(manifest['records'])==14
assert sha(ROOT/'experiments_iclr/prediction_manifest.json')==manifest['original_prediction_manifest_sha256']
for r in manifest['records']:
    path=ROOT/r['projection_path']
    assert sha(path)==r['projection_sha256']
    table=ROOT/r['source_table_path'];assert sha(table)==r['source_table_sha256']
    with table.open() as f:
        rows=list(csv.DictReader(f))
    matched=[x for x in rows if x['prediction_file']==r['original_path']]
    assert matched==[r['source_row']]
    row=matched[0]
    with np.load(path,allow_pickle=False) as a:
        assert set(a.files)==set(r['retained_arrays']) and 'member_logits' not in a.files
        z=a['pooled_logits'].astype(np.float64);y=a['y_true']
        assert z.shape==(len(y),18) and np.isfinite(z).all()
        assert np.array_equal(z.argmax(-1),a['ensemble_pred'])
        assert a['member_pred'].shape==(4,len(y))
        assert len(np.unique(a['node_index']))==len(y)
        member_acc=float((a['member_pred']==y[None]).mean())
        acc=float((z.argmax(-1)==y).mean())
        peak=z.max(-1)
        ce=float((peak+np.log(np.exp(z-peak[:,None]).sum(-1))-z[np.arange(len(y)),y]).mean())
        p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        assert np.max(np.abs(p-a['ensemble_prob']))<1e-5
        for actual,k,tol in [(acc,'test_acc',1e-7),(member_acc,'mean_member_acc',1e-7),(ce,'test_loss',1e-5)]:
            assert abs(actual-float(row[k]))<tol,(r['original_path'],k,actual,row[k])
print('ALL14_CONTROL_POOLED_SCORE_CHECK_PASS')
