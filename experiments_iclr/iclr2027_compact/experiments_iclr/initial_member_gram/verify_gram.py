"""Check all retained Gram matrices and scalar arithmetic, without raw gradients."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'MANIFEST.json').read_text())
for name,digest in manifest['sha256'].items():
    assert sha(ROOT/name)==digest,name
report=json.loads((ROOT/'INITIAL_MEMBER_GRADIENT_GRAM.json').read_text())
audit=json.loads((ROOT/'INITIAL_MEMBER_GRAM_ROOT_AUDIT.json').read_text())
assert report['source_sha256']==sha(ROOT/'analyze_initial_member_gram.py')
assert audit['report_sha256']==sha(ROOT/'INITIAL_MEMBER_GRADIENT_GRAM.json')
assert audit['input_manifest_sha256']==report['input_manifest_sha256']
assert audit['status']=='ALL12_RAW_GRAM_AND_FIRST_ORDER_IDENTITIES_PASS'
expected={(d,s) for d in ('cora','wikics','actor','chameleon_filtered') for s in range(3)}
assert {(r['dataset'],r['seed']) for r in report['rows']}==expected
assert len(report['rows'])==len(audit['records'])==12
for row in report['rows']:
    g=np.asarray(row['gram'],dtype='float64')
    assert g.shape==(4,4) and np.isfinite(g).all()
    assert np.allclose(g,g.T,rtol=0,atol=1e-10)
    assert np.linalg.eigvalsh(g).min()>=-1e-10 and np.all(g.diagonal()>0)
    norms=np.sqrt(g.diagonal())
    cos=g/np.outer(norms,norms)
    off=cos[np.triu_indices(4,1)]
    ratio=g.sum()/np.trace(g)
    assert 0<=ratio<=4+1e-12
    assert np.allclose(cos,row['pairwise_cosine'],rtol=1e-10,atol=1e-10)
    assert np.isclose(off.mean(),row['mean_offdiagonal_cosine'],rtol=1e-10,atol=1e-10)
    assert (off<0).sum()==row['negative_pair_count']
    assert (g.sum(1)<0).sum()==row['members_with_adverse_graph_only_sgd_first_order_loss_change']
    a=next(r for r in audit['records'] if (r['dataset'],r['seed'])==(row['dataset'],row['seed']))
    assert a['raw_sha256']==row['source_sha256']
    for actual,key in [(ratio,'descent_magnitude_ratio'),(g.sum()/16,'tied_descent_magnitude'),
                       (np.trace(g)/16,'untied_descent_magnitude'),((g.sum()-np.trace(g))/16,'cross_term')]:
        assert np.isclose(actual,a[key],rtol=1e-10,atol=1e-10)
print('ALL12_RETAINED_GRAM_ARITHMETIC_PASS; original gradient recomputation requires omitted arrays or new source execution')
