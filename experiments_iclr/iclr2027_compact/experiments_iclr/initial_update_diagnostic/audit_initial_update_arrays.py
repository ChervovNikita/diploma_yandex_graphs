"""Independent NumPy audit and post hoc sign-magnitude sensitivity.

The original 12-row initial-update diagnostic was frozen prospectively.
The extra thresholds below are post hoc numeric interpretation checks, not
a new predeclared endpoint or a rule for selecting graph/seed results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--study',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
raw=a.study/'initial_update_raw_gradients'
manifest=json.loads((raw/'manifest.json').read_text())
frozen=a.study/'INITIAL_UPDATE_DIAGNOSTIC_FREEZE.json'
result=a.study/'INITIAL_UPDATE_DIAGNOSTIC_RESULTS.json'
assert manifest['original_diagnostic_freeze_sha256']==sha(frozen)
assert manifest['original_diagnostic_result_sha256']==sha(result)
prior={(r['dataset'],r['seed']):r for r in json.loads(result.read_text())['rows']}
assert len(prior)==12
assert set(prior)=={(g,s) for g in ['cora','wikics','actor','chameleon_filtered'] for s in range(3)}
rows=[]
for entry in manifest['rows']:
    key=(entry['dataset'],entry['seed'])
    original=prior[key]
    path=raw/entry['file']
    assert sha(path)==entry['sha256']
    with np.load(path,allow_pickle=False) as z:
        g=z['member_graph_gradients'].copy()
        actual_t=z['actual_tied_graph_update'].astype(np.float64)
        actual_s=z['actual_sync_graph_update'].astype(np.float64)
    assert g.shape==(4,165120) and g.dtype==np.float32
    assert actual_t.shape==actual_s.shape==(165120,)
    assert np.isfinite(g).all() and np.isfinite(actual_t).all() and np.isfinite(actual_s).all()
    mean=g.mean(axis=0)
    t=(-np.float32(.001)*mean/(np.abs(mean)+np.float32(1e-8))).astype(np.float64)
    s=(-np.float32(.001)*g/(np.abs(g)+np.float32(1e-8))).mean(axis=0).astype(np.float64)
    metrics={
        'update_cosine':float(t@s/(np.linalg.norm(t)*np.linalg.norm(s))),
        'opposite_sign_fraction_all_graph_coordinates':float(np.mean(t*s<0)),
        'sync_zero_update_fraction_all_graph_coordinates':float(np.mean(s==0)),
        'tied_graph_update_l2_norm':float(np.linalg.norm(t)),
        'sync_graph_update_l2_norm':float(np.linalg.norm(s)),
        'sync_over_tied_l2_norm_ratio':float(np.linalg.norm(s)/np.linalg.norm(t)),
        'relative_l2_update_difference':float(np.linalg.norm(t-s)/np.linalg.norm(t)),
        'mean_gradient_dot_tied_update':float(mean.astype(np.float64)@t),
        'mean_gradient_dot_sync_update':float(mean.astype(np.float64)@s)}
    max_error=max(abs(v-float(original[k])) for k,v in metrics.items())
    assert max_error<1e-10,(key,max_error)
    err_t=float(np.max(np.abs(actual_t-t)))
    err_s=float(np.max(np.abs(actual_s-s)))
    assert max(err_t,err_s)<=1e-5
    sensitivity=[]
    for tau in [0,1e-7,1e-6,1e-5,1e-4]:
        formula_opposite=(t*s<0)&(np.abs(t)>tau)&(np.abs(s)>tau)
        actual_opposite=(actual_t*actual_s<0)&(np.abs(actual_t)>tau)&(np.abs(actual_s)>tau)
        sensitivity.append({'minimum_magnitude':tau,
            'analytic_opposite_fraction_all_coordinates':float(formula_opposite.mean()),
            'actual_opposite_fraction_all_coordinates':float(actual_opposite.mean()),
            'both_analytic_and_actual_opposite_fraction_all_coordinates':float((formula_opposite&actual_opposite).mean())})
    rows.append({'dataset':key[0],'seed':key[1],'source_array_sha256':entry['sha256'],
        'original_metric_max_abs_recalculation_error':max_error,
        'actual_tied_formula_max_abs_error':err_t,
        'actual_sync_formula_max_abs_error':err_s,
        **metrics,'posthoc_magnitude_sensitivity':sensitivity})
assert len(rows)==12
out={'status':'PASS','scope':'all12 frozen train-label-only initializations; NumPy recomputation from postfreeze raw exports',
    'sensitivity_status':'thresholds selected post hoc to distinguish tiny cancellation residuals from larger opposite updates',
    'original_freeze_sha256':sha(frozen),'original_result_sha256':sha(result),
    'raw_export_manifest_sha256':sha(raw/'manifest.json'),'auditor_sha256':sha(Path(__file__)),
    'rows':rows}
a.output.write_text(json.dumps(out,indent=2)+'\n')
for tau_index,tau in enumerate([0,1e-7,1e-6,1e-5,1e-4]):
    values=[100*r['posthoc_magnitude_sensitivity'][tau_index]['actual_opposite_fraction_all_coordinates'] for r in rows]
    print(f'Actual opposite updates with both magnitudes > {tau:g}: {min(values):.3f}% to {max(values):.3f}%')
print('PASS: all12 array hashes, original metrics, and analytic-versus-actual update comparisons.')
