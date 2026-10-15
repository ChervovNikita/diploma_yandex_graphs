"""Summarize completed selected-VALID reports without loading tensors/models."""
import hashlib
import json
from pathlib import Path
import statistics


here = Path(__file__).resolve().parent
previous = here.parent / 'complete_route_diagnostic_v1/RESULT.json'
parent = json.loads(previous.read_text())
rows, receipts = [], []
for path in sorted((here / 'fetched').glob('fit_*/RESULT.json')):
    value = json.loads(path.read_text())
    assert value['status'] == 'complete' and value['TEST_accessed'] is False
    assert value['actual_work']['optimizer_steps'] == 0
    assert value['actual_work']['native_route_forwards_completed'] == 20
    assert value['sanity_against_saved_export']['passed']
    metrics = value['metrics']
    mean_member = statistics.mean(x['rocauc'] for x in metrics['members'])
    controls = {name: next(x for x in parent['rows']
                           if x['seed'] == value['seed'] and x['method'] == name)
                for name in ('I4', 'gnnm_boundary_native')}
    rows.append({
        'seed': value['seed'], 'selected_epoch': value['selected_epoch'],
        'mean_member_AUC_percent': 100 * mean_member,
        'pooled_AUC_percent': 100 * metrics['pooled']['rocauc'],
        'pool_minus_mean_member_AUC_pp': 100 * (metrics['pooled']['rocauc'] - mean_member),
        'mean_member_BCE': statistics.mean(x['BCE'] for x in metrics['members']),
        'pooled_BCE': metrics['pooled']['BCE'],
        'mean_route_CKA': statistics.mean(x['centered_linear_CKA']
                                         for x in value['hidden_geometry']['pairwise']),
        'pooled_wrong_graphs_at_logit_zero': value['error_overlap']['pooled_wrong_graphs'],
        'all_members_wrong_graphs_at_logit_zero': value['error_overlap']['all_members_wrong_graphs'],
        'member_AUC_delta_pp_vs_base': 100 * (mean_member - controls['gnnm_boundary_native']['mean_member_AUC']),
        'member_AUC_delta_pp_vs_I4': 100 * (mean_member - controls['I4']['mean_member_AUC']),
        'pool_AUC_delta_pp_vs_base': 100 * (metrics['pooled']['rocauc'] - controls['gnnm_boundary_native']['pool_AUC']),
        'pool_AUC_delta_pp_vs_I4': 100 * (metrics['pooled']['rocauc'] - controls['I4']['pool_AUC']),
        'controls': controls,
    })
    receipts.append({'path': str(path.relative_to(here)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert sorted(row['seed'] for row in rows) == [101, 202, 303]
result = {
    'schema': 'BN-selected-route-descriptive-summary-v1', 'rows': rows,
    'mean': {key: statistics.mean(row[key] for row in rows)
             for key in rows[0] if key not in {'seed', 'selected_epoch', 'controls'}},
    'parent_receipt': {'path': str(previous.relative_to(here.parent.parent)),
                       'sha256': hashlib.sha256(previous.read_bytes()).hexdigest()},
    'inputs': receipts, 'summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'independent_training_replicates': 3, 'TEST_accessed': False,
    'scope': 'selected VALID diagnostics, different selected epochs, no causal or confirmation claim',
}
(here / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'mean': result['mean'], 'per_seed': [
    {k: row[k] for k in ('seed', 'member_AUC_delta_pp_vs_base', 'member_AUC_delta_pp_vs_I4',
                        'mean_member_AUC_percent', 'pooled_AUC_percent')}
    for row in rows]}))
