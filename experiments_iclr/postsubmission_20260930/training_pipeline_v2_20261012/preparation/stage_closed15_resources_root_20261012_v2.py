"""Stage source/provider cases with inherited zero-update boundaries."""
import base64
import hashlib
import json
from pathlib import Path
import zlib
from prepare_molecular_packets_allocation import GUARD, invoke

BASE = Path(__file__).resolve().parent
PHASE = BASE.parents[1]
HERE = BASE / 'closed15_resource_stage_root_20261012_v2'
REMOTE = GUARD + r'''
import gzip
rows=json.loads(zlib.decompress(base64.b64decode(sys.argv[1])))
stage=phase/'closed15_resource_stage_root_20261012_v2';assert not stage.exists();stage.mkdir()
source=stage/'source';source.mkdir()
for row in rows:
 raw=base64.b64decode(row['data']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 (source/row['name']).write_bytes(raw)
proposal=json.loads((source/'PROPOSAL.json').read_text());assert len(proposal['requested_cases'])==14
prior=json.loads((phase/'native_wrapper_rotation_resource_stage_20261012_v1/ENGINEERING_ADMISSION.json').read_text())
def receipt(path):
 raw=path.read_bytes();return {'path':str(path.relative_to(phase)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def bound(row):
 path=phase/row['path'];assert path.resolve().is_relative_to(phase);assert receipt(path)==row;return path
runtime=phase/'remaining_fitted_label20_source_batch_20261012_v3/runtime'
sources={};studies={};shapes=dict(prior['node_bank_role_shapes'])
for case in proposal['requested_cases']:
 path=phase/case['study_path'];raw=path.read_bytes();study=json.loads(gzip.decompress(raw))
 assert hashlib.sha256(json.dumps(study,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==case['study_spec_hash']
 run=next(run for run in study['runs'] if run['run_id']==case['source_run_id'])
 assert hashlib.sha256(json.dumps(run,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==case['source_run_hash']
 for row in study['source']['files']:bound(row);sources[row['path']]=row
 studies[case['case_id']]=receipt(path)
 if study['dataset'] not in shapes:
  for row in study['data'].get('files',[]):
   if row['path'].endswith('.json'):
    custody=json.loads(bound(row).read_text())
    if custody.get('schema')=='node-provider-role-custody-v1' and custody.get('dataset_id')==study['dataset']:
     shapes[study['dataset']]={role:max(s['role_counts'][role] for s in custody['splits']) for role in ('cal','evaluate')}
     shapes[study['dataset']]['custody_receipt']=row
     break
for key in ('role_optimizer_cap_helper','tensor_commit_tree_helper'):bound(prior[key])
helper=receipt(source/'qualify_closed15.py')
admission={'schema':'root-approved-zero-update-resource-admission-v1','root_source_review_confirmed':True,
 'helper_sha256':helper['sha256'],'proposal_receipt':receipt(source/'PROPOSAL.json'),
 'scientific_GO':False,'max_actual_optimizer_updates':0,'approved_case_ids':[case['case_id'] for case in proposal['requested_cases']],
 'case_study_receipts':studies,'runtime_package':str(runtime.relative_to(phase)),
 'runtime_dependency_receipts':[receipt(path) for path in sorted(runtime.glob('*.py'))],
 'source_dependency_receipts':list(sources.values()),
 'role_optimizer_cap_helper':prior['role_optimizer_cap_helper'],'tensor_commit_tree_helper':prior['tensor_commit_tree_helper'],
 'node_bank_role_shapes':shapes,'two_process_pairs_approved':False,'no_automatic_retry':True,
 'scope':'Finite full source/provider constructors and complete gradient callbacks with optimizer commits denied; no selector targets or benchmark scores'}
(stage/'ENGINEERING_ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
print(json.dumps({'helper_sha256':helper['sha256'],'admission':receipt(stage/'ENGINEERING_ADMISSION.json'),
 'cases':14,'source_families':sorted({case['family'] for case in proposal['requested_cases']}),'models':0,'CUDA_callbacks':0,'GO':False}))
'''


def main():
    HERE.mkdir()
    original = json.loads((PHASE / 'remaining_fitted_label15_resource_proposal_20261012_v2/PROPOSAL.json').read_text())
    cases = []
    for case in original['cases']:
        run = case['run']
        cfg = run['config']
        cases.append({'case_id': case['id'], 'family': cfg['ported_family'],
            'dataset': Path(case['source_study']['path']).stem,
            'study_path': case['source_study']['path'], 'study_spec_hash': case['source_study']['spec_hash'],
            'source_run_id': case['run_id'],
            'source_run_hash': hashlib.sha256(json.dumps(run, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
            'model_graph_shape': case['model_graph'], 'shape_config': {'classes': cfg['classes'], 'members': cfg['members']},
            'timeout_seconds': 300, 'source_optimizer_constructor_calls': case['source_optimizer_constructor_calls']})
    prior_closure = json.loads((BASE / 'closed15_resource_results_root_20261012_v1/CLOSURE.json').read_text())
    failed = {r['case_id'] for r in prior_closure['attempts'] if r['exit_code'] != 0}
    assert len(failed) == 14 and all(r['failure_log_tail'].splitlines()[-1].endswith('Undeclared project binary input') for r in prior_closure['attempts'] if r['exit_code'] != 0)
    cases = [case for case in cases if case['case_id'] in failed]
    assert {case['case_id'] for case in cases} == failed
    assert len({case['case_id'] for case in cases}) == 14
    # Dataset labels derive from the declared source graph/feature shape.
    for case in cases:
        case['dataset'] = {11701: 'wikics', 7650: 'amazon-photo', 19717: 'pubmed'}[case['model_graph_shape']['nodes']]
    proposal = {'schema': 'closed15-root-finite-zero-update-resource-v1', 'requested_cases': cases,
                'source_proposal': 'remaining_fitted_label15_resource_proposal_20261012_v2/PROPOSAL.json',
                'actual_optimizer_updates': 0, 'scientific_fits': 0, 'GO': False}
    helper = PHASE / 'closed15_generic_resource_source_root_20261012_v2/qualify_closed15.py'
    inputs = {'PROPOSAL.json': (json.dumps(proposal, indent=2) + '\n').encode(), 'qualify_closed15.py': helper.read_bytes()}
    rows = [{'name': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'data': base64.b64encode(raw).decode()}
            for name, raw in inputs.items()]
    (HERE / 'PROPOSAL.json').write_bytes(inputs['PROPOSAL.json'])
    (HERE / 'ROOT_REVIEW.json').write_text(json.dumps({'helper_sha256': rows[1]['sha256'],
        'source_provider_cases': 14, 'callbacks_per_case': 2, 'maximum_actual_optimizer_updates': 0,
        'learning_source_changed': False, 'scientific_GO': False}, indent=2) + '\n')
    result = invoke(REMOTE, [base64.b64encode(zlib.compress(json.dumps(rows).encode(), 9)).decode()], HERE, 'STAGE', 120)
    assert result['exit_code'] == 0, 'Preserve staging failure'
    value = json.loads(result['stdout'])
    (HERE / 'STAGE_RECEIPT.json').write_text(json.dumps(value, indent=2) + '\n')
    print(result['stdout'])


if __name__ == '__main__':
    main()
