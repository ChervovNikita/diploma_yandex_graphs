"""Stdlib-only source/seal/scope/template checks. No ML or data execution."""
import ast
import hashlib
import json
from pathlib import Path
from source_ops import contract,scopes,NAMESPACE_KEYS

ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent/'learnable_internal_be_contrastive_multitask_suite_20261007_v5'
EXPECTED='ecff016e9a68af1625293bf0166d1428bca20d7c2ac12b64b37914dda5c3af3b'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha(SUITE/'MANIFEST.json')!=EXPECTED:raise ValueError('Exact v5 seal')
    sealed=json.loads((SUITE/'MANIFEST.json').read_text())
    for row in sealed['files']:
        path=SUITE/row['path']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('V5 source changed')
    parsed=[]
    for path in sorted(ROOT.glob('*.py')):
        ast.parse(path.read_text());parsed.append(path.name)
    expected=json.loads((ROOT/'EXTRACTED_SCOPES.json').read_text())
    if contract(SUITE/'run.py')!=expected['scopes']:raise ValueError('Named source scopes changed')
    if list(NAMESPACE_KEYS)!=expected['explicit_namespace_keys']:raise ValueError('Explicit captures changed')
    selected=scopes(SUITE/'run.py')
    calls=[ast.unparse(node.func) for statement in selected['TRAIN_batch'] for node in ast.walk(statement) if isinstance(node,ast.Call)]
    if calls.count('loss.backward')!=1 or calls.count('opt.step')!=1 or 'alignment_loss' not in calls or 'residual_member_contrast' not in calls:
        raise ValueError('Full source training scope not selected')
    templates=[]
    for path in sorted((ROOT/'jobs').glob('*.json')):
        job=json.loads(path.read_text())
        if job.get('task')!='collab' or job.get('role_schema')!='internal-be-official-role-projection-v3':
            raise ValueError('Collab-only temporal-role template')
        for key in ('root_resource_execution_authorized','qualifier_source_review_approved','resource_scope_adopted'):
            if job.get(key) is not False:raise ValueError('Template enabled')
        for key in ('TEST_access','automatic_retry','fit_authorized','export_authorized'):
            if job.get(key) is not False:raise ValueError('Closed role/task boundary changed')
        config=json.loads((SUITE/'configs'/(job['task']+'.json')).read_text())
        if job['arm'] not in config['arms'] or job['seed'] not in config['pilot_seeds'] or job['config']['sha256']!=sha(SUITE/'configs'/(job['task']+'.json')):
            raise ValueError('Unchanged registered source task/arm/config/seed')
        if job['hard_seconds']!=3600 or job['active_compute_seconds']!=3590 or job['cleanup_grace_seconds']!=10:
            raise ValueError('Disabled template caps')
        templates.append(path.name)
    if len(templates)!=1:raise ValueError('One disabled Collab template')
    report={'schema':'internal-be-resource-qualifier-static-check-v1','suite_manifest_sha256':EXPECTED,
        'python_files_AST_parsed':parsed,'disabled_templates':templates,'exact_source_scopes':expected['scopes'],
        'framework_imports':0,'data_or_label_payloads_opened':0,'GPU_work':False,'remote_access':False,
        'numerical_checks_run':False,'resource_qualification_passed':False,'fit_authorized':False,
        'static_checks_passed':True}
    (ROOT/'STATIC_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'static_checks_passed':True,'framework_imports':0,'resource_qualification_passed':False}))


if __name__=='__main__':main()
