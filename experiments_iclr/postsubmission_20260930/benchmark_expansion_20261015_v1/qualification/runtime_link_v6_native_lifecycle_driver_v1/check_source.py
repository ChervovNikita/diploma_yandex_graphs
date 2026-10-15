"""Source-only audit: frozen inputs, exact finite invocation schedule and custody gates."""
import ast
import hashlib
import json
from pathlib import Path
import support

HERE=Path(__file__).resolve().parent
EXPANSION=HERE.parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=[]
    for folder,expected in [('runtime_link_v6',support.RUNTIME_SHA),
        ('qualification/runtime_link_v6_native_lifecycle_v1',support.PLAN_SHA),
        ('interaction_link_ncnc_v1',support.METHOD_SHA)]:
        root=EXPANSION/folder
        assert digest(root/'MANIFEST.json')==expected
        for row in json.loads((root/'MANIFEST.json').read_text())['files']:
            path=(root/row['path']).resolve();assert path.is_relative_to(root.resolve())
            assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
        checks.append({'name':'frozen-package-'+folder,'passed':True})
    trees={p.name:ast.parse(p.read_text()) for p in HERE.glob('*.py')}
    checks.append({'name':'all-driver-sources-parse','passed':True})
    plan=json.loads((EXPANSION/'qualification/runtime_link_v6_native_lifecycle_v1/PLAN.json').read_text())
    expected={'native_constructions_attempted':plan['maximum_native_constructions'],
        'TRAIN_epochs_started':plan['maximum_TRAIN_epochs'],'updates_attempted':plan['maximum_optimizer_attempts'],
        'optimizer_attempts':plan['maximum_optimizer_attempts'],'reverse_attempts':plan['maximum_native_reverse_collections'],
        'VALID_attempts':plan['maximum_whole_VALID_calls'],'mean_exports_attempted':plan['maximum_selected_exports']}
    assert support.LIMITS==expected
    checks.append({'name':'all-hard-work-limits-match-frozen-plan','passed':True})
    calls=[n for n in ast.walk(trees['qualify.py']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='invoke']
    schedule=[(ast.literal_eval(n.args[0]),ast.literal_eval(n.args[1]),[kw.arg for kw in n.keywords]) for n in calls]
    assert schedule==[('controlled_completed_epoch1',17,['interrupt']),('resume_remaining_epoch2',17,['resume']),('stopped_finalization_resume',0,['resume'])]
    checks.append({'name':'exact17-plus17-plus0-invocation-schedule','passed':True})
    for variant in ('CPU','CUDA'):
        parent=json.loads((EXPANSION/f'qualification/runtime_link_v5_native_lifecycle_v1/JOB_{variant}.json').read_text())
        child=json.loads((EXPANSION/f'qualification/runtime_link_v6_native_lifecycle_v1/JOB_{variant}.json').read_text())
        parent['id']=parent['id'].replace('runtime-link-v5','runtime-link-v6')
        parent['link_policy_module']=parent['link_policy_module'].replace('runtime_link_v5','runtime_link_v6')
        assert child==parent
    checks.append({'name':'successor-JOBs-only-runtime-path-and-own-identity-change','passed':True})
    parent=json.loads((EXPANSION/'qualification/runtime_link_v5_native_lifecycle_v1/PLAN.json').read_text())
    for key,value in parent.items():
        if key in ('schema','runtime_manifest_sha256'):continue
        assert plan[key]==value,(key,plan[key],value)
    checks.append({'name':'all-parent-native-work-device-budgets-and-floating-policy-identical','passed':True})
    for filename in ('qualify.py','supervise.py'):
        main=next(n for n in trees[filename].body if isinstance(n,ast.FunctionDef) and n.name=='main')
        assert ast.unparse(main.body[0])=='guard()'
    checks.append({'name':'both-entrypoints-guard-before-project-read-or-tensor-import','passed':True})
    assert "claim.open('x')" in (HERE/'supervise.py').read_text()
    assert "os.killpg(child.pid,signal.SIGTERM)" in (HERE/'supervise.py').read_text()
    assert "os.killpg(child.pid,signal.SIGKILL)" in (HERE/'supervise.py').read_text()
    checks.append({'name':'exclusive-one-attempt-claim-and-TERM-KILL-watchdog-present','passed':True})
    return {'schema':'runtime-link-v6-native-driver-source-check-v1','status':'passed','checks_count':len(checks),
        'checks':checks,'runtime_manifest_sha256':support.RUNTIME_SHA,'plan_manifest_sha256':support.PLAN_SHA,
        'source_only':True,'native_models':0,'native_forward_calls':0,'native_optimizer_updates':0,'TEST_accessed':False}


if __name__=='__main__':print(json.dumps(main(),indent=2))
