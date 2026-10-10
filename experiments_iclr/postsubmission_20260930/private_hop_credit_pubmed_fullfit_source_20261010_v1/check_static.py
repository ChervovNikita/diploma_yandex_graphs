"""AST/hash/metadata checks only. Never import fit, method or numerical modules."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
CONDITIONS=('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview')
SEEDS=(9101,9203,9307)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def epoch_loop(path,function):
    tree=ast.parse(path.read_text())
    body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
    return ast.dump(next(n for n in ast.walk(body) if isinstance(n,ast.For) and
                         isinstance(n.target,ast.Name) and n.target.id=='epoch'),include_attributes=False)


def main():
    parsed=[]
    for path in sorted(HERE.glob('*.py')):
        tree=ast.parse(path.read_text(),filename=str(path));parsed.append(path.name)
        for node in tree.body:
            if isinstance(node,ast.Import):
                assert not any(n.name.split('.')[0] in ('torch','numpy','scipy','torch_geometric') for n in node.names)
            if isinstance(node,ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in ('torch','numpy','scipy','torch_geometric')
    assert epoch_loop(HERE/'m4_fit.py','fit')==epoch_loop(PHASE/'combination_masked_context_pubmed_stage1_source_20261010_v1/train.py','fit')
    assert epoch_loop(HERE/'m1_fit.py','fit_body')==epoch_loop(PHASE/'combination_pubmed_strong_reference_source_20261010_v2/train.py','fit_body')
    roster=json.loads((HERE/'ROSTER.json').read_text())
    expected=[f'seed{s}__{c}' for s in SEEDS for c in CONDITIONS]
    assert [r['record_id'] for r in roster]==expected and len(roster)==18
    for row in roster:
        assert row['max_epochs']==2000 and row['patience']==250 and row['lambda_fixed']==.5 and row['TEST_access'] is False
    release_paths=list(HERE.glob('*RELEASE*DISABLED.json'))+list((HERE/'releases_disabled').glob('*.json'))
    for path in release_paths:
        value=json.loads(path.read_text());assert value['enabled'] is False
        assert value.get('TEST_access',False) is False and value['automatic_retry'] is False
    plan=json.loads((HERE/'OWNER_PLAN_TEMPLATE_DISABLED.json').read_text())
    assert plan['enabled'] is False and plan['_completed']==[] and [r['record_id'] for r in plan['records']]==expected
    assert plan['owner_sha256']==sha(HERE/'queue.py')
    assert plan['reused_owner_source_sha256']==sha(PHASE/'pubmed_factor1_controls_source_20261010_v1/queue.py')
    for row in plan['records']:
        assert row['entrypoint']['sha256']==sha(HERE/'run.py') and row['entry_args']==['--mode','science']
    bindings=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        file=(PHASE/row['path']).resolve(strict=True)
        assert file.is_relative_to(PHASE) and sha(file)==row['sha256'] and file.stat().st_size==row['bytes']
    tree=ast.parse((PHASE/'private_hop_credit_source_20261010_v1/adapter.py').read_text())
    enabled=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENABLED' for t in n.targets))
    assert ast.literal_eval(enabled.value) is False
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    assert protocol['conditions']==list(CONDITIONS) and protocol['seeds']==list(SEEDS)
    assert protocol['enabled'] is False and protocol['science_enabled'] is False and protocol['TEST_access'] is False
    if (HERE/'SOURCE_MANIFEST.json').exists():
        for row in json.loads((HERE/'SOURCE_MANIFEST.json').read_text())['files']:
            file=(HERE/row['path']).resolve(strict=True)
            assert file.is_relative_to(HERE) and sha(file)==row['sha256'] and file.stat().st_size==row['bytes']
    result=dict(passed=True,parsed=parsed,exact_M4_epoch_loop=True,exact_M1_epoch_loop=True,
                fixed18=True,all_release_templates_disabled=True,hop_disk_disabled=True,
                bound_source_files=len(bindings['files']),numerical_imports=False,numerical_execution=False,
                data_model_checkpoint_reads=False,remote_commands_or_jobs=False)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
