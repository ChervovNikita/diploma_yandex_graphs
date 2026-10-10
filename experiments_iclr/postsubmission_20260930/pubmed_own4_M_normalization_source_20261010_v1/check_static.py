"""Stdlib AST/hash/disabled-metadata checks only; never import scientific source."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;PHASE=HERE.parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    trees={f.name:ast.parse(f.read_text()) for f in HERE.glob('*.py')}
    for tree in trees.values():
        for n in tree.body:
            if isinstance(n,ast.Import):assert not any(x.name in ('torch','numpy','scipy') for x in n.names)
            if isinstance(n,ast.ImportFrom):assert (n.module or '').split('.')[0] not in ('torch','numpy','scipy')
    adapter=(HERE/'adapter.py').read_text();tree=trees['adapter.py']
    assert ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENABLED' for t in n.targets))) is False
    assert not any(isinstance(n,ast.FunctionDef) and n.name=='train_step' for n in ast.walk(tree))
    changed=[n.target.slice.value for n in ast.walk(tree) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Subscript) and isinstance(n.target.value,ast.Name) and n.target.value.id=='group']
    assert sorted(changed)==['eps','weight_decay']
    roster=json.loads((HERE/'ROSTER.json').read_text())
    assert [r['seed'] for r in roster]==[9101,9203,9307] and all(r['condition']=='shared4_own_M_normalized' and r['M']==4 and r['private_decay_divisor']==4 and r['private_epsilon_divisor']==4 and r['max_epochs']==2000 and r['patience']==250 for r in roster)
    for f in (HERE/'releases_disabled').glob('*.json'):
        r=json.loads(f.read_text());assert r['enabled'] is False and r['TEST_access'] is False and r['automatic_retry'] is False
    for f in HERE.glob('*OWNER_PLAN_TEMPLATE_DISABLED.json'):
        v=json.loads(f.read_text());assert v['enabled'] is False and v['_completed']==[] and v['owner_sha256']==sha(HERE/'queue.py')
        for r in v['records']:assert r['entrypoint']['sha256']==sha(HERE/'run.py')
    for r in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        p=(PHASE/r['path']).resolve(strict=True);assert p.is_relative_to(PHASE) and sha(p)==r['sha256'] and p.stat().st_size==r['bytes']
    if (HERE/'SOURCE_MANIFEST.json').exists():
        for r in json.loads((HERE/'SOURCE_MANIFEST.json').read_text())['files']:
            assert sha(HERE/r['path'])==r['sha256'] and (HERE/r['path']).stat().st_size==r['bytes']
    print(json.dumps(dict(passed=True,parsed=sorted(trees),only_private_eps_and_decay_assignments=True,native_train_step_not_overridden=True,
        fixed3=True,templates_disabled=True,immutable_sources_hash_verified=True,numerical_imports=False,numerical_execution=False,
        data_model_checkpoint_prediction_reads=False,remote_jobs=False)))


if __name__=='__main__':main()
