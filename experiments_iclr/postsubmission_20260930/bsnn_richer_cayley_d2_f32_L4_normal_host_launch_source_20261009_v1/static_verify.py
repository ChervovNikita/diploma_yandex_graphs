"""Source/AST/hash only; no numerical, launcher, owner, provider or data import."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
STDLIB = {'argparse','ast','datetime','hashlib','importlib','json','math','os','pathlib','resource','signal','socket','subprocess','time'}


def read(path): return json.loads(Path(path).read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(tree,name): return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def same(a,b): return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def check(ok,name):
    if not ok: raise AssertionError(name)
    return name


def verify():
    checks=[]
    pins=read(HERE/'SOURCE_BINDINGS.json')
    for p in HERE.glob('*.py'):
        tree=ast.parse(p.read_text(),filename=str(p))
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
                check(all(v.split('.')[0] in STDLIB for v in names),'Stdlib-only source entry:'+p.name)
        checks.append('AST parse and stdlib-only top-level imports:'+p.name)
    original=ast.parse((HERE/'original_qualification_SUPERVISOR.py.txt').read_text())
    own=ast.parse((HERE/'SUPERVISOR.py').read_text())
    for name in ('identity','same','query','write'):
        checks.append(check(same(function(own,name),function(original,name)),'Exact original custody helper:'+name))
    main,prior=function(own,'main'),function(original,'main')
    loops=[n for n in ast.walk(main) if isinstance(n,ast.While)]
    priorloops=[n for n in ast.walk(prior) if isinstance(n,ast.While)]
    checks.append(check(len(loops)==len(priorloops)==1 and same(loops[0],priorloops[0]),'Exact original own-worker monitoring loop'))
    own_cleanup=next(n for n in ast.walk(main) if isinstance(n,ast.ExceptHandler) and isinstance(n.type,ast.Name) and n.type.id=='BaseException')
    prior_cleanup=next(n for n in ast.walk(prior) if isinstance(n,ast.ExceptHandler) and isinstance(n.type,ast.Name) and n.type.id=='BaseException')
    checks.append(check(same(own_cleanup,prior_cleanup),'Exact original failure/identity-bound group signaling and reap'))
    for name in ('launch.py','SUPERVISOR.py'):
        text=(HERE/name).read_text()
        checks.append(check("if not args.execute:" in text and "return" in text,'Default inactive:'+name))
    for row in pins['files']:
        p=(PHASE/row['path']).resolve(strict=True)
        check(p.is_relative_to(PHASE) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Pinned source:'+row['path'])
    checks.append('Every bound old activation source/qualifier/richer source byte unchanged')
    for key in ('qualifier','richer_wrapper'):
        root=PHASE/pins[key]['directory']
        check(sha(root/'MANIFEST.json')==pins[key]['manifest_sha256'],'Target manifest:'+key)
        check(sha(root/'SEAL.json')==pins[key]['seal_file_sha256'],'Target seal:'+key)
        for row in read(root/'MANIFEST.json')['files']:
            p=(root/row['path']).resolve(strict=True)
            check(p.is_relative_to(root) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Target source payload:'+row['path'])
        checks.append('All sealed target payloads unchanged:'+key)
    for name in ('ROOT_RELEASE_TEMPLATE_DISABLED.json','ROOT_OWNER_RECEIPT_TEMPLATE_DISABLED.json','QUALIFICATION_ADOPTION_TEMPLATE_DISABLED.json'):
        value=read(HERE/name)
        check(value.get('enabled',value.get('released',value.get('adopted_by_root'))) is False,'Disabled:'+name)
    checks.append('Owner/resource/adoption release templates inactive')
    if (HERE/'MANIFEST.json').exists():
        for row in read(HERE/'MANIFEST.json')['files']:
            p=(HERE/row['path']).resolve(strict=True)
            check(p.is_relative_to(HERE) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Own manifest:'+row['path'])
        seal=read(HERE/'SEAL.json')
        check(seal['source_only'] is True and seal['execution_enabled'] is False and sha(HERE/'MANIFEST.json')==seal['manifest_sha256'],'Own inactive seal')
        checks.append('Own sealed payload hashes pass')
    return dict(status='passed',checks=checks,source_only=True,numerical_host_model_provider_or_data_import=False,
        activation_release_results_terminal_or_new_GNNM_outcomes_opened=False,execution_enabled=False)


if __name__=='__main__': print(json.dumps(verify(),indent=2)+'\n')
