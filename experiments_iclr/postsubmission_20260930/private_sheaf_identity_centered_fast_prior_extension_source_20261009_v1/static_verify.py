"""AST/source-byte assessment only; no runner/model/numerical/data import."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
IMPORTS={'argparse','ast','bank','gc','hashlib','importlib','json','pathlib','resource','support','sys','time'}


def read(p):return json.loads(p.read_text())


def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def check(value,name):
    if not value:raise AssertionError(name)
    return name


def parsed(p):return ast.parse(p.read_text(),filename=str(p))


def fn(tree,name):return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)


def methods(node,name):return [v for v in ast.walk(node) if isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr==name]


def exact_files(root,rows):
    for row in rows:
        p=(root/row['path']).resolve(strict=True)
        check(p.is_relative_to(root) and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Exact bytes: '+row['path'])


def verify():
    checks=[];pins=read(HERE/'SOURCE_BINDINGS.json');protocol=read(HERE/'PROTOCOL.json')
    for p in sorted(HERE.glob('*.py')):
        tree=parsed(p)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                check(all(name.split('.')[0] in IMPORTS for name in names),'Deferred numerical/model imports: '+p.name)
        checks.append('AST/stdlib-local imports: '+p.name)
    core=PHASE/pins['scientific_core_directory']
    check(sha(core/'SEAL.json')==pins['scientific_core_seal_sha256']=='bc6264234cc003c5e89cc46bedf045e0edfbf097ec35b79a8d194fd0c3d11a22','Original vanilla V2 exact required control')
    new=parsed(HERE/'centered_fit.py');old=parsed(core/'shared_fit.py')
    checks.append(check(ast.dump(fn(new,'evaluate'),include_attributes=False)==ast.dump(fn(old,'evaluate'),include_attributes=False),
        'Full native selected/evaluation/serving path AST exactly V2'))
    update=fn(new,'train_update');loop=next(n for n in update.body if isinstance(n,ast.For))
    back=methods(loop,'backward')
    check(len(back)==1 and isinstance(back[0].func.value,ast.BinOp) and isinstance(back[0].func.value.op,ast.Add)
        and isinstance(back[0].func.value.left,ast.BinOp) and isinstance(back[0].func.value.left.op,ast.Div)
        and back[0].func.value.left.right.value==4 and isinstance(back[0].func.value.right,ast.Name) and back[0].func.value.right.id=='prior',
        'Each backward own NLL/4 plus its member centered prior')
    checks.append(check(not methods(loop,'step') and len(methods(update,'step'))==len(methods(update,'zero_grad'))==1,
        'Four old-parameter streamed backwards then one deduplicated Adam, unchanged finite/version guards'))
    prior=fn(new,'member_prior');returned=prior.body[-1].value
    check(isinstance(returned,ast.BinOp) and isinstance(returned.op,ast.Mult)
        and isinstance(returned.left,ast.BinOp) and isinstance(returned.left.op,ast.Div)
        and isinstance(returned.left.right,ast.BinOp) and returned.left.right.left.value==2 and returned.left.right.right.value==4,
        'Exact lambda/(2M) prior factor with M4')
    check(any(isinstance(n,ast.BinOp) and isinstance(n.op,ast.Sub) and isinstance(n.right,ast.Constant) and n.right.value==1 for n in ast.walk(prior)),
        'Identity-centered squared-distance factor prior')
    groups=fn(new,'parameter_groups')
    check(len([n for n in ast.walk(groups) if isinstance(n,ast.keyword) and n.arg=='weight_decay'])==3
        and any(isinstance(n,ast.keyword) and n.arg=='weight_decay' and isinstance(n.value,ast.Constant) and n.value.value==0.0 for n in ast.walk(groups)),
        'Three groups, unchanged native slow incidence/other decay and fast decay0')
    fit=fn(new,'fit');text=(HERE/'centered_fit.py').read_text()
    check('member_weighted_centered_prior=own_prior' in text and 'total_weighted_centered_prior=sum(own_prior)' in text
        and 'mean_own_TRAIN_nll=sum(own_nll)/4' in text,'Likelihood and explicit prior separately recorded')
    checks.append('No factor/init/native topology or context change; exact native slow source and V2 optimizer/selector/checkpoint/restore/serving structure retained')
    original_try=next(n for n in fn(old,'fit').body if isinstance(n,ast.Try));new_try=next(n for n in fit.body if isinstance(n,ast.Try))
    def restores(body):
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and n.value.value=='fresh_selected_state_reconstruction')
        end=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and n.value.value=='persist_fresh_selected_serving_reference')
        return [ast.dump(n,include_attributes=False) for n in body[start:end]]
    checks.append(check(restores(original_try.body)==restores(new_try.body),'Exact V2 selected model/optimizer/RNG reconstruction, fresh scores and recorded materiality AST retained'))
    run_source=(HERE/'runner.py').read_text();support=(HERE/'support.py').read_text()
    check('independent.run' not in run_source and 'native_fit' not in run_source and 'torch.load' not in support,
        'No additional independent fit/replay or checkpoint tensor read for reused origins')
    checks.append(check('validate_revision(release, pins)' in support and 'validate_origins(release, receipt, config, pins)' in support
        and "scope['declared_before_any_scientific_fit'] is True" in support and "scope['required_logical_records']==21" in support
        and "complete['complete'] is True" in support,'Pre-SCI co-primary revision, original18 completion/origins/cost and all21 closure enforced'))
    check(protocol['seeds']==[7409,8501,9607] and protocol['new_shared_fits']==3 and protocol['required_logical_records']==21
        and protocol['centered_prior_coefficient']==0.0005 and protocol['centered_fast_optimizer_decay']==0.0
        and protocol['context_regularizer']==0 and protocol['configuration_search'] is False,'Exactly one stated centered recipe, three paired fits, no search')
    check(read(HERE/'RELEASE_TEMPLATE_DISABLED.json')['enabled'] is False and read(HERE/'REVISED_SCOPE_TEMPLATE_DISABLED.json')['enabled'] is False
        and read(HERE/'ENGINEERING_OUTLINE.json')['execution_enabled'] is False,'Inactive source/revised scope/engineering outline')
    checks.append('Prior-only displacement probe checks zero at identity, nonzero away, restoring gradient, no slow gradients and exact parameter restoration; unexecuted')
    exact_files(PHASE,pins['files'])
    preserved=[]
    for row in pins['immutable_packets']:
        root=PHASE/row['directory'];check(sha(root/'MANIFEST.json')==row['manifest_sha256'],'Prior manifest: '+row['directory'])
        payloads=read(root/'MANIFEST.json')['files'];exact_files(root,payloads)
        preserved.append(dict(directory=row['directory'],verified_payloads=len(payloads),manifest_sha256=row['manifest_sha256']))
    checks.append('All10 prior sealed packets,146 pinned source/evidence files and existing root pilot/accounting/Rank1 scopes unchanged')
    if (HERE/'MANIFEST.json').exists():
        exact_files(HERE,read(HERE/'MANIFEST.json')['files']);seal=read(HERE/'SEAL.json')
        checks.append(check(seal['source_only'] is True and seal['execution_enabled'] is False and sha(HERE/'MANIFEST.json')==seal['manifest_sha256'],'Own inactive seal verifies'))
    return dict(status='passed',assessment='AST and exact source/metadata hashes only',checks=checks,preserved_packets=preserved,
        source_only=True,numerical_model_or_provider_import=False,numerical_execution=False,array_or_checkpoint_access=False,
        comparative_outcome_access=False,old_sources_reports_pilot_or_pointers_modified=False,server_or_installation_actions=False,
        new_primary_paper_credit=0,scientific_or_engineering_activation=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2)+'\n')
