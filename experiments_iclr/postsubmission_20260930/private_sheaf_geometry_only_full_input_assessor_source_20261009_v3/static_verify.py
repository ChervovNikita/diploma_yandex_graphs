"""Stdlib AST and source/metadata hashes only, never import the assessor/core."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
IMPORTS={'argparse','ast','gc','hashlib','importlib','json','pathlib','reference','resource','support','sys','time'}


def read(path): return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
    return digest.hexdigest()


def check(value,name):
    if not value: raise AssertionError(name)
    return name


def parsed(path): return ast.parse(path.read_text(),filename=str(path))


def function(tree,name): return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)


def method_calls(node,name):
    return [value for value in ast.walk(node) if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr==name]


def exact_files(root,rows):
    for row in rows:
        path=(root/row['path']).resolve(strict=True)
        check(path.is_relative_to(root) and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Exact payload: '+row['path'])


def verify():
    checks=[]; pins=read(HERE/'SOURCE_BINDINGS.json'); protocol=read(HERE/'PROTOCOL.json')
    for path in sorted(HERE.glob('*.py')):
        tree=parsed(path)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module]
                check(all(name.split('.')[0] in IMPORTS for name in names),'Stdlib/local imports only: '+path.name)
        checks.append('AST parses with deferred numerical/model imports: '+path.name)
    core=PHASE/pins['scientific_core_directory']
    checks.append(check(sha(core/'SEAL.json')==pins['scientific_core_seal_sha256']=='bc6264234cc003c5e89cc46bedf045e0edfbf097ec35b79a8d194fd0c3d11a22',
                        'Exact unchanged V2 scientific seal and algorithm/helper source'))
    exact_files(PHASE,pins['files'])
    reference=parsed(HERE/'reference.py'); accumulator=function(reference,'accumulate')
    member_loop=next(node for node in ast.walk(accumulator) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='member')
    backward=method_calls(member_loop,'backward')
    checks.append(check(len(backward)==1 and isinstance(backward[0].func.value,ast.BinOp) and isinstance(backward[0].func.value.op,ast.Div)
        and isinstance(backward[0].func.value.right,ast.Constant) and backward[0].func.value.right.value==4
        and not method_calls(member_loop,'step'),'Each sequential complete M1 uses own NLL/4 backward with no M1 Adam/decay'))
    calls=method_calls(reference,'SharedBELinear')
    check(len(calls)==1 and len(method_calls(reference,'_replace'))==1 and not method_calls(reference,'deepcopy'),
          'M1 fresh original factory and same incidence-only factor wrapper; no bank/tape cloning')
    text=(HERE/'reference.py').read_text()
    checks.append(check("'members.'+str(member)+'.' if name in private_names else 'members.0.'" in text
        and 'accumulator[target].add_(gradient)' in text and 'counts[target] += 1' in text
        and 'del value,gradient' in text and 'model = None' in text and 'released_before_next_M1=True' in text,
        'CPU common slow/four contributions and matching private/one contribution mapping; GPU body/cache/tape released before next'))
    update=function(reference,'apply_accumulated_update')
    check(len(method_calls(update,'step'))==1 and not method_calls(update,'backward') and not method_calls(update,'train'),
          'Tape-free same V2 bank Adam once on accumulated gradients')
    assessor=parsed(HERE/'assessor.py'); run=function(assessor,'run')
    calls=method_calls(run,'train_update')
    checks.append(check(len(calls)==1 and isinstance(calls[0].func.value,ast.Name) and calls[0].func.value.id=='fit'
        and len(method_calls(run,'optimizer'))==3 and not method_calls(run,'joint_update'),
        'Actual sealed V2 train_update and exact V2 optimizer reused; failed joint method absent'))
    source=(HERE/'assessor.py').read_text()
    checks.append(check('register_forward_pre_hook' in source and 'handle.remove()' in source
        and 'forward_rng[index]=helpers.capture_rng' in source and 'backward_end_rng[index]=helpers.capture_rng' in source,
        'Read-only actual member forward-start/backward-end RNG captured; hooks removed before serving'))
    check('roc_auc_score' not in source and 'metrics(' not in source and 'evaluate(' not in source
        and "del arrays['valid_y'],arrays['valid_index']" in source,'No scientific/VALID scorer or metric, VALID arrays dropped')
    check(all(token in source for token in ('model_parameter_buffer_restore_exact=True','optimizer_restore_exact=True','training_RNG_restore_exact=True',
        'q.cache_ownership','q.verify_topology','q.equivalent_topology','q.structural','q.label_free_serving','q.serving_materiality')),
        'All actual V2 ownership/cache/topology/exact reconstruction and label-free serving checks retained')
    checks.append(check('source_only_preparation=False' in source and "source_seal_sha256=pins['scientific_core_seal_sha256']" in source
        and 'assessor_source_seal_sha256=sha' in source and 'reference_source_sha256=sha' in source,
        'Activated record is numerical; qualification targets V2 scientific seal and separately hashes V3 assessor/reference'))
    startup=next(node for node in run.body if isinstance(node,ast.Try))
    init_index=next(i for i,node in enumerate(startup.body) if method_calls(node,'init'))
    reset_index=next(i for i,node in enumerate(startup.body) if method_calls(node,'reset_peak_memory_stats'))
    checks.append(check(init_index<reset_index,'Explicit CUDA initialization/device before peak reset'))
    release=read(HERE/'RELEASE_TEMPLATE_DISABLED.json')
    checks.append(check(release['enabled'] is False and release['new_post_failure_resource_plan'] is False
        and release['source_seal_sha256']==pins['scientific_core_seal_sha256'] and release['reference_source_sha256']==sha(HERE/'reference.py'),
        'Separate V3 release inactive, requiring a prospective post-failure resource plan'))
    check(protocol['expected_TRAIN_forwards']==protocol['expected_backwards']==protocol['expected_label_free_serving_forwards']==8
        and protocol['expected_Adam_steps']==2 and protocol['expected_original_constructors']==7
        and protocol['reference_bodies_live_at_once']==protocol['autograd_graphs_live_at_once']==1
        and protocol['failed_joint_method_repeated'] is False and protocol['automatic_retry'] is False,
        'Bounded work accounting and no joint-method repetition/retry')
    prior=read(PHASE/pins['prior_failure']['path']); terminal=prior['metadata']['TERMINAL.json']
    check(sha(PHASE/pins['prior_failure']['path'])==pins['prior_failure']['sha256'] and terminal['complete'] is False
        and terminal['exit_code']==-15 and terminal['partial_files_and_costs_retained'] is True,
        'Exact failed V2 resource attempt/cost/proof remains preserved')
    preserved=[]
    for row in pins['immutable_packets']:
        root=PHASE/row['directory']
        check(sha(root/'MANIFEST.json')==row['manifest_sha256'],'Exact immutable manifest: '+row['directory'])
        payloads=read(root/'MANIFEST.json')['files']; exact_files(root,payloads)
        preserved.append(dict(directory=row['directory'],manifest_sha256=row['manifest_sha256'],verified_payloads=len(payloads)))
    checks.append('All9 preceding sealed packets unchanged, including V2 scientific core;130 pinned source/evidence files verify')
    if (HERE/'MANIFEST.json').exists():
        exact_files(HERE,read(HERE/'MANIFEST.json')['files'])
        seal=read(HERE/'SEAL.json')
        checks.append(check(seal['source_only'] is True and seal['execution_enabled'] is False and seal['manifest_sha256']==sha(HERE/'MANIFEST.json'),
                            'Own inactive assessor manifest/seal verifies'))
    return dict(status='passed',verification='stdlib AST and exact source/metadata byte hashes only',checks=checks,preserved_packets=preserved,
        assessor_source_only=True,V2_scientific_source_changed=False,failed_V2_attempt_rewritten=False,model_or_numeric_provider_import=False,
        numerical_execution=False,data_array_or_checkpoint_access=False,scientific_metric_access=False,server_or_installation_actions=False,execution_enabled=False)


if __name__=='__main__': print(json.dumps(verify(),indent=2)+'\n')
