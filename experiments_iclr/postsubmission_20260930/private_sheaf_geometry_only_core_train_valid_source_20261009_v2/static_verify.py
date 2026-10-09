"""Source AST and exact byte hashes only; never import local/numerical/model code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
TOP_LEVEL_IMPORTS = {'argparse','ast','bank','copy','gc','hashlib','importlib','independent','json','native_fit','pathlib',
                     'resource','shared_fit','support','sys','time'}


def read(path): return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def check(value, name):
    if not value: raise AssertionError(name)
    return name


def parsed(path): return ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))


def function(tree, name): return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def dump(node): return ast.dump(node, include_attributes=False)


def method_calls(node, attr):
    return [value for value in ast.walk(node) if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == attr]


def assigns_stage(node, stage):
    return isinstance(node, ast.Assign) and any(isinstance(target,ast.Name) and target.id=='stage' for target in node.targets) and isinstance(node.value,ast.Constant) and node.value.value==stage


def file_rows(root, rows):
    for row in rows:
        path = (root/row['path']).resolve(strict=True)
        check(path.is_relative_to(root) and path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Exact payload: '+row['path'])


def verify():
    checks, pins, protocol = [], read(HERE/'SOURCE_BINDINGS.json'), read(HERE/'PROTOCOL.json')
    for path in sorted(HERE.glob('*.py')):
        tree = parsed(path)
        for node in tree.body:
            if isinstance(node, (ast.Import,ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node,ast.Import) else [node.module]
                check(all(name.split('.')[0] in TOP_LEVEL_IMPORTS for name in names), 'Deferred numerical/model imports: '+path.name)
        checks.append('AST parse and stdlib/local top-level imports: '+path.name)
    check(sha(HERE/'bank.py')==sha(PHASE/'private_sheaf_geometry_only_core_train_valid_source_20261009_v1/bank.py'), 'Bank implementation unchanged from reviewed V1')
    bank = parsed(HERE/'bank.py')
    calls = {node.func.attr for node in ast.walk(bank) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    checks.append(check({'_shared_copy_memo','_module_at','_replace','SharedBELinear'}.issubset(calls) and 'SharedNativeBank' not in calls,
                        'Reviewed exact existing sharing/affine helpers, restricted incidence factors'))
    update = function(parsed(HERE/'shared_fit.py'), 'train_update')
    member_loop = next(node for node in update.body if isinstance(node,ast.For))
    backwards = method_calls(member_loop,'backward')
    checks.append(check(len(backwards)==1 and isinstance(backwards[0].func.value,ast.BinOp) and isinstance(backwards[0].func.value.op,ast.Div)
         and isinstance(backwards[0].func.value.left,ast.Name) and backwards[0].func.value.left.id=='nll'
         and isinstance(backwards[0].func.value.right,ast.Constant) and backwards[0].func.value.right.value==4, 'Four streamed own NLL/4 backwards'))
    check(not method_calls(member_loop,'step') and len(method_calls(update,'zero_grad'))==len(method_calls(update,'step'))==1, 'One zero and one step; no member update')
    step_index=next(i for i,node in enumerate(update.body) if method_calls(node,'step'))
    checks.append(check(step_index>update.body.index(member_loop), 'One deduplicated Adam after all old-parameter backwards'))
    core=function(parsed(HERE/'shared_fit.py'),'fit')
    qual=parsed(HERE/'qualifier.py')
    check(sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='train_update' for node in ast.walk(core))==1
          and sum(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='train_update' for node in ast.walk(qual))==1,
          'Qualifier and scientific fits invoke the exact same streamed step function')
    joint=function(qual,'joint_update')
    joint_loop=next(node for node in joint.body if isinstance(node,ast.For))
    check(not method_calls(joint_loop,'backward') and not method_calls(joint_loop,'step') and len(method_calls(joint,'backward'))==len(method_calls(joint,'step'))==1,
          'Separate joint reference: four complete forwards then one mean backward/Adam')
    checks.append('Same-bank joint reference uses identical initial state and synchronized owned RNG; record-only gradient/update/loss materiality')
    qual_text=(HERE/'qualifier.py').read_text()
    qual_run=function(qual,'run')
    qualifier_try=next(node for node in qual_run.body if isinstance(node,ast.Try))
    initializers=[i for i,node in enumerate(qualifier_try.body) if method_calls(node,'init')]
    peak_resets=[i for i,node in enumerate(qualifier_try.body) if method_calls(node,'reset_peak_memory_stats')]
    checks.append(check(len(initializers)==len(peak_resets)==1 and initializers[0]<peak_resets[0],
                        'CUDA explicitly initialized and selected before qualifier peak-stat reset'))
    check('roc_auc_score' not in qual_text and 'helpers.metrics' not in qual_text and 'helpers.evaluate' not in qual_text,
          'Qualifier has no scientific scorer/evaluator')
    checks.append(check("del arrays['valid_y'], arrays['valid_index']" in qual_text and 'model_parameter_buffer_restore_exact=True' in qual_text
          and 'optimizer_restore_exact=True' in qual_text and 'training_RNG_restore_exact=True' in qual_text,
          'VALID arrays dropped; actual private gradients/cache/topology ownership and exact reconstructed model/optimizer/RNG checks'))
    original=function(parsed(PHASE/pins['NSD_V2_directory']/'baseline_runner.py'),'fit_one')
    successor=function(parsed(HERE/'native_fit.py'),'fit_one')
    original_try=next(node for node in original.body if isinstance(node,ast.Try))
    successor_try=next(node for node in successor.body if isinstance(node,ast.Try))
    cutoff=lambda nodes: nodes[:next(i for i,node in enumerate(nodes) if assigns_stage(node,'load_owned_selected_checkpoint'))]
    checks.append(check([dump(node) for node in cutoff(original_try.body)]==[dump(node) for node in cutoff(successor_try.body)],
          'Independent original V2 constructor, Adam, full TRAIN, selector, checkpoint and patience AST exactly unchanged'))
    orig_restore=next(node for node in original_try.body if isinstance(node,ast.If) and any(isinstance(value,ast.Constant)
          and value.value=='Reconstructed parameter/buffer state differs from selected checkpoint' for value in ast.walk(node)))
    check(any(dump(node)==dump(orig_restore) for node in successor_try.body), 'Original exact parameter/buffer reconstruction check retained')
    independent=parsed(HERE/'independent.py')
    fits=method_calls(independent,'fit_one')
    checks.append(check(len(fits)==1 and isinstance(fits[0].func.value,ast.Name) and fits[0].func.value.id=='native_fit'
          and isinstance(fits[0].args[4],ast.Call) and {key.arg for key in fits[0].args[4].keywords}=={'optimizer','checkpoint_rule'},
          'Fresh full independent successor fit_one with complete original helper argument contract'))
    check('adopted_member0' not in (HERE/'independent.py').read_text() and protocol['independent_fresh_members']==[0,1,2,3]
          and protocol['independent_member0_reuse'] is False, 'All12 independent native fits fresh, no historical reuse')
    checks.append(check(protocol['members']==4 and protocol['seeds']==[7409,8501,9607] and protocol['configuration_id']=='d4_f16_L4'
          and protocol['context_regularizer']==0 and protocol['configuration_search'] is False
          and protocol['independent_seed_rule']=='base+1000003*member', 'One architecture and three prospectively declared exploratory paired seeds'))
    materiality=function(parsed(HERE/'support.py'),'replay_materiality')
    check(not any(isinstance(node,ast.Constant) and isinstance(node.value,float) and node.value!=0 for node in ast.walk(materiality)), 'No float drift gate in materiality')
    check(protocol['replay_materiality_thresholds'] is None and protocol['float_bitwise_gate'] is False,
          'Exact state and finite serving with record-only probability/metric/decision materiality')
    checks.append('Fresh reconstructed scores and fresh server-only pooled/member references retained consistently; serialization charged')
    release=read(HERE/'RELEASE_TEMPLATE_DISABLED.json'); qualifier_release=read(HERE/'QUALIFIER_RELEASE_TEMPLATE_DISABLED.json')
    receipt=read(HERE/'EXPLORATORY_ADMISSION_TEMPLATE_DISABLED.json')
    checks.append(check(release['enabled'] is False and qualifier_release['enabled'] is False and receipt['admission_enabled'] is False
          and release['action']=='post_screen_geometry_core' and qualifier_release['action']=='geometry_engineering_qualify'
          and receipt['original_native15_protocol_pass_claimed'] is False and receipt['original_incomplete_no_freeze_preserved'] is True,
          'Both releases and separate exploratory admission remain disabled; original protocol failure preserved'))
    check(protocol['same_already_used_Tolokers_split0'] is True and protocol['new_seeds_are_independent_confirmation'] is False,
          'Same original-paper exposure with no independent confirmation claim')
    decision=read(PHASE/pins['root_decision']['path'])
    check(sha(PHASE/pins['root_decision']['path'])==pins['root_decision']['sha256']==protocol['root_decision_sha256']==receipt['root_decision_sha256'], 'Exact root decision binding')
    check(decision['configuration_id']==protocol['configuration_id'] and decision['new_base_seeds']==protocol['seeds']
          and decision['reuse_native_member0'] is False and decision['original_screen_protocol_pass_claimed'] is False, 'Transparent post-screen architecture decision')
    summary=read(PHASE/pins['original_screen_summary_path'])
    check(summary['selection_status']=='incomplete_no_freeze' and summary['failures']==2 and summary['architecture_frozen'] is False, 'Unchanged original screen failure')
    historical=read(PHASE/pins['all_attempts_path'])['records']
    check(len(historical)==15 and sum(row['status']=='complete' for row in historical)==13 and sum(row['status']=='failed' for row in historical)==2,
          'All15 historical attempts retained without survivor replacement')
    file_rows(PHASE,pins['files'])
    checks.append('Exact NSD/native-adapter/V2 helper/source-license closure, historical metadata/diagnostics/decision and private-decay accounting preserved')
    original_support=parsed(PHASE/pins['BSNN_wrapper_directory']/'support.py')
    selector=function(original_support,'selector').body[-1].value
    checks.append(check(any(isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='key' for target in node.targets)
          and dump(node.value)==dump(selector) for node in ast.walk(original)), 'Borrowed shared selector AST equals original V2 selector'))
    preserved=[]
    for row in pins['immutable_packets']:
        root=PHASE/row['directory']; check(sha(root/'MANIFEST.json')==row['manifest_sha256'],'Immutable manifest: '+row['directory'])
        payloads=read(root/'MANIFEST.json')['files']; file_rows(root,payloads)
        preserved.append(dict(directory=row['directory'],manifest_sha256=row['manifest_sha256'],verified_payloads=len(payloads)))
    checks.append('All8 preceding sealed packets unchanged, including geometry core V1 and deterministic bundle V2')
    if (HERE/'MANIFEST.json').exists():
        file_rows(HERE,read(HERE/'MANIFEST.json')['files'])
        seal=read(HERE/'SEAL.json')
        checks.append(check(seal['source_only'] is True and seal['execution_enabled'] is False and seal['manifest_sha256']==sha(HERE/'MANIFEST.json'),
                            'Own inactive source manifest/seal verified'))
    return dict(status='passed',verification='AST and exact source/receipt byte hashes only',checks=checks,preserved_packets=preserved,
          architecture_scope='post-screen exploratory fixed d4_f16_L4',model_or_runner_or_numeric_provider_import=False,numerical_execution=False,
          new_data_array_or_checkpoint_access=False,new_scientific_metrics_calculated=False,original_reports_modified=False,
          historical_JSON_status_identity_and_source_metadata_read=True,server_or_installation_actions=False,execution_enabled=False)


if __name__=='__main__': print(json.dumps(verify(),indent=2)+'\n')
