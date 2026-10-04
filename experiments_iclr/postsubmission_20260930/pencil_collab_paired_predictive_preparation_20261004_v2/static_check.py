"""Local stdlib AST/hash checks only; never imports native/numerical code or arrays."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
PARENT = PHASE/'pencil_collab_resource_qualifier_preparation_20261004_v3'
V1 = PHASE/'pencil_collab_paired_predictive_preparation_20261004_v1'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def calls(parsed, name):
    return [node for node in ast.walk(parsed) if isinstance(node, ast.Call) and ast.unparse(node.func) == name]


def check():
    python_files = sorted(HERE.rglob('*.py'))
    parsed = {str(path.relative_to(HERE)): tree(path) for path in python_files}
    native = json.loads((HERE/'NATIVE_SOURCE_BINDINGS.json').read_text())
    assert len(native['files']) == 20
    for row in native['files']:
        value = (HERE/row['path']).read_bytes()
        assert len(value) == row['bytes'] and digest(value) == row['sha256']
        assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() == row['git_blob_sha1']
        assert value == (PHASE/row['origin']).read_bytes()
    unchanged_payloads = ['data_adapter.py','NATIVE_SOURCE_BINDINGS.json','OFFICIAL_CONFIG_REFERENCE.yaml',
        'metadata/config.json','metadata/DATA_AUTHORITY.json','metadata/RUNTIME_AUTHORITY.json',
        'ACTUAL_DEPENDENCY_BINDING.json','native/.project-root'] + [r['path'] for r in native['files']]
    for name in unchanged_payloads:
        assert (HERE/name).read_bytes() == (PARENT/name).read_bytes(), name
    inherited = ('fsync_directory','durable_write','process_identity',
                 'held_identity','kill_owned','read_json','inventory')
    before = {n.name:n for n in tree(PARENT/'supervise.py').body if isinstance(n,ast.FunctionDef)}
    after = {n.name:n for n in parsed['supervise.py'].body if isinstance(n,ast.FunctionDef)}
    for name in inherited:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    v1_functions = {n.name:n for n in tree(V1/'supervise.py').body if isinstance(n,ast.FunctionDef)}
    for name in v1_functions:
        if name != 'members_of_session':
            assert ast.dump(v1_functions[name]) == ast.dump(after[name]), name
    assert (HERE/'worker.py').read_bytes() == (V1/'worker.py').read_bytes()
    common = (HERE/'common.py').read_text()
    common_restored = common.replace("            and release.get('allocator_policy') == plan['allocator_policy']\n",'').replace(
        "            and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'\n",'')
    assert common_restored == (V1/'common.py').read_text()
    proc_fixture_checks = check_exit_race_sampler(after['members_of_session'])
    inputs = json.loads((HERE/'INPUT_BINDINGS.json').read_text())['inputs']
    for row in inputs:
        path = PHASE/row['path']
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        value = path.read_bytes()
        assert len(value) == row['bytes'] and digest(value) == row['sha256']
    # Parent source closure contains source, configuration and text/JSON evidence only.
    parent_manifest = json.loads((PARENT/'MANIFEST.json').read_text())
    for row in parent_manifest['files']:
        assert Path(row['path']).suffix not in ('.pt','.npy','.npz','.gz','.pkl')
        value = (PARENT/row['path']).read_bytes()
        assert len(value) == row['bytes'] and digest(value) == row['sha256']
    evidence = json.loads((HERE/'evidence/V3_RESOURCE_SUMMARY.json').read_text())
    assert evidence['adoption_eligible'] is True and evidence['physical_exit_code'] == 0
    assert evidence['physical_session_closed'] is True and evidence['supervisor_identity'] is None
    assert (HERE/'evidence/V3_RESOURCE_SUMMARY.json').read_bytes() == (PHASE/'pencil_collab_resource_qualifier_execution_root_20261004_v3/owned_monitor02/SUMMARY.json').read_bytes()
    assert (HERE/'evidence/V3_ROOT_RESOURCE_ADOPTION.json').read_bytes() == (PHASE/'pencil_collab_resource_qualifier_root_adoption_20261004_v3/ROOT_ADOPTION.json').read_bytes()
    plan = json.loads((HERE/'PLAN.json').read_text())
    prior = json.loads((PARENT/'PLAN.json').read_text())
    assert plan['status'] == 'DISABLED_UNEXECUTED' and plan['scientific_fit_released'] is False
    assert plan['seeds'] == [0,1,2] and plan['grid_search'] is False and plan['max_concurrent_fits'] == 1
    assert plan['workload']['native_epochs'] == plan['workload']['full_scientific_epochs'] == 20
    assert plan['workload']['seeds'] == [0,1,2] and plan['workload']['scientific_fits'] == 3
    assert plan['workload']['native_epoch_indices'] == list(range(20))
    assert plan['workload']['batch_size_per_rank'] == 1024 and plan['workload']['gradient_accumulation_per_rank'] == 8
    assert plan['workload']['num_workers'] == 12 and plan['workload']['world_size'] == 1
    assert plan['distribution_versions'] == prior['distribution_versions'] and plan['added_package_modules'] == prior['added_package_modules']
    for name in ('host_RSS_bytes','cuda_allocated_bytes'):
        assert plan['caps'][name] == prior['caps'][name]
    assert plan['caps']['cuda_reserved_bytes'] == 80*1024**3
    assert plan['allocator_policy']['environment_variable'] == 'PYTORCH_CUDA_ALLOC_CONF'
    assert plan['allocator_policy']['value'] == 'expandable_segments:True'
    v1_plan = json.loads((V1/'PLAN.json').read_text())
    restored = dict(plan);restored.pop('allocator_policy');restored.pop('v1_failure_metadata')
    restored['caps'] = dict(plan['caps'],cuda_reserved_bytes=v1_plan['caps']['cuda_reserved_bytes'])
    restored['execution_directory'] = v1_plan['execution_directory']
    assert restored['protocol_differences'][:-2] == v1_plan['protocol_differences']
    restored['protocol_differences'] = restored['protocol_differences'][:-2]
    assert restored == v1_plan
    assert plan['caps']['wall_seconds'] == 14400 and plan['caps']['output_bytes'] == 2*1024**3
    assert plan['loader_pin_memory_policy'] == prior['loader_pin_memory_policy']
    release = json.loads((HERE/'ROOT_RELEASE_TEMPLATE.json').read_text())
    assert release['status'] == 'DISABLED_TEMPLATE_NOT_AUTHORIZATION' and release['root_authorization_reference'] is None
    assert release['plan_sha256'] == digest((HERE/'PLAN.json').read_bytes())
    assert release['workload'] == plan['workload'] and release['caps'] == plan['caps']
    assert release['allocator_policy'] == plan['allocator_policy']
    assert release['VALID_selection_only'] is True and release['scores_for_selection'] is True
    assert release['TEST_reads'] is False and release['heldout_release'] is False and release['state_donor'] is False
    dependency = json.loads((HERE/'DEPENDENCY_ADMISSION_TEMPLATE.json').read_text())
    assert dependency['status'] == 'DISABLED_TEMPLATE_NOT_ADMISSION' and dependency['installed_inventory_complete'] is False
    assert dependency['required_released_status'] == 'ROOT_ADMITTED_FOR_NATIVE_PENCIL_PREDICTIVE_FITS'
    assert dependency['distribution_versions'] == plan['distribution_versions']
    command = json.loads((HERE/'PROPOSED_COMMAND.json').read_text())
    assert command['numeric_launch_authorized'] is False and command['execution_order'] == [0,1,2]
    env = dict(command['environment'])
    assert env.pop('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
    assert env == json.loads((V1/'PROPOSED_COMMAND.json').read_text())['environment']
    assert command['maximum_concurrent_fits'] == 1 and command['supplied_client'] is False
    assert command['full_selected_state_transfer_to_local_client'] is False
    for seed, row in enumerate(command['commands']):
        assert row['seed'] == seed and row['argv'][-2:] == ['--seed',str(seed)]
        assert row['argv'][1] == '-B' and row['argv'][2].endswith(HERE.name+'/supervise.py')
    worker = (HERE/'worker.py').read_text()
    worker_ast = parsed['worker.py']
    assert not calls(worker_ast,'run_lp.main') and not calls(worker_ast,'torch.load')
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in ('get_edge_split','load_checkpoint','save_checkpoint','Popen','setsid','setpgrp') for n in ast.walk(worker_ast))
    constructors = calls(worker_ast,'ShaDowKHopSeqFromEdgesMapDataset')
    assert [[k.value.value for k in n.keywords if k.arg == 'data_split'] for n in constructors] == [['train'],['valid']]
    for name in ('run_lp.build_loaders','run_lp.train_loop','run_lp.evaluate_loop','run_lp.get_model','torch.optim.AdamW'):
        assert len(calls(worker_ast,name)) == 1, name
    assert 'for epoch in range(configs.num_epochs)' in worker and 'configs.num_epochs == 20' in worker
    build = calls(worker_ast,'run_lp.build_loaders')[0]
    train = calls(worker_ast,'run_lp.train_loop')[0]
    evaluate = calls(worker_ast,'run_lp.evaluate_loop')[0]
    assert ast.unparse(next(k.value for k in build.keywords if k.arg == 'test_dataset_raw')) == 'None'
    assert ast.unparse(next(k.value for k in build.keywords if k.arg == 'epoch')) == 'epoch'
    assert {k.arg:ast.unparse(k.value) for k in train.keywords}['gradient_accumulation_steps'] == '8'
    assert {k.arg:ast.unparse(k.value) for k in train.keywords}['max_num_samples'] == '-1'
    assert {k.arg:ast.unparse(k.value) for k in evaluate.keywords}['evaluator'] == 'evaluator'
    assert {k.arg:ast.unparse(k.value) for k in evaluate.keywords}['compute_loss'] == 'False'
    assert {k.arg:ast.unparse(k.value) for k in evaluate.keywords}['check_sequential_indices'] == 'True'
    pin_assignments = [n for n in ast.walk(worker_ast) if isinstance(n,ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr == 'pin_memory']
    assert len(pin_assignments) == 2 and all(n.value.value is False and build.end_lineno < n.lineno < train.lineno < evaluate.lineno for n in pin_assignments)
    assert 'eval_test=False,only_eval_test=False,eval_every=1,save_only_improve=True' in worker
    assert "improved=score>best" in worker and "first_best=max(range(20),key=lambda index:history[index]['VALID_hits50'])" in worker
    for text in ('model_state_dict=model.state_dict()','optimizer_state_dict=optimizer.state_dict()',
        'python=random.getstate()','numpy=dict(','torch_cpu=torch.get_rng_state()',
        'torch_cuda=torch.cuda.get_rng_state_all()','train_loader.generator.get_state()',
        'valid_loader.generator.get_state()',"native_scores.dtype in (torch.bfloat16,torch.float32)",
        'float_scores=native_scores.float()','torch.equal(float_scores.to(native_scores.dtype),native_scores)',
        'SELECTED_FULL_STATE.pt','VALID_POSITIVE_SCORES.npy','VALID_NEGATIVE_SCORES.npy'):
        assert text in worker, text
    init = calls(worker_ast,'dist.init_process_group')[0]
    assert {k.arg:ast.unparse(k.value) for k in init.keywords}['world_size'] == '1'
    assert {k.arg:ast.unparse(k.value) for k in init.keywords}['rank'] == '0'
    assert {k.arg:ast.unparse(k.value) for k in init.keywords}['init_method'] == 'rendezvous.as_uri()'
    assert "torch.backends.cuda.matmul.allow_tf32 = True" in worker and "use_bf16=True" in worker
    assert "fresh_native_scratch_per_seed" in worker and "historical_Collab_TEST_consumed=True" in worker
    supervisor = (HERE/'supervise.py').read_text()
    popen = calls(parsed['supervise.py'],'subprocess.Popen')
    assert len(popen) == 1 and next(k.value for k in popen[0].keywords if k.arg == 'start_new_session').value is True
    assert "ACTIVE_FIT.json" in supervisor and "FIT_ATTEMPT_SPENT.json" in supervisor
    assert "if physical_complete and reaped:" in supervisor and "active.unlink()" in supervisor
    assert "os.WNOWAIT" in supervisor and "implicit_Popen_destructor_reap_disabled=True" in supervisor
    assert "terminal['completed_scientific_fits']=1 if success else 0" in supervisor
    assert not calls(parsed['supervise.py'],'torch.load') and not calls(parsed['supervise.py'],'np.load')
    assert plan['historical_Collab_TEST_consumed'] is True and plan['exact_author_reproduction'] is False
    for name in ('WORKER_MONITOR_FAILURE.json','FAILURE_METADATA.json'):
        assert (HERE/'evidence'/('V1_'+name)).read_bytes() == (PHASE/'pencil_collab_paired_predictive_execution_root_20261004_v1'/name).read_bytes()
    manifest_present = (HERE/'MANIFEST.json').exists()
    if manifest_present:
        manifest = json.loads((HERE/'MANIFEST.json').read_text())
        rows = manifest['files']
        assert {r['path'] for r in rows} == {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name not in ('MANIFEST.json','SEAL.json')}
        for row in rows:
            value = (HERE/row['path']).read_bytes()
            assert len(value) == row['bytes'] and digest(value) == row['sha256']
    return dict(status='AUTHOR_SOURCE_CHECKS_PASS_ONLY',AST_parsed_files=len(python_files),
        native_byte_identical_files=20,inherited_ownership_functions=list(inherited),bound_external_inputs=len(inputs),
        v1_worker_byte_identical=True,only_supervisor_sampler_function_changed=True,
        stdlib_mock_proc_exit_race_checks=proc_fixture_checks,allocator_policy_before_Torch_import_source=True,
        reserved_cap_80GiB_allocated_70GiB_all_other_caps_unchanged=True,
        parent_source_closure_rehashed_unchanged=True,manifest_present_at_check=manifest_present,
        fresh_seeds=[0,1,2],native_epochs_each=20,strict_first_tie_VALID_Hits50_selection=True,
        full_selected_model_optimizer_RNG_source_present=True,native_score_values_lossless_export_source=True,
        three_fit_commands_disabled=True,numerical_execution=False,native_or_numerical_modules_imported=False,
        server_actions=False,graph_score_checkpoint_payload_reads=False,execution_client_prepared=False,
        independent_source_review=False,numeric_execution_authorized=False,
        limitation='Local source checks do not prove runtime, scientific completion or exact numerical reproduction.')


def check_exit_race_sampler(node):
    """Isolated stdlib mock checks; never reads /proc or controls a real process."""
    class Status:
        def __init__(self,text):self.text=text
        def read_text(self):return self.text
    class Entry:
        name='101'
        def __init__(self,text):self.text=text
        def __truediv__(self,key):
            assert key=='status';return Status(self.text)
    class Proc:
        def __init__(self,text):self.text=text
        def iterdir(self):return [Entry(self.text)]
    initial=dict(pid=101,start_ticks=55,session=100,group=100,state='R')
    cases=[
        ('live_RSS',dict(initial,state='S'),'VmRSS:\t7 kB\n','row'),
        ('R_to_Z_no_RSS',dict(initial,state='Z'),'','row'),
        ('still_live_no_RSS',dict(initial,state='R'),'','fail'),
        ('duplicate_live_RSS',dict(initial,state='S'),'VmRSS:\t7 kB\nVmRSS:\t8 kB\n','fail'),
        ('disappeared',None,'','omit'),
        ('reused_birth',dict(initial,start_ticks=56),'VmRSS:\t7 kB\n','omit'),
        ('changed_session',dict(initial,session=200),'VmRSS:\t7 kB\n','omit'),
        ('changed_owned_group',dict(initial,group=200),'VmRSS:\t7 kB\n','row'),
    ]
    for name,current,text,outcome in cases:
        observations=iter([initial,current]);calls=[]
        def identity(pid):
            calls.append(pid);return next(observations)
        def require(condition,message):
            if not condition:raise RuntimeError(message)
        namespace=dict(Path=lambda path:Proc(text),process_identity=identity,require=require)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated sampler source>','exec'),namespace)
        try:
            rows=namespace['members_of_session'](100)
        except RuntimeError as error:
            assert outcome=='fail' and str(error)=='Owned live RSS observation missing',name
        else:
            assert outcome!='fail',name
            if outcome=='omit':assert rows==[],name
            else:
                assert len(rows)==1 and {k:rows[0][k] for k in current}==current,name
                assert rows[0]['RSS_bytes']==(7*1024 if text else 0),name
        assert calls==[101,101],name
    return dict(passed=len(cases),real_proc_access=False,real_process_or_signal_operations=False)


if __name__ == '__main__':
    print(json.dumps(check(),indent=2,sort_keys=True,allow_nan=False))
