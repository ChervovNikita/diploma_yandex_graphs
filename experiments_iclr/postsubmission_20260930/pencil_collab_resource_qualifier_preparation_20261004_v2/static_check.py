"""Source-only author checks. Does not import native modules, dependencies or arrays."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def main():
    for path in HERE.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    native = json.loads((HERE/'NATIVE_SOURCE_BINDINGS.json').read_text())
    for row in native['files']:
        value = (HERE/row['path']).read_bytes()
        assert len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256']
        assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest() == row['git_blob_sha1']
    # Verify the inherited process/session ownership, physical completion and publication core stays identical.
    before = ast.parse((PHASE/'graph_count_conditioned_pattern_minimal_gradient_preparation_20261004_v4/supervise.py').read_text())
    after = ast.parse((HERE/'supervise.py').read_text())
    before_functions = {node.name:node for node in before.body if isinstance(node,ast.FunctionDef)}
    after_functions = {node.name:node for node in after.body if isinstance(node,ast.FunctionDef)}
    unchanged = ('fsync_directory','durable_write','process_identity','members_of_session','held_identity','kill_owned','read_json','inventory')
    for name in unchanged:
        assert ast.dump(before_functions[name]) == ast.dump(after_functions[name]), name
    worker = (HERE/'worker.py').read_text()
    calls = [node for node in ast.walk(ast.parse(worker)) if isinstance(node,ast.Call)]
    assert not any(isinstance(node.func,ast.Attribute) and node.func.attr in ('get_edge_split','save_checkpoint') for node in calls)
    assert not any(isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                   and node.func.value.id=='run_lp' and node.func.attr=='main' for node in calls)
    split_values = [keyword.value.value for node in calls for keyword in node.keywords
                    if keyword.arg=='data_split' and isinstance(keyword.value,ast.Constant)]
    assert split_values == ['train','valid'], split_values
    assert 'evaluator=None' in worker and 'compute_loss=False' in worker and 'check_sequential_indices=True' in worker
    assert 'max_num_samples=-1' in worker and 'gradient_accumulation_steps=8' in worker
    plan = json.loads((HERE/'PLAN.json').read_text())
    assert plan['status'] == 'DISABLED_UNEXECUTED' and plan['workload']['world_size']==1
    assert json.loads((HERE/'ROOT_RELEASE_TEMPLATE.json').read_text())['status'] != 'APPROVED'
    installation = json.loads((HERE/'DEPENDENCY_INSTALL_PLAN.json').read_text())
    assert installation['status'] == 'DISABLED_ROOT_ACTION_ONLY_NOT_INSTALLED'
    assert installation['new_packages_count'] == 16 and installation['implicit_dependency_resolution_at_install'] is False
    assert installation['target'].endswith('/.gnnm_runtime/pencil_extra_v1/site')
    assert '--no-deps' in installation['proposed_install_argv'] and '--require-hashes' in installation['proposed_install_argv']
    assert '--upgrade' not in installation['proposed_install_argv']
    requirements = (HERE/'dependency_requirements.txt').read_bytes()
    assert hashlib.sha256(requirements).hexdigest() == installation['requirements_sha256']
    assert len(requirements.decode().splitlines()) == 16
    assert not any(row['name'].lower().replace('_','-') in ('torch','numpy','torch-geometric','torch-sparse','torch-scatter')
                   or row['name'].lower().startswith(('nvidia-','cuda')) for row in installation['new_packages'])
    for row in json.loads((HERE/'INPUT_BINDINGS.json').read_text())['inputs']:
        value = (PHASE/row['path']).read_bytes()
        assert len(value) == row['bytes'] and hashlib.sha256(value).hexdigest() == row['sha256']
    v1 = PHASE/'pencil_collab_resource_qualifier_preparation_20261004_v1'
    original_plan = json.loads((v1/'PLAN.json').read_text())
    assert plan['caps'] == original_plan['caps'] and plan['workload'] == original_plan['workload']
    assert plan['distribution_versions'] == original_plan['distribution_versions']
    unchanged_payloads = ['common.py','data_adapter.py','OFFICIAL_CONFIG_REFERENCE.yaml',
                          'metadata/config.json','metadata/RUNTIME_AUTHORITY.json','metadata/DATA_AUTHORITY.json']
    unchanged_payloads += [row['path'] for row in native['files']] + ['native/.project-root']
    for name in unchanged_payloads:
        assert (HERE/name).read_bytes() == (v1/name).read_bytes(), name
    supervisor_ast = ast.parse((HERE/'supervise.py').read_text())
    assigns = {node.targets[0].id:node.value for node in ast.walk(supervisor_ast)
               if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
    command_ast = assigns['command']
    assert isinstance(command_ast,ast.List) and len(command_ast.elts)==7
    assert ast.unparse(command_ast.elts[0])=='sys.executable' and command_ast.elts[1].value=='-B'
    assert 'worker.py' in ast.unparse(command_ast.elts[2])
    assert not any(isinstance(x,ast.Constant) and x.value in ('-m','torch.distributed.run') for x in command_ast.elts)
    env_ast = assigns['rank_environment']
    assert {k.arg:k.value.value for k in env_ast.keywords} == {'RANK':'0','LOCAL_RANK':'0','WORLD_SIZE':'1'}
    popen = [node for node in ast.walk(supervisor_ast) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='subprocess.Popen']
    assert len(popen)==1
    keywords = {item.arg:item.value for item in popen[0].keywords}
    assert keywords['start_new_session'].value is True and ast.unparse(keywords['env'])=='child_environment'
    worker_ast = ast.parse(worker)
    init = [node for node in ast.walk(worker_ast) if isinstance(node,ast.Call)
            and isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='dist.init_process_group']
    assert len(init)==1
    init_kw = {item.arg:item.value for item in init[0].keywords}
    assert init[0].args[0].value=='nccl' and init_kw['rank'].value==0 and init_kw['world_size'].value==1
    assert ast.unparse(init_kw['init_method'])=='rendezvous.as_uri()'
    assert "rendezvous = child/'RANK0_RENDEZVOUS'" in worker and 'rendezvous.unlink(missing_ok=True)' in worker
    assert not any(isinstance(node.func,ast.Attribute) and node.func.attr in ('setsid','setpgrp','Popen')
                   for node in ast.walk(worker_ast) if isinstance(node,ast.Call))
    scientific_names = {'load_data','filter_by_year','ShaDowKHopSeqFromEdgesMapDataset','get_feature_dim',
                        'get_model','build_loaders','train_loop','evaluate_loop','AdamW'}
    def scientific_calls(text):
        output=[]
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node,ast.Call): continue
            name=node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            if name in scientific_names: output.append(ast.dump(node))
        return output
    assert scientific_calls(worker)==scientific_calls((v1/'worker.py').read_text())
    dependency = json.loads((HERE/'ACTUAL_DEPENDENCY_BINDING.json').read_text())
    assert dependency['new_packages']==16 and dependency['core_versions_unchanged'] is True
    assert dependency['PLAN_distribution_versions_match_observed'] is True
    assert dependency['DISTRIBUTION_ADMISSION']['sha256']=='3c52edf196945f7610ff7b243bc709d3264b8ae40106224d87121be744f41d65'
    assert dependency['INSTALLED_FILE_INVENTORY']['sha256']=='f5052723eeff9ae66616413fda3f30ef6cee69a6ade3d682379243fa83162d7a'
    return dict(status='AUTHOR_SOURCE_CHECKS_PASS_ONLY',AST_parsed_files=len(list(HERE.rglob('*.py'))),
                native_byte_identical_files=len(native['files']),unchanged_supervisor_functions=list(unchanged),
                hash_pinned_new_wheels=16,existing_core_upgrade_planned=False,installation_executed_by_v2_author=False,existing_root_overlay_installation_observed=True,
                TEST_array_access=False,target_dependencies_imported=False,numerical_workload=False,
                independent_source_review=False,numeric_execution_authorized=False,
                direct_held_worker_source_topology=True,explicit_rank0_world1=True,private_file_rendezvous=True,
                native_scientific_call_AST_unchanged=True,caps_workload_distribution_versions_unchanged=True,
                actual_OS_process_ownership_observed=False,new_environment_or_package_inspection=False,
                limitation='Source assertions do not prove native package/operator/model behavior or resource success.')


if __name__ == '__main__':
    result = main()
    (HERE/'AUTHOR_SOURCE_CHECK.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))
