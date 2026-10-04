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
    return dict(status='AUTHOR_SOURCE_CHECKS_PASS_ONLY',AST_parsed_files=len(list(HERE.rglob('*.py'))),
                native_byte_identical_files=len(native['files']),unchanged_supervisor_functions=list(unchanged),
                hash_pinned_new_wheels=16,existing_core_upgrade_planned=False,installation_executed=False,
                TEST_array_access=False,target_dependencies_imported=False,numerical_workload=False,
                independent_source_review=False,numeric_execution_authorized=False,
                limitation='Source assertions do not prove native package/operator/model behavior or resource success.')


if __name__ == '__main__':
    result = main()
    (HERE/'AUTHOR_SOURCE_CHECK.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))
