"""Standard-library source/metadata checks; never imports a worker or tensor library."""
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def callname(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return callname(node.value) + '.' + node.attr
    return ''


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(dict(name=name, status='PASS'))

    source = (HERE / 'qualify_artifact.py').read_text()
    tree = ast.parse(source)
    compile(tree, str(HERE / 'qualify_artifact.py'), 'exec')
    for path in (HERE / 'pinned_ogb').glob('*.py'):
        compile(ast.parse(path.read_text()), str(path), 'exec')
    check('AST_compile_without_import_or_execution', True)
    plan = json.loads((HERE / 'PLAN.json').read_text())
    pins = json.loads((HERE / 'SOURCE_PINS.json').read_text())
    release = json.loads((HERE / 'ROOT_RELEASE_TEMPLATE.json').read_text())
    for path in HERE.glob('*.json'):
        json.loads(path.read_text())
    inputs = json.loads((HERE / 'INPUTS.json').read_text())['inputs']
    for row in inputs:
        path = ROOT / row['path']
        check('input:' + row['path'], path.stat().st_size == row['bytes'] and sha(path) == row['sha256'])
    for row in pins['OGB_sources']:
        path = HERE / row['copied_path']
        check('copy:' + row['path'], path.stat().st_size == row['bytes'] and sha(path) == row['sha256'])
    with (HERE / 'pinned_ogb/master.csv').open() as stream:
        rows = csv.reader(stream)
        column = next(rows).index('ogbl-ddi')
        metadata = {row[0]: row[column] for row in rows}
    check('official_DDI_metadata', metadata == plan['official_metadata']
          == json.loads((HERE / 'DDI_OFFICIAL_METADATA.json').read_text())['values'])
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    gate = next(n.lineno for n in main_node.body if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call) and callname(n.value.func) == 'gate')
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    numerical = [n for n in imports if isinstance(n, ast.Import)
                 and any(a.name in ('numpy', 'torch') for a in n.names)]
    check('numerical_imports_after_gate', len(numerical) == 2
          and all(n in list(ast.walk(main_node)) and n.lineno > gate for n in numerical))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = ('torch.cuda.', 'torch.nn.', 'torch.optim.', 'requests.', 'subprocess.', 'urllib.')
    check('no_GPU_model_network_loader', not any(callname(n.func).startswith(forbidden)
          or callname(n.func) in ('PygLinkPropPredDataset', 'Evaluator', 'BaseModel', 'exec', 'eval', 'z.extract', 'z.extractall') for n in calls))
    check('no_OGB_or_model_import', not any((getattr(n, 'module', '') or '').startswith(
          ('ogb', 'torch_geometric', 'native', 'requests', 'subprocess', 'socket', 'urllib')) for n in imports))
    member_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'MEMBERS' for t in n.targets))
    members = ('ddi/split/target/train.pt', 'ddi/split/target/valid.pt', 'ddi/raw/num-node-list.csv.gz',
               'ddi/raw/num-edge-list.csv.gz', 'ddi/raw/edge.csv.gz')
    check('five_exact_literal_members', ast.literal_eval(member_node.value) == tuple(plan['selected_members']) == members)
    loads = {ast.unparse(n) for n in calls if callname(n.func) == 'torch.load'}
    expected_loads = {"torch.load(selected / 'train.pt', map_location='cpu', weights_only=False)",
                      "torch.load(selected / 'valid.pt', map_location='cpu', weights_only=False)",
                      "torch.load(temporary, map_location='cpu', weights_only=True)"}
    check('only_TRAIN_VALID_and_output_loads', loads == expected_loads)
    check('literal_bounded_zip_stream', sum(callname(n.func) == 'z.open' for n in calls) == 1)
    raw_tree = ast.parse((HERE / 'pinned_ogb/read_graph_raw.py').read_text())
    homogeneous = next(n for n in raw_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'read_csv_graph_raw')

    def inverse_assignments(node):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(node) if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Name)
                and n.targets[0].value.id == 'duplicated_edge']

    check('exact_homogeneous_native_inverse_assignments', inverse_assignments(tree) == inverse_assignments(homogeneous)
          and len(inverse_assignments(tree)) == 2 and 'np.repeat(raw_edge, 2, axis=1)' in source)
    saves = [n for n in calls if callname(n.func) == 'torch.save']
    equality = [n for n in calls if callname(n.func) == 'require' and any(isinstance(a, ast.Constant)
                and a.value == 'Native raw graph does not equal unique loop-free TRAIN; no substitution' for a in n.args)]
    check('membership_equality_before_serialization', len(saves) == len(equality) == 1 and equality[0].lineno < saves[0].lineno)
    groups = {
        'ambiguity_and_central_binding': ['for name in MEMBERS:', 'recorded == observed', 'len(names) == len(set(names))', "'split_dict.pt'"],
        'native_dtype_order_absent_weights': ['value.dtype == np.int64', 'value = torch.from_numpy(value)', "set(train_raw) == {'edge'}", 'graph_edge_weight=None', 'saved.dtype == original.dtype', 'torch.equal(saved, original)'],
        'fixed_VALID_global_candidates': ["set(valid_raw) == {'edge', 'edge_neg'}", 'TRAIN_VALID_positive_overlap=overlap_count(train_keys, valid_keys, torch)', 'TRAIN_VALID_negative_overlap=overlap_count(train_keys, negative_keys, torch)', 'VALID_positive_negative_overlap=overlap_count(valid_keys, negative_keys, torch)', 'len(negatives) >= 100', 'TEST_disjointness_numerically_checked=False'],
        'single_graph_and_TRAIN_binding': ["nodes == plan['expected_nodes'] and records == plan['expected_TRAIN_records']", 'rows == records', "plan['TRAIN_tensor_bytes_sha256']"],
        'strict_artifact_schema': ["{'schema', 'num_nodes', 'graph_edge_index', 'graph_edge_weight', 'train', 'valid'}", "roundtrip['graph_edge_weight'] is None", "set(roundtrip['train']) == {'edge'}", "set(roundtrip['valid']) == {'edge', 'edge_neg'}"],
        'CPU_bounds_and_failure_custody': ["os.environ.get('CUDA_VISIBLE_DEVICES') == ''", 'torch.set_num_threads(2)', 'torch.set_num_interop_threads(1)', 'gunzip_bounded', 'INCOMPLETE_NONQUALIFYING', 'sha(central) == central_sha', 'Final selected member custody differs', 'require(not output.exists()'],
    }
    for name, snippets in groups.items():
        check(name, all(snippet in source for snippet in snippets))
    check('disabled_exact_release', release['status'] == 'DISABLED_TEMPLATE' and release['root_authorization_reference'] == ''
          and release['source_manifest_sha256'] == 'UNSET' and release['plan_sha256'] == sha(HERE / 'PLAN.json')
          and plan['status'] == 'DISABLED_SOURCE_PREPARATION' and plan['automatic_retry'] is False)
    cost = json.loads((HERE / 'PROSPECTIVE_COST.json').read_text())
    check('named_cost_arithmetic', cost['native_graph_edge_index_bytes'] == 34173152 and cost['TRAIN_tensor_bytes'] == 17086576
          and cost['named_artifact_tensor_bytes_upper_bound'] == 83259728 and cost['selected_plus_artifact_named_disk_bound_bytes'] == 469762048
          and cost['new_network_bytes'] == 0)
    check('validator_stdlib_only', not any(name in sys.modules for name in ('torch', 'numpy', 'torch_geometric', 'ogb')))
    result = dict(status='PASS_STATIC_SOURCE_AND_METADATA_ONLY', UTC=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(HERE / 'qualify_artifact.py'), plan_sha256=sha(HERE / 'PLAN.json'),
                  checks_passed=len(checks), checks=checks, scientific_payloads_opened=False,
                  numerical_modules_imported=False, worker_or_OGB_source_imported=False, server_or_network_actions=False,
                  numerical_graph_equivalence_qualified=False, artifact_produced=False, training_runtime_or_budget_qualified=False,
                  limitation='Source/AST and metadata validation only; root execution must produce data qualification and observed resource evidence.')
    (HERE / 'STATIC_VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], checks_passed=len(checks), source_sha256=result['source_sha256'])))


if __name__ == '__main__':
    main()
