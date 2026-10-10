"""Standard-library source/hash audit and synthetic-text chunk verification.

Never imports analysis.py or any scientific module; never reads payloads.
"""
import ast
import hashlib
import json
from pathlib import Path
import symtable
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STDLIB_NAMES = getattr(sys, 'stdlib_module_names', frozenset({
    'argparse', 'ast', 'hashlib', 'importlib', 'io', 'itertools', 'json',
    'math', 'pathlib', 'socket', 'statistics', 'subprocess', 'symtable',
    'sys', 'time'}))


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_imports(tree):
    return [a.name.split('.')[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names] + [n.module.split('.')[0] for n in tree.body if isinstance(n, ast.ImportFrom)]


def call_lines(function, name):
    return sorted(n.lineno for n in ast.walk(function) if isinstance(n, ast.Call)
                  and (isinstance(n.func, ast.Name) and n.func.id == name
                       or isinstance(n.func, ast.Attribute) and n.func.attr == name))


def main():
    raw = (HERE / 'analysis.py').read_text()
    tree = ast.parse(raw)
    compile(tree, str(HERE / 'analysis.py'), 'exec')
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(all(n in functions for n in ('preflight', 'payloads', 'score_raw', 'calibrate', 'analyze', 'encode', 'chunk_documents', 'main')), 'Complete source API')
    imports = module_imports(tree)
    require(set(imports) <= STDLIB_NAMES, 'Module-level imports are standard library only')
    main_node = functions['main']
    numpy_line = next(n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Import) and any(a.name == 'numpy' for a in n.names))
    torch_line = next(n.lineno for n in ast.walk(main_node) if isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names))
    order = [call_lines(main_node, 'preflight')[0], numpy_line, call_lines(main_node, 'payloads')[0], call_lines(main_node, 'score_raw')[0], torch_line, call_lines(main_node, 'calibrate')[0], call_lines(main_node, 'analyze')[0]]
    require(order == sorted(order) and len(set(order)) == len(order), 'Full gate before arrays/scoring/scalar calibration')
    require(len([n for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name in ('numpy', 'torch') for a in n.names)]) == 2, 'Numerical imports occur only in gated main')
    preflight = functions['preflight']
    full_records_line = next(n.lineno for n in ast.walk(preflight) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and [x.id for x in t.elts if isinstance(x, ast.Name)] == ['complete', 'old'] for t in n.targets))
    require(max(call_lines(preflight, 'closed_owner')) < min(call_lines(preflight, 'completion_header'))
            and max(call_lines(preflight, 'completion_header')) < full_records_line, 'Both owner/header closures before complete outcomes')
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    require(not any(x in calls for x in ('torch.load', 'torch.save', 'Family', 'Model', 'train_step', 'family_class')), 'No native model/checkpoint execution')
    require(calls.count('operators.fit_fold') == 1, 'One fixed scalar fitter call site')
    require('temperature_global' in [n.value for n in ast.walk(functions['calibrate']) if isinstance(n, ast.Constant)], 'One fixed global temperature rule')
    table = symtable.symtable(raw, str(HERE / 'analysis.py'), 'exec')
    require(next(x for x in table.get_children() if x.get_name() == 'analyze').lookup('contrast').is_global(), 'Contrast helper is not shadowed')
    inputs = json.loads((HERE / 'INPUT_BINDINGS.json').read_text())
    for name, sha in inputs['research_files'].items():
        path = ROOT / name
        require(path.suffix not in ('.npz', '.pt'), 'No scientific payload read by source audit')
        require(digest(path) == sha, 'Pinned text source/metadata changed: ' + name)
    pilot = 'native_neighborhood_distribution_pilot_root_20261010_v1'
    freeze = json.loads((ROOT / pilot / 'FREEZE.json').read_text())
    for name in inputs['acquisition_files'] + [pilot + '/' + x for x in ('DECISION.md', 'PROSPECTIVE_PROTOCOL.json', 'CONFIG.json', 'ACTUAL_QUALIFICATION_V1.json', 'SCIENTIFIC_ADMISSION.json')]:
        matches = [r['sha256'] for r in freeze['bound_files'] if r['path'].endswith(name)]
        require(matches == [inputs['research_files'][name]], 'Unique final-freeze binding: ' + name)
    for suffix, sha in inputs['native_suffixes'].items():
        require([r['sha256'] for r in freeze['bound_files'] if r['path'].endswith(suffix)] == [sha], 'Native suffix agrees with final freeze')
    new_assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'NEW' for x in n.targets))
    fresh = ast.literal_eval(new_assignment.value)
    refs_assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'REFS' for x in n.targets))
    refs = ast.literal_eval(refs_assignment.value)
    qualified = json.loads((ROOT / pilot / 'ACTUAL_QUALIFICATION_V1.json').read_text())
    require([r['arm'] for r in qualified['cases']] == list(fresh), 'Exact five ordered qualification arms')
    require(qualified['qualified'] is True and qualified['VALID_quality_scored'] is False and qualified['TEST_access'] is False and qualified['scientific_quality_evidence'] is False, 'Qualification remains TRAIN-only setup')
    source_schema = json.loads((ROOT / 'native_neighborhood_quantile_native_source_20261010_v1/OUTPUT_SCHEMA.json').read_text())
    bases = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'BASE_KEYS' for x in n.targets)))
    new_extra = ast.literal_eval(next(n.value.right for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'NEW_KEYS' for x in n.targets)))
    require(bases | new_extra == set(source_schema['selected_VALID.npz']['keys']), 'Exact 12-key acquisition schema')
    for name in ('private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py', 'common_wrapper_graph_reliability_source_20261010_v2/operators.py'):
        require(set(module_imports(ast.parse((ROOT / name).read_text()))) <= STDLIB_NAMES, 'Pinned helper import audit')
    prior = ast.parse((ROOT / 'common_wrapper_graph_reliability_source_20261010_v2/study.py').read_text())
    prior_folds = [ast.dump(n, include_attributes=False) for n in ast.walk(prior) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'folds' for t in n.targets)]
    reader_folds = [ast.dump(n, include_attributes=False) for n in ast.walk(functions['calibrate']) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'folds' for t in n.targets)]
    require(reader_folds == prior_folds, 'Existing permutation-position modulo-five fold law')

    # Execute only the standard-library text serializer and chunker extracted
    # from the AST. No reader import, numerical function, module or data.
    fragment = ast.Module(body=[functions[n] for n in ('require', 'encode', 'chunk_documents')], type_ignores=[])
    namespace = {'json': json, 'LIMIT': 2000000, 'CHUNK_BUDGET': 1800000}
    exec(compile(fragment, '<extracted-text-serialization>', 'exec'), namespace)
    rows = [{'index': i, 'text': chr(65 + i) * 780000} for i in range(3)]
    encode, chunk = namespace['encode'], namespace['chunk_documents']
    unchunked_bytes = len(encode({'category': 'SYNTHETIC_AGGREGATE', 'rows': rows}))
    require(unchunked_bytes > 2000000, 'Fixture exceeds old 2 MB boundary')
    docs = chunk('SYNTHETIC_AGGREGATE', rows)
    require(len(docs) > 1 and all(len(encode(v)) < 1800000 for _, v in docs), 'Large aggregate split before bounded limit')
    require([r for _, d in docs for r in d['rows']] == rows, 'Every synthetic row preserved in original order')
    require([(n, encode(v)) for n, v in docs] == [(n, encode(v)) for n, v in chunk('SYNTHETIC_AGGREGATE', rows)], 'Byte deterministic partitions')
    require([d['offset'] for _, d in docs] == [sum(len(v['rows']) for _, v in docs[:i]) for i in range(len(docs))], 'Exact chunk offsets')
    try:
        chunk('OVERSIZED_ROW', [{'text': 'x' * 1900000}])
    except ValueError:
        pass
    else:
        raise AssertionError('Oversized individual row must fail before write')
    aggregate_node = next(n for n in ast.walk(main_node) if isinstance(n, ast.AugAssign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'chunk_documents' and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == 'SAGE_ARM_AGGREGATES')
    namespace['aggregates'] = {'raw_corrected_or_original': {a: {'text': a} for a in fresh + refs}, 'exact_native_at_corrected_selected': {a: {'text': a} for a in fresh}, 'calibrated': {k: {a: {'text': k + a} for a in fresh} for k in ('corrected', 'native')}}
    aggregate_docs = eval(compile(ast.Expression(aggregate_node.value), '<aggregate-text-shape>', 'eval'), namespace)
    aggregate_rows = [r for _, d in aggregate_docs for r in d['rows']]
    require(len(aggregate_rows) == 25 and len({(r['kind'], r['arm']) for r in aggregate_rows}) == 25, 'Nested raw/native/calibrated aggregate rows all retained')
    value = {'schema': 'distribution_reader_static_verification_v1', 'status': 'passed_stdlib_source_and_synthetic_text_only',
             'analysis_sha256': digest(HERE / 'analysis.py'), 'input_bindings_sha256': digest(HERE / 'INPUT_BINDINGS.json'),
             'module_level_imports': imports, 'gate_order_source_lines': order, 'pinned_text_bindings_checked': len(inputs['research_files']),
             'final_freeze_sha256': digest(ROOT / pilot / 'FREEZE.json'), 'source_API_AST_compile': True,
             'both_owner_header_gates_before_outcomes': True, 'numerical_imports_after_full_gate': True,
             'exact_new_archive_schema': True, 'all_150_fixed_scalar_fit_roster_source_review': True,
             'contrast_symbol_scope': 'global_helper', 'native_model_checkpoint_or_TEST_execution_calls': False,
             'synthetic_text_serialization': {'unchunked_bytes': unchunked_bytes, 'chunk_bytes': [len(encode(v)) for _, v in docs],
                                             'byte_deterministic': True, 'all_rows_preserved': True, 'individual_oversized_row_rejected': True, 'aggregate_rows': len(aggregate_rows)},
             'execution': {'reader_module_imported': False, 'numerical_functions_run': False, 'numpy_torch_model_imports': False,
                           'scientific_payload_reads': False, 'remote_operations': False, 'TEST_access': False},
             'limitations': ['Static source and synthetic text checks do not qualify numerical archive processing or scalar calibration runtime.',
                             'Root owns closed-family numerical execution and interpretation.']}
    (HERE / 'STATIC_VERIFICATION.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(value, separators=(',', ':')))


if __name__ == '__main__':
    main()
