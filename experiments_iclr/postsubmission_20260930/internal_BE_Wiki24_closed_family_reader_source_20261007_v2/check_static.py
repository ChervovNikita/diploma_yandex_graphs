"""AST/source-only checks; does not import or execute prepared programs."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    trees = {name: ast.parse((ROOT / name).read_text(), filename=name)
             for name in ('gate.py', 'extract_selected_metadata.py', 'read_wiki24.py')}
    forbidden_modules = {'models', 'data', 'run', 'runtime', 'selection', 'numpy', 'torch_geometric', 'ogb'}
    imports, forbidden_calls = {}, []
    for name, tree in trees.items():
        imports[name] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports[name].extend(a.name for a in node.names)
            if isinstance(node, ast.ImportFrom): imports[name].append(node.module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {
                'load_state_dict', 'member_forward', 'backward', 'step', 'Popen', 'kill', 'killpg', 'eval'}:
                forbidden_calls.append((name, node.lineno, node.func.attr))
        assert not (set(imports[name]) & forbidden_modules)
    assert not forbidden_calls
    assert 'torch' not in imports['gate.py'] and 'torch' not in imports['read_wiki24.py']
    extractor = (ROOT / 'extract_selected_metadata.py').read_text()
    assert extractor.index("preflight(args.activation") < extractor.index('        import torch')
    assert "map_location='cpu', weights_only=False" in extractor
    assert "checkpoint_deserialization_attempts" in extractor and 'EXTRACTION_COST.json' in extractor
    assert "accuracy(own['metric']) == member_VALID[member]" not in extractor
    assert 'own_selected_member_diagnostics' in extractor and 'final_bank_minus_own_saved_member_VALID' in extractor
    assert "own['global'] == member_global[member]" in extractor
    assert "accuracy(bank['VALID']) == selected_VALID" in extractor
    gate = (ROOT / 'gate.py').read_text()
    assert "len(rows) == 24" in gate and "freeze.get('epochs') == 1100" in gate and "freeze.get('steps') == 1100" in gate
    assert "sha(selected) == freeze.get('selected_sha256') == custody.get('selected_sha256')" in gate
    reader = (ROOT / 'read_wiki24.py').read_text()
    assert "preflight(args.activation" in reader and reader.index('preflight(args.activation') < reader.index('export = read(export_path)')
    assert "if len(observed) != 3:" in reader and '4.302652729911275' in reader
    assert "('DESCRIPTIVE', 'be_init_contrastive', 'single')" in reader
    assert "('DESCRIPTIVE', 'be_init_contrastive', 'independent4')" in reader
    for name in ('EXTRACT_TEMPLATE_DISABLED.json', 'READER_TEMPLATE_DISABLED.json'):
        cfg = json.loads((ROOT / name).read_text())
        assert cfg['stage_enabled'] is False and cfg['root_comparative_opening_authorized'] is False
    refs = json.loads((ROOT / 'SOURCE_BINDINGS.json').read_text())['references']
    for row in refs:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in trees['gate.py'].body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                 and isinstance(node.value, (ast.Constant, ast.List, ast.Set))}
    for source_name, pin in [('DRIVER', 'DRIVER_SHA'), ('SUITE', 'SUITE_SHA'), ('QUALIFIER', 'QUALIFIER_SHA')]:
        row = next(row for row in refs if row['path'].endswith(constants[source_name] + '/MANIFEST.json'))
        assert constants[pin] == row['sha256']
    result = dict(passed=True, checks='AST syntax, imports/calls, gate ordering/pins, full horizon, charge fields, missing-value policy, distinct-event diagnostics, descriptive baseline labels, disabled templates and frozen local source hashes',
                  scientific_modules_imported=False, prepared_programs_imported_or_executed=False,
                  server_or_data_or_checkpoint_or_predictive_access=False,
                  runtime_verified=False, qualification_or_scientific_adoption=False)
    (ROOT / 'STATIC_CHECKS.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
