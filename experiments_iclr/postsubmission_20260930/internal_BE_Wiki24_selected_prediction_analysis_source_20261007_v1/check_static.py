"""AST/source-seal inspection only; never imports prepared programs or numerical code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    checked = []
    for path in sorted(ROOT.glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        for item in tree.body:
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in item.names] if isinstance(item, ast.Import) else [item.module or '']
                if any(name.split('.')[0] in {'numpy', 'torch', 'torch_geometric', 'ogb', 'models', 'data', 'factors', 'selection'} for name in names):
                    raise ValueError('Numerical/model import at module scope: ' + path.name)
        checked.append(path.name)
    template = json.loads((ROOT / 'RELEASE_TEMPLATE_DISABLED.json').read_text())
    for key in ('stage_enabled', 'root_execution_authorized', 'source_review_approved',
                'owner_and_children_terminal', 'trusted_checkpoint_deserialization_authorized',
                'prediction_collection_and_analysis_cost_charged'):
        if template[key] is not False:
            raise ValueError('Prepared execution must remain disabled')
    if template['maximum_member_forwards'] != 78 or template['TEST_access'] is not False or template['automatic_retry'] is not False:
        raise ValueError('Fixed bounded no-TEST/no-retry source contract')
    collector = (ROOT / 'collect.py').read_text(); analysis = (ROOT / 'analyse.py').read_text()
    for anchor in ('model.member_forward(batch, member)', 'model.eval()', 'with torch.no_grad():',
                   'member_probability = logits.softmax(-1)', 'pooled = member_probability.mean(0)',
                   "saved['body_global']", "model.set_global(saved['global'])", "state['arm'] == 'be_init'",
                   "baseline_first", "metadata_extraction_cost", "MAX_FORWARDS = 78"):
        if anchor not in collector:
            raise ValueError('Missing source contract: ' + anchor)
    for forbidden in ('runtime.admit(', '.backward(', '.step(', 'torch.optim.', 'selection.local_transition('):
        if forbidden in collector:
            raise ValueError('Training/reselection work in prediction collector')
    for anchor in ('baseline_common_competitor', 'net_repairs=repairs - harms', 'VALID_selection_optimism',
                   "'independent_TEST_evidence': False", 'allow_pickle=False', 'smallest_qualifying_class'):
        if anchor not in analysis:
            raise ValueError('Missing analysis contract: ' + anchor)
    references = json.loads((ROOT / 'SOURCE_BINDINGS.json').read_text())['references']
    for row in references:
        path = ROOT.parent / row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Inspected source binding changed: ' + row['path'])
    if (ROOT / 'MANIFEST.json').exists():
        for row in json.loads((ROOT / 'MANIFEST.json').read_text())['files']:
            path = ROOT / row['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256'] or path.stat().st_size != row['bytes']:
                raise ValueError('Packet source seal changed: ' + row['path'])
    print(json.dumps(dict(AST_parsed=checked, disabled_release=True, source_bindings_verified=len(references),
        no_prepared_program_import_or_execution=True, no_numerical_import=True, no_runtime_or_result_verification=True), sort_keys=True))


if __name__ == '__main__':
    main()
