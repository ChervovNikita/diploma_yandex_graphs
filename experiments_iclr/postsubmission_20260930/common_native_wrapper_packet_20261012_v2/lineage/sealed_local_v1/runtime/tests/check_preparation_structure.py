"""New-port metadata/source QA only; no numerics, tensors, results or remote I/O."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[1]
PACKET = RUNTIME.parent
ROOT = PACKET.parent
sys.path.insert(0, str(RUNTIME))
from adapters.common_native_wrapper import ARMS, BACKBONES, DATASETS, I4, SOURCES, ast_hash, bound, render_sources
from compile_common_native_wrapper_matrix import COMPARATIVE_SEEDS, DOMAIN, SEARCH_SEEDS, SOURCE_ANCHOR, bind_metadata
from preparation_guard import require_admitted_plan
from spec import fingerprint, validate


def rejected(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return True
    raise AssertionError('Invalid candidate was admitted')


def main():
    for name in ('torch', 'numpy', 'torch_geometric'):
        assert name not in sys.modules
    bindings = json.loads((PACKET / 'SOURCE_BINDINGS.json').read_text())
    source_copy = json.loads((PACKET / 'SOURCE_COPY_RECEIPTS.json').read_text())
    for name, row in source_copy.items():
        assert row['byte_identical'] and row['original']['sha256'] == row['frozen']['sha256'] == SOURCES[name][1]
        assert (ROOT / row['original']['path']).read_bytes() == bound(ROOT, bindings[name]).read_bytes()
    family = next(n for n in ast.parse(bound(ROOT, bindings['family']).read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Family')
    originals = {n.name: n for n in family.body if isinstance(n, ast.FunctionDef)}
    ast_receipts = []
    for dataset, features, classes in [('wikics', 300, 10), ('amazon-photo', 745, 8)]:
        rendered = render_sources(ROOT, bindings, features, classes)
        builder = ast.parse(rendered['build']).body[0]
        extracted = {n.name: n for n in builder.body}
        for method in ('seeds', 'scope', 'logits'):
            assert ast.dump(extracted[method]) == ast.dump(originals[method])
        # Invert exactly the three declared anchors and prove full method equality.
        class InvertProvider(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self' and node.func.attr == 'Model':
                    assert node.args[2].value == features and node.args[4].value == classes
                    node.args[2].value, node.args[4].value = 300, 10
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'randint':
                    assert [n.value for n in node.args[2].elts] == [1, features]
                    node.args[2].elts[1].value = 300
                return node
        assert ast.dump(InvertProvider().visit(extracted['make'])) == ast.dump(originals['make'])
        helpers = {n.name: n for n in ast.parse(rendered['train']).body}
        for key, name in [('base_train', 'train_step'), ('gnnm_train', 'train_step_tabm')]:
            original = next(n for n in ast.parse(bound(ROOT, bindings[key]).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(helpers[name]) == ast.dump(original)
        branch = next(n for n in ast.walk(originals['fit']) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'bank').body
        assert ast.dump(ast.Module(body=helpers['packed_train_step'].body[:-1], type_ignores=[])) == ast.dump(ast.Module(body=branch, type_ignores=[]))
        custody, _, _, _ = bind_metadata(ROOT, PACKET / 'metadata' / dataset / 'CUSTODY.json', PACKET / 'metadata' / dataset / 'RECEIPT.json')
        assert len(custody['splits']) == DATASETS[dataset]
        ast_receipts.append({'dataset': dataset, 'provider_changes_only': True, 'unchanged_seed_rng_and_forward_AST': True,
            'literal_base_TABM_and_packed_train_AST': True, 'hashes': rendered['source_method_AST_hashes']})
    rejected(lambda: render_sources(ROOT, bindings, 0, 8))
    rejected(lambda: render_sources(ROOT, bindings, 745, 2))
    poison = json.loads(json.dumps(bindings))
    poison['family']['sha256'] = '0' * 64
    rejected(lambda: render_sources(ROOT, poison, 745, 8))
    plan = json.loads((PACKET / 'prepared/PLAN.json').read_text())
    summary = json.loads((PACKET / 'prepared/SUMMARY.json').read_text())
    assert fingerprint(plan) == summary['plan_hash'] and len(plan['studies']) == 6
    assert plan['scientific_launch_enabled'] is False and plan['human_GO_required'] is True
    assert plan['scientific_roster_approved'] is False and plan['resource_admission_status'] == 'pending_root_qualification'
    rejected(lambda: require_admitted_plan(plan, {}))
    # Entrypoint guard must reject before it tries any real payload receipt.
    from launcher import load_plan
    from run_study import require_phase_approval
    rejected(lambda: load_plan(ROOT, PACKET / 'prepared/PLAN.json', PACKET / 'does_not_exist_GO.json'))
    rejected(lambda: require_phase_approval(ROOT,
        {'study_id': 'guard-fixture', 'release': {'global_plan_gates_required': True}, 'source': {'source_commit_status': 'pending_root_owned'}},
        {'config': {'stage': 'search'}}, {'approved_plan_path': str((PACKET / 'prepared/PLAN.json').relative_to(ROOT))}))
    forged = dict(plan, scientific_launch_enabled=True)
    rejected(lambda: require_admitted_plan(forged, {}))
    admitted_preview = dict(forged, scientific_roster_approved=True, resource_admission_status='qualified')
    keys, physical, kinds, per_study = set(), 0, Counter(), []
    for row in plan['studies']:
        study = validate(json.loads(gzip.decompress((ROOT / row['path']).read_bytes())))
        assert fingerprint(study) == row['spec_hash'] and study['source']['source_commit_status'] == 'pending_root_owned'
        rejected(lambda: require_admitted_plan(admitted_preview, {study['study_id']: study}))
        assert study['stage'] == 'development' and study['data']['prior_exposure']['evaluation_access_status'] == 'encountered_development'
        backbone = next(r['config']['backbone'] for r in study['runs'] if r['kind'] == 'model_fit')
        assert backbone in BACKBONES
        assert study['data']['evidence']['split_roster'] == list(range(DATASETS[study['dataset']]))
        runs = {run['run_id']: run for run in study['runs']}
        assert len(runs) == len(study['runs'])
        search = [r for r in runs.values() if r['kind'] == 'model_fit' and r['config']['stage'] == 'search']
        assert len(search) == 108
        for arm in ARMS:
            rows = [r for r in search if r['config']['wrapper_arm'] == arm]
            assert len(rows) == 18 and {r['seed'] for r in rows} == set(SEARCH_SEEDS)
            for seed in SEARCH_SEEDS:
                assert [{k: r['config'][k] for k in ('learning_rate', 'dropout')} for r in rows if r['seed'] == seed] == DOMAIN
                assert sum({k: r['config'][k] for k in SOURCE_ANCHOR} == SOURCE_ANCHOR for r in rows if r['seed'] == seed) == 1
            selector = runs['select-' + arm]
            assert len(selector['parents']) == 18 and set(selector['parents']) == {r['run_id'] for r in rows}
        comparative = [r for r in runs.values() if r['kind'] == 'model_fit' and r['config']['stage'] == 'comparative']
        assert len(comparative) == DATASETS[study['dataset']] * 3 * 14
        for split in range(DATASETS[study['dataset']]):
            for seed in COMPARATIVE_SEEDS:
                block = [r for r in comparative if r['split'] == split and r['seed'] == seed]
                assert len(block) == 14
                exports = [r for r in runs.values() if r['kind'] == 'ensemble_assembly' and r['split'] == split and r['seed'] == seed]
                assert len(exports) == 8
                for bank_arm, arm in I4.items():
                    identifier = 'bank-sp' + str(split) + '-' + bank_arm + '-s' + str(seed)
                    members = [runs[i] for i in runs[identifier]['config']['fit_parents']]
                    assert len(members) == 4 and {r['body'] for r in members} == set(range(4))
                    assert len({seed + 1000003 * r['body'] for r in members}) == 4
                    assert all(r['config'] == members[0]['config'] for r in members)
                    assert all(r['config']['wrapper_arm'] == arm and r['config']['members'] == 1 and r['config']['selected_config_from'] == 'select-' + arm for r in members)
                    m1 = 'compare-sp' + str(split) + '-' + arm + '-s' + str(seed)
                    assert m1 not in runs[identifier]['parents'] and all(i != m1 for i in runs[identifier]['config']['fit_parents'])
        done, active = set(), set()
        def visit(identifier):
            assert identifier in runs and identifier not in active
            if identifier in done:
                return
            active.add(identifier)
            for parent in runs[identifier]['parents']:
                visit(parent)
            active.remove(identifier)
            done.add(identifier)
        for run in runs.values():
            visit(run['run_id'])
            keys.add(study['study_id'] + '/' + run['run_id'])
            kinds.update([run['kind']])
            if run['kind'] == 'model_fit':
                physical += 1
                assert run['budget'] == {'unit': 'optimizer_updates', 'policy': 'full_budget', 'maximum': 5000}
                assert run['selection']['checkpoint0'] is False and run['selection']['evaluation_interval'] == 1 and run['selection']['ties'] == 'first'
                assert run['config']['is_original_gnnm'] == (run['config']['wrapper_arm'] == 'original_gnnm')
                assert run['config']['fusion'] == ('mean_logits' if run['config']['is_original_gnnm'] else 'mean_probabilities')
                assert all('sealed/' not in row['path'] for row in run['config']['node_inputs'].values())
            if run['kind'] == 'calibration':
                assert run['config']['steps'] == 500 and run['config']['placement'] == 'after_serving_pool'
        per_study.append({'study_id': study['study_id'], 'run_records': len(runs), 'acyclic_dependencies': True,
            'whole_mask_roster': True, 'fresh_genuine_I4_and_own_M1': True, 'original_GNNM_present': True})
    placements = [key for group in plan['groups'] for key in group['run_keys']]
    assert len(placements) == len(set(placements)) and set(placements) == keys
    assert physical == 4428 and dict(kinds) == {'model_fit': 4428, 'selection': 36, 'ensemble_assembly': 2160, 'calibration': 2160, 'evaluation': 2160}
    assert summary['counts']['logical_backward_calls'] == 27810000 and summary['counts']['full_view_native_TRAIN_forwards'] == 44820000
    gates = plan['global_gates']
    assert gates['whole_search']['source_stages'] == ['search', 'search_selection'] and gates['whole_search']['required_for_stages'] == ['comparative']
    assert gates['whole_comparative']['required_for_stages'] == ['bank_export', 'calibration', 'evaluation']
    assert gates['whole_calibration']['required_for_stages'] == ['evaluation']
    for name in ('torch', 'numpy', 'torch_geometric'):
        assert name not in sys.modules
    result = {'schema': 'common-wrapper-new-port-structure-QA-v1', 'status': 'passed', 'plan_hash': fingerprint(plan),
        'source_AST': ast_receipts, 'studies': per_study, 'counts': summary['counts'],
        'negative_checks': ['wrong dimensions', 'binary classes', 'source hash poison', 'disabled plan', 'unapproved roster', 'pending source commit',
            'launcher disabled before payload access', 'direct child disabled before execution'],
        'real_tensor_reads': 0, 'scientific_fits': 0, 'scientific_scores': 0, 'numerical_imports': 0, 'CUDA_operations': 0,
        'remote_operations': 0, 'git_operations': 0, 'old_closed_audits_rerun': False,
        'numerical_source_QA': 'separate new-port synthetic check; not claimed by this metadata-only receipt'}
    (PACKET / 'VALIDATION_STRUCTURE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'plan_hash', 'counts', 'real_tensor_reads', 'scientific_fits')}, indent=2))


if __name__ == '__main__':
    main()
