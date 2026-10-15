"""Freeze qualified BUDDY production source bindings; no model/data imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E, P = HERE.parent, HERE.parent.parent


def pin(path, expected=None):
    path = Path(path).resolve()
    assert path.is_relative_to(P) and path.suffix in {'.py', '.json', '.md', '.txt', '.html', '.xml'}
    assert not {'results', 'datasets', 'processed', 'working_data', 'raw'} & set(path.relative_to(P).parts)
    raw = path.read_bytes()
    if path.suffix == '.py': ast.parse(raw)
    value = {'path_relative_to_phase': str(path.relative_to(P)), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if expected:
        for k in ('bytes', 'sha256'):
            if k in expected: assert value[k] == expected[k], str(path)
    return value


def main():
    target = HERE / 'OVERLAY_v3.json'
    assert not target.exists()
    hpath = E / 'qualification/BUDDY_PRODUCTION_SUCCESSORS_HANDOFF_v1.json'
    qpath = E / 'qualification/BUDDY_PRODUCTION_SEED1_QUALIFICATION_07.json'
    h, q = (json.loads(p.read_text()) for p in (hpath, qpath))
    assert q['status'] == q['supervisor_status'] == 'passed'
    assert (q['optimizer_updates'], q['whole_VALID_calls'], q['constructions'], q['forwards']) == (2, 3, 2, 32)
    assert q['full_campaign_GO'] is False and q['TEST_accessed'] is False
    sources = {}
    def remember(path, expected=None):
        value = pin(path, expected); sources[value['path_relative_to_phase']] = value
        return value
    accepted = remember(qpath)
    remember(hpath)
    for r in q['files']: remember(E / r['path'], r)
    seen = set()
    def package(folder, expected=None):
        folder = Path(folder).resolve()
        if folder in seen: return
        seen.add(folder)
        remember(folder / 'MANIFEST.json', expected)
        m = json.loads((folder / 'MANIFEST.json').read_text())['files']
        if isinstance(m, dict): m = [{'path': k, 'sha256': v} for k, v in m.items()]
        for r in m: remember(folder / r['path'], r)
        for name in ('SOURCES.json', 'BINDINGS.json'):
            if not (folder / name).exists(): continue
            d = json.loads((folder / name).read_text())
            for r in d.get('source_files', []) + d.get('native_sources', []):
                if 'path_relative_to_phase' in r: remember(P / r['path_relative_to_phase'], r)
            for r in d.get('packages', []):
                if 'path' in r and 'manifest_sha256' in r:
                    package(E / r['path'], {'sha256': r['manifest_sha256']})
    registry = json.loads((HERE / 'REGISTRY_v1.json').read_text())
    parent_overlay = json.loads((HERE / 'OVERLAY_v2.json').read_text())
    entries = {(r['cell_id'], r['method_id']): r for r in registry['entries']}
    entries.update({(r['cell_id'], r['method_id']): r for r in parent_overlay['entries']})
    aliases = {'link_endpoint_responsibility_v2': 'link_endpoint_responsibility_private_v2',
               'link_query_responsibility_v2': 'link_query_iid_responsibility_private_v2'}
    wanted = {'link_endpoint_responsibility_v2', 'link_query_responsibility_v2',
              'aggregation_mean_globalJ_v1', 'aggregation_global_globalJ_v1', 'aggregation_conditional_globalJ_v1',
              'interaction_geometry_input', 'interaction_responsibility_private_pool',
              'interaction_input_meanJ', 'interaction_input_conditionalJ', 'gnnm_private_bn_state_v3'}
    new = []
    for method in sorted(wanted):
        runtime_method = aliases.get(method, method)
        source = next(r for r in h['source_bindings'] if runtime_method in r['methods'])
        package(E / source['package'], source['manifest'])
        cb = remember(E / source['callback']['path'], source['callback'])
        mf = remember(E / source['manifest']['path'], source['manifest'])
        review = remember(E / source['source_review']['path'], source['source_review'])
        assert 'create_adapter' in {n.name for n in ast.parse((P / cb['path_relative_to_phase']).read_text()).body if isinstance(n, ast.FunctionDef)}
        row = copy.deepcopy(entries[('collab-buddy', method)])
        row.update(callback={**cb, 'function': 'create_adapter'}, session_transform=None,
            source_manifest=mf, method_settings=source['required_method_settings'], runtime_procedure=runtime_method,
            base_procedure='gnnm_boundary_native', supported_seeds=list(range(10)),
            status='source_prepared_not_runtime_admitted', source_context_scope='production_native_provider; actual runtime seed before construction',
            source_review=review, provider_input_qualification=remember(E / h['provider_input_qualification']['path'], h['provider_input_qualification']),
            representative_qualification=accepted,
            qualification_scope={'actual_executed_method': 'interaction_input_conditionalJ', 'actual_executed_seed': 1,
                'this_binding_actual_representative': method == 'interaction_input_conditionalJ',
                'this_binding_native_coverage': 'representative seed1 two-update CPU optimization/restore' if method == 'interaction_input_conditionalJ' else 'source-equivalent unchanged method functions; not individually executed on the new provider',
                'all_seeds_executed': False, 'whole_epoch_qualified': False, 'selected_runtime_qualified': False, 'CUDA_qualified': False},
            native_CPU_gate='accepted representative seed1 conditional interaction; other recipes retain source-equivalence scope',
            pending=['BUDDY full native TRAIN epoch and own selected-runtime/export/resume',
                     'actual callback protocol/runtime binding and device qualification outside representative scope',
                     'measured resource/cost/concurrency admission', 'human full-campaign GO'])
        for k in ('unsupported_seed_reason', 'context_manifest_receipt', 'missing_reason'):
            row.pop(k, None)
        assert not {'context_manifest', 'context_manifest_sha256'} & set(row['method_settings'])
        new.append(row)
    assert len(new) == 10
    value = {'schema': 'expanded-fit-BUDDY-production-source-overlay-v3',
        'parent_registry': pin(HERE / 'REGISTRY_v1.json'), 'parent_overlay': pin(HERE / 'OVERLAY_v2.json'),
        'parent_readiness': pin(HERE / 'compact_generated_v2/READINESS.json'),
        'accepted_representative_qualification': accepted, 'source_successors_handoff': pin(hpath),
        'entries': new, 'source_files': sorted(sources.values(), key=lambda r: r['path_relative_to_phase']),
        'source_context_blocks_closed_by_successor': 90, 'existing_seed0_bindings_rebound': 10,
        'representative_actual_method_seed': ['interaction_input_conditionalJ', 1],
        'other_recipes_individual_native_pass_claim': False, 'runnable': False,
        'runtime_link_successor_and_protocol_admission_still_required': True, 'resource_admission': False,
        'scientific_launch_enabled': False, 'campaign_GO': False,
        'published_parent': 'c58a59ffe00c75e81082973139234a41b907c09b'}
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with target.open('xb') as out: out.write(raw)
    print(json.dumps({'overlay': pin(target), 'bindings': 10, 'source_files': len(sources), 'native_work': 0}))


if __name__ == '__main__': main()
