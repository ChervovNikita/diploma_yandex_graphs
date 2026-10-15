"""Freeze explicit source bindings from audited aliases. Stdlib; never import ports.

This local builder writes one fresh registry. Later source ports can be supplied
as explicit complete replacement entries through --supplement in a new generation.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent
REQUIRED = {
    'runtime_v4/MANIFEST.json': '6ea4735aac075314e66d0f18a46873d430f98b1803f8452ad24ba4e39ce91d98',
    'portable_private_input_v2/MANIFEST.json': '21ad8f2c644a1833f0015dfc4c0c0193bcfc9598b6f84d98b9d686330bed8115',
    'interaction_link_ncn_v2/MANIFEST.json': '8f6cc6ca60214b47653ba4c305f2b98dfae503c08ae52ee32b30de67ed132450',
    'interaction_link_ncnc_v1/MANIFEST.json': 'e7fb265ea3c3650d7ba4c24157adda04b20cdba88ad46ebc54a498b9b892b0c1',
    'context_affinity_link_port_v2/MANIFEST.json': '4a43ff70bbe99e5d1db0c26972866abdc260de43c3791dabf6215405a5e79b2e',
    'context_affinity_ncnc_port_v1/MANIFEST.json': '1748b647dee6de992261dfc96b8d1b0b210307d786ce61b7da4f7d05a1b0b526',
    'context_affinity_BUDDY_port_v1/MANIFEST.json': '4b15c6fae17d668bf4c567e223b42ca32f24582cc081dfd2c0c90f0a9835d64d',
    'buddy_private_bn_state_v1/MANIFEST.json': 'b2bfe9e1c2c4050d5f1352864811a0a7fdcc9cd62e7ec6df1e7e217e25bc5cc4',
    'buddy_private_bn_state_v1/callback.py': 'bf316ab2b59751c6c60274cd5f27768b42f75a1da473c5fb1e716a80105dfaec',
    'private_bn_state_v3/banks.py': 'd90a74bf98be32408cad306f35ab5ebd1ec80153981a0b7f92cba07e7f447fea',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def pin(path):
    path = Path(path).resolve()
    assert path.is_relative_to(PHASE)
    assert path.suffix in {'.py', '.json', '.md', '.txt', '.xml'} or path.name == '.gitignore'
    raw = path.read_bytes()
    if path.suffix == '.py':
        ast.parse(raw, filename=str(path))
    result = {'path_relative_to_phase': str(path.relative_to(PHASE)), 'bytes': len(raw),
              'sha256': hashlib.sha256(raw).hexdigest()}
    expected = REQUIRED.get(str(path.relative_to(EXPANSION))) if path.is_relative_to(EXPANSION) else None
    if expected:
        assert result['sha256'] == expected, 'Frozen required source changed: '+str(path)
    return result


def exported(relative, function):
    path = EXPANSION/relative
    names = {n.name for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert function in names, str(path)+':'+function
    return {**pin(path), 'function': function}


def manifest(package):
    return pin(EXPANSION/package/'MANIFEST.json')


def replace_port(row, package, procedure):
    row.update(callback=exported(package+'/callback.py', 'create_adapter'),
               session_transform=None, source_manifest=manifest(package), runtime_procedure=procedure,
               base_procedure='gnnm_boundary_native', method_settings={'manifest_sha256': manifest(package)['sha256']},
               status='source_prepared_not_runtime_admitted', supported_seeds='all_proposed_source_seeds', pending=[])


def freeze(output_name, supplement=None):
    assert Path(output_name).name == output_name and output_name.endswith('.json')
    target = HERE/output_name
    assert not target.exists(), 'Create a fresh registry; preserve the ancestor'
    audit_path = EXPANSION/'expanded_binding_audit_v1/BINDING_AUDIT.json'
    roster_path = EXPANSION/'expanded_study_design_v1/PROPOSED_ROSTER.json'
    audit = json.loads(audit_path.read_text())
    roster = json.loads(roster_path.read_text())
    families = {x['id']: x['task_family'] for x in roster['cells']}
    entries = copy.deepcopy(audit['entries'])
    context = json.loads((EXPANSION/'link_expansion_ports_BUDDY_v2/CPU_RESULTS.json').read_text())['cases'][0]['result']['materialized_context_manifest']
    assert context['sha256'] == '53bae3eb6c840fbc2ab9a1e68561cd621d120183f990c533665fe3d291bf558a'
    for row in entries:
        cell, method = row['cell_id'], row['method_id']
        row['audited_binding_sha256'] = hashlib.sha256(canonical(row)).hexdigest()
        row['supported_seeds'] = 'all_proposed_source_seeds'
        row['source_binding_only'] = True
        row['pending'] = ['complete native epoch/selection/resume/device and selected-serving qualification',
                          'provider/split and measured resource admission', 'human full-campaign GO']
        if method.startswith('portable_input_'):
            row['method_settings'].update(rank_cap=8, frame_policy='disjoint_if_fit_else_separate_orthonormal_frames')
        if families[cell] == 'ogb' and method == 'task_geometry_alignment_variance':
            row['helper_source_manifests'] = [manifest('molecular'), pin(EXPANSION/'SOURCE_RELEASE.json')]
        if families[cell] != 'link':
            continue
        architecture = cell.removeprefix('collab-').upper()
        if method.startswith('context_affinity_'):
            package, prefix = {
                'NCN': ('context_affinity_link_port_v2', 'context_affinity_link_'),
                'NCNC': ('context_affinity_ncnc_port_v1', 'context_affinity_ncnc_'),
                'BUDDY': ('context_affinity_BUDDY_port_v1', 'context_affinity_BUDDY_'),
            }[architecture]
            version = '_v2' if architecture == 'NCN' else '_v1'
            replace_port(row, package, prefix+method.removeprefix('context_affinity_')+version)
        elif method.startswith('interaction_') and architecture in {'NCN', 'NCNC'}:
            replace_port(row, 'interaction_link_ncn_v2' if architecture == 'NCN' else 'interaction_link_ncnc_v1', method)
        elif method == 'gnnm_private_bn_state_v3' and architecture == 'BUDDY':
            replace_port(row, 'buddy_private_bn_state_v1', method)
            row['method_settings'].update(banks_sha256=pin(EXPANSION/'private_bn_state_v3/banks.py')['sha256'],
                context_manifest=context['path'], context_manifest_sha256=context['sha256'])
            row['supported_seeds'] = [0]
            row['unsupported_seed_reason'] = 'authentic_BUDDY_native_year2007_private_BN_context_only_seed0'
            row['context_manifest_receipt'] = context
        elif architecture == 'BUDDY' and (method.startswith('aggregation_') or method in {'link_endpoint_responsibility_v2', 'link_query_responsibility_v2'}):
            row['method_settings'].update(context_manifest=context['path'], context_manifest_sha256=context['sha256'])
            row['supported_seeds'] = [0]
            row['unsupported_seed_reason'] = 'authentic_BUDDY_native_year2007_C6_responsibility_context_only_seed0'
            row['context_manifest_receipt'] = context
        if architecture == 'BUDDY' and method.startswith('interaction_'):
            row['missing_reason'] = 'separate_native_BUDDY_interaction_factory_not_frozen'
        row['pending'] = ['runtime_v4_link_selected_mean_export_successor_required',
                          'full native epoch/selection/resume/device and selected-serving qualification',
                          'provider/split and measured resource admission', 'human full-campaign GO']
    supplement_pin = None
    if supplement:
        supplement = Path(supplement).resolve()
        supplement_pin = pin(supplement)
        value = json.loads(supplement.read_text())
        assert value['schema'] == 'expanded-binding-explicit-supplement-v1'
        overrides = {(r['cell_id'], r['method_id']): r for r in value['entries']}
        known = {(r['cell_id'], r['method_id']) for r in entries}
        assert set(overrides).issubset(known)
        entries = [copy.deepcopy(overrides.get((r['cell_id'], r['method_id']), r)) for r in entries]
    # Direct bindings and their frozen Python dependencies are recorded once.
    sources = {}
    def remember(row):
        value = pin(PHASE/row['path_relative_to_phase'])
        assert all(value[key] == row[key] for key in ('bytes', 'sha256')), row
        sources[value['path_relative_to_phase']] = value
    packages = set()
    for row in entries:
        for key in ('callback', 'session_transform'):
            value = row.get(key)
            if value:
                source_path = PHASE/value['path_relative_to_phase']
                names = {n.name for n in ast.parse(source_path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                assert value['function'] in names
                actual = {**pin(source_path), 'function': value['function']}
                assert all(actual[k] == value[k] for k in ('bytes', 'sha256', 'function')), value
                remember(value)
        for value in [row.get('source_manifest')] + row.get('helper_source_manifests', []):
            if value:
                remember(value)
                if Path(value['path_relative_to_phase']).name == 'MANIFEST.json':
                    packages.add((PHASE/value['path_relative_to_phase']).parent)
        for name, expected in row.get('method_settings', {}).get('dependency_sha256', {}).items():
            value = pin(EXPANSION/name); assert value['sha256'] == expected; remember(value)
    # These ancestors implement native lifecycles and shared helper laws.
    packages.update(EXPANSION/name for name in ('runtime_v4', 'link', 'molecular', 'private_bn_state_v3',
                                               'context_affinity_v1', 'portable_private_input_v2'))
    for package in sorted(packages):
        mf = package/'MANIFEST.json'
        remember(pin(mf))
        for value in json.loads(mf.read_text()).get('files', []):
            relative = value.get('path', '')
            if not relative.endswith('.py'):
                continue
            path = (PHASE.parents[1]/relative).resolve() if relative.startswith('experiments_iclr/') else (package/relative).resolve()
            actual = pin(path)
            assert actual['sha256'] == value['sha256'] and actual['bytes'] == value['bytes'], str(path)
            remember(actual)
    for name in ('link_native_integration_v1/session.py', 'link_native_integration_v1/epoch.py',
                 'link_policy_v1/policy.py', 'link_policy_v1/BINDINGS.json'):
        remember(pin(EXPANSION/name))
    assert len(entries) == 511 and sum(r['scientific_fits_represented'] for r in entries) == 5110
    result = {'schema': 'expanded-fit-source-registry-v1', 'generation': output_name.removesuffix('.json'),
        'input_audit': pin(audit_path), 'proposed_roster': pin(roster_path),
        'native_settings': pin(EXPANSION/'expanded_study_design_v1/NATIVE_SETTINGS.json'),
        'inherited_cell_bindings': pin(EXPANSION/'job_binding_preparation_v1/CELL_BINDINGS.json'),
        'runtime_manifest': manifest('runtime_v4'), 'runtime_package': EXPANSION.name+'/runtime_v4',
        'runtime_use': 'validate_job metadata only; models/adapters/transforms never imported',
        'supplement': supplement_pin, 'entries': entries, 'source_files': sorted(sources.values(), key=lambda r:r['path_relative_to_phase']),
        'future_generation_contract': 'Use complete explicit replacement entries in a fresh registry, verify all frozen source hashes, then compile to a fresh output directory. Never overwrite registry or generated ancestors.',
        'runnable': False, 'full_campaign_GO': False}
    raw = (json.dumps(result, indent=2, sort_keys=True)+'\n').encode()
    with target.open('xb') as out: out.write(raw)
    print(json.dumps({'registry': str(target), 'sha256': hashlib.sha256(raw).hexdigest(), 'bindings': len(entries), 'pinned_source_files': len(sources)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='REGISTRY_v1.json')
    parser.add_argument('--supplement')
    args = parser.parse_args()
    freeze(args.output, args.supplement)
