"""Source/JSON chain audit only; no host, tensor, remote or process calls."""
import hashlib
import ast
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent
PHASE = EXPANSION.parent


def package(value):
    path = (PHASE / value['path_relative_to_phase']).resolve()
    assert path.is_relative_to(EXPANSION)
    raw = path.read_bytes()
    assert len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256']
    manifest = json.loads(raw)
    for row in manifest['files']:
        source = (path.parent / row['path']).resolve()
        assert source.is_relative_to(path.parent)
        data = source.read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    return manifest


def main():
    chain = json.loads((HERE / 'SOURCE_CHAIN.json').read_text())
    expected = {
        'final_selected_scoring_v4': 'f7af2c6b164ead73188f62905f2eb796320c05bfe17029fdde6995624e30f0f6',
        'buddy_private_bn_production_v3': 'bad72516cb3d0093c845846bdef18a3ed5935894843e1c6bdc120ce3833e77e8',
        'final_scoring_integration_v2': '256cb2d8f36959497213753b2b2c63deba1e55abf895809e91d297ca5111d896',
        'combined_campaign_orchestration_v3': 'da16b320ec4425286072dec2e711bc63e27ade63902fbf6fe99745f591d0f632'}
    for name, value in chain['source_manifests'].items():
        assert value['sha256'] == expected[name]
        package(value)
    for value in chain['unchanged_training_authorities'].values():
        package(value)
    package(chain['native_qualification_plan'])
    assert chain['native_qualification_plan']['sha256'] == '0ec833351a8c062a75b714968a54d605ad16d104c1c9a3ea60d0363db0e22cfd'
    past = {'final_scoring_integration_v1': 'c6b453a2b3fdb2dd2adb9a106656c403b82805b7a04d0e3632edf9010a65ad97',
            'combined_campaign_orchestration_v2': 'f45f3659cece95e5d7d9251051ce128bd5a64644da45e0ecbd71d67a24b64cc0'}
    for name, sha in past.items():
        path = EXPANSION / name / 'MANIFEST.json'
        package({'path_relative_to_phase': str(path.relative_to(PHASE)), 'bytes': path.stat().st_size, 'sha256': sha})
    assert (EXPANSION / 'final_selected_scoring_v4/contracts.py').read_bytes() == (EXPANSION / 'final_selected_scoring_v2/contracts.py').read_bytes()
    actual = json.loads((EXPANSION / 'FINAL_POPULATION_ROSTER_RUN_14.json').read_text())['result']
    for field in ('roster', 'registry', 'population_index'):
        assert chain['existing_actual_allocation_artifacts'][field] == actual[field]
        # Remote receipts are origin metadata only, never opened by this check.
        assert actual[field]['path'].startswith('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/')
    assert chain['counts'] == actual['counts'] and chain['final_tasks'] == 6730 and chain['population_presets'] == 27
    assert chain['roster_or_registry_regeneration'] is False
    assert chain['scientific_training_fits_added'] == chain['actual_GO_artifacts_generated'] == chain['portable_or_producer_registry_edits'] == 0
    sys.path.insert(0, str(EXPANSION))
    from final_scoring_integration_v2.common import pin, verify_source
    verify_source(EXPANSION.parent.parent.parent, pin(EXPANSION / 'final_scoring_integration_v2/MANIFEST.json'))
    prepared = json.loads((EXPANSION / 'combined_campaign_orchestration_v3/PREPARATION_STATUS.json').read_text())
    assert prepared['scorer_binding']['sha256'] == expected['final_selected_scoring_v4']
    assert prepared['final_runtime_binding']['sha256'] == expected['final_scoring_integration_v2']
    assert prepared['inference_overlay_binding']['sha256'] == expected['buddy_private_bn_production_v3']
    assert prepared['native_runtime_readiness'] == 'pending' and 'torch' not in sys.modules
    for name in ('qualify_orchestration.py', 'metadata_supervisor_fixture.py'):
        tree = ast.parse((HERE / name).read_text(), filename=name)
        assert all(not item.name.startswith(('torch', 'torch_geometric')) for node in ast.walk(tree) if isinstance(node, ast.Import) for item in node.names)
    driver = (HERE / 'qualify_orchestration.py').read_text()
    assert '--execute-engineering' in driver and "require(len(result['cases']) == 8" in driver
    assert "== 3, 'Exactly three metadata-only supervisors'" in driver
    print(json.dumps({'schema': 'coherent-final-source-chain-check-v1', 'status': 'PASS',
        'source_packages_bound': 4, 'existing_population_contracts': 27, 'existing_roster_tasks': 6730,
        'roster_registry_regenerated': False, 'past_packages_and_portable_preserved': True,
        'source_ABI_unchanged': True, 'executable_engineering_driver_prepared_not_run': True, 'native_runtime_readiness': 'pending',
        'models': 0, 'forwards': 0, 'optimizer_updates': 0, 'supervisors_started': 0,
        'remote_calls': 0, 'TEST_accessed': False, 'full_campaign_GO': False}, indent=2))


if __name__ == '__main__':
    main()
