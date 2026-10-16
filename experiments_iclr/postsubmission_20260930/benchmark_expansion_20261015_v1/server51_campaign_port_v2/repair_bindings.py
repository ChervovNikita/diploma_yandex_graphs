"""Finite BUDDY execution-source rebinding; canonical scientific fields stay fixed."""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent.parent
REPAIR = HERE.parent / 'buddy_provider_portability_20261016_v1'
REPAIR_SHA = '2d9c0eca0ddb5c738b477acb63d143f2fe5e864803c326c983f5f79dcd4dd712'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relative_receipt(path):
    raw = path.read_bytes()
    return {'path_relative_to_phase': str(path.relative_to(PHASE)),
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


@lru_cache(maxsize=1)
def repair_metadata():
    """Pin the reviewed finite patchset once per metadata/worker process."""
    manifest_pin = relative_receipt(REPAIR / 'MANIFEST.json')
    require(manifest_pin['sha256'] == REPAIR_SHA, 'Frozen BUDDY repair manifest differs')
    manifest = json.loads((REPAIR / 'MANIFEST.json').read_text())
    rows = [manifest_pin]
    for row in manifest['files']:
        path = REPAIR / row['path']
        require(path.resolve().is_relative_to(REPAIR), 'Repair source escapes package')
        actual = relative_receipt(path)
        require(all(actual[k] == row[k] for k in ('bytes', 'sha256')), 'Repair source member differs')
        rows.append(actual)
    patchset = json.loads((REPAIR / 'PATCHSET.json').read_text())
    for row in patchset['files']:
        path = PHASE / row['path_relative_to_phase']
        require(path.resolve().is_relative_to(PHASE) and relative_receipt(path) == row,
                'BUDDY successor source bytes differ')
        rows.append(copy.deepcopy(row))
    delta = json.loads((REPAIR / 'REGISTRY_DELTA.json').read_text())
    require(delta['descriptor_fields_changed'] == ['callback_module', 'callback_sha256',
            'method_settings.manifest_sha256', 'method_settings.provider_manifest_sha256'],
            'Only reviewed BUDDY deployment fields may change')
    return delta, rows, manifest_pin


def repair_sources():
    return copy.deepcopy(repair_metadata()[1])


def replacement_for(job, root, phase_relative, *, successor=False):
    delta, _, _ = repair_metadata()
    phase = PurePosixPath(root) / phase_relative
    key = 'new_callback' if successor else 'old_callback'
    return next((row for row in delta['replacements']
                 if job.get('callback_module') == str(phase / row[key]['path_relative_to_phase'])), None)


def rebind_buddy(job, root, phase_relative):
    """Return a fresh descriptor with four reviewed source fields changed only."""
    row = replacement_for(job, root, phase_relative)
    if row is None:
        return copy.deepcopy(job), None
    delta, _, manifest = repair_metadata()
    require(job.get('family') == 'link' and job.get('recipe', {}).get('architecture', '').upper() == 'BUDDY'
            and job.get('TEST_access') is False, 'Only explicit closed-TEST BUDDY descriptors may rebind')
    settings = job.get('method_settings', {})
    require(job['callback_sha256'] == row['old_callback']['sha256']
            and settings.get('manifest_sha256') == row['old_manifest']['sha256']
            and settings.get('provider_manifest_sha256') == delta['provider']['old_manifest']['sha256']
            and settings.get('static_provider_sha256') == delta['provider']['static_provider']['sha256'],
            'Canonical BUDDY callback/provider identity differs')
    require(not any(job.get(k) for k in ('session_transform', 'extension_factory', 'session_extension')),
            'The repaired BUDDY callback owns its unchanged session directly')
    result = copy.deepcopy(job)
    changes = [
        {'pointer': '/callback_module', 'from': job['callback_module'],
         'to': str(PurePosixPath(root) / phase_relative / row['new_callback']['path_relative_to_phase'])},
        {'pointer': '/callback_sha256', 'from': job['callback_sha256'], 'to': row['new_callback']['sha256']},
        {'pointer': '/method_settings/manifest_sha256', 'from': settings['manifest_sha256'],
         'to': row['new_manifest']['sha256']},
        {'pointer': '/method_settings/provider_manifest_sha256', 'from': settings['provider_manifest_sha256'],
         'to': delta['provider']['new_manifest']['sha256']},
    ]
    for change in changes:
        fields = change['pointer'].lstrip('/').split('/')
        target = result
        for name in fields[:-1]:
            target = target[name]
        target[fields[-1]] = change['to']
    return result, {'schema': 'BUDDY-execution-source-rebinding-v1', 'repair_manifest': copy.deepcopy(manifest),
                    'field_changes': changes, 'canonical_scientific_fields_changed': False,
                    'old_package': row['old_package'], 'new_package': row['new_package'],
                    'actual_session_source': copy.deepcopy(row['new_session'])}


def actual_link_extension(job):
    """Resolve the declared authentic callback extension without model imports."""
    import importlib.util
    path = HERE / 'runtime_link/link_authority.py'
    spec = importlib.util.spec_from_file_location('_port2_reporting_link_authority', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    callback = PurePosixPath(job['callback_module'])
    entry = module.REGISTRY.get(callback.parent.name)
    if entry is None:
        return None
    require(callback.name == entry['callback_name'] and job['callback_sha256'] == entry['callback_sha256']
            and job.get('method_settings', {}).get('manifest_sha256') == entry['manifest_sha256'],
            'Reporting callback must match the authentic link source registry')
    return {'path': str(callback.with_name(entry['extension_name']))}
