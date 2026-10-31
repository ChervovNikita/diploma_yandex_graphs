"""Scalar source identity for the existing genuine composed producer."""
import copy
import hashlib
import json
from pathlib import Path

RUNTIME_NAMES = ('run_fit.py', 'adapters.py', 'bookkeeping.py', 'common.py', 'controllers.py', 'source_rng.py')
BASE_KEYS = {'runtime_sha256', 'policy_sha256', 'callback_sha256', 'extension_sha256', 'dependency_sha256'}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def verify_sources(authority):
    root = Path(authority['project_root']).resolve()
    producer = Path(authority['producer_manifest']['path']).parent
    phase = producer.parent.parent.parent
    for pin in authority['source_files']:
        require(Path(pin['path']).resolve().is_relative_to(root) and receipt(pin['path']) == pin,
                'Composed reporting source receipt differs')
    manifest = json.loads(Path(authority['producer_manifest']['path']).read_text())
    require(manifest['schema'] == 'server51-composed-normalization-runtime-source-v6', 'Wrong genuine producer schema')
    for row in manifest['files']:
        path = (producer / row['path']).resolve()
        require(path.is_relative_to(producer), 'Producer member escapes source package')
        now = receipt(path)
        require(all(now[key] == row[key] for key in ('bytes', 'sha256')), 'Genuine producer member changed')
    bindings = json.loads((producer / 'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        path = (phase / row['path_relative_to_phase']).resolve()
        require(path.is_relative_to(root), 'Genuine producer dependency escapes project')
        now = receipt(path)
        require(all(now[key] == row[key] for key in ('bytes', 'sha256')), 'Genuine producer dependency changed')


def runtime_witness(job, spec, actual, *, verify_dependency_bytes=True):
    """Authenticate observed runtime metadata; never construct an export or adapter."""
    authority = spec['composed_runtime_authority']
    require(authority['schema'] == 'normalization-reporting-runtime-authority-v1'
            and job.get('normalization_composition') == authority['normalization_composition']
            and digest(job) == spec['job_hash'], 'Composed reporting job authority differs')
    require(isinstance(actual, dict) and set(actual) == BASE_KEYS | {'schema', 'composition_producer'}
            and actual['schema'] == 'server51-composed-normalization-genuine-runtime-binding-v2',
            'Actual genuine composed runtime identity is missing or malformed')
    producer = actual['composition_producer']
    require(isinstance(producer, dict) and set(producer) == {'source_manifest', 'entrypoint', 'normalization_source', 'unchanged_helper_runtime', 'link_extension_authority'},
            'Genuine composed producer fields differ')
    require(producer['source_manifest'] == authority['producer_manifest']
            and producer['entrypoint'] == authority['entrypoint']
            and producer['normalization_source'] == authority['normalization_source']
            and producer['link_extension_authority'] == authority['link_extension_authority']
            and actual['runtime_sha256'] == authority['runtime_sha256'], 'Genuine composed source identity differs')
    helper = producer['unchanged_helper_runtime']
    require(isinstance(helper, dict) and set(helper) == BASE_KEYS and helper['runtime_sha256'] == authority['helper_runtime_sha256'],
            'Unchanged native helper source identity differs')
    for key in ('callback_sha256', 'extension_sha256', 'policy_sha256'):
        require(actual[key] == helper[key] == authority[key], 'Composed/native helper binding differs: ' + key)
    require(actual['dependency_sha256'] == helper['dependency_sha256']
            and isinstance(actual['dependency_sha256'], dict) and actual['dependency_sha256'],
            'Actual composed/native dependency map differs or is missing')
    if verify_dependency_bytes:
        root = Path(authority['project_root']).resolve()
        for name, sha in actual['dependency_sha256'].items():
            path = Path(name).resolve()
            require(path.is_relative_to(root) and path.suffix in ('.py', '.json'),
                    'Only genuine project source dependency bytes may be verified')
            require(receipt(path)['sha256'] == sha, 'Observed runtime source dependency bytes changed')
    return {'schema': 'observed-genuine-composed-reporting-runtime-v1',
            'authority_sha256': digest(authority), 'job_hash': spec['job_hash'],
            'runtime_binding': copy.deepcopy(actual), 'native_calls_added': 0}
