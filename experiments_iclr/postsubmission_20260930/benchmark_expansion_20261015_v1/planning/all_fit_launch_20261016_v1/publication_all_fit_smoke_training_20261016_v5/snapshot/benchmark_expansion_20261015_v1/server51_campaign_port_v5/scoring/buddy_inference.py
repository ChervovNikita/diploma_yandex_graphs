"""Explicit inference attachment for the actual repaired BUDDY v5 producer.

Training metadata is immutable. The v5 session intentionally retains v2's
metadata.port literal because its numerical/session source is byte-identical.
Actual class, callback, manifest and source receipts authenticate v5 here.
Ownership operations are borrowed from the exact frozen v3 inference source.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
PRODUCER = EXPANSION / 'buddy_private_bn_production_v5'
BORROWED = EXPANSION / 'buddy_private_bn_production_v3'
PRODUCER_MANIFEST = '878d0c03650c5fc5613b614b967a3139b422dc54ea64b89132ae1bb99f9bcacb'
PRODUCER_CALLBACK = '811dd1dc2e020fe1028d3f10cabe53a29d613d6fb1923c0e656d8af1ea277e8d'
PRODUCER_SESSION = '4dc3cc611c4247a74e63be9f40157c05c77ff3b48e7aa5b14fea619b38688d59'
METHOD = 'gnnm_private_bn_state_v3'
INFERENCE_PORT = 'server51_campaign_port_v5/scoring'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def authenticate_producer(session, job, root):
    root = Path(root).resolve()
    require(PRODUCER.is_relative_to(root), 'Actual project root required')
    parent = sys.modules.get(type(session).__module__)
    require(parent is not None and Path(getattr(parent, '__file__', '')).resolve() == PRODUCER / 'session.py'
            and type(session) is parent.BUDDYPrivateBN, 'Exact authenticated v5 producer class required')
    require(receipt(PRODUCER / 'session.py')['sha256'] == PRODUCER_SESSION
            and receipt(PRODUCER / 'MANIFEST.json')['sha256'] == PRODUCER_MANIFEST,
            'Actual v5 producer source differs')
    metadata = session.method_metadata
    require(metadata.get('schema') == 'native-BUDDY-private-BN-production-source-v1'
            and metadata.get('method_id') == METHOD and metadata.get('port') == 'buddy_private_bn_production_v2',
            'Preserve the byte-identical producer metadata without relabeling')
    require(job == session.job and job.get('procedure') == METHOD and job.get('TEST_access') is False,
            'Exact restored closed-TEST producer descriptor required')
    callback = Path(job['callback_module'])
    callback = (callback if callback.is_absolute() else root / callback).resolve()
    require(callback == PRODUCER / 'callback.py' and job.get('callback_sha256') == PRODUCER_CALLBACK
            and receipt(callback)['sha256'] == PRODUCER_CALLBACK
            and job['method_settings']['manifest_sha256'] == PRODUCER_MANIFEST,
            'Actual v5 callback and manifest required')
    pins = parent.bindings(root, PRODUCER_MANIFEST)
    require(metadata.get('source_receipts') == pins, 'Restored v5 producer source closure differs')
    return parent, pins


def bind_inference_owner(session, job, root, expected_manifest_sha256):
    parent, producer_pins = authenticate_producer(session, job, root)
    own_manifest = receipt(HERE / 'MANIFEST.json')
    require(own_manifest['sha256'] == expected_manifest_sha256, 'Exact port5 scoring manifest required')
    own = json.loads((HERE / 'MANIFEST.json').read_text())
    for row in own['files']:
        path = (HERE / row['path']).resolve()
        require(path.is_relative_to(HERE), 'Scoring source escapes package')
        now = receipt(path)
        require(all(now[k] == row[k] for k in ('bytes', 'sha256')), 'Scoring source changed')
    borrowed_pin = receipt(BORROWED / 'MANIFEST.json')
    dependencies = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())['files']
    require(any(row['path_relative_to_phase'] == str((BORROWED / 'MANIFEST.json').relative_to(EXPANSION.parent))
                and all(row[k] == borrowed_pin[k] for k in ('bytes', 'sha256')) for row in dependencies),
            'Borrowed inference source must be pinned by port5 scoring')
    name = '_port5_borrowed_BUDDY_inference_' + borrowed_pin['sha256']
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BORROWED / 'session.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    inference = sys.modules[name]
    require(Path(inference.__file__).resolve() == BORROWED / 'session.py', 'Borrowed source module conflict')
    borrowed_pins = inference.bindings(root, borrowed_pin['sha256'])
    require('_inference_role' not in session.__dict__ and 'inference_metadata' not in session.__dict__,
            'Attach exactly one inference API')
    require(session._operation is None and session._state_operation is None and session.manager._active is None,
            'Attach only to an idle canonical restored session')
    session.assert_identity()
    expected_handles = 1 + len(session.manager.names) + 2 * (2 + len(session.manager.names))
    require(len(session._handles) == expected_handles, 'Complete authentic producer guards required')
    for name in ('_route', '_next_batch', '_hydrate'):
        require(getattr(getattr(getattr(session.native, name), '__func__', None), '__module__', None) == parent.__name__,
                'Producer native ownership method source changed')
    original_metadata = copy.deepcopy(session.method_metadata)
    session._inference_role = session._inference_calls = None
    # This is the same separately authorized ownership attachment as the frozen
    # inference API; only the genuine producer authentication above is new.
    for handle in session._handles:
        handle.remove()
    session._handles = []
    for name in ('_inference_buffers', 'inference_operation', 'operation', 'state', 'load_state', 'close'):
        setattr(session, name, types.MethodType(getattr(inference.BUDDYPrivateBN, name), session))
    inference.BUDDYPrivateBN._install_ownership_guards(session)
    require(session.method_metadata == original_metadata, 'Inference attachment cannot rewrite training metadata')
    session.inference_metadata = {'schema': 'bound-BUDDY-private-BN-inference-api-v2',
        'producer_package': PRODUCER.name, 'producer_metadata_port': original_metadata['port'],
        'producer_manifest': receipt(PRODUCER / 'MANIFEST.json'), 'inference_port': INFERENCE_PORT,
        'borrowed_inference_port': BORROWED.name, 'inference_manifest': own_manifest,
        'attachment_source': receipt(__file__), 'borrowed_inference_manifest': borrowed_pin,
        'source_receipts': [*producer_pins, *borrowed_pins, own_manifest, receipt(__file__)],
        'producer_metadata_changed': False}
    return copy.deepcopy(session.inference_metadata)


def owned_model_state(session, root):
    """Read parameters and private banks through the genuine owned state API.

    The native checkpoint call obeys the producer's export hooks. Only model
    state and private banks enter the fingerprint; replay ledgers, sampler,
    optimizer and RNG snapshots are excluded from the comparison.
    """
    authenticate_producer(session, session.job, root)
    declaration = getattr(session, 'inference_metadata', {})
    require(declaration.get('schema') == 'bound-BUDDY-private-BN-inference-api-v2'
            and declaration.get('inference_port') == INFERENCE_PORT
            and declaration.get('producer_package') == PRODUCER.name,
            'Authenticate the owned v5 inference API before fingerprinting')
    saved = session.state()
    require(saved.get('schema') == 'native-BUDDY-private-BN-production-state-v1'
            and saved.get('method') == session.method_metadata, 'Genuine private-BN owned snapshot required')
    provider = saved['native_integration']
    require(provider.get('schema') == 'BUDDY-production-native-provider-state-v1', 'Genuine provider snapshot required')
    native = provider['native_integration']
    require(native.get('schema') == 'actual-native-link-extension-state-v1', 'Genuine native integration snapshot required')
    return {'model': native['native']['model'], 'private_BN_banks': saved['private_BN_banks']}
