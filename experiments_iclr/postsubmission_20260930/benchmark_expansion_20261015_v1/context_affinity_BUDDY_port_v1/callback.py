"""Explicit factual-context BUDDY callback around the unchanged native source."""
import copy
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def source():
    name = '_context_affinity_BUDDY_port_v1_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / 'session.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def create_adapter(torch, job, policy, root):
    library = source()
    root, job = Path(root).resolve(), copy.deepcopy(job)
    if job.get('family') != 'link' or job.get('TEST_access') is not False or job.get('procedure') not in library.METHODS:
        raise ValueError('Explicit BUDDY factual-context method and closed TEST required')
    if getattr(policy, 'family', None) != 'link' or getattr(policy, 'track', None) == 'source_reference':
        raise ValueError('Explicit matched native link controller required')
    if any(job.get(key) for key in ('session_transform', 'extension_factory', 'session_extension')):
        raise ValueError('Standalone native BUDDY context callback only')
    settings = set(job.get('method_settings', {}))
    if settings not in ({'manifest_sha256'}, {'manifest_sha256', 'context_manifest', 'context_manifest_sha256'}):
        raise ValueError('Only fixed source manifest and optional authenticated seed0 native-input audit are accepted')
    actual = library.receipt(__file__)
    declared = Path(job['callback_module'])
    declared = (declared if declared.is_absolute() else root / declared).resolve()
    if declared != Path(actual['path']) or job.get('callback_sha256') != actual['sha256']:
        raise ValueError('Pin this actual outer callback explicitly')
    declared_root = job['recipe'].get('_repository_root')
    if declared_root is None or Path(declared_root).resolve() != root:
        raise ValueError('Bind actual repository root explicitly')
    if str(job['recipe'].get('architecture', '')).upper() != 'BUDDY':
        raise ValueError('This port accepts the actual native BUDDY recipe only')
    pins = library.bindings(root, job['method_settings']['manifest_sha256'])
    path = HERE.parent / 'link_native_integration_v1/session.py'
    native = library.load(path)
    base_job = copy.deepcopy(job)
    base_job.update(procedure='gnnm_boundary_native', base_procedure='gnnm_boundary_native',
        callback_module=str(path), callback_sha256=library.receipt(path)['sha256'], method_settings={})
    adapter = native.create_adapter(torch, base_job, policy, root)
    adapter.session = library.BUDDYContextSession(adapter.session, job, pins)
    adapter.model, adapter.optimizers = adapter.session.model, adapter.session.optimizers
    adapter.callback_receipt, adapter.extension_receipt = actual, library.receipt(HERE / 'session.py')
    return adapter
