"""Explicit new NCNC callback; frozen runtime and native bridge are unchanged."""
import copy
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent


def source():
    name = '_link_expansion_ports_NCNC_v1_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE/'session.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def create_adapter(torch, job, policy, root):
    library = source(); root = Path(root).resolve(); job = copy.deepcopy(job)
    if job.get('family') != 'link' or job.get('TEST_access') is not False or job.get('procedure') not in library.METHODS:
        raise ValueError('Explicit new link method and closed TEST required')
    if getattr(policy, 'family', None) != 'link' or getattr(policy, 'track', None) == 'source_reference':
        raise ValueError('Explicit matched native link controller required')
    if any(job.get(key) for key in ('session_transform','extension_factory','session_extension')):
        raise ValueError('Standalone new native callback; extension combinations are not supported')
    if set(job.get('method_settings', {})) != {'manifest_sha256'}:
        raise ValueError('Pin only the exact fixed new port manifest; no implicit settings')
    actual = library.receipt(__file__)
    declared = Path(job['callback_module']); declared = (declared if declared.is_absolute() else root/declared).resolve()
    if declared != Path(actual['path']) or job.get('callback_sha256') != actual['sha256']:
        raise ValueError('Pin this actual outer callback explicitly')
    declared_root = job['recipe'].get('_repository_root')
    if declared_root is None or Path(declared_root).resolve() != root:
        raise ValueError('Bind actual repository root explicitly')
    if str(job['recipe'].get('architecture', '')).upper() != 'NCNC':
        raise NotImplementedError('This separate callback accepts the actual NCNC recipe only')
    pins = library.bindings(root, job['method_settings']['manifest_sha256'])
    native_path = HERE.parent/'link_native_integration_v1/session.py'
    native = library.load(native_path)
    base_job = copy.deepcopy(job)
    base_job.update(procedure='gnnm_boundary_native', base_procedure='gnnm_boundary_native',
        callback_module=str(native_path), callback_sha256=library.receipt(native_path)['sha256'], method_settings={})
    adapter = native.create_adapter(torch, base_job, policy, root)
    adapter.session = library.NCNCExpandedSession(adapter.session, job, pins)
    adapter.model, adapter.optimizers = adapter.session.model, adapter.session.optimizers
    adapter.callback_receipt = actual
    adapter.extension_receipt = library.receipt(HERE/'session.py')
    return adapter
