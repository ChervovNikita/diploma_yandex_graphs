"""Explicit new NCN interaction factory, with one fresh native base."""
import copy
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent


def source():
    name = '_explicit_interaction_link_ncn_v1_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,HERE/'session.py')
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return sys.modules[name]


def create_adapter(torch,job,policy,root):
    library = source(); root = Path(root).resolve(); job = copy.deepcopy(job)
    if job.get('family') != 'link' or job.get('procedure') not in library.METHODS \
            or job.get('base_procedure') != 'gnnm_boundary_native' or job.get('TEST_access') is not False:
        raise ValueError('Explicit new NCN interaction and original boundary base with TEST closed required')
    if getattr(policy,'family',None) != 'link' or getattr(policy,'track',None) == 'source_reference':
        raise ValueError('An interaction cannot be called a source-reference baseline')
    if any(job.get(key) for key in ('session_transform','extension_factory','session_extension')):
        raise ValueError('No standalone transform chain; only this explicit native factory')
    if set(job.get('method_settings',{})) != {'manifest_sha256'}:
        raise ValueError('Only exact fixed interaction manifest accepted')
    actual = library.receipt(__file__)
    declared = Path(job['callback_module']); declared = (declared if declared.is_absolute() else root/declared).resolve()
    if declared != Path(actual['path']) or job.get('callback_sha256') != actual['sha256']:
        raise ValueError('Pin the actual outer interaction callback')
    declared_root = job['recipe'].get('_repository_root')
    if declared_root is None or Path(declared_root).resolve() != root:
        raise ValueError('Explicit actual repository root required')
    pins,declaration = library.bindings(root,job['method_settings']['manifest_sha256'])
    inner_path = HERE.parent/'link_native_integration_v1/session.py'; native = library.load(inner_path)
    base_job = copy.deepcopy(job)
    base_job.update(procedure='gnnm_boundary_native',base_procedure='gnnm_boundary_native',
        callback_module=str(inner_path),callback_sha256=library.receipt(inner_path)['sha256'],method_settings={})
    adapter = native.create_adapter(torch,base_job,policy,root)
    adapter.session = library.NCNInteractionSession(adapter.session,job,pins,declaration)
    adapter.model,adapter.optimizers = adapter.session.model,adapter.session.optimizers
    adapter.callback_receipt = actual; adapter.extension_receipt = library.receipt(HERE/'session.py')
    return adapter
