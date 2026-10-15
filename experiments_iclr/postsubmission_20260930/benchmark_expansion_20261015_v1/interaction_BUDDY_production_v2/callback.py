"""Explicit new BUDDY interaction factory, with one fresh native base."""
import copy
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent


def source():
    name = '_explicit_interaction_BUDDY_production_v2_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,HERE/'session.py')
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return sys.modules[name]


def create_adapter(torch,job,policy,root):
    library = source(); root = Path(root).resolve(); job = copy.deepcopy(job)
    if job.get('family') != 'link' or job.get('procedure') not in library.METHODS \
            or job.get('base_procedure') != 'gnnm_boundary_native' or job.get('TEST_access') is not False:
        raise ValueError('Explicit new BUDDY interaction and original boundary base with TEST closed required')
    if getattr(policy,'family',None) != 'link' or getattr(policy,'track',None) == 'source_reference':
        raise ValueError('An interaction cannot be called a source-reference baseline')
    if any(job.get(key) for key in ('session_transform','extension_factory','session_extension')):
        raise ValueError('No standalone transform chain; only this explicit native factory')
    if set(job.get('method_settings',{})) != {'manifest_sha256','provider_manifest_sha256','static_provider_sha256','provider_mode'}:
        raise ValueError('Only exact fixed interaction manifest accepted')
    if str(job.get('recipe',{}).get('architecture','')).upper()!='BUDDY':
        raise ValueError('Native BUDDY recipe required')
    if type(job.get('seed')) is not int or job['seed'] < 0 or job['method_settings']['provider_mode'] != 'production_native':
        raise ValueError('Actual nonnegative fit seed and explicit production_native provider required')
    actual = library.receipt(__file__)
    declared = Path(job['callback_module']); declared = (declared if declared.is_absolute() else root/declared).resolve()
    if declared != Path(actual['path']) or job.get('callback_sha256') != actual['sha256']:
        raise ValueError('Pin the actual outer interaction callback')
    declared_root = job['recipe'].get('_repository_root')
    if declared_root is None or Path(declared_root).resolve() != root:
        raise ValueError('Explicit actual repository root required')
    pins,declaration = library.bindings(root,job['method_settings']['manifest_sha256'])
    inner_path = HERE.parent/'buddy_native_provider_v1/callback.py'; native = library.load(inner_path)
    base_job = copy.deepcopy(job)
    base_job.update(procedure='gnnm_boundary_native',base_procedure='gnnm_boundary_native',
        callback_module=str(inner_path),callback_sha256=library.receipt(inner_path)['sha256'],method_settings={'manifest_sha256':job['method_settings']['provider_manifest_sha256'],'static_provider_sha256':job['method_settings']['static_provider_sha256'],'provider_mode':'production_native'})
    adapter = native.create_adapter(torch,base_job,policy,root)
    adapter.session = library.BUDDYInteractionSession(adapter.session,job,pins,declaration)
    adapter.model,adapter.optimizers = adapter.session.model,adapter.session.optimizers
    adapter.callback_receipt = actual; adapter.extension_receipt = library.receipt(HERE/'session.py')
    return adapter
