"""New native-base entrypoint with explicit production input authority."""
import copy
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def source():
    name='_BUDDY_production_native_provider_v1'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,HERE/'provider.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def create_adapter(torch,job,policy,root):
    library=source();root=Path(root).resolve();job=copy.deepcopy(job)
    if job.get('family')!='link' or job.get('procedure')!='gnnm_boundary_native' or job.get('base_procedure')!='gnnm_boundary_native' or job.get('TEST_access') is not False:
        raise ValueError('Native boundary base with closed TEST only; candidate methods require new integrations')
    if getattr(policy,'family',None)!='link' or getattr(policy,'track',None)=='source_reference':raise ValueError('Explicit matched native link controller required')
    if any(job.get(key) for key in ('session_transform','extension_factory','session_extension')):raise ValueError('Explicit provider factory only')
    if set(job.get('method_settings',{}))!={'manifest_sha256','static_provider_sha256','provider_mode'} or job['method_settings']['provider_mode']!='production_native':
        raise ValueError('Exact new provider/static manifests and production_native authority required')
    if type(job.get('seed'))is not int or job['seed']<0 or str(job.get('recipe',{}).get('architecture','')).upper()!='BUDDY':
        raise ValueError('Explicit actual nonnegative BUDDY fit seed required')
    if Path(job['recipe'].get('_repository_root','')).resolve()!=root:raise ValueError('Bind actual repository root')
    actual=library.receipt(__file__);declared=Path(job['callback_module']);declared=(declared if declared.is_absolute() else root/declared).resolve()
    if str(declared)!=actual['path'] or job.get('callback_sha256')!=actual['sha256']:raise ValueError('Pin new production callback')
    pins=library.bindings(root,job['method_settings']['manifest_sha256'])
    inner=HERE.parent/'link_native_integration_v1/session.py';native=library.load(inner)
    base_job=copy.deepcopy(job);base_job.update(callback_module=str(inner),callback_sha256=library.receipt(inner)['sha256'],method_settings={})
    adapter=native.create_adapter(torch,base_job,policy,root)
    adapter.session=library.NativeProviderSession(adapter.session,job,pins)
    adapter.model,adapter.optimizers=adapter.session.model,adapter.session.optimizers
    adapter.callback_receipt=actual;adapter.extension_receipt=library.receipt(HERE/'provider.py')
    return adapter
