"""Standalone prepared BUDDY private-BN callback; no automatic execution."""
import copy
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent


def source():
    name = '_buddy_private_bn_production_v3_source'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,HERE/'session.py')
        module = importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def create_adapter(torch,job,policy,root):
    library=source();root=Path(root).resolve();job=copy.deepcopy(job)
    if job.get('family')!='link' or job.get('procedure')!=library.METHOD or job.get('base_procedure')!='gnnm_boundary_native' or job.get('TEST_access') is not False:
        raise ValueError('Exact standalone private-BN link method and closed TEST required')
    if getattr(policy,'family',None)!='link' or getattr(policy,'track',None)=='source_reference':
        raise ValueError('Matched native link controller required')
    if any(job.get(key) for key in ('session_transform','extension_factory','session_extension')):
        raise ValueError('No implicit composition with standalone private-BN callback')
    if set(job.get('method_settings',{}))!={'manifest_sha256','banks_sha256','provider_manifest_sha256','static_provider_sha256','provider_mode'}:
        raise ValueError('Pin exactly this port, frozen v3 banks and explicit compact native provider')
    if str(job.get('recipe',{}).get('architecture','')).upper()!='BUDDY':
        raise NotImplementedError('This separate source port accepts native BUDDY only')
    if Path(job['recipe'].get('_repository_root','')).resolve()!=root:
        raise ValueError('Bind the explicit actual repository root')
    if type(job.get('seed')) is not int or job['seed'] < 0 or job['method_settings']['provider_mode'] != 'production_native':
        raise ValueError('Actual nonnegative fit seed and explicit production_native provider required')
    actual=library.receipt(__file__);declared=Path(job['callback_module']);declared=(declared if declared.is_absolute() else root/declared).resolve()
    if str(declared)!=actual['path'] or job.get('callback_sha256')!=actual['sha256']:
        raise ValueError('Pin this actual outer callback')
    pins=library.bindings(root,job['method_settings']['manifest_sha256'])
    native_path=HERE.parent/'buddy_native_provider_v2/callback.py';native=library.load(native_path)
    base_job=copy.deepcopy(job);base_job.update(procedure='gnnm_boundary_native',base_procedure='gnnm_boundary_native',
        callback_module=str(native_path),callback_sha256=library.receipt(native_path)['sha256'],method_settings={'manifest_sha256':job['method_settings']['provider_manifest_sha256'],'static_provider_sha256':job['method_settings']['static_provider_sha256'],'provider_mode':'production_native'})
    adapter=native.create_adapter(torch,base_job,policy,root)
    adapter.session=library.BUDDYPrivateBN(adapter.session,job,pins)
    adapter.model,adapter.optimizers=adapter.session.model,adapter.session.optimizers
    adapter.callback_receipt=actual;adapter.extension_receipt=library.receipt(HERE/'session.py')
    return adapter
