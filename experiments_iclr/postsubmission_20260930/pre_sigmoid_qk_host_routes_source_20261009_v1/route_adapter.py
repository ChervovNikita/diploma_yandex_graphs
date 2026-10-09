"""Inactive host-aware callbacks around the immutable Q/K/F installer.

One identical installer on every route. Mathematical/operator code is untouched.
No launcher, link creation, data acquisition, remote contact or source mutation.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def bound(row):
    path = (PHASE / row['path']).resolve(strict=True)
    require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
            'Changed same-source/role file: ' + row['path'])
    return path


def sealed(root, digest):
    require(sha(root / 'MANIFEST.json') == digest and read(root / 'SEAL.json')['manifest_sha256'] == digest,
            'Changed immutable source seal')
    for row in read(root / 'MANIFEST.json')['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Changed source payload')


def sources(route_id):
    seal = read(HERE / 'SEAL.json')
    require(seal['source_only'] is True and seal['execution_enabled'] is False, 'Inactive route source only')
    sealed(HERE, seal['manifest_sha256'])
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    route = read(HERE / 'ROUTES.json')['routes'][route_id]
    require(PHASE == Path(route['phase']).resolve(strict=True) and socket.gethostname() == route['hostname'],
            'Explicit declared host and normal repository phase')
    sealed(PHASE / pins['math_directory'], pins['math_manifest_sha256'])
    sealed(PHASE / pins['cell_directory'], pins['cell_manifest_sha256'])
    require(sha(PHASE / pins['public_directory'] / 'MANIFEST.json') == pins['public_manifest_sha256'], 'Exact original public manifest')
    for row in read(PHASE / pins['public_directory'] / 'MANIFEST.json')['files']:
        path = (PHASE / pins['public_directory'] / row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE / pins['public_directory']) and sha(path) == row['sha256']
                and path.stat().st_size == row['bytes'], 'Exact original public source')
    for row in pins['source_files']: bound(row)
    return route, pins, seal['manifest_sha256']


def verify_route(route, session=None):
    pythonpath = os.environ.get('PYTHONPATH', '')
    require(Path(sys.executable).resolve() == Path(route['python']).resolve()
            and ([] if not pythonpath else pythonpath.split(os.pathsep)) == route['PYTHONPATH']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == route['GPU_uuid'], 'Explicit interpreter/path/physical GPU route')
    if route.get('python_sha256'):
        require(sha(Path(sys.executable).resolve()) == route['python_sha256'], 'Saved normal 77 interpreter bytes')
    if session is not None:
        require(str(session.device) == 'cuda:0' and str(session.torch.__version__) == route['torch']
                and session.np.__version__ == route['numpy'] and session.torch.cuda.device_count() == 1
                and session.torch.cuda.get_device_name(0) == route['GPU_name'], 'Same torch/numpy/float32 native contract')
        if route.get('provider_site'):
            site = Path(route['provider_site']).resolve()
            require(Path(session.torch.__file__).resolve().is_relative_to(site)
                    and Path(session.np.__file__).resolve().is_relative_to(site), 'Saved normal 77 provider origin')


def verify_roles(route, pins):
    """Runtime-only hashes. Root stages explicit in-repo aliases separately."""
    result = {}
    for kind, row in pins['roles'].items():
        alias = PHASE / row['path']
        target = PHASE / route['author_roles'][kind]
        require(alias.resolve(strict=True).is_relative_to(PHASE)
                and target.resolve(strict=True).is_relative_to(PHASE), 'Role alias/target remain in normal phase')
        if route['hostname'] == 'peptide' and alias.is_symlink():
            require(alias.resolve() == target.resolve(), 'Explicit 77 same-hash author-role link target')
        require(target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], 'Existing author role bytes')
        result[kind] = bound(row)
    return result


def make_session(*, route_id, operator, kind, seed, device='cuda:0', later_execution_authorized=False):
    require(later_execution_authorized is True, 'Inactive route installer; separate root authorization required')
    route, pins, manifest = sources(route_id)
    verify_route(route)
    math = module(PHASE / pins['math_directory'] / 'operator_adapter.py', '_route_private_immutable_math')
    paths = dict(native=bound(pins['native']), public_manifest=PHASE / pins['public_directory'] / 'MANIFEST.json',
                 runtime=HERE / 'ROUTES.json')
    public_root = PHASE / pins['public_directory']
    # Replace only private-instance file/host callbacks. make_session, install,
    # _attach, _global_forward and _streamed_f_step execute unchanged source bytes.
    math._sources = lambda: (paths, public_root, pins['math_manifest_sha256'])
    math._route = lambda paths, session=None: verify_route(route, session)
    session = math.make_session(operator=operator, kind=kind, seed=seed, device=device,
                                later_execution_authorized=True)
    session.operator_binding.update(route_id=route_id, route_adapter_manifest_sha256=manifest,
        route_adapter_program_sha256=sha(__file__), route_catalog_sha256=sha(HERE / 'ROUTES.json'),
        physical_GPU_uuid=route['GPU_uuid'], hostname=route['hostname'], runtime=route,
        torch_module_path=str(Path(session.torch.__file__).resolve()),
        numpy_module_path=str(Path(session.np.__file__).resolve()),
        mathematical_source_changed=False, cross_host_bitwise_parity_required=False)
    return session
