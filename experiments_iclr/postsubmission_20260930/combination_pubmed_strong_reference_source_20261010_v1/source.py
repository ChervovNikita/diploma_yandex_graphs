"""Hash-bound V3 loading and project artifact admission. Stdlib at import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def inside(path, existing=True):
    value = Path(path)
    if not value.is_absolute(): value = PHASE/value
    value = value.resolve(strict=existing)
    if value == PHASE or not value.is_relative_to(PHASE):
        raise ValueError('Artifact must remain inside the bound project phase')
    return value


def bind(row):
    path = inside(row['path'])
    if not path.is_file() or sha(path) != row['sha256']:
        raise ValueError('Bound artifact changed: '+str(row.get('path')))
    if row.get('bytes') is not None and path.stat().st_size != row['bytes']:
        raise ValueError('Bound artifact byte count changed')
    return path


def verify_sources():
    bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']: bind(row)
    folder = PHASE/bindings['prototype_folder']
    manifest = json.loads(bind(bindings['prototype_manifest']).read_text())
    for row in manifest['files']:
        path = (folder/row['path']).resolve(strict=True)
        if not path.is_relative_to(folder) or sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Exact V3 payload changed')
    dependency_bindings = json.loads((folder/'SOURCE_BINDINGS.json').read_text())
    for row in dependency_bindings['dependencies'].values(): bind(row)
    return bindings


def verify_manifest(digest):
    path = HERE/'SOURCE_MANIFEST.json'
    if sha(path) != digest: raise ValueError('Exact sealed strong-reference source manifest required')
    for row in json.loads(path.read_text())['files']:
        file = (HERE/row['path']).resolve(strict=True)
        if not file.is_relative_to(HERE) or sha(file) != row['sha256'] or file.stat().st_size != row['bytes']:
            raise ValueError('Strong-reference source payload changed')
    return verify_sources()


def load_v3(numerical=False):
    """Load exact existing files; no copied or edited method/provider code."""
    bindings = verify_sources()
    folder = PHASE/bindings['prototype_folder']
    saved = {name:sys.modules.get(name) for name in ('native', 'plan')}
    loaded = {}
    try:
        for name in ('native', 'plan') + (('method',) if numerical else ()):
            spec = importlib.util.spec_from_file_location('_strong_reference_exact_v3_'+name, folder/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[name] = module
            if name in saved: sys.modules[name] = module
        return loaded
    finally:
        for name, previous in saved.items():
            if previous is None: sys.modules.pop(name, None)
            else: sys.modules[name] = previous


def fingerprint(array):
    header = json.dumps(dict(dtype=array.dtype.str, shape=list(array.shape)), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header+b'\n'+array.tobytes(order='C')).hexdigest()
