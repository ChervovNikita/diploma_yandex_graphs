"""JSON/source authorities only; no model imports or runtime admission."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), 'Integration path escapes project')
    return path


def pin(path, root=None):
    path = Path(path).resolve() if root is None else inside(path, root)
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def verify(value, root, *, json_metadata=False):
    require(set(value) == {'path', 'bytes', 'sha256'}, 'Exact immutable receipt required')
    require(pin(value['path'], root) == value, 'Integration authority changed: ' + value['path'])
    if json_metadata:
        require(Path(value['path']).suffix == '.json', 'JSON metadata required')
        return json.loads(Path(value['path']).read_text())
    return Path(value['path'])


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_source(root, manifest_pin):
    require(manifest_pin['path'] == str(HERE / 'MANIFEST.json'), 'Exact integration source required')
    manifest = verify(manifest_pin, root, json_metadata=True)
    require(manifest['schema'] == 'final-scoring-integration-source-v1', 'Wrong integration manifest')
    for row in manifest['files']:
        verify({'path': str(HERE / row['path']), 'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        verify({'path': str(EXPANSION / row['path_relative_to_expansion']), 'bytes': row['bytes'], 'sha256': row['sha256']}, root)
    return manifest
