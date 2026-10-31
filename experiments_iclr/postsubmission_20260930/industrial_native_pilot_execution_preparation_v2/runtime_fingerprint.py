"""Future metadata/dependency closure observation; never installs packages."""
from __future__ import annotations
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys

TOP_REQUIREMENTS = {
    'torch': '==2.4.0+cu118', 'dgl': '==2.4.0+cu118', 'numpy': '==2.2.4', 'scipy': '==1.15.2',
    'scikit-learn': '==1.6.1', 'delu': '==0.0.26', 'PyYAML': '==6.0.2',
    'pandas': '==2.3.0', 'packaging': '==25.0', 'torchdata': '==0.8.0',
}


def fingerprint():
    if platform.python_version() != '3.12.9':
        raise RuntimeError('Candidate pilot runtime follows published Python 3.12.9 metadata; '
                           'a different runtime needs a source-versioned qualification policy')
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    queue = [Requirement(name + spec) for name, spec in TOP_REQUIREMENTS.items()]
    distributions = {}
    edges = []
    while queue:
        requirement = queue.pop(0)
        if requirement.marker is not None and not requirement.marker.evaluate({'extra': ''}):
            continue
        name = canonicalize_name(requirement.name)
        distribution = metadata.distribution(requirement.name)
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            raise RuntimeError('Dependency version violates requirement: ' + str(requirement))
        if name in distributions:
            continue
        record = distribution.read_text('RECORD')
        meta = distribution.read_text('METADATA')
        if record is None or meta is None:
            raise RuntimeError('Wheel metadata closure unavailable: ' + name)
        distributions[name] = {
            'version': distribution.version,
            'record_sha256': hashlib.sha256(record.encode()).hexdigest(),
            'metadata_sha256': hashlib.sha256(meta.encode()).hexdigest(),
        }
        for raw in distribution.requires or ():
            child = Requirement(raw)
            if child.marker is None or child.marker.evaluate({'extra': ''}):
                edges.append({'parent': name, 'requires': str(child)})
                queue.append(child)
    return {'schema': 'native-runtime-closure-v2', 'python': platform.python_version(),
            'implementation': platform.python_implementation(), 'machine': platform.machine(),
            'platform': platform.platform(), 'executable': sys.executable,
            'distributions': dict(sorted(distributions.items())), 'dependency_edges': edges,
            'note': 'Runtime resolved transitive closure is fingerprinted; exact wheel/native-library '
                    'availability and operational image custody are unqualified during source preparation.'}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_runtime_image(root='/'):
    """Hash every declared library file; do not traverse mounted worker volumes.

    This checks image integrity and qualification/grid identity. A manifest made
    by the assembler cannot independently certify the custody of its inputs.
    """
    root = Path(root)
    manifest_path = root / 'RUNTIME_IMAGE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'candidate-allowlist-runtime-image-v2':
        raise RuntimeError('Runtime image manifest schema differs')
    excluded = {'inputs', 'sources', 'outputs', 'tmp', 'dev', 'proc'}
    declared = {}
    for item in manifest['payload']:
        relative = Path(item['path'])
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or relative.parts[0] in excluded or item['path'] in declared):
            raise RuntimeError('Runtime image manifest path is outside library scope')
        declared[item['path']] = item
    actual = {}
    def walk(directory):
        for path in directory.iterdir():
            relative = path.relative_to(root).as_posix()
            if path.parent == root and (path.name in excluded or path.name == manifest_path.name):
                continue
            if path.is_symlink():
                actual[relative] = {'path': relative, 'symlink': os.readlink(path)}
                # In chroot this resolves within the allowlist OS root; dangling
                # or circular links signal an incomplete relocatable image.
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise RuntimeError('Unresolved runtime-image symlink: ' + relative) from error
                resolved_relative = resolved.relative_to(root)
                if resolved_relative.parts and resolved_relative.parts[0] in excluded:
                    raise RuntimeError('Runtime library symlink enters a worker mount')
            elif path.is_dir():
                walk(path)
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as file:
                    for block in iter(lambda: file.read(1024 * 1024), b''):
                        digest.update(block)
                actual[relative] = {'path': relative, 'bytes': path.stat().st_size,
                                    'sha256': digest.hexdigest()}
            else:
                raise RuntimeError('Special file in runtime library image: ' + relative)
    walk(root)
    if actual != declared:
        raise RuntimeError('Runtime library image file/symlink payload differs from its manifest')
    return {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'payload_identity_sha256': canonical_sha(actual), 'payload_files': len(actual),
            'actual_library_payload_verified': True,
            'custody_limit': 'Hashes bind actual image bytes; dedicated source-prefix custody is a root obligation.'}


def verify_source_packet(root):
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    seal = json.loads((root / 'SEAL.json').read_text())
    if hashlib.sha256((root / 'MANIFEST.json').read_bytes()).hexdigest() != seal['manifest_sha256']:
        raise RuntimeError('Source packet manifest hash differs')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise RuntimeError('Source packet contains a symlink or special file')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != {item['path'] for item in manifest['payload']} | {'MANIFEST.json', 'SEAL.json'}:
        raise RuntimeError('Source packet has undeclared or missing files')
    for item in manifest['payload']:
        if Path(item['path']).is_absolute() or '..' in Path(item['path']).parts:
            raise RuntimeError('Source manifest has an invalid relative path')
        path = root / item['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
            raise RuntimeError('Source packet file differs: ' + item['path'])
    return {'manifest_sha256': seal['manifest_sha256'],
            'seal_sha256': hashlib.sha256((root / 'SEAL.json').read_bytes()).hexdigest(),
            'matrix_sha256': hashlib.sha256((root / 'CONFIG_MATRIX.json').read_bytes()).hexdigest()}
