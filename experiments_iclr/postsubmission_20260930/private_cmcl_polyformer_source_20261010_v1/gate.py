"""Sealed source/review binding, checked before the session's numerical imports."""
import json
from .caps import CLOSED
from .native import HERE, PHASE, sha, require


def source_gate(identity, caps=CLOSED):
    caps.require('source_bound')
    seal = json.loads((HERE/'SEAL.json').read_text())
    require(seal['source_only'] is True and seal['execution_enabled'] is False
            and seal['manifest_sha256'] == sha(HERE/'MANIFEST.json')
            and identity['source_seal_sha256'] == sha(HERE/'SEAL.json')
            and identity['root_review_sha256'] == caps.root_review_sha256,
            'Explicit immutable source/review identity')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
        path = (HERE/row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Unchanged sealed candidate source')
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['dependencies'].values():
        path = (PHASE/row['path']).resolve(strict=True)
        require(path.is_relative_to(PHASE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Unchanged exact native/factor/state source')
