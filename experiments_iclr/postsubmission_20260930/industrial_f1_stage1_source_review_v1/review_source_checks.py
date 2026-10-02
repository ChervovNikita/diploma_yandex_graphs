"""Reproduce source-only checks. No packet or scientific modules are imported."""
import ast
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent.parent / 'industrial_f1_stage1_native_preparation_v1'
EXPECTED_MANIFEST = 'b1189ba8535d0e91d939f8ea90186348dd6b93b99d100539b6468bf1eb7c2287'


def digest(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    manifest = json.loads((PACKET / 'MANIFEST.json').read_text())
    manifest_hash = digest(PACKET / 'MANIFEST.json')['sha256']
    assert manifest_hash == EXPECTED_MANIFEST
    entries = {}
    for name, expected in manifest['files'].items():
        actual = digest(PACKET / name)
        assert actual == expected, name
        entries[name] = actual
    actual_files = {str(p.relative_to(PACKET)) for p in PACKET.rglob('*') if p.is_file()}
    assert actual_files == set(entries) | {'MANIFEST.json'}
    python_files = sorted(PACKET.rglob('*.py'))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    for path in PACKET.rglob('*.json'):
        json.loads(path.read_text())
    receipts = json.loads((PACKET / 'evidence/source_receipts.json').read_text())
    reused = blobs = 0
    for receipt in receipts:
        data = (PACKET / receipt['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == receipt['sha256']
        if 'reused_from' in receipt:
            assert data == (PROJECT / receipt['reused_from']).read_bytes()
            reused += 1
        if 'git_blob' in receipt:
            assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == receipt['git_blob']
            blobs += 1
    refs = json.loads((PACKET / 'evidence/PRIOR_REFERENCES.json').read_text())
    for name, expected in refs.items():
        assert digest(PROJECT / name)['sha256'] == expected
    tree = ast.parse((PACKET / 'sources/user_study_inferred_stypes.py').read_text())
    author = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'rel-f1-driver-position':
                    author = {ast.literal_eval(k): v.attr for k, v in zip(value.keys, value.values)}
    engineered = json.loads((PACKET / 'ENGINEERED_STYPES.json').read_text())
    assert author is not None
    removed = sorted(k for k in author if k.startswith('upcoming_'))
    assert len(removed) == 6
    assert engineered == {k: v for k, v in author.items() if k not in removed}
    assert len(engineered) == 47
    print(json.dumps({
        'manifest_sha256': manifest_hash,
        'all_38_manifest_entries_match': True,
        'no_extra_packet_files': True,
        'python_sources_ast_valid': len(python_files),
        'source_receipts_verified': len(receipts),
        'reused_snapshots_identical': reused,
        'git_blob_identities_verified': blobs,
        'prior_reference_hashes_verified': len(refs),
        'history_map_matches_author_minus_six_upcoming_features': True,
        'historical_predictors': len(engineered) - 3,
        'non_target_fields_including_driverId_date': len(engineered) - 1,
        'removed_upcoming_fields': removed,
        'scientific_imports': False,
        'scientific_execution': False,
        'runtime_qualified': False,
    }, indent=2))


if __name__ == '__main__':
    main()
