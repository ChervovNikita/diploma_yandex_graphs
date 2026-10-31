"""Stdlib-only syntax and saved-source integrity check; no scientific imports."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    packet = Path(__file__).resolve().parent
    count = 0
    for path in packet.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
        count += 1
    for path in packet.rglob('*.json'):
        json.loads(path.read_text())
    receipts = json.loads((packet / 'evidence/source_receipts.json').read_text())
    for item in receipts:
        data = (packet / item['file']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['file']
        if 'git_blob' in item:
            assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['git_blob']
    print(json.dumps({'python_sources_ast_valid': count,
                      'source_receipts_verified': len(receipts),
                      'scientific_execution': False, 'runtime_qualified': False}))


if __name__ == '__main__':
    main()
