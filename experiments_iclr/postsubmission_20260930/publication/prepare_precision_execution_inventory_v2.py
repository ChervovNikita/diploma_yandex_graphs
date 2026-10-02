"""Snapshot an explicit reviewed source/evidence inventory; exclude scientific binaries."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

PHASE = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.py', '.json', '.md', '.txt', '.log', '.jsonl', '.csv', '.sh', '.patch', '.diff', '.toml', '.html', '.xml'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--readme', type=Path, required=True)
    parser.add_argument('--root', action='append', required=True)
    parser.add_argument('--file', action='append', default=[])
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9_]+', args.snapshot) and re.fullmatch(r'[0-9a-f]{40}', args.head)
    out = PHASE / 'publication' / args.snapshot
    out.mkdir(exist_ok=False)
    selected = set()
    for relative in args.root:
        root = PHASE / relative
        assert root.resolve().is_relative_to(PHASE) and root.is_dir() and not root.is_symlink()
        selected.update(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                        and (p.suffix in EXTENSIONS or p.name == '.gitignore') and '__pycache__' not in p.parts)
    for relative in args.file:
        path = PHASE / relative
        assert path.resolve().is_relative_to(PHASE) and path.is_file() and not path.is_symlink()
        assert path.suffix in EXTENSIONS or path.name == '.gitignore'
        selected.add(path)
    for name in ('PUBLIC_STATUS.md', 'RESEARCH_STATE.md', 'research_ledger.json',
                 'publication/prepare_precision_execution_inventory_v2.py'):
        selected.add(PHASE / name)
    selected.update(p for p in (PHASE / 'protocols').glob('authorized_fetch_15[2-9].json'))
    rows = []
    for path in sorted(selected):
        assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
        data = path.read_bytes()
        assert len(data) < 10 * 1024**2 and b'\0' not in data
        data.decode('utf-8')
        assert not re.search(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----', data)
        if path.suffix == '.py':
            ast.parse(data, filename=str(path))
        if path.suffix == '.json':
            json.loads(data)
        relative = path.relative_to(PHASE)
        snapshot = out / 'files' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open('xb') as stream:
            stream.write(data)
        rows.append(dict(target='experiments_iclr/postsubmission_20260930/' + relative.as_posix(),
                         source_snapshot=snapshot.relative_to(PHASE).as_posix(),
                         sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    readme = args.readme.resolve()
    assert readme.is_relative_to(PHASE)
    data = readme.read_bytes()
    (out / 'files' / 'README.md').write_bytes(data)
    rows.append(dict(target='README.md', source_snapshot=(out / 'files' / 'README.md').relative_to(PHASE).as_posix(),
                     sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    inventory = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
                     branch='codex/postsubmission-research-20260930', expected_head=args.head,
                     files=rows, scientific_utility_claim=False, original_score_files_included=False,
                     credentials_or_binary_artifacts_included=False)
    (out / 'INVENTORY.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), inventory=str(out / 'INVENTORY.json'))))

if __name__ == '__main__':
    main()
