"""Run the unchanged validation verifier after unpacking exact trace records.

The upload packs all result/trace records together without changing their bytes.
The complete author stage may keep the same files as ordinary files instead.
"""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, tarfile, tempfile

ROOT=Path(__file__).resolve().parent

def unpack(target):
    original=json.loads((ROOT/'COMPACT_BUNDLE_MANIFEST.json').read_text())['file_sha256']
    expected={n:h for n,h in original.items() if n.startswith('results/')}
    packing=json.loads((ROOT/'PACKED_TRACES_MANIFEST.json').read_text())
    archive=ROOT/'CELL_RECORDS.tar.xz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==packing['archive_sha256']
    assert packing['records']==expected
    found=set()
    with tarfile.open(archive,'r:xz') as source:
        for item in source:
            assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
            assert item.name in expected and item.name not in found
            data=source.extractfile(item).read()
            assert hashlib.sha256(data).hexdigest()==expected[item.name]
            out=target/item.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
            found.add(item.name)
    assert found==set(expected)

if __name__=='__main__':
    if (ROOT/'results').exists():
        code=subprocess.run([sys.executable,str(ROOT/'verify_compact_tuning.py')]).returncode
    else:
        with tempfile.TemporaryDirectory(prefix='graph_trace_check_') as name:
            target=Path(name)/'study'
            shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns('CELL_RECORDS.tar.xz','__pycache__'))
            unpack(target)
            code=subprocess.run([sys.executable,str(target/'verify_compact_tuning.py')]).returncode
    sys.exit(code)
