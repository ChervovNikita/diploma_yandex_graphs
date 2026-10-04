"""Rehydrate only named exact sealed literature assets through the pinned resolver."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
SSH = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2']
REMOTE = '''
from pathlib import Path
import hashlib,json,subprocess,sys
repo,root,gpu,manifest_pin,seal_pin,relative,size,digest=sys.argv[1:]
repo,root=Path(repo),Path(root);size=int(size)
query=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
assert query.stdout.strip().splitlines()==[gpu],'Wrong one-GPU archive route'
assert repo.resolve()==repo and root.resolve()==root and root.is_relative_to(repo)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'ARCHIVE_MANIFEST.json')==manifest_pin and sha(root/'ARCHIVE_SEAL.json')==seal_pin
manifest=json.loads((root/'ARCHIVE_MANIFEST.json').read_text())
rows={row['archive_path']:row for row in manifest['files']}
assert relative in rows and rows[relative]['bytes']==size and rows[relative]['sha256']==digest
path=root/relative
assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
assert path.stat().st_size==size and sha(path)==digest
with path.open('rb') as stream:
 for block in iter(lambda:stream.read(1048576),b''):sys.stdout.buffer.write(block)
sys.stdout.buffer.flush()
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', action='append', default=[], help='Exact phase-relative asset path from RESOLVER.json')
    parser.add_argument('--packet', action='append', default=[], help='Exact packet name; restore its offloaded raw assets')
    args = parser.parse_args()
    resolver = json.loads((HERE / 'RESOLVER.json').read_text())
    rows = {row['original_phase_path']: row for row in resolver['assets']}
    selected = set(args.path)
    for packet in args.packet:
        assert packet in resolver['packets'], 'Unknown packet'
        selected.update(name for name in rows if name.startswith(packet + '/'))
    assert selected and selected <= rows.keys(), 'Choose known named assets or packets'
    restored = []
    for name in sorted(selected):
        row = rows[name]
        target = PHASE / name
        assert target.resolve().is_relative_to(PHASE) and not target.is_symlink()
        if target.exists():
            assert target.is_file() and target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], 'Existing asset differs; refusing overwrite'
            restored.append({'path': name, 'status': 'ALREADY_EXACT'})
            continue
        command = shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', REMOTE,
            resolver['repository'], resolver['archive_directory'], resolver['GPU_UUID'],
            resolver['archive_manifest_sha256'], resolver['archive_seal_sha256'],
            row['archive_path'], str(row['bytes']), row['sha256']])
        received = subprocess.run([*SSH, resolver['ssh_destination'], command],
                                  capture_output=True, timeout=90)
        assert received.returncode == 0, received.stderr.decode(errors='replace')
        assert len(received.stdout) == row['bytes'] and hashlib.sha256(received.stdout).hexdigest() == row['sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.rehydrate-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(received.stdout)
                stream.flush()
                os.fsync(stream.fileno())
            assert not target.exists(), 'Target appeared during retrieval'
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink()
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
        restored.append({'path': name, 'status': 'REHYDRATED_EXACT', 'bytes': row['bytes'], 'sha256': row['sha256']})
    receipt = {'UTC': datetime.now(timezone.utc).isoformat(), 'assets': restored,
               'route': resolver['ssh_destination'], 'scientific_execution': False}
    path = HERE / ('REHYDRATION_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'assets': len(restored), 'receipt': str(path)}, sort_keys=True))


if __name__ == '__main__':
    main()
