"""Stage inspected plaintext through direct authorized allocation SSH."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from host_extension import SHARDS, config, ROOT

HERE = Path(__file__).resolve().parent
REMOTE = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_gpu1_gpu6_20261030_v1'
SSH = ['ssh', '-T', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt', '-p', '2222',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']


def main():
    (HERE / 'hosts').mkdir(exist_ok=True)
    for shard in SHARDS:
        (HERE / 'hosts' / (shard.replace('/', '_') + '.json')).write_text(json.dumps(config(shard), indent=2) + '\n')
    files = sorted([*HERE.glob('*.py'), *HERE.glob('hosts/*.json')])
    manifest = {'schema': 'allocation-three-device-additive-source-v1', 'files': [
        {'path': str(path.relative_to(HERE)), 'bytes': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]}
    (HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(HERE / 'MANIFEST.json')
    receipts = []
    for local in files:
        target = REMOTE / local.relative_to(HERE)
        command = ('test "$(hostname)" = anogena-2-0 && '
                   'test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = GPU-44039938-fd82-41d2-fefd-de71514e2fac && '
                   'test ! -e ' + shlex.quote(str(REMOTE / 'PARTITION.json')) + ' && '
                   'mkdir -p ' + shlex.quote(str(target.parent)) + ' && '
                   'cat > ' + shlex.quote(str(target)) + ' && sha256sum ' + shlex.quote(str(target)))
        raw = local.read_bytes()
        result = subprocess.run([*SSH, command], input=raw, capture_output=True, timeout=45)
        digest = hashlib.sha256(raw).hexdigest()
        if result.returncode or result.stdout.decode().split()[0] != digest:
            raise RuntimeError('Plaintext staging failed for ' + str(local) + ': ' + result.stderr.decode())
        receipts.append({'argv': [*SSH, command], 'source': str(local), 'destination': str(target),
                         'sha256': digest, 'bytes': len(raw), 'exit_code': result.returncode})
    receipt = {'time': time.time(), 'status': 'PASS', 'files': receipts,
               'source_manifest_sha256': hashlib.sha256((HERE / 'MANIFEST.json').read_bytes()).hexdigest()}
    (HERE / 'STAGING.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'files': len(files), 'source_manifest_sha256': receipt['source_manifest_sha256']}))


if __name__ == '__main__':
    main()
