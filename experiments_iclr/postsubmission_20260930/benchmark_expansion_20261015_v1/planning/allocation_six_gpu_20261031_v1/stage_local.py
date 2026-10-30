"""Stage inspected plaintext through direct authorized allocation SSH."""
import hashlib
import json
from pathlib import Path
import sys
import time
from host_extension import SHARDS, config, ROOT

HERE = Path(__file__).resolve().parent
REMOTE = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/planning/allocation_six_gpu_20261031_v1'
sys.path.insert(0, str(HERE.parent.parent / 'campaign_readable_coordinator_20261021_v2'))
from readable_transport import Transport


def main():
    (HERE / 'hosts').mkdir(exist_ok=True)
    for shard in SHARDS:
        (HERE / 'hosts' / (shard.replace('/', '_') + '.json')).write_text(json.dumps(config(shard), indent=2) + '\n')
    files = sorted([*HERE.glob('*.py'), *HERE.glob('hosts/*.json')])
    manifest = {'schema': 'allocation-six-device-additive-source-v1', 'files': [
        {'path': str(path.relative_to(HERE)), 'bytes': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]}
    (HERE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(HERE / 'MANIFEST.json')
    receipts = []
    transport = Transport(HERE / 'transport_audit')
    transport.call('allocation', 'test ! -e ' + str(REMOTE / 'PARTITION.json') + '\n')
    for local in files:
        target = REMOTE / local.relative_to(HERE)
        receipt = transport.stage_plaintext('allocation', local, target)
        receipts.append({'source': str(local), **receipt})
    receipt = {'time': time.time(), 'status': 'PASS', 'files': receipts,
               'source_manifest_sha256': hashlib.sha256((HERE / 'MANIFEST.json').read_bytes()).hexdigest()}
    (HERE / 'STAGING.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'files': len(files), 'source_manifest_sha256': receipt['source_manifest_sha256']}))


if __name__ == '__main__':
    main()
