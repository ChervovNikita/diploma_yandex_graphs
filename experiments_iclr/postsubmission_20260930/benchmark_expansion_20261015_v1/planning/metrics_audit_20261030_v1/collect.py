"""Collect frozen completed scalar receipts; do not change either server."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
EXPANSION = HERE.parent.parent
sys.path.insert(0, str(EXPANSION / 'campaign_readable_coordinator_20261021_v2'))
from readable_transport import Transport, checked_hash
from remote_backend import ROOTS, EXPANSION_REL

parser = argparse.ArgumentParser()
parser.add_argument('host', choices=['allocation', 'server51'])
args = parser.parse_args()
local_source = EXPANSION / 'planning/progress_20261028_v1/inspect_completed_scores.py'
remote_source = Path(ROOTS[args.host]) / EXPANSION_REL / 'planning/progress_20261028_v1/inspect_completed_scores.py'
workers = ['allocation_gpu0'] if args.host == 'allocation' else ['server51_gpu0', 'server51_gpu1']
command = checked_hash(remote_source, hashlib.sha256(local_source.read_bytes()).hexdigest())
support = EXPANSION / 'planning/future_queue_audit_20261024_v1/queue_support.py'
remote_support = Path(ROOTS[args.host]) / EXPANSION_REL / 'planning/future_queue_audit_20261024_v1/queue_support.py'
command += checked_hash(remote_support, hashlib.sha256(support.read_bytes()).hexdigest())
command += shlex.join(['python3', '-B', str(remote_source), '--root', ROOTS[args.host], '--workers', *workers]) + '\n'
raw = Transport(HERE / 'commands').call(args.host, command, timeout=55)
document = json.loads(raw)
destination = HERE / (args.host + '-completed-scores.json')
with destination.open('xb') as stream:
    stream.write(raw)
print(json.dumps({'host': args.host, 'utc': document['utc'], 'completed': len(document['completed_rows']),
                  'scalar_issues': document['scalar_integrity_issues'], 'output': str(destination)}))
