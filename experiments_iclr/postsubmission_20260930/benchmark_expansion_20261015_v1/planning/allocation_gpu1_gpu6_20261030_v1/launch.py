"""Start one additive queue in tmux without changing existing sessions."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
from host_extension import SHARDS, bind, pin, require, ROOT
from partition import source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=SHARDS, required=True)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    bind(args.worker); source(args.source_sha256)
    check = json.loads((HERE / ('PREFLIGHT-' + args.worker.replace('/', '_') + '.json')).read_text())
    require(check['status'] == 'PASS' and check['source'] == pin(HERE / 'MANIFEST.json'), 'Current preflight PASS required')
    require(json.loads((HERE / 'SHARED_FLOCK_PASS.json').read_text())['status'] == 'PASS',
            'Cross-host shared filesystem lock qualification required')
    session = 'gnnm-share-' + args.worker.replace('/', '-') + '-20261030'
    exists = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0
    require(not exists, 'Queue tmux session already exists; inspect, never duplicate it')
    log = HERE / ('OWNER-' + args.worker.replace('/', '_') + '.log')
    command = [str(ROOT / '.venv/bin/python'), '-B', '-u', str(HERE / 'shard_worker.py'),
               '--worker', args.worker, '--source-sha256', args.source_sha256,
               '--process-title-dependency-dir', str(E / 'campaign_process_title_dependency_20261021_v1')]
    subprocess.check_call(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT)])
    subprocess.check_call(['tmux', 'set-option', '-t', session, 'remain-on-exit', 'on'])
    script = shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.check_call(['tmux', 'send-keys', '-t', session, '-l', script])
    subprocess.check_call(['tmux', 'send-keys', '-t', session, 'Enter'])
    result = {'session': session, 'worker': args.worker, 'argv': command,
              'source': pin(HERE / 'MANIFEST.json'), 'log': str(log), 'launched': time.time(),
              'status': 'tmux_queue_dispatched', 'existing_sessions_changed': False}
    (HERE / ('LAUNCH-' + args.worker.replace('/', '_') + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
