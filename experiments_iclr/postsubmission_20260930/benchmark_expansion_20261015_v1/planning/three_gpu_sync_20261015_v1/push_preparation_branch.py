"""Push an exact preparation branch while preserving the pilot's live HEAD.

Run interactively on the verified allocation. The key passphrase is requested
by ssh-add, never accepted in arguments or saved in this program's receipts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
KEY = '/home/jovyan/.ssh/the_github'


def main():
    if socket.gethostname() != 'anogena-2-0':
        raise ValueError('Wrong scientific host')
    observed = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    if observed != [UUID]:
        raise ValueError('Wrong GPU inventory')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit-receipt', required=True)
    parser.add_argument('--push-receipt', required=True)
    args = parser.parse_args()
    commit_path, output = Path(args.commit_receipt).resolve(), Path(args.push_receipt).resolve()
    if not commit_path.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Confined commit receipt and fresh output required')
    value = json.loads(commit_path.read_text())
    if not value['live_head_unchanged'] or not value['live_index_unchanged']:
        raise ValueError('Expected an isolated preparation commit')

    def git(*args, env=None):
        return subprocess.check_output(['git', *args], cwd=ROOT, env=env, text=True).strip()

    ref = 'refs/heads/' + value['branch']
    live = git('rev-parse', 'HEAD')
    if live != value['parent'] or git('symbolic-ref', 'HEAD') == ref or git('rev-parse', ref) != value['commit']:
        raise ValueError('Live or preparation branch changed')
    if git('remote', 'get-url', 'origin') != 'git@github.com:ChervovNikita/diploma_yandex_graphs.git':
        raise ValueError('Unexpected Git remote')
    agent_socket = ROOT / '.git/p51.sock'
    if agent_socket.exists() or len(str(agent_socket).encode()) >= 104:
        raise ValueError('Fresh confined agent socket required')
    raw = subprocess.check_output(['ssh-agent', '-a', str(agent_socket), '-s'], cwd=ROOT, text=True)
    sock = re.search(r'SSH_AUTH_SOCK=([^;]+);', raw)
    pid = re.search(r'SSH_AGENT_PID=(\d+);', raw)
    if not sock or not pid or sock[1] != str(agent_socket):
        raise ValueError('Unexpected ssh-agent response')
    env = dict(os.environ, SSH_AUTH_SOCK=sock[1], SSH_AGENT_PID=pid[1])
    env['GIT_SSH_COMMAND'] = shlex.join([
        'ssh', '-p', '443', '-o', 'Hostname=ssh.github.com', '-o', 'HostKeyAlias=github.com',
        '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'IdentityAgent=' + str(agent_socket), '-i', KEY,
    ])
    report = {'started_UTC': datetime.now(timezone.utc).isoformat(), 'branch': value['branch'],
              'commit': value['commit'], 'full_campaign_GO': False, 'verified': False,
              'passphrase_recorded': False,
              'commit_receipt_sha256': hashlib.sha256(commit_path.read_bytes()).hexdigest()}
    try:
        subprocess.run(['ssh-add', KEY], cwd=ROOT, env=env, check=True, timeout=180)
        push = subprocess.run(['git', 'push', 'origin', ref + ':' + ref], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=180)
        report.update(returncode=push.returncode, stdout=push.stdout, stderr=push.stderr)
        if push.returncode:
            raise RuntimeError('Preparation branch push failed')
        remote = git('ls-remote', '--exit-code', 'origin', ref, env=env).split()
        if remote != [value['commit'], ref] or git('rev-parse', 'HEAD') != live:
            raise ValueError('Published ref or unchanged live HEAD verification failed')
        report.update(verified=True, live_head_unchanged=True, remote_ref=ref)
    except Exception as error:
        report.update(failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        subprocess.run(['ssh-agent', '-k'], cwd=ROOT, env=env, capture_output=True, check=False)
        report['terminal_UTC'] = datetime.now(timezone.utc).isoformat()
        with output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
