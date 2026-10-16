"""Push the exact all-fit source commit while preserving the pilot checkout.

Root runs this interactively on the verified allocation. ssh-add requests the
key passphrase directly; this helper never accepts or records a passphrase.
Publication does not launch training or assert that any smoke has passed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import socket
import subprocess

ROOT = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
KEY = '/home/jovyan/.ssh/the_github'
ORIGIN = 'git@github.com:ChervovNikita/diploma_yandex_graphs.git'
BRANCH = 'codex/all-fit-smoke-training-20261016'
PARENT = 'ef2f2d4c254e0873c28915d422a5931234c52698'
LIVE_HEAD = '48329b19941ea869c439e5a220a3df0e30aa1237'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_route():
    require(socket.gethostname() == 'anogena-2-0', 'Wrong scientific host')
    observed = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    require(observed == [UUID], 'Wrong GPU inventory')


def fresh_agent_socket(root):
    path = root / ('.git/af-' + secrets.token_hex(4) + '.sock')
    require(path.resolve().is_relative_to(root) and path.parent.is_dir(),
            'Repository-confined agent socket required')
    require(not os.path.lexists(path) and len(str(path).encode()) < 104,
            'Fresh short agent socket required')
    return path


def push_source(commit_receipt, push_receipt):
    verify_route()
    root = ROOT.resolve()
    commit_path, output = Path(commit_receipt).resolve(), Path(push_receipt).resolve()
    require(commit_path.is_relative_to(root) and output.is_relative_to(root),
            'Confined commit and push receipts required')
    require(not os.path.lexists(output) and output.parent.is_dir(), 'Fresh push receipt required')
    commit_raw = commit_path.read_bytes()
    value = json.loads(commit_raw)
    require(value.get('live_head_unchanged') is True and value.get('live_index_unchanged') is True
            and value.get('working_tree_checkout_performed') is False,
            'Expected an isolated source commit')
    require(value.get('publication_only') is True and value.get('training_launch_performed') is False
            and value.get('conditional_full_training_GO') is True,
            'Preserve the conditional all-fit training authorization')
    require(value.get('branch') == BRANCH and value.get('parent') == PARENT
            and value.get('live_head') == LIVE_HEAD, 'Unexpected publication identity')
    require(isinstance(value.get('commit'), str) and re.fullmatch('[0-9a-f]{40}', value['commit']),
            'Exact commit ID required')

    def git(*args, env=None):
        return subprocess.check_output(['git', *args], cwd=root, env=env, text=True).strip()

    ref = 'refs/heads/' + value['branch']
    symbolic = git('symbolic-ref', 'HEAD')
    require(symbolic != ref, 'Publication branch must not be live')

    def verify_local():
        require(git('rev-parse', 'HEAD') == value['live_head']
                and git('symbolic-ref', 'HEAD') == symbolic
                and git('rev-parse', ref) == value['commit'], 'Live or source branch changed')
        require(git('rev-parse', value['commit'] + '^') == value['parent'],
                'Source commit parent changed')
        require(git('remote', 'get-url', '--all', 'origin') == ORIGIN
                and git('remote', 'get-url', '--push', '--all', 'origin') == ORIGIN,
                'Unexpected Git remote')

    verify_local()
    index = Path(git('rev-parse', '--git-path', 'index'))
    if not index.is_absolute():
        index = root / index
    index_before = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    agent_socket = fresh_agent_socket(root)
    raw = subprocess.check_output(['ssh-agent', '-a', str(agent_socket), '-s'], cwd=root, text=True)
    sock = re.search(r'SSH_AUTH_SOCK=([^;]+);', raw)
    pid = re.search(r'SSH_AGENT_PID=(\d+);', raw)
    require(sock and pid and sock[1] == str(agent_socket), 'Unexpected ssh-agent response')
    env = dict(os.environ, SSH_AUTH_SOCK=sock[1], SSH_AGENT_PID=pid[1])
    env['GIT_SSH_COMMAND'] = shlex.join([
        'ssh', '-p', '443', '-o', 'Hostname=ssh.github.com', '-o', 'HostKeyAlias=github.com',
        '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'IdentityAgent=' + str(agent_socket), '-i', KEY,
    ])
    report = {'started_UTC': datetime.now(timezone.utc).isoformat(), 'branch': value['branch'],
              'commit': value['commit'], 'parent': value['parent'], 'live_head': value['live_head'],
              'conditional_full_training_GO': True, 'publication_only': True,
              'training_launch_performed': False, 'verified': False, 'passphrase_recorded': False,
              'commit_receipt_sha256': hashlib.sha256(commit_raw).hexdigest()}
    try:
        subprocess.run(['ssh-add', KEY], cwd=root, env=env, check=True, timeout=180)
        verify_local()
        before = git('ls-remote', '--exit-code', 'origin', ref, env=env).split()
        require(before == [value['parent'], ref], 'Remote publication parent changed')
        push = subprocess.run(
            ['git', 'push', '--no-follow-tags', 'origin', value['commit'] + ':' + ref],
            cwd=root, env=env, capture_output=True, text=True, timeout=180)
        report.update(returncode=push.returncode, stdout=push.stdout, stderr=push.stderr)
        require(push.returncode == 0, 'Source branch push failed')
        remote = git('ls-remote', '--exit-code', 'origin', ref, env=env).split()
        require(remote == [value['commit'], ref], 'Published exact ref verification failed')
        verify_local()
        index_after = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
        require(index_before == index_after, 'Live index changed during push')
        report.update(verified=True, live_head_unchanged=True, live_index_unchanged=True, remote_ref=ref)
    except Exception as error:
        report.update(failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        subprocess.run(['ssh-agent', '-k'], cwd=root, env=env, capture_output=True, check=False)
        report['terminal_UTC'] = datetime.now(timezone.utc).isoformat()
        with output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit-receipt', required=True)
    parser.add_argument('--push-receipt', required=True)
    args = parser.parse_args()
    print(json.dumps(push_source(args.commit_receipt, args.push_receipt)))
