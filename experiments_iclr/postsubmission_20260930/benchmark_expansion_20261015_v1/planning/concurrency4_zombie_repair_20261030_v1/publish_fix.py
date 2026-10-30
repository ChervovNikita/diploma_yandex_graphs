"""Publish additive scheduler source without changing the live checkout or index."""
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
EXPANSION = ROOT / 'experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1'
HERE = EXPANSION / 'planning/concurrency4_zombie_repair_20261030_v1'
PARENT = '11f897a336dc1f64f75c0d6bb73f5ebee36d5ef2'
BRANCH = 'codex/zombie-handover-fix-20261030'
ORIGIN = 'git@github.com:ChervovNikita/diploma_yandex_graphs.git'
KEY = '/home/jovyan/.ssh/the_github'


def require(value, message):
    if not value:
        raise ValueError(message)


def git(*args, env=None, data=None):
    return subprocess.check_output(['git', *args], cwd=ROOT, env=env, input=data, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    require(socket.gethostname() == 'anogena-2-0', 'Wrong host')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
            == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Wrong GPU')
    require(git('remote', 'get-url', '--all', 'origin') == ORIGIN
            and git('remote', 'get-url', '--push', '--all', 'origin') == ORIGIN, 'Unexpected remote')
    head, symbolic = git('rev-parse', 'HEAD'), git('symbolic-ref', 'HEAD')
    live_index = Path(git('rev-parse', '--git-path', 'index'))
    if not live_index.is_absolute():
        live_index = ROOT / live_index
    index_before = digest(live_index)
    isolated = HERE / 'publication.index'
    require(not isolated.exists(), 'Publication already prepared; inspect its receipt')
    require(git('rev-parse', PARENT) == PARENT, 'Publication parent missing')
    env = dict(os.environ, GIT_INDEX_FILE=str(isolated))
    git('read-tree', PARENT, env=env)
    package = EXPANSION / 'campaign_concurrency_20261029_v1'
    manifest = json.loads((package / 'MANIFEST.json').read_text())
    require(digest(package / 'MANIFEST.json') == '82b056580848c7074d6d9e2a3aa9a13f298c2123256d0d9f07c9139ba6bbe912', 'Source changed')
    files = [HERE / name for name in ('finish_handover.py', 'test_finish_handover.py', 'STATUS.md')]
    for row in manifest['files']:
        require(digest(package / row['path']) == row['sha256'], 'Source member changed')
    files += [Path(__file__)]
    entries = []
    for path in files:
        relative = str(path.relative_to(ROOT))
        blob = git('hash-object', '-w', '--', str(path))
        entries.append('100644 ' + blob + '\t' + relative)
    git('update-index', '--index-info', env=env, data='\n'.join(entries) + '\n')
    tree = git('write-tree', env=env)
    commit = git('commit-tree', tree, '-p', PARENT, '-m', 'Fix drained queue handover for defunct supervisor processes')
    ref = 'refs/heads/' + BRANCH
    git('update-ref', ref, commit, '0' * 40)
    report = {'branch': BRANCH, 'commit': commit, 'parent': PARENT,
              'live_head': head, 'live_index_unchanged': digest(live_index) == index_before,
              'working_tree_checkout_performed': False, 'files': [str(p.relative_to(ROOT)) for p in files],
              'push_verified': False, 'credential_saved': False}
    (HERE / 'COMMIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'commit': commit, 'branch': BRANCH}), flush=True)
    agent_socket = ROOT / ('.git/c4-' + secrets.token_hex(4) + '.sock')
    require(len(str(agent_socket).encode()) < 104 and not agent_socket.exists(), 'Invalid agent socket')
    raw = subprocess.check_output(['ssh-agent', '-a', str(agent_socket), '-s'], text=True)
    sock = re.search(r'SSH_AUTH_SOCK=([^;]+);', raw)
    pid = re.search(r'SSH_AGENT_PID=(\d+);', raw)
    require(sock and pid and sock[1] == str(agent_socket), 'Unexpected agent response')
    agent_env = dict(os.environ, SSH_AUTH_SOCK=sock[1], SSH_AGENT_PID=pid[1])
    agent_env['GIT_SSH_COMMAND'] = shlex.join(['ssh', '-p', '443', '-o', 'Hostname=ssh.github.com',
        '-o', 'HostKeyAlias=github.com', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes',
        '-o', 'UpdateHostKeys=no', '-o', 'StrictHostKeyChecking=yes', '-o', 'IdentitiesOnly=yes',
        '-o', 'IdentityAgent=' + str(agent_socket), '-i', KEY])
    try:
        subprocess.run(['ssh-add', KEY], cwd=ROOT, env=agent_env, check=True, timeout=180)
        require(not git('ls-remote', 'origin', ref, env=agent_env), 'New publication branch already exists')
        subprocess.run(['git', 'push', '--no-follow-tags', 'origin', commit + ':' + ref],
                       cwd=ROOT, env=agent_env, check=True, timeout=180)
        require(git('ls-remote', '--exit-code', 'origin', ref, env=agent_env).split() == [commit, ref], 'Remote ref mismatch')
        require(git('rev-parse', 'HEAD') == head and git('symbolic-ref', 'HEAD') == symbolic
                and digest(live_index) == index_before, 'Live checkout/index changed')
        report.update(push_verified=True, live_head_unchanged=True, live_index_unchanged=True)
    finally:
        subprocess.run(['ssh-agent', '-k'], env=agent_env, capture_output=True)
        (HERE / 'PUSH.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
