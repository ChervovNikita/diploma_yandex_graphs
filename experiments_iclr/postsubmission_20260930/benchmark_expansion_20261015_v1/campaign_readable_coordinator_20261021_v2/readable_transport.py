"""Literal SSH / MacLink run argument vectors; source staging remains plaintext."""
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from remote_backend import HOSTS, ROOTS, exact_write, identity, json_bytes, require

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
LOGIN = 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru'
IDENTITY = '/Users/alex/.ssh/mlspace__private_key_anogena.txt'
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=yes',
               '-o', 'UpdateHostKeys=no']
ALLOCATION_SSH = ['/usr/bin/ssh', '-tt', '-e', 'none', '-p', '2222', '-i', IDENTITY, *SSH_OPTIONS, LOGIN]
ALLOCATION_SCP = ['/usr/bin/scp', '-P', '2222', '-i', IDENTITY, *SSH_OPTIONS]
SERVER51_SSH = ['/usr/bin/ssh', '-S', '/Users/shmelev/Downloads/reverse_maclink_daily/.gnnm51-control.sock',
                *SSH_OPTIONS, 'shmelev@192.168.18.51']
MACLINK = [str(WORKSPACE / 'reverse_maclink/maclink-env/bin/python'), '-B',
           str(WORKSPACE / 'reverse_maclink/maclink.py'), 'run', '--']


def guard_shell(host):
    hostname, gpu_uuids = HOSTS[host]
    lines = ['set -eu', 'test "$(hostname)" = ' + shlex.quote(hostname),
             'test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = ' + shlex.quote('\n'.join(gpu_uuids))]
    if host == 'allocation':
        lines.append('stty raw -echo')  # Preserve JSON stdout bytes on the mandated SSH PTY.
    return '\n'.join(lines) + '\n'


def checked_hash(path, sha):
    require(len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'Invalid checksum')
    return 'test "$(sha256sum -- ' + shlex.quote(str(path)) + ' | cut -d\' \' -f1)" = ' + sha + '\n'


def remote_argv(host, command):
    require(host in ROOTS, 'Unknown route')
    command = guard_shell(host) + command
    argv = [*ALLOCATION_SSH, command] if host == 'allocation' else [*MACLINK, *SERVER51_SSH, command]
    # MacLink sends the argument vector in a JSON frame without transforming commands.
    require(len(json.dumps(argv).encode()) < 900000, 'Command exceeds reviewed MacLink frame bound')
    return argv


class Transport:
    def __init__(self, audit):
        self.audit = Path(audit)
        self.audit.mkdir(parents=True, exist_ok=True)

    def run(self, argv, timeout=180):
        tag = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
        # The exact executable arguments are saved before dispatch; no credentials are included.
        exact_write(self.audit / (tag + '-COMMAND.json'), json_bytes({'argv': argv, 'cwd': str(WORKSPACE)}))
        try:
            result = subprocess.run(argv, cwd=WORKSPACE, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            exact_write(self.audit / (tag + '-RESULT.json'), json_bytes({'status': 'unknown_delivery',
                        'automatic_resubmission': False, 'inspect_before_retry': True}))
            raise RuntimeError('Transport timed out; delivery is uncertain. Inspect recorded command before retry: ' + tag)
        exact_write(self.audit / (tag + '-RESULT.json'), json_bytes({'returncode': result.returncode,
                    'stdout': identity(result.stdout), 'stderr': result.stderr.decode('utf-8', errors='replace')}))
        require(result.returncode == 0, 'Transport failed: ' + result.stderr.decode('utf-8', errors='replace'))
        return result.stdout

    def call(self, host, command, timeout=180):
        return self.run(remote_argv(host, command), timeout)

    def stage_plaintext(self, host, source, destination):
        """Stage named text with SCP on allocation or visible heredocs on server51."""
        source, destination = Path(source), Path(destination)
        raw = source.read_bytes()
        raw.decode('utf-8')
        require(raw.endswith(b'\n'), 'Plaintext staging requires a final newline')
        require(destination.is_relative_to(Path(ROOTS[host])), 'Staging escapes authorized project')
        temporary = destination.with_name(destination.name + '.incoming-' + uuid.uuid4().hex)
        parent, temp, target = map(lambda p: shlex.quote(str(p)), (destination.parent, temporary, destination))
        prepare = ('test "$(realpath -m -- ' + parent + ')" = ' + parent + '\n' +
                   'mkdir -p -- ' + parent + '\n' + 'test ! -e ' + temp + '\n' + 'test ! -L ' + temp + '\n')
        self.call(host, prepare)
        if host == 'allocation':
            # Ordinary SCP carries the actual source file; the preceding guarded call
            # checked the literal scientific destination before this project write.
            self.run([*ALLOCATION_SCP, str(source), LOGIN + ':' + str(temporary)])
        else:
            # run has no stdin channel; visible heredocs keep every source line readable.
            lines, block = raw.decode('utf-8').splitlines(keepends=True), ''
            blocks = []
            for line in lines:
                require(len(line.encode()) <= 48000, 'Plaintext line exceeds bounded heredoc size')
                if block and len((block + line).encode()) > 48000:
                    blocks.append(block)
                    block = ''
                block += line
            if block:
                blocks.append(block)
            offset = 0
            for block in blocks:
                delimiter = 'READABLE_FILE_' + uuid.uuid4().hex
                require(delimiter not in block.splitlines(), 'Heredoc delimiter collision')
                if offset == 0:
                    prefix = 'set -C\ncat > ' + temp
                else:
                    prefix = 'test "$(wc -c < ' + temp + ')" -eq ' + str(offset) + '\ncat >> ' + temp
                self.call(host, prefix + " <<'" + delimiter + "'\n" + block + delimiter + '\n')
                offset += len(block.encode())
        finalize = (checked_hash(temporary, identity(raw)['sha256']) + 'test ! -L ' + target + '\n' +
                    'if test -e ' + target + '; then\n  cmp -- ' + temp + ' ' + target + '\nelse\n' +
                    '  chmod 444 -- ' + temp + '\n  ln -- ' + temp + ' ' + target + '\nfi\n' +
                    'rm -f -- ' + temp + '\n' + checked_hash(destination, identity(raw)['sha256']))
        self.call(host, finalize)
        # Read back the staged text before any invocation. It must be the reviewed bytes.
        staged = self.call(host, 'cat -- ' + target + '\n')
        require(staged == raw, 'Staged plaintext differs from reviewed local source')
        return {'path': str(destination), **identity(raw), 'plaintext_readback_verified': True}
