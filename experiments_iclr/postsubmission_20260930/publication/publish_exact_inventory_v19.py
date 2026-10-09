"""Keep the v18 inventory contract; transfer through a private raw SSH terminal.

This gateway currently requires a PTY. Only that connection's terminal mode is
changed, before any payload is sent. No host settings, files outside the project,
credentials, inventory semantics, or scientific jobs are changed.
"""
import hashlib
import importlib.util
from pathlib import Path
import shlex
import subprocess
import types

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'publish_exact_inventory_v18.py'
BASE_SHA256 = 'd4c2ab5d481deb7a424c1088d740bf9c3ce86c73db46a6643b2954f183eb3f24'

def terminal_transport(argv, *, input, capture_output, text, timeout):
    assert argv[0] == 'ssh' and capture_output and text and isinstance(input, str)
    fields = shlex.split(argv[-1])
    assert fields[:5] == ['/usr/bin/python3', '-I', '-S', '-B', '-c']
    remote = fields[5]
    old = 'payload=json.loads(zlib.decompress(base64.b64decode(sys.stdin.read())))'
    assert remote.count(old) == 1
    new = '''import termios,tty
terminal_before=termios.tcgetattr(0)
tty.setraw(0)
print('GNNM_INVENTORY_PAYLOAD_READY',flush=True)
encoded=sys.stdin.read(int(sys.argv[4]))
termios.tcsetattr(0,termios.TCSANOW,terminal_before)
payload=json.loads(zlib.decompress(base64.b64decode(encoded)))'''
    fields[5] = remote.replace(old, new)
    fields.append(str(len(input)))
    command = [argv[0], '-tt', *argv[1:-1], shlex.join(fields)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        ready = process.stdout.readline().strip()
        if ready != 'GNNM_INVENTORY_PAYLOAD_READY':
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(command, process.returncode or 1,
                                               ready + '\n' + stdout, stderr)
        try:
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

def main():
    assert BASE_SHA256 and hashlib.sha256(SOURCE.read_bytes()).hexdigest() == BASE_SHA256
    spec = importlib.util.spec_from_file_location('_unchanged_v18_inventory_contract', SOURCE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    base.subprocess = types.SimpleNamespace(run=terminal_transport)
    base.__file__ = __file__  # Receipts identify this actual transport successor.
    base.main()

if __name__ == '__main__':
    main()
