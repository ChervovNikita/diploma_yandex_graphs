"""Use the v19 inventory transport with hostname/GPU checks before project reads."""
import hashlib
import importlib.util
from pathlib import Path
import shlex
import types

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'publish_exact_inventory_v19.py'

def main():
    spec = importlib.util.spec_from_file_location('_v19_literal_guarded_transport', SOURCE)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    original_transport = prior.terminal_transport

    def guarded_transport(argv, **kwargs):
        fields = shlex.split(argv[-1])
        assert fields[:5] == ['/usr/bin/python3', '-I', '-S', '-B', '-c']
        marker = 'repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];operation=sys.argv[3]'
        assert fields[5].count(marker) == 1
        replacement = """import socket
assert socket.gethostname()=='anogena-2-0'
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==[sys.argv[2]]
repo=pathlib.Path(sys.argv[1]);uuid=sys.argv[2];operation=sys.argv[3]"""
        fields[5] = fields[5].replace(marker, replacement)
        return original_transport([*argv[:-1], shlex.join(fields)], **kwargs)

    base_path = HERE / 'publish_exact_inventory_v18.py'
    assert hashlib.sha256(base_path.read_bytes()).hexdigest() == prior.BASE_SHA256
    base_spec = importlib.util.spec_from_file_location('_unchanged_v18_inventory_contract', base_path)
    base = importlib.util.module_from_spec(base_spec)
    base_spec.loader.exec_module(base)
    base.subprocess = types.SimpleNamespace(run=guarded_transport)
    base.__file__ = __file__
    base.main()

if __name__ == '__main__':
    main()
