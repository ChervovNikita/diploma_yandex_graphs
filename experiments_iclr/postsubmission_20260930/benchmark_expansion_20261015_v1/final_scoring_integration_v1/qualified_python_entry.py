"""Reuse actual lexical Python validation; allow only fixed final observer."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    portable = Path(__file__).resolve().parent.parent / 'portable_dispatcher_v3'
    sys.path.insert(0, str(portable))
    from host_binding import guard, load_config
    config = guard()
    from qualified_python_entry import validate_runtime_environment
    root = Path(config['project_root'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True); parser.add_argument('--environment-sha256', required=True)
    parser.add_argument('--entrypoint', required=True); parser.add_argument('--entrypoint-sha256', required=True)
    parser.add_argument('runtime_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    environment = Path(args.environment).resolve()
    if not environment.is_relative_to(root) or hashlib.sha256(environment.read_bytes()).hexdigest() != args.environment_sha256:
        raise ValueError('Exact original lexical environment required')
    validate_runtime_environment(json.loads(environment.read_text()))
    entry = Path(args.entrypoint).resolve()
    if entry != Path(__file__).with_name('runtime_entry.py').resolve() or not entry.is_relative_to(root) \
            or hashlib.sha256(entry.read_bytes()).hexdigest() != args.entrypoint_sha256:
        raise ValueError('Only fixed source-pinned final observer entry allowed')
    if not args.runtime_args or args.runtime_args[0] != '--':
        raise ValueError('Explicit fixed final arguments required')
    sys.argv = [str(entry), *args.runtime_args[1:]]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
