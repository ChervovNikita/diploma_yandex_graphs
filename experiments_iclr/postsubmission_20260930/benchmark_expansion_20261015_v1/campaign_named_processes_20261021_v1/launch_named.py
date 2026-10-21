"""Set the campaign process title, then run a named plaintext Python script."""
import argparse
import hashlib
import importlib
from pathlib import Path
import runpy
import sys

PROCESS_TITLE = 'AI Auto Research Pipeline'
DEPENDENCY_FILES = (
    ('setproctitle/__init__.py', 1409, '9b0cf124bc1815788b818152ae715262c6ed9105fdc67f7eb686446d4b0334cc'),
    ('setproctitle/_setproctitle.cpython-311-x86_64-linux-gnu.so', 69064,
     'ee9f9a53d1424fa4aba918a2d7375c1074e05625973e7b7fcdde0189eafa128c'),
)


def load_library(dependency_dir=None):
    receipts = []
    previous_path = list(sys.path)
    try:
        if dependency_dir is not None:
            directory = Path(dependency_dir).resolve()
            if not directory.is_dir():
                raise ValueError('Process title dependency directory is missing')
            for relative, size, digest in DEPENDENCY_FILES:
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError('Process title dependency escapes project directory')
                raw = path.read_bytes()
                if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                    raise ValueError('Verified process title dependency changed: ' + str(path))
                receipts.append({'path': str(path), 'bytes': size, 'sha256': digest})
            sys.path.insert(0, str(directory))
        library = importlib.import_module('setproctitle')
        if dependency_dir is not None:
            if Path(library.__file__).resolve() != directory / 'setproctitle/__init__.py' or library.__version__ != '1.3.6':
                raise ValueError('Wrong process title library imported')
    finally:
        sys.path[:] = previous_path
    return library, receipts


def verify_dependency(dependency_dir):
    """Read/import the pinned dependency without naming or starting any process."""
    library, receipts = load_library(dependency_dir)
    return {'version': library.__version__, 'dependency_files': receipts, 'process_title': PROCESS_TITLE}


def set_process_title(dependency_dir=None):
    library, _ = load_library(dependency_dir)
    library.setproctitle(PROCESS_TITLE)
    if library.getproctitle() != PROCESS_TITLE:
        raise RuntimeError('Exact campaign process title was not applied')


def run_native(native_argv, dependency_dir=None):
    if not native_argv or not Path(native_argv[0]).is_file():
        raise ValueError('A named readable native Python script is required')
    set_process_title(dependency_dir)
    sys.argv[:] = native_argv
    # Match Python's script invocation lookup directory. No environment change.
    sys.path[0] = str(Path(native_argv[0]).resolve().parent)
    runpy.run_path(native_argv[0], run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-dir')
    parser.add_argument('native_argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    native = args.native_argv
    if native and native[0] == '--':
        native = native[1:]
    run_native(native, args.dependency_dir)


if __name__ == '__main__':
    main()
