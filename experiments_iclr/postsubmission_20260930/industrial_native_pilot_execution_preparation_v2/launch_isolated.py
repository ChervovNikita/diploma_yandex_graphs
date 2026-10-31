"""UNEXECUTED concrete unshare launch. Never installs or builds a runtime.

The runtime image is a separately reviewed OS root tree, not a venv or host root.
It must contain /opt/native/bin/python and its complete libraries, no private
data/home/caches. Worker gets only read-only public/compact-source input files,
read-only this source packet, and a fresh dedicated output volume. No host-root,
home, custodian, raw archive, targets, host network or host-process bind is made.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
from closed_inputs import INPUT_FILES, PUBLIC_INPUT_FILES
from runtime_fingerprint import verify_source_packet


def directory(value):
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('Required directory is missing or a symlink')
    return path.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('isolation-probe', 'qualify', 'run-nine'))
    parser.add_argument('--runtime-image', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--namespace-root', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--gpu-device')
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if os.name != 'posix' or not Path('/proc/self/mountinfo').exists():
        raise RuntimeError('Requires Linux unshare with user/mount/PID/network namespaces')
    runtime, inputs, outputs = map(directory, (args.runtime_image, args.inputs, args.outputs))
    sources = Path(__file__).resolve().parent
    namespace_root = Path(args.namespace_root)
    if namespace_root.is_symlink() or namespace_root.exists():
        raise ValueError('Namespace root must be a new repo-owned directory per phase')
    namespace_root = namespace_root.resolve()
    namespace_root.relative_to(sources.parent)
    if sources in namespace_root.parents or inputs in namespace_root.parents or outputs in namespace_root.parents:
        raise ValueError('Namespace root must be outside readonly sources/inputs and output volume')
    namespace_root.mkdir(mode=0o700)
    if runtime == Path('/') or any(path == runtime or runtime in path.parents for path in (inputs, outputs, sources)):
        raise ValueError('Runtime image cannot be host root or contain input/output/source trees')
    if inputs == outputs or inputs in outputs.parents or outputs in inputs.parents:
        raise ValueError('Worker output and readonly inputs must be disjoint')
    if any(path == sources or sources in path.parents or path in sources.parents for path in (inputs, outputs)):
        raise ValueError('Worker inputs/outputs and source packet must be disjoint')
    verify_source_packet(sources)
    for private in ('home', 'root', 'workspace', 'mnt', 'media', 'srv'):
        path = runtime / private
        if path.is_symlink() or (path.is_dir() and any(path.iterdir())):
            raise ValueError('Runtime image contains a private/data mount directory: ' + private)
    interpreter = runtime / 'opt/native/bin/python'
    # Absolute /opt/... links are intentionally interpreted after chroot.
    if not (interpreter.is_file() or interpreter.is_symlink()):
        raise ValueError('Reviewed runtime-image interpreter is absent')
    actual = set()
    for path in inputs.rglob('*'):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError('Input bundle includes symlink/special file')
        if path.is_file():
            actual.add(path.relative_to(inputs).as_posix())
    if actual != (PUBLIC_INPUT_FILES if args.mode == 'isolation-probe' else INPUT_FILES):
        raise ValueError('Input allowlist differs; archive/raw targets cannot be mounted')
    if any(path.is_symlink() for path in outputs.rglob('*')):
        raise ValueError('Output volume contains a symlink')
    if args.mode in ('isolation-probe', 'qualify') and any(outputs.iterdir()):
        raise ValueError('Probe/qualification starts in a fresh dedicated output volume')
    if args.mode == 'run-nine' and {path.name for path in outputs.iterdir()} != {'qualification'}:
        raise ValueError('Native grid output volume must contain only completed qualification')
    unshare = shutil.which('unshare')
    if unshare is None or shutil.which('mount') is None:
        raise RuntimeError('unshare/mount unavailable; no unisolated fallback')
    import sys
    command = [unshare, '--user', '--map-root-user', '--mount', '--pid', '--fork',
               '--net', '--ipc', '--uts', '--kill-child', sys.executable,
               str(sources / 'namespace_bootstrap.py'), args.mode,
               '--runtime-image', str(runtime), '--inputs', str(inputs), '--outputs', str(outputs),
               '--sources', str(sources), '--namespace-root', str(namespace_root), '--device', args.device]
    if args.device == 'cuda:0':
        import re
        if not args.gpu_device or not re.fullmatch(r'/dev/nvidia[0-9]+', args.gpu_device):
            raise ValueError('Exactly one approved physical GPU device path is required')
        if not args.gpu_uuid or not re.fullmatch(r'GPU-[0-9a-fA-F-]+', args.gpu_uuid):
            raise ValueError('Approved GPU UUID is required')
        command += ['--gpu-device', args.gpu_device, '--gpu-uuid', args.gpu_uuid]
    elif args.gpu_device or args.gpu_uuid:
        raise ValueError('CPU phase does not accept GPU device scope')
    result = subprocess.run(command, check=False, stdin=subprocess.DEVNULL, close_fds=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
