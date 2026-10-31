"""UNEXECUTED trusted mount bootstrap, standard-library only before chroot.

Scientific modules import only after exec inside the new allowlist root. The
bootstrap necessarily starts with host path visibility to create mounts; it
opens no archive/targets/model. Do not run it directly: parent unshare creates
user,mount,PID,network,IPC,UTS namespaces first. Any mount failure aborts.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess


def drop_capabilities():
    """Remove chroot/mount escape privileges before scientific Python exec."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    last_cap = int(Path('/proc/sys/kernel/cap_last_cap').read_text())
    for cap in range(last_cap + 1):
        if libc.prctl(24, cap, 0, 0, 0) != 0:  # PR_CAPBSET_DROP
            raise OSError(ctypes.get_errno(), 'Cannot drop capability bounding set')
    if libc.prctl(47, 4, 0, 0, 0) != 0:  # PR_CAP_AMBIENT/CLEAR_ALL
        raise OSError(ctypes.get_errno(), 'Cannot clear ambient capabilities')
    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32),
                   ('inheritable', ctypes.c_uint32)]
    header, data = Header(0x20080522, 0), (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), 'Cannot drop active capabilities')
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), 'Cannot set no_new_privs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('isolation-probe', 'qualify', 'run-nine'))
    for name in ('runtime-image', 'inputs', 'outputs', 'sources', 'namespace-root', 'device'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--gpu-device')
    parser.add_argument('--gpu-uuid')
    args = parser.parse_args()
    if os.getpid() != 1 or os.getuid() != 0:
        raise RuntimeError('Bootstrap requires unshare mapped-root PID namespace')
    mount = shutil.which('mount')
    if mount is None:
        raise RuntimeError('Mount utility missing')
    root = Path(args.namespace_root)

    def call(*arguments):
        subprocess.run([mount, *map(str, arguments)], check=True, stdin=subprocess.DEVNULL, close_fds=True)

    def bind(source, destination, readonly):
        call('--bind', source, destination)
        if readonly:
            call('-o', 'remount,bind,ro', destination)

    call('--make-rprivate', '/')
    bind(args.runtime_image, root, True)
    for name in ('inputs', 'sources', 'outputs', 'tmp', 'dev', 'proc'):
        if not (root / name).is_dir() or (root / name).is_symlink():
            raise RuntimeError('Runtime tree lacks an empty mount target: ' + name)
    bind(args.inputs, root / 'inputs', True)
    bind(args.sources, root / 'sources', True)
    bind(args.outputs, root / 'outputs', False)
    call('-t', 'tmpfs', '-o', 'mode=1777', 'tmpfs', root / 'tmp')
    call('-t', 'tmpfs', '-o', 'mode=755', 'tmpfs', root / 'dev')
    for name in ('null', 'zero', 'random', 'urandom'):
        target = root / 'dev' / name
        target.touch()
        bind(Path('/dev') / name, target, False)
    if args.device == 'cuda:0':
        if not args.gpu_device or not args.gpu_uuid:
            raise RuntimeError('One approved GPU scope missing')
        for source in (args.gpu_device, '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools'):
            path = Path(source)
            if not path.exists():
                raise RuntimeError('Required CUDA device node missing: ' + source)
            target = root / 'dev' / path.name
            target.touch()
            bind(path, target, False)
    call('-t', 'proc', 'proc', root / 'proc')
    os.chroot(root)
    os.chdir('/outputs')
    for path in ('/tmp/home', '/tmp/cache'):
        Path(path).mkdir(mode=0o700)
    drop_capabilities()
    # Clear inherited host credentials/configuration and close inherited fds.
    os.closerange(3, 65536)
    environment = {'PATH': '/opt/native/bin:/usr/bin', 'PYTHONNOUSERSITE': '1',
                   'HOME': '/tmp/home', 'XDG_CACHE_HOME': '/tmp/cache',
                   'PYTHONDONTWRITEBYTECODE': '1', 'DGLBACKEND': 'pytorch',
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = ['/opt/native/bin/python', '/sources/run_native.py', args.mode, '--device', args.device]
    if args.device == 'cuda:0':
        environment['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
        command += ['--expected-gpu-uuid', args.gpu_uuid]
    os.execve(command[0], command, environment)


if __name__ == '__main__':
    main()
