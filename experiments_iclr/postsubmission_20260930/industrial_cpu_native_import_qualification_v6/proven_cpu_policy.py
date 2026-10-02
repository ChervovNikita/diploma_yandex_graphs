"""UNEXECUTED CPU-only Landlock ABI1/seccomp fixture diagnostic.

Run only through the reviewed root wrapper. Fresh system Python starts in six
verified namespaces; trusted stdlib startup precedes restriction. No GPU,
project data, archive, label, model or scientific package is accessed.
"""
from __future__ import annotations
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import stat
import sys
import time

LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long
LIBC.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                      ctypes.c_ulong, ctypes.c_ulong]
LIBC.prctl.restype = ctypes.c_int
NAMESPACES = ('user', 'mnt', 'pid', 'net', 'ipc', 'uts')
PUBLIC = b'public synthetic fixture\n'
EXCLUDED = b'excluded synthetic fixture\n'
FIXED_RUNS = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/'
                  'experiments_iclr/postsubmission_20260930/industrial_content_sandbox_prototype_v1/root_runs')
# ABI1's complete handled filesystem mask; writable grants exclude special
# file creation and symlinks. No ABI2+ rights are requested.
FS_ALL_ABI1 = (1 << 13) - 1
FS_READ = (1 << 2) | (1 << 3)
FS_OUTPUT = FS_READ | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8)
EPERM = errno.EPERM


def raw(number, *args):
    ctypes.set_errno(0)
    converted = [ctypes.c_long(value) if isinstance(value, int) else value
                 for value in args]
    value = LIBC.syscall(ctypes.c_long(number), *converted)
    return value, ctypes.get_errno() if value < 0 else 0


def checked_syscall(number, *args):
    value, error = raw(number, *args)
    if value < 0:
        raise OSError(error, 'Required syscall failed: ' + str(number))
    return value


class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


class Filter(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint16), ('jt', ctypes.c_uint8),
                ('jf', ctypes.c_uint8), ('k', ctypes.c_uint32)]


class Program(ctypes.Structure):
    _fields_ = [('len', ctypes.c_uint16), ('filter', ctypes.POINTER(Filter))]


def prctl(option, argument=0):
    ctypes.set_errno(0)
    if LIBC.prctl(option, argument, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Required prctl failed: ' + str(option))


def drop_capabilities():
    last = int(Path('/proc/sys/kernel/cap_last_cap').read_text())
    if not 0 <= last <= 63:
        raise RuntimeError('Unexpected capability range')
    for capability in range(last + 1):
        prctl(24, capability)  # PR_CAPBSET_DROP; active SETPCAP remains until capset.
    ctypes.set_errno(0)
    if LIBC.prctl(47, 4, 0, 0, 0) != 0:  # PR_CAP_AMBIENT_CLEAR_ALL
        raise OSError(ctypes.get_errno(), 'Cannot clear ambient capabilities')
    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Data(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32),
                    ('inheritable', ctypes.c_uint32)]
    header, data = Header(0x20080522, 0), (Data * 2)()
    checked_syscall(126, ctypes.byref(header), ctypes.byref(data))  # capset
    prctl(38, 1)  # PR_SET_NO_NEW_PRIVS
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                  if ':' in line)
    for key in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'):
        if int(fields[key].strip(), 16) != 0:
            raise RuntimeError('Capability retained: ' + key)
    if fields['NoNewPrivs'].strip() != '1':
        raise RuntimeError('no_new_privs was not observed')


def restrict_landlock(readonly, writable):
    abi = checked_syscall(444, ctypes.c_void_p(), 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock ABI1 unavailable')
    attr = Ruleset(FS_ALL_ABI1)
    ruleset = checked_syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    try:
        for path, rights in [(p, FS_READ) for p in readonly] + [(p, FS_OUTPUT) for p in writable]:
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                rule = PathBeneath(rights, descriptor)
                checked_syscall(445, ruleset, 1, ctypes.byref(rule), 0)
            finally:
                os.close(descriptor)
        checked_syscall(446, ruleset, 0)
    finally:
        os.close(ruleset)
    return abi


def seccomp_program(pin):
    """Forward-only classic BPF; fixed x86_64 ABI and occupied FD pin.

    Kernel FD parameters are int/unsigned int, so comparisons intentionally
    use their low 32 bits. No pathname or pointed-to-structure assumptions.
    """
    instructions, labels = [], {}
    def label(name):
        if name in labels:
            raise RuntimeError('Duplicate BPF label')
        labels[name] = len(instructions)
    def emit(code, value=0, yes=None, no=None):
        instructions.append((code, value, yes, no))
    def load(offset):
        emit(0x20, offset)  # BPF_LD | BPF_W | BPF_ABS
    def branch(code, value, yes, no):
        emit(code, value, yes, no)
    load(4)  # seccomp_data.arch
    branch(0x15, 0xC000003E, 'arch_ok', 'kill')
    label('arch_ok')
    load(0)  # seccomp_data.nr
    branch(0x45, 0x40000000, 'kill', 'native')  # reject x32 syscall bit
    label('native')
    branch(0x35, 453, 'deny', 'known')  # refuse later/unreviewed syscall additions
    label('known')
    denied = {
        59, 322,                 # execve, execveat
        76,                      # truncate; ftruncate(77) retained by FD provenance
        90, 91, 268, 452,         # chmod/fchmod/fchmodat/fchmodat2
        92, 93, 94, 260,          # chown/fchown/lchown/fchownat
        132, 235, 261, 280,       # utime/utimes/futimesat/utimensat
        188, 189, 190, 197, 198, 199,  # set/remove xattr variants
        101, 310, 311, 438,       # ptrace/process_vm_*/pidfd_getfd
        425, 426, 427,           # io_uring setup/enter/register
        155, 161, 165, 166, 272, 308,  # pivot_root/chroot/mount/umount2/unshare/setns
        428, 429, 430, 431, 432, 442,  # open_tree/move_mount/fsopen/fsconfig/fsmount/mount_setattr
        436, 437,                # close_range, openat2
        47, 299,                 # recvmsg, recvmmsg: no foreign-FD reception
    }
    for index, number in enumerate(sorted(denied)):
        continuation = 'deny_next_' + str(index)
        branch(0x15, number, 'deny', continuation)
        label(continuation)
    dispatch = {2: 'open_flags', 257: 'openat_flags', 16: 'ioctl_pin',
                3: 'pin_mutation', 72: 'pin_mutation', 33: 'replace_pin',
                292: 'replace_pin', 41: 'unix_socket', 53: 'unix_socket'}
    for index, (number, target) in enumerate(sorted(dispatch.items())):
        continuation = 'dispatch_next_' + str(index)
        branch(0x15, number, target, continuation)
        label(continuation)
    emit(0x05, 'allow')  # BPF_JA
    for name, offset in [('open_flags', 24), ('openat_flags', 32)]:
        label(name)
        load(offset)
        branch(0x45, os.O_TRUNC, name + '_mode', 'allow')
        label(name + '_mode')
        emit(0x54, os.O_ACCMODE)  # BPF_ALU | BPF_AND | BPF_K
        branch(0x15, os.O_RDONLY, 'deny', 'allow')
    label('ioctl_pin')
    load(16)  # args[0].low
    branch(0x15, pin, 'allow', 'deny')
    label('pin_mutation')
    load(16)
    branch(0x15, pin, 'deny', 'allow')  # every fcntl on pin is refused
    label('replace_pin')
    load(24)  # dup2/dup3 args[1]: newfd
    branch(0x15, pin, 'deny', 'allow')
    label('unix_socket')
    load(16)
    branch(0x15, 1, 'deny', 'allow')  # AF_UNIX; existing sockets were closed
    label('deny')
    emit(0x06, 0x00050000 | EPERM)  # SECCOMP_RET_ERRNO
    label('kill')
    emit(0x06, 0x80000000)  # SECCOMP_RET_KILL_PROCESS
    label('allow')
    emit(0x06, 0x7FFF0000)  # SECCOMP_RET_ALLOW
    assembled = []
    for index, (code, value, yes, no) in enumerate(instructions):
        if code == 0x05:
            value = labels[value] - index - 1
        jt = labels[yes] - index - 1 if yes is not None else 0
        jf = labels[no] - index - 1 if no is not None else 0
        if not 0 <= jt <= 255 or not 0 <= jf <= 255 or not 0 <= value <= 0xFFFFFFFF:
            raise RuntimeError('Invalid forward BPF jump')
        assembled.append(Filter(code, jt, jf, value))
    array = (Filter * len(assembled))(*assembled)
    program = Program(len(assembled), array)
    checked_syscall(317, 1, 0, ctypes.byref(program))  # seccomp SET_MODE_FILTER
    return len(assembled)


def reviewed_stdio_and_pin():
    null = os.stat('/dev/null')
    if not stat.S_ISCHR(null.st_mode) or null.st_rdev != os.makedev(1, 3):
        raise RuntimeError('Linux /dev/null must be character device major 1 minor 3')
    incoming = os.fstat(0)
    if not stat.S_ISCHR(incoming.st_mode) or incoming.st_rdev != null.st_rdev:
        raise RuntimeError('stdin must be the reviewed /dev/null endpoint')
    if fcntl.fcntl(0, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise RuntimeError('stdin must be readonly')
    for descriptor in (1, 2):
        if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
            raise RuntimeError('stdout/stderr must be trusted wrapper pipes')
        if fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE != os.O_WRONLY:
            raise RuntimeError('stdout/stderr must be write-only')
    pin = os.open('/dev/null', os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    observed = os.fstat(pin)
    if pin < 3 or not stat.S_ISCHR(observed.st_mode) or observed.st_rdev != null.st_rdev:
        raise RuntimeError('Only a fresh validated /dev/null character FD may be pinned')
    # No finite cutoff: close_range must work or the prototype fails closed.
    if pin > 3:
        checked_syscall(436, 3, pin - 1, 0)
    checked_syscall(436, pin + 1, 0xFFFFFFFF, 0)
    return pin, observed.st_rdev


