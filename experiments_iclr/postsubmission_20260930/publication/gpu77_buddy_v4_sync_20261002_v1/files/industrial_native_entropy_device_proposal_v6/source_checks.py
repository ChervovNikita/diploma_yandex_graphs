"""Source and synthetic stdlib mocks only; no launcher/device/kernel calls."""
from __future__ import annotations
import ast
import ctypes
import errno
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
V5 = BASE / 'industrial_cpu_native_import_qualification_v5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


def synthetic_case(node, name, *, approved=True, device_mode=stat.S_IFCHR,
                   major=1, minor=9, regular_mode=stat.S_IFREG,
                   open_errno=None, add_errno=None, abi=1):
    calls, opens, closes, rules = [], [], [], []
    objects = {}
    # These are Linux x86_64 flags, independent of the local OS constants.
    flags = dict(O_PATH=0x200000, O_CLOEXEC=0x80000,
                 O_NOFOLLOW=0x20000, O_DIRECTORY=0x10000)

    def mock_open(path, value):
        path = str(path)
        opens.append((path, value))
        if path == '/dev/urandom':
            assert value == flags['O_PATH'] | flags['O_CLOEXEC'] | flags['O_NOFOLLOW']
            if open_errno is not None:
                raise OSError(open_errno, 'synthetic device open failure')
            mode, device = device_mode, (major << 8) | minor
        elif path in ('runtime', 'listing', 'output'):
            assert value == flags['O_PATH'] | flags['O_CLOEXEC'] | flags['O_NOFOLLOW'] | flags['O_DIRECTORY']
            mode, device = stat.S_IFDIR, 0
        elif path == 'library':
            assert value == flags['O_PATH'] | flags['O_CLOEXEC'] | flags['O_NOFOLLOW']
            mode, device = regular_mode, 0
        else:
            raise AssertionError('Unreviewed mock path: ' + path)
        descriptor = 100 + len(opens)
        objects[descriptor] = SimpleNamespace(st_mode=mode, st_rdev=device, path=path)
        return descriptor

    def mock_close(descriptor):
        closes.append(descriptor)
        assert descriptor == 90 or descriptor in objects

    def mock_syscall(number, *args):
        calls.append(number)
        if number == 444:
            return abi if args[-1] == 1 else 90
        if number == 445:
            rule = args[2]._obj
            descriptor = rule.parent_fd
            path = objects[descriptor].path
            rules.append({'path': path, 'rights': rule.allowed_access})
            if path == '/dev/urandom' and add_errno is not None:
                raise OSError(add_errno, 'synthetic device rule failure')
            return 0
        if number == 446:
            assert rules[-1] == {'path': '/dev/urandom', 'rights': 1 << 2}
            return 0
        raise AssertionError('Unreviewed mock syscall')

    mock_os = SimpleNamespace(**flags, open=mock_open, close=mock_close,
                              fstat=lambda fd: objects[fd],
                              major=lambda device: device >> 8,
                              minor=lambda device: device & 0xff)
    policy = SimpleNamespace(checked_syscall=mock_syscall,
        Ruleset=Ruleset, PathBeneath=PathBeneath, FS_ALL_ABI1=(1 << 13) - 1,
        FS_READ=(1 << 2) | (1 << 3),
        FS_OUTPUT=(1 << 2) | (1 << 3) | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8))
    namespace = dict(ctypes=ctypes, os=mock_os, stat=stat,
                     PUBLIC_DEVICE_GRANT_APPROVED=approved)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<exact-proposed-function-with-mocks>', 'exec'), namespace)
    caught, result = None, None
    try:
        result = namespace['restrict_native'](policy, ['runtime'], ['library'], ['listing'], ['output'])
    except (OSError, RuntimeError) as error:
        caught = error
    expected_failure = (not approved or device_mode != stat.S_IFCHR or major != 1 or minor != 9
                        or regular_mode != stat.S_IFREG or open_errno is not None
                        or add_errno is not None or abi < 1)
    assert (caught is not None) == expected_failure
    if not approved:
        assert calls == opens == closes == rules == []
    elif abi < 1:
        assert calls == [444] and opens == closes == rules == []
    else:
        assert 90 in closes
        assert all(fd in closes for fd in objects)
        if expected_failure:
            assert 446 not in calls
        else:
            observed_abi, identity = result
            assert observed_abi == 1
            assert identity == {'path': '/dev/urandom', 'object_type': 'character_device',
                'major': 1, 'minor': 9, 'allowed_access_fs': 1 << 2,
                'actual_fstat_validated': True, 'read_file_only': True,
                'write_file_granted': False, 'inherited_entropy_FD': False}
            assert rules == [dict(path='runtime', rights=policy.FS_READ),
                             dict(path='library', rights=1 << 2),
                             dict(path='listing', rights=1 << 3),
                             dict(path='output', rights=policy.FS_OUTPUT),
                             dict(path='/dev/urandom', rights=1 << 2)]
            assert calls.count(446) == 1
    if open_errno is not None or add_errno is not None:
        assert caught.errno == (open_errno if open_errno is not None else add_errno)
    return dict(case=name, passed=True, expected_failure=expected_failure,
                real_device_or_kernel_access=False)


def run_checks():
    assert not any(name.split('.')[0] in {'torch', 'delu', 'numpy', 'scipy', 'pandas'} for name in sys.modules)
    source = (HERE / 'native_import_worker.proposed.py').read_text()
    tree = ast.parse(source)
    compile(tree, 'native_import_worker.proposed.py', 'exec')
    reversed_source = source
    changes = json.loads((HERE / 'DELTA_REVERSAL.json').read_text())
    for old, new in reversed(changes):
        assert reversed_source.count(new) == 1
        reversed_source = reversed_source.replace(new, old, 1)
    assert reversed_source == (V5 / 'native_import_worker.py').read_text()
    assert 'PUBLIC_DEVICE_GRANT_APPROVED = False' in source
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    block = next(node for node in main.body if isinstance(node, ast.Try))
    assert isinstance(block.body[0], ast.If)
    assert ast.unparse(block.body[0].test) == 'not PUBLIC_DEVICE_GRANT_APPROVED'
    assert isinstance(block.body[0].body[0], ast.Raise)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'restrict_native')
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node))
    cases = [('dormant_approval_guard', dict(approved=False)),
             ('exact_char_1_9_READ_FILE', {}),
             ('reject_regular_device_path', dict(device_mode=stat.S_IFREG)),
             ('reject_symlink_device_path', dict(device_mode=stat.S_IFLNK)),
             ('reject_char_major2', dict(major=2)),
             ('reject_dev_random_minor8', dict(minor=8)),
             ('keep_regular_library_type_strict', dict(regular_mode=stat.S_IFCHR)),
             ('propagate_device_open_EACCES', dict(open_errno=errno.EACCES)),
             ('propagate_device_rule_EPERM', dict(add_errno=errno.EPERM)),
             ('require_landlock_ABI1', dict(abi=0))]
    rows = [synthetic_case(node, name, **kwargs) for name, kwargs in cases]
    preserved = json.loads((HERE / 'PREDECESSORS_PRESERVED_HASHES.json').read_text())['files']
    for item in preserved:
        path = BASE / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    guards = {name: sha(V5 / name) for name in ('proven_cpu_policy.py', 'root_outer.py', 'IMPORT_CONTRACT.json')}
    assert guards['proven_cpu_policy.py'] == '36f26ad757b88c3ff86bf39fac5ee5020cc3765b269ec7d8e2f7c8aeefef63a0'
    assert guards['root_outer.py'] == '67ee2a5cb25794439b251402c5da0de3608577dc9a865b2202b2f454cf9febca'
    assert guards['IMPORT_CONTRACT.json'] == 'd8f57b8aa74f7022233d2ec81d54b9ab91f00097be69524d8b7f9407f5b0ee1c'
    return dict(schema='native-entropy-device-proposal-v6-source-checks',
        status='PASSED_DORMANT_SOURCE_AND_STDLIB_MOCKS_ONLY',
        actual_proposed_function_mock_cases=rows,
        full_worker_reversal_byte_identical_to_v5=True,
        original_guard_hashes_unchanged=guards,
        preserved_predecessor_files=len(preserved),
        proposed_fixed_device='/dev/urandom', proposed_major=1, proposed_minor=9,
        proposed_access_fs=1 << 2, activation_approved=False, activated=False,
        scientific_imports_entropy_reads_remote_calls_or_kernel_execution=False)


if __name__ == '__main__':
    result = run_checks()
    print(json.dumps(result, indent=2))
