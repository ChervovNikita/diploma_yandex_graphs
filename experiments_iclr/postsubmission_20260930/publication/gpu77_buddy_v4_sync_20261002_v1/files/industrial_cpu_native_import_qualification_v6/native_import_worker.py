"""UNEXECUTED CPU import qualification; trusted stdlib startup, then restriction.

No dataset loading, model construction, tensors, fits or CUDA initialization is
requested. Scientific/native imports occur only after Landlock and seccomp.
"""
from __future__ import annotations
import ctypes
import errno
import hashlib
import importlib
from importlib import metadata, util
import json
import operator
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time

# Root-admitted read-only OS entropy dependency; see sealed ROOT_ADMISSION.json.
PUBLIC_DEVICE_GRANT_APPROVED = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restrict_native(policy, directories, files, directory_listing, writable):
    """ABI1 directory policy plus individually allowed public/library files.

    Directory/output rights and ABI1 mask match the proved CPU policy. File
    rules omit READ_DIR and O_DIRECTORY; public parent directories get only
    READ_DIR, so added label files acquire no read permission.
    """
    if not PUBLIC_DEVICE_GRANT_APPROVED:
        raise RuntimeError('Proposed /dev/urandom grant has no activation approval')
    abi = policy.checked_syscall(444, ctypes.c_void_p(), 0, 1)
    if abi < 1:
        raise RuntimeError('Landlock ABI1 unavailable')
    attr = policy.Ruleset(policy.FS_ALL_ABI1)
    ruleset = policy.checked_syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    grants = ([(p, policy.FS_READ, True) for p in directories]
              + [(p, 1 << 2, False) for p in files]
              + [(p, 1 << 3, True) for p in directory_listing]
              + [(p, policy.FS_OUTPUT, True) for p in writable])
    try:
        for path, rights, directory in grants:
            flags = os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW
            if directory:
                flags |= os.O_DIRECTORY
            descriptor = os.open(path, flags)
            try:
                mode = os.fstat(descriptor).st_mode
                if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
                    raise RuntimeError('Actual rule object type differs: ' + str(path))
                rule = policy.PathBeneath(rights, descriptor)
                policy.checked_syscall(445, ruleset, 1, ctypes.byref(rule), 0)
            finally:
                os.close(descriptor)
        # Fixed public entropy endpoint; regular-file rules above stay strict.
        device_descriptor = os.open('/dev/urandom', os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            observed_device = os.fstat(device_descriptor)
            if (not stat.S_ISCHR(observed_device.st_mode)
                    or os.major(observed_device.st_rdev) != 1
                    or os.minor(observed_device.st_rdev) != 9):
                raise RuntimeError('Expected /dev/urandom character device major1 minor9')
            device_rule = policy.PathBeneath(1 << 2, device_descriptor)
            policy.checked_syscall(445, ruleset, 1, ctypes.byref(device_rule), 0)
        finally:
            os.close(device_descriptor)
        policy.checked_syscall(446, ruleset, 0)
    finally:
        os.close(ruleset)
    return abi, {'path': '/dev/urandom', 'object_type': 'character_device',
                 'major': 1, 'minor': 9, 'allowed_access_fs': 1 << 2,
                 'actual_fstat_validated': True, 'read_file_only': True,
                 'write_file_granted': False, 'inherited_entropy_FD': False}


def diagnostic_text(value, *, representation=False, limit=16384):
    """Bounded exception text, with no source files or frame locals accessed."""
    try:
        text = repr(value) if representation else str(value)
    except BaseException as error:
        text = '<exception text unavailable: ' + type(error).__name__ + '>'
    return text[:limit], len(text) > limit


def exception_diagnostics(error):
    """Both exception edges, including suppressed context, and stack metadata.

    A frame's code filename/function/line is enough to locate the import failure.
    No linecache, source-line lookup, frame globals/locals or package import is
    used while reporting. Native loader errno/message survives in inner errors.
    """
    pending, identities, nodes = [error], {id(error): 0}, []
    max_nodes, max_frames = 32, 80
    chain_truncated = False
    for exception in pending:
        index = identities[id(exception)]
        message, message_truncated = diagnostic_text(exception)
        representation, repr_truncated = diagnostic_text(exception, representation=True)
        frames, frame_count = [], 0
        tb = exception.__traceback__
        while tb is not None:
            frame_count += 1
            frames.append({'file': tb.tb_frame.f_code.co_filename,
                           'function': tb.tb_frame.f_code.co_name, 'line': tb.tb_lineno})
            if len(frames) > max_frames:
                frames.pop(0)  # Keep the innermost useful frames if unusually deep.
            tb = tb.tb_next
        node = {'id': index, 'type': type(exception).__name__,
                'type_module': type(exception).__module__, 'message': message,
                'repr': representation, 'message_truncated': message_truncated,
                'repr_truncated': repr_truncated, 'frames_outer_to_inner': frames,
                'frame_count': frame_count, 'frames_truncated': frame_count > max_frames,
                'suppress_context': bool(exception.__suppress_context__)}
        attributes = {}
        for name in ('errno', 'filename', 'filename2', 'name', 'path'):
            value = getattr(exception, name, None)
            if value is not None:
                attributes[name] = value if type(value) is int else diagnostic_text(value)[0]
        node['attributes'] = attributes
        for name, linked in [('cause_id', exception.__cause__), ('context_id', exception.__context__)]:
            if linked is None:
                node[name] = None
            elif id(linked) in identities:
                node[name] = identities[id(linked)]
            elif len(pending) < max_nodes:
                identities[id(linked)] = len(pending)
                node[name] = len(pending)
                pending.append(linked)
            else:
                node[name] = 'TRUNCATED'
                chain_truncated = True
        nodes.append(node)
    lines = ['Exception graph and traceback frames (source lines and frame locals not read):']
    for node in nodes:
        lines.append('Exception #' + str(node['id']) + ' cause=' + str(node['cause_id'])
                     + ' context=' + str(node['context_id']) + ' suppress_context=' + str(node['suppress_context']))
        lines.append('Traceback (outermost to innermost; last 80 frames at most):')
        for frame in node['frames_outer_to_inner']:
            lines.append('  File ' + repr(frame['file']) + ', line ' + str(frame['line'])
                         + ', in ' + frame['function'])
        lines.append(node['type_module'] + '.' + node['type'] + ': ' + node['message'])
        if node['attributes']:
            lines.append('  Exception attributes: ' + json.dumps(node['attributes'], sort_keys=True))
    traceback_text = '\n'.join(lines)
    text_limit = 131072
    return {'schema': 'CPU-import-exception-diagnostics-v2', 'root_exception_id': 0,
            'nodes': nodes, 'chain_truncated': chain_truncated, 'max_nodes': max_nodes,
            'max_frames_per_exception': max_frames,
            'traceback_text': traceback_text[:text_limit],
            'traceback_text_truncated': len(traceback_text) > text_limit,
            'source_lines_read': False, 'frame_locals_or_globals_captured': False,
            'native_syscall_cause_verified': False}


def entropy_probe(policy):
    """Eight-byte OS entropy observations only; no bytes or RNG state retained."""
    buffer = ctypes.create_string_buffer(8)
    try:
        count, error = policy.raw(318, ctypes.cast(buffer, ctypes.c_void_p), 8, 0)
    finally:
        ctypes.memset(buffer, 0, 8)
    observed = {'has_os_getrandom': hasattr(os, 'getrandom'),
                'raw_getrandom': {'syscall': 318, 'requested_bytes': 8, 'flags': 0,
                                  'return_count': int(count), 'errno': int(error)},
                'entropy_bytes_recorded': False}
    try:
        value = os.urandom(8)
        observed['os_urandom8'] = {'succeeded': len(value) == 8, 'return_count': len(value)}
        del value
    except (OSError, NotImplementedError) as error:
        observed['os_urandom8'] = {'succeeded': False, 'exception_type': type(error).__name__,
                                  'errno': getattr(error, 'errno', None)}
    return observed


def make_os_entropy_provider(raw):
    """Exact byte API backed only by native getrandom; no optional fallback."""
    audit = {'requests': 0, 'requested_bytes': 0, 'syscalls': 0,
             'partial_reads': 0, 'EINTR_retries': 0, 'hard_errors': 0,
             'entropy_bytes_recorded': False}

    def urandom(n, /):
        n = operator.index(n)
        if n < 0:
            raise ValueError('negative argument not allowed')
        if n > sys.maxsize:
            raise OverflowError('Python int too large to convert to C ssize_t')
        audit['requests'] += 1
        audit['requested_bytes'] += n
        if n == 0:
            return b''
        buffer = ctypes.create_string_buffer(n)
        offset = 0
        try:
            while offset < n:
                requested = min(n - offset, 256)
                audit['syscalls'] += 1
                count, error = raw(318, ctypes.byref(buffer, offset), requested, 0)
                if count < 0:
                    if error == errno.EINTR:
                        audit['EINTR_retries'] += 1
                        continue
                    audit['hard_errors'] += 1
                    raise OSError(error, 'getrandom OS entropy request failed')
                if count == 0 or count > requested:
                    audit['hard_errors'] += 1
                    raise OSError(errno.EIO, 'getrandom returned an invalid byte count')
                if count < requested:
                    audit['partial_reads'] += 1
                offset += count
            return ctypes.string_at(buffer, n)
        finally:
            ctypes.memset(buffer, 0, n)

    return urandom, audit


def activate_os_entropy_provider(policy, contract, worker_source_sha256):
    """Explicit stdlib compatibility adapter, called only after restriction."""
    original = os.urandom
    if getattr(original, '__module__', None) != 'posix':
        raise RuntimeError('Expected unmodified pinned interpreter os.urandom')
    loaded_random = sys.modules.get('random')
    random_path = Path(contract['runtime_image']) / 'opt/python/lib/python3.12/random.py'
    if loaded_random is not None:
        if Path(loaded_random.__file__).resolve(strict=True) != random_path:
            raise RuntimeError('Already-loaded random is not the pinned stdlib module')
        if getattr(loaded_random, '_urandom', None) is not original:
            raise RuntimeError('Already-loaded random entropy alias differs')
    provider, audit = make_os_entropy_provider(policy.raw)
    os.urandom = provider
    if loaded_random is not None:
        loaded_random._urandom = provider
    return {'schema': 'trusted-getrandom-OS-entropy-adapter-v1',
            'worker_source_sha256': worker_source_sha256,
            'factory': 'make_os_entropy_provider', 'activation': 'activate_os_entropy_provider',
            'activation_stage': 'AFTER_LANDLOCK_AND_SECCOMP_BEFORE_SCIENTIFIC_IMPORTS',
            'syscall': 318, 'flags': 0, 'max_bytes_per_syscall': 256,
            'os_urandom_rebound': os.urandom is provider,
            'random_loaded_at_activation': loaded_random is not None,
            'random_urandom_rebound': loaded_random is not None and loaded_random._urandom is provider,
            'future_stdlib_random_import_uses_os_alias': loaded_random is None,
            'deterministic_substitute': False, 'optional_fallback': False,
            'additional_filesystem_or_device_grants': False, 'inherited_entropy_FD': False,
            'scientific_seed_policy_changed': False, 'audit': audit}


def main():
    result = {'schema': 'CPU-native-import-qualification-v6', 'status': 'STARTING',
              'diagnostic_only_revision': 2,
              'equivalent_to_v2_chroot_mount_contract': False, 'metadata_visibility_disclosed': True,
              'GPU_ioctl_compatibility_claimed': False, 'dataset_load_requested': False,
              'model_construction_or_fit_requested': False, 'GPU_compute_requested': False,
              'public_member_contents_read': False, 'restrictions_installed': False,
              'imports': [], 'boundaries': [], 'automatic_retry': False}
    audit_fd = None
    started = time.monotonic()
    try:
        if not PUBLIC_DEVICE_GRANT_APPROVED:
            raise RuntimeError('Dormant v6 proposal: public-device grant is unapproved')
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        root = Path(sys.argv[1]); host = json.loads(sys.argv[2]); pins = json.loads(sys.argv[3])
        bindings_sha = sys.argv[4]
        if sys.platform != 'linux' or platform.machine() != 'x86_64' or platform.python_version() != '3.12.9':
            raise RuntimeError('Pinned Linux x86_64 Python3.12.9 required')
        trusted = root / 'trusted_source'
        for name in ('proven_cpu_policy.py', 'native_import_worker.py', 'IMPORT_CONTRACT.json'):
            if digest(trusted / name) != pins[name]:
                raise RuntimeError('Retained source/contract pin differs: ' + name)
        if Path(__file__) != trusted / 'native_import_worker.py':
            raise RuntimeError('Worker source location differs')
        contract = json.loads((trusted / 'IMPORT_CONTRACT.json').read_text())
        fixed = Path(contract['phase']) / contract['run_container']
        if root.parent != fixed or root.resolve(strict=True) != root:
            raise RuntimeError('Fresh fixed project run required')
        if digest(root / 'RUN_BINDINGS.json') != bindings_sha:
            raise RuntimeError('Preflight binding changed')
        bindings = json.loads((root / 'RUN_BINDINGS.json').read_text())
        observed = {name: os.readlink('/proc/self/ns/' + name) for name in ('user','mnt','pid','net','ipc','uts')}
        if set(host) != set(observed) or os.getuid() != 0 or os.getpid() != 1 or any(host[k] == observed[k] for k in host):
            raise RuntimeError('Distinct six namespaces and mapped-root PID1 required')
        if len(list(Path('/proc/self/task').iterdir())) != 1:
            raise RuntimeError('Single-threaded trusted startup required')
        scientific = ('packaging','numpy','scipy','pandas','yaml','delu','torch','torchdata','dgl','sklearn','native_model')
        if any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in scientific):
            raise RuntimeError('Scientific modules present before restriction')
        spec = util.spec_from_file_location('proven_cpu_policy', trusted / 'proven_cpu_policy.py')
        policy = util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        sys.path[:] = contract['sys_path']
        sys.dont_write_bytecode = True
        os.chdir(root / 'outputs')
        result['boundaries'].append({'stage': 'TRUSTED_STDLIB_READY', 'seconds': time.monotonic()-started,
                                    'scientific_modules_present': False})
        policy.drop_capabilities()
        pin, device = policy.reviewed_stdio_and_pin()
        result['entropy_probes'] = {'before_restrictions': entropy_probe(policy)}
        abi, random_device = restrict_native(policy, bindings['readonly_directories'], bindings['readonly_files'],
                              bindings['directory_listing_only'], bindings['writable_directories'])
        result['boundaries'].append({'stage': 'LANDLOCK_INSTALLED', 'seconds': time.monotonic()-started})
        if pin > 3:
            policy.checked_syscall(436, 3, pin - 1, 0)
        policy.checked_syscall(436, pin + 1, 0xFFFFFFFF, 0)
        count = policy.seccomp_program(pin)
        result.update(restrictions_installed=True, kernel_landlock_abi=abi, used_filesystem_ABI=1,
                      public_device_read_grant=random_device, explicit_policy_difference_from_v5=True,
                      seccomp_BPF_instructions=count, pinned_FD=pin, pinned_character_device='major1 minor3 only',
                      source_pins=pins, bindings_sha256=bindings_sha,
                      host_namespaces=host, worker_namespaces=observed,
                      readonly_directories=bindings['readonly_directories'], readonly_files=bindings['readonly_files'],
                      directory_listing_only=bindings['directory_listing_only'], writable_directories=bindings['writable_directories'],
                      sys_path=list(sys.path), sanitized_environment=dict(os.environ))
        result['boundaries'].append({'stage': 'SECCOMP_INSTALLED_BEFORE_SCIENTIFIC_IMPORTS', 'seconds': time.monotonic()-started})
        result['entropy_probes']['after_restrictions'] = entropy_probe(policy)
        result['entropy_provider'] = activate_os_entropy_provider(policy, contract, pins['native_import_worker.py'])
        audit_fd = os.open(root / 'outputs/python_audit.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        audit_count = 0
        audit_limit = 2048
        def audit(event, args):
            nonlocal audit_count
            if event not in ('open', 'import', 'ctypes.dlopen', 'socket.__new__', 'subprocess.Popen', 'os.exec') or audit_count >= audit_limit:
                return
            audit_count += 1
            detail = str(args[0])[:2048] if args else ''
            row = {'kind': 'Python_audit_event_not_complete_syscall_trace', 'event': event,
                   'argument_0': detail, 'pid': os.getpid(), 'seconds': time.monotonic()-started}
            if event == 'import':
                row['resolved_filename_argument'] = str(args[1])[:2048] if len(args) > 1 else None
            os.write(audit_fd, (json.dumps(row, sort_keys=True) + '\n').encode())
        sys.addaudithook(audit)
        os.write(audit_fd, (json.dumps({'kind':'process_boundary','stage':'LANDLOCK_AND_SECCOMP_INSTALLED',
                                       'pid':os.getpid(),'uid':os.getuid(),'BPF_instructions':count})+'\n').encode())
        probes = []
        for path in [str(root / 'excluded_fixture/blocked.txt'), *contract['forbidden_content_paths']]:
            try:
                descriptor = os.open(path, os.O_RDONLY)
            except OSError as error:
                accepted = (errno.EACCES,) if path == str(root / 'excluded_fixture/blocked.txt') else (errno.EACCES,errno.ENOENT)
                if error.errno not in accepted:
                    raise
                probes.append({'path': path, 'operation': 'open_read_without_reading_bytes', 'errno': error.errno})
            else:
                os.close(descriptor)
                raise RuntimeError('Forbidden path opened: ' + path)
        for path in [Path(contract['native_source']) / 'native_model.py', Path(contract['public_inputs']) / 'tolokers-2/features.csv']:
            try:
                descriptor = os.open(path, os.O_WRONLY)
            except OSError as error:
                if error.errno != errno.EACCES:
                    raise
                probes.append({'path': str(path), 'operation': 'open_write_without_writing_bytes', 'errno': error.errno})
            else:
                os.close(descriptor)
                raise RuntimeError('Readonly content acquired write FD: ' + str(path))
        # Only the public metadata manifest is read; no CSV/array parser is called.
        if digest(Path(contract['public_inputs']) / contract['public_manifest_name']) != bindings['public_manifest_sha256']:
            raise RuntimeError('Public manifest changed after restriction')
        result['public_manifest_metadata_hash_rechecked_after_restriction'] = True
        syscall_observations = []
        for number, arguments in [(76,(os.fsencode(root/'excluded_fixture/blocked.txt'),0)),
                                  (2,(os.fsencode(root/'excluded_fixture/blocked.txt'),os.O_RDONLY|os.O_TRUNC,0)),
                                  (437,(-100,0,0,0))]:
            returned, error = policy.raw(number,*arguments)
            syscall_observations.append({'syscall':number,'returned':returned,'errno':error})
            if returned != -1 or error != errno.EPERM:
                raise RuntimeError('Covered truncation/open syscall boundary differs')
        result['explicit_syscall_boundary_observations'] = syscall_observations
        os.write(audit_fd,(json.dumps({'kind':'explicit_syscall_boundary_observations',
                                     'observations':syscall_observations})+'\n').encode())
        result['content_boundary_probes'] = probes
        allowed_roots = [Path(p).resolve(strict=True) for p in bindings['readonly_directories']]
        for name in contract['imports']:
            entry = {'module': name, 'start_seconds': time.monotonic()-started, 'after_restrictions': True}
            result['imports'].append(entry)
            os.write(audit_fd,(json.dumps({'kind':'import_boundary','stage':'START','module':name,
                                         'pid':os.getpid(),'seconds':entry['start_seconds']})+'\n').encode())
            module = importlib.import_module(name)
            origin = Path(module.__file__).resolve(strict=True)
            if not any(origin == directory or directory in origin.parents for directory in allowed_roots):
                raise RuntimeError('Imported module outside pinned readonly directories: ' + str(origin))
            entry.update(origin=str(origin), completed=True, end_seconds=time.monotonic()-started)
            os.write(audit_fd,(json.dumps({'kind':'import_boundary','stage':'COMPLETE','module':name,
                                         'origin':str(origin),'seconds':entry['end_seconds']})+'\n').encode())
        from packaging.specifiers import SpecifierSet
        versions = {}
        for name, expected in contract['selected_top_versions'].items():
            actual = metadata.version(name)
            if not SpecifierSet('==' + expected).contains(actual, prereleases=True):
                raise RuntimeError('Selected package version differs: ' + name + ' ' + actual)
            versions[name] = actual
        torch = sys.modules['torch']
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA unexpectedly initialized during CPU imports')
        result.update(distribution_versions=versions, CUDA_initialized=False,
                      torch_build_CUDA=str(torch.version.cuda), Python_audit_events=audit_count,
                      Python_audit_limit=audit_limit, native_C_syscalls_fully_traced=False)
        current = os.fstat(pin)
        returned, error = policy.raw(16, pin, 0, 0)
        if not stat.S_ISCHR(current.st_mode) or current.st_rdev != device or returned != -1 or error != errno.ENOTTY:
            raise RuntimeError('Character-device pin changed after native imports')
        output_fd = os.open(root / 'outputs/provenance_fixture.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(output_fd, b'CPU import diagnostic only\n')
            returned, error = policy.raw(16, output_fd, 0, 0)
            if returned != -1 or error != errno.EPERM:
                raise RuntimeError('Unpinned ioctl allowed after imports')
        finally:
            os.close(output_fd)
        result['postimport_pin_and_unpinned_ioctl_observations'] = {'pin_ioctl_errno': errno.ENOTTY, 'unpinned_ioctl_errno': errno.EPERM}
        result['boundaries'].append({'stage': 'IMPORTS_COMPLETED_WITH_CONTENT_POLICY', 'seconds': time.monotonic()-started})
        result['status'] = 'PASSED_CPU_NATIVE_IMPORTS_ONLY'
    except BaseException as error:
        result.update(status='REFUSED_OR_FAILED_CPU_IMPORTS', error_type=type(error).__name__, error=str(error))
        if result['imports'] and not result['imports'][-1].get('completed'):
            result['imports'][-1]['failed'] = True
        try:
            diagnostics = exception_diagnostics(error)
            result['exception_diagnostics'] = diagnostics
            os.write(2, (diagnostics['traceback_text'] + '\n').encode('utf-8', errors='backslashreplace'))
            if audit_fd is not None:
                os.write(audit_fd, (json.dumps({'kind': 'exception_diagnostics',
                    'after_restrictions': result['restrictions_installed'],
                    'diagnostics': diagnostics}, sort_keys=True) + '\n').encode())
        except BaseException as diagnostic_error:
            # Retain the original outer error even if diagnostics themselves fail.
            result['exception_diagnostic_reporting_error'] = {
                'type': type(diagnostic_error).__name__, 'message': diagnostic_text(diagnostic_error)[0]}
    finally:
        if audit_fd is not None:
            result.update(Python_audit_events=audit_count, Python_audit_limit=audit_limit,
                          native_C_syscalls_fully_traced=False)
            os.close(audit_fd)
    result['seconds'] = time.monotonic()-started
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result['status'] == 'PASSED_CPU_NATIVE_IMPORTS_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
