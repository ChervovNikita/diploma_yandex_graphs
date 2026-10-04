#!/usr/bin/env python3
"""Disabled, separately released owner of one fabricated count-conditioned CPU child.

This observer imports only the standard library. It never imports the candidate.
The only numerical child calls exact bucketed_oracles.qualify_bucketed (119 cases).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
CANDIDATE = HERE  # The separately reviewed QA driver directory; subject is pinned below.
CANDIDATE_MANIFEST = '47d8fc112f82d05bfce84548f86538e2b259b67efaf3635c5d2de42c567a434b'
CORE_MANIFEST = '92ae9f79c15cf6de53e39f06c241780b8089d652d59070cf9d23ed521e2740cf'
EXECUTION = RESEARCH / 'exact_cb_support_bucket_cpu_qa_execution_root_20261004_v1'
OUT = EXECUTION / 'owned_supervisor/run01'
CHILD_OUT = EXECUTION / 'fabricated_cpu/run01'
ROOT_RELEASE = EXECUTION / 'ROOT_RELEASE_owned_supervisor.json'
INNER_RELEASE = EXECUTION / 'ROOT_RELEASE_fabricated_cpu.json'
ROOT_LOCK = EXECUTION / '.ROOT_LOCK_fabricated_cpu_owned_supervisor'
CAPS = {'wall_seconds': 900, 'peak_RSS_bytes': 4 * 1024**3}
POLL_SECONDS = .1
PS_TIMEOUT_SECONDS = .5
CLEANUP_SECONDS = 5.
PS_ARGV = ['/bin/ps', '-axo', 'pid=,ppid=,pgid=,rss=,lstart=,stat=']
EXPECTED_CASES = {'ragged_analytic_cases': 96, 'masked_ESP_cases': 12, 'zero_cases': 6, 'plan_cases': 5}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def pinned_json(path, pin):
    require(path.is_file() and not path.is_symlink() and sha(path) == pin, 'Pinned JSON differs: ' + str(path))
    require(path.stat().st_size <= 32 * 1024**2, 'JSON observation exceeds 32 MiB')
    return json.loads(path.read_text())


def verify_rows(base, rows):
    require(isinstance(rows, list) and rows, 'Source closure is empty')
    seen = set()
    for row in rows:
        path = base / row['path']
        require(row['path'] not in seen and path.resolve().is_relative_to(base.resolve())
                and path.is_file() and not path.is_symlink(), 'Invalid source closure path')
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Source closure changed: ' + str(path))
        seen.add(row['path'])


def source_gate(manifest_pin):
    manifest = pinned_json(HERE / 'MANIFEST.json', manifest_pin)
    require(manifest['source_only'] is True and manifest['execution_authorized'] is False,
            'Supervisor source preparation flags differ')
    verify_rows(HERE, manifest['files'])
    binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
    require(binding['candidate_manifest_sha256'] == CANDIDATE_MANIFEST
            and binding['core_manifest_sha256'] == CORE_MANIFEST, 'Candidate/core binding differs')
    verify_rows(RESEARCH, binding['external_input_pins'])
    core_review = binding['reused_core_independent_review']
    source_review(RESEARCH / core_review['path'], core_review['sha256'], CORE_MANIFEST)


def source_review(path, pin, manifest_pin):
    require(path.resolve().is_relative_to(RESEARCH) and not path.is_symlink(), 'Review outside project source closure')
    review = pinned_json(path, pin)
    require(review.get('status') == 'PASS' and review.get('candidate_manifest_sha256') == manifest_pin
            and review.get('execution_authorized') is False and not review.get('blocking_findings'),
            'Independent exact source PASS with no blocking findings required')


def preflight(path, release_pin):
    require(path == ROOT_RELEASE and path.resolve() == ROOT_RELEASE and not path.is_symlink(), 'Exact root supervisor release required')
    release = pinned_json(path, release_pin)
    require(release.get('schema') == 'exact-CB-bucket-physical-cpu-root-release-v1'
            and release.get('status') == 'APPROVED' and release.get('root_authorization_reference')
            and release.get('authorized_stages') == ['fabricated_cpu_owned_supervisor']
            and release.get('caps') == CAPS and release.get('automatic_retry') is False
            and release.get('scientific_fit_admitted') is False and release.get('GPU_data_access') is False
            and release.get('TEST_supported') is False, 'CPU supervisor release absent')
    require(release.get('candidate_manifest_sha256') == CANDIDATE_MANIFEST, 'Fixed candidate pin differs')
    source_gate(release['supervisor_manifest_sha256'])
    source_review(Path(release['independent_supervisor_review_path']), release['independent_supervisor_review_sha256'],
                  release['supervisor_manifest_sha256'])
    require(Path(release['inner_cpu_release_path']) == INNER_RELEASE and INNER_RELEASE.resolve() == INNER_RELEASE,
            'Exact child release required')
    inner = pinned_json(INNER_RELEASE, release['inner_cpu_release_sha256'])
    require(inner.get('schema') == 'exact-CB-bucket-fabricated-cpu-root-release-v1'
            and inner.get('status') == 'APPROVED' and inner.get('authorized_stages') == ['fabricated_cpu']
            and inner.get('root_authorization_reference') and inner.get('caps') == CAPS
            and inner.get('source_manifest_sha256') == CANDIDATE_MANIFEST
            and inner.get('core_manifest_sha256') == CORE_MANIFEST
            and inner.get('qa_manifest_sha256') == release['supervisor_manifest_sha256']
            and inner.get('threads') == 2 and inner.get('interop_threads') == 1
            and inner.get('CUDA_VISIBLE_DEVICES') == ''
            and inner.get('independent_qa_review_path') == release['independent_supervisor_review_path']
            and inner.get('independent_qa_review_sha256') == release['independent_supervisor_review_sha256']
            and inner.get('automatic_retry') is False and inner.get('scientific_fit_admitted') is False
            and inner.get('GPU_data_access') is False and inner.get('TEST_supported') is False,
            'Exact inner fabricated CPU release absent')
    source_review(Path(inner['independent_source_review_path']), inner['independent_source_review_sha256'], CANDIDATE_MANIFEST)
    profile = release['runtime_profile']
    require(profile['platform'] == sys.platform and sys.platform in ('darwin', 'linux')
            and sys.version_info >= (3, 10) and profile['ps_executable'] == PS_ARGV[0]
            and sha(Path(profile['ps_executable'])) == profile['ps_executable_sha256'], 'Pinned observer platform differs')
    interpreter = Path(profile['python_executable'])
    require(interpreter.is_absolute() and interpreter.resolve() == Path(sys.executable).resolve()
            and sha(interpreter) == profile['python_executable_sha256'], 'Pinned supervisor interpreter differs')
    require(inner['runtime_profile']['python_executable'] == str(interpreter)
            and inner['runtime_profile']['python_executable_sha256'] == profile['python_executable_sha256']
            and inner['runtime_profile']['torch_distribution_version'], 'Pinned child runtime differs')
    require(signal.getsignal(signal.SIGCHLD) == signal.SIG_DFL and signal.getitimer(signal.ITIMER_REAL) == (0., 0.),
            'Fresh default SIGCHLD/no active timer required to preserve unreaped child identity')
    command = [str(interpreter), '-I', '-B', str(CANDIDATE / 'qualify_cpu.py'), '--root-release', str(INNER_RELEASE),
               '--release-sha256', release['inner_cpu_release_sha256']]
    command_pin = hashlib.sha256(json.dumps(command, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    require(release['child_argv'] == command and release['child_command_sha256'] == command_pin
            and release['child_cwd'] == str(CANDIDATE), 'Exact child command/cwd differs')
    require(OUT.resolve() == OUT and CHILD_OUT.resolve() == CHILD_OUT and ROOT_LOCK.resolve() == ROOT_LOCK,
            'Output/lock ancestors must be canonical without symlinks')
    require(not OUT.exists() and not CHILD_OUT.exists() and not ROOT_LOCK.exists(), 'Fresh sole CPU attempt required; no retry')
    return release, inner, command, command_pin


def census(child_pid):
    # Only identity/RSS fields are requested; no unrelated process command text.
    environment = dict(os.environ, LC_ALL='C', LANG='C')
    completed = subprocess.run(PS_ARGV, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               check=True, timeout=PS_TIMEOUT_SECONDS, env=environment)
    rows = []
    for line in completed.stdout.decode('ascii', errors='strict').splitlines():
        fields = line.split()
        require(len(fields) == 10, 'Unexpected pinned ps identity layout')
        pid, ppid, pgid, rss_kib = map(int, fields[:4])
        require(pid >= 0 and ppid >= 0 and pgid >= 0 and rss_kib >= 0, 'Invalid ps identity/RSS values')
        if pid == 0:
            continue
        try:
            sid = os.getsid(pid)
        except ProcessLookupError:
            # A process exiting during the observer snapshot contributes no
            # later live process; an owned identity gap is recorded separately.
            if pid == child_pid or pgid == child_pid:
                raise RuntimeError('Owned process disappeared during census')
            continue
        if sid == child_pid or pid == child_pid or ppid == child_pid:
            rows.append({'pid': pid, 'ppid': ppid, 'pgid': pgid, 'sid': sid,
                         'start_lstart': ' '.join(fields[4:9]), 'state': fields[9], 'rss_bytes': rss_kib * 1024})
    return rows


def held_child_signal(child_pid, reason, actions):
    require(type(child_pid) is int and child_pid > 0, 'Only a positive held direct-child PID may be signaled')
    # The direct child has never been waitpid-reaped. Its PID cannot be reused.
    # A dedicated session cannot admit unrelated processes into this group.
    try:
        sid, pgid = os.getsid(child_pid), os.getpgid(child_pid)
        if sid == child_pid and pgid == child_pid:
            os.killpg(child_pid, signal.SIGKILL)
            target = 'held_owned_child_process_group'
        else:
            os.kill(child_pid, signal.SIGKILL)
            target = 'held_direct_child_before_session_handshake'
        actions.append({'UTC': utc(), 'reason': reason, 'target': target, 'pid_or_pgid': child_pid, 'signal': 'SIGKILL'})
    except ProcessLookupError:
        actions.append({'UTC': utc(), 'reason': reason, 'target': 'held_child_absent', 'pid': child_pid})


def collected_result(inner, physical, terminal):
    result = {'status': 'INCOMPLETE_OR_MISSING', 'qualification': None, 'child_final_custody': None, 'reason': None}
    qualification, final = CHILD_OUT / 'QUALIFICATION.json', CHILD_OUT / 'FINAL_CUSTODY.json'
    failure = CHILD_OUT / 'FAILURE.json'
    if failure.is_file() and not failure.is_symlink():
        result['status'] = 'COLLECTED_FAILED'
        result['failure'] = {'path': str(failure), 'sha256': sha(failure), 'bytes': failure.stat().st_size}
    if not qualification.is_file() or not final.is_file():
        result['reason'] = 'Actual child qualification and FINAL_CUSTODY links are both required; partial output is not PASS'
        return result
    result['qualification'] = {'path': str(qualification), 'sha256': sha(qualification), 'bytes': qualification.stat().st_size}
    result['child_final_custody'] = {'path': str(final), 'sha256': sha(final), 'bytes': final.stat().st_size}
    try:
        q = pinned_json(qualification, result['qualification']['sha256'])
        f = pinned_json(final, result['child_final_custody']['sha256'])
        require(physical == 'REAPED_OWNED_SESSION_EMPTY' and terminal['child_exit_code'] == 0
                and not terminal['stop_reason'] and not failure.exists(), 'Physical/failed child outcome cannot qualify')
        require(terminal.get('child_session_handshake') == {'pid': terminal['child_pid'], 'sid': terminal['child_pid'],
                'pgid': terminal['child_pid'], 'child_command_sha256': terminal['child_command_sha256']},
                'Actual dedicated child handshake required')
        require(q.get('schema') == 'exact-CB-bucket-fabricated-cpu-qualification-v1'
                and q.get('status') == 'PASS' and q.get('source_manifest_sha256') == CANDIDATE_MANIFEST
                and q.get('core_manifest_sha256') == CORE_MANIFEST
                and q.get('qa_manifest_sha256') == terminal['supervisor_manifest_sha256']
                and q.get('root_release_sha256') == terminal['inner_cpu_release_sha256']
                and q.get('runtime_profile') == inner['runtime_profile']
                and q.get('torch_version') == inner['runtime_profile']['torch_distribution_version']
                and q.get('actual_module_bindings_checked') is True and q.get('complete_runtime_closure_claim') is False,
                'Bucket child source/runtime/binding identity differs')
        require(all(q.get(name) is False for name in ('GPU_data_access', 'scientific_fit_admitted', 'TEST_supported',
                    'native_full_batch_resource_qualification', 'frozen_four_arm_screen_changed')), 'Child claim flags differ')
        value = q.get('inclusive_wall_seconds')
        require(type(value) in (int, float) and 0 <= value <= CAPS['wall_seconds'], 'Child reported wall differs')
        require(type(q.get('peak_RSS_bytes')) is int and 0 <= q['peak_RSS_bytes'] <= CAPS['peak_RSS_bytes'], 'Child reported RSS differs')
        r = q['result']
        require(r.get('cases') == EXPECTED_CASES and r.get('CPU_QA_result_not_source_preparation') is True
                and r.get('native_neural_or_full65536_resource_qualified') is False
                and r.get('methodological_or_inference_efficiency_novelty_claim') is False, 'Exact bucket oracle completion/claims differ')
        reports = r.get('comparison_reports')
        require(isinstance(reports, list) and len(reports) == 3288
                and len({row['label'] for row in reports}) == 3288, 'Exact distinct bucket comparison reports incomplete')
        for row in reports:
            tolerance = (1e-10, 1e-9) if row['dtype'] == 'torch.float64' else (1.52587890625e-5, 1.52587890625e-5)
            require(row['dtype'] in ('torch.float32', 'torch.float64')
                    and (row['atol'], row['rtol']) == tolerance
                    and type(row['elements']) is int and row['elements'] >= 0
                    and type(row['maximum_absolute_difference']) in (int, float) and 0 <= row['maximum_absolute_difference'] < float('inf')
                    and type(row['maximum_fraction_of_allowed_error']) in (int, float)
                    and 0 <= row['maximum_fraction_of_allowed_error'] <= 1, 'Fixed oracle report failed or tolerance changed')
        source = CHILD_OUT / 'SOURCE_CUSTODY.json'
        runtime = CHILD_OUT / 'RUNTIME_SCOPE.json'
        source_value = pinned_json(source, sha(source))
        runtime_value = pinned_json(runtime, sha(runtime))
        binding = json.loads((HERE / 'SOURCE_BINDING.json').read_text())
        require(source_value.get('candidate_manifest_sha256') == CANDIDATE_MANIFEST
                and source_value.get('core_manifest_sha256') == CORE_MANIFEST
                and source_value.get('qa_manifest_sha256') == terminal['supervisor_manifest_sha256']
                and source_value.get('module_load_order') == binding['module_load_order']
                and source_value.get('actual_module_bindings_checked') is True, 'Actual module source custody differs')
        expected_modules = [dict(row, actual_file=str(RESEARCH / row['path'])) for row in binding['module_load_order']]
        require(source_value.get('before_QA') == expected_modules
                and runtime_value.get('source_modules_after_QA') == expected_modules
                and runtime_value.get('python_executable') == inner['runtime_profile']['python_executable']
                and runtime_value.get('torch_version') == inner['runtime_profile']['torch_distribution_version']
                and runtime_value.get('torch_distribution_version') == inner['runtime_profile']['torch_distribution_version']
                and runtime_value.get('threads') == 2 and runtime_value.get('interop_threads') == 1
                and runtime_value.get('CUDA_VISIBLE_DEVICES') == ''
                and runtime_value.get('complete_runtime_closure_claim') is False, 'Actual source/runtime scope differs')
        require(f.get('stage') == 'fabricated_cpu' and f.get('completed') is True, 'Child custody completion differs')
        rows = f['files']
        require(isinstance(rows, list) and rows and any(row['path'] == 'QUALIFICATION.json' for row in rows), 'Child custody lacks qualification')
        verify_rows(CHILD_OUT, rows)
        names = {row['path'] for row in rows}
        require(len(names) == len(rows) and all(Path(name).name == name for name in names), 'Child custody filenames invalid')
        actual = {path.name for path in CHILD_OUT.iterdir() if path.name != 'FINAL_CUSTODY.json'}
        require(actual == names and all(path.is_file() and not path.is_symlink() for path in CHILD_OUT.iterdir()),
                'Unlisted, partial or nested child output cannot qualify')
        require(terminal['physical_child_wall_seconds'] <= CAPS['wall_seconds']
                and terminal['observed_aggregate_session_peak_RSS_bytes'] <= CAPS['peak_RSS_bytes'], 'Independent observed caps differ')
        result['status'] = 'COLLECTED_PASS'
    except BaseException as error:
        result['status'] = 'COLLECTED_FAILED' if failure.exists() else 'UNQUALIFIED_COLLECTED_OUTPUT'
        result['reason'] = type(error).__name__ + ': ' + str(error)
    return result


def snapshot_files(root):
    rows = []
    if root.exists():
        for path in sorted(root.rglob('*')):
            require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Custody path is not owned plain output')
            if path.is_file():
                rows.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    return rows


def supervise(release, inner, command, command_pin, release_pin):
    EXECUTION.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(ROOT_LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    lock = {'schema': 'graph-count-conditioned-pattern-cpu-root-lock-v1', 'created_UTC': utc(),
            'supervisor_pid': os.getpid(), 'root_release_sha256': release_pin,
            'supervisor_manifest_sha256': release['supervisor_manifest_sha256'],
            'candidate_manifest_sha256': CANDIDATE_MANIFEST, 'automatic_retry': False}
    os.write(lock_fd, (json.dumps(lock, sort_keys=True) + '\n').encode()); os.fsync(lock_fd)
    directory_fd = os.open(EXECUTION, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    OUT.mkdir(parents=True, mode=0o700)
    directory_fd = os.open(OUT.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    terminal = dict(lock, schema='graph-count-conditioned-pattern-owned-physical-cpu-terminal-v1',
                    status='IN_PROGRESS', child_argv=command, child_command_sha256=command_pin,
                    child_cwd=str(CANDIDATE), inner_cpu_release_sha256=release['inner_cpu_release_sha256'],
                    caps=CAPS, monitoring={'poll_seconds': POLL_SECONDS, 'ps_timeout_seconds': PS_TIMEOUT_SECONDS,
                    'RSS_semantics': 'Independent observed aggregate RSS of the owned session; sampled, not an instantaneous OS RSS limit',
                    'cleanup_seconds_after_stop': CLEANUP_SECONDS}, child_pid=None, child_exit_code=None,
                    waitpid_status=None, stop_reason=None, observed_aggregate_session_peak_RSS_bytes=0,
                    owned_identity_ledger=[], termination_actions=[], physical_child_wall_seconds=None,
                    execution_authorized_by_this_source=False, scientific_fit_admitted=False, TEST_supported=False,
                    GPU_data_access=False, automatic_retry=False)
    atomic_json(OUT / 'STARTED.json', terminal)
    child_pid, read_fd = None, None
    started, stopped_at = time.monotonic(), None
    identity, observations = None, 0
    errors = []
    physical = 'NOT_STARTED'
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def stop(reason):
        nonlocal stopped_at
        if terminal['stop_reason'] is None:
            terminal['stop_reason'] = reason
            stopped_at = time.monotonic()
        if child_pid is not None and terminal['waitpid_status'] is None:
            held_child_signal(child_pid, reason, terminal['termination_actions'])

    def interrupted(signum, frame):
        # A fork child can inherit this handler before resetting its signals.
        # It must exit directly and never run the parent-owned cleanup path.
        if os.getpid() != terminal['supervisor_pid']:
            os._exit(128 + signum)
        stop('WALL' if signum == signal.SIGALRM else 'SUPERVISOR_SIGNAL_' + str(signum))

    try:
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_inheritable(read_fd, False); os.set_inheritable(write_fd, False)
        stdout_fd = os.open(OUT / 'CHILD_STDOUT.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        stderr_fd = os.open(OUT / 'CHILD_STDERR.txt', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        require(terminal['stop_reason'] is None, 'Recorded setup stop forbids fork')
        child_pid = os.fork()
        if child_pid == 0:
            try:
                if stopped_at is not None:
                    os._exit(126)
                os.close(read_fd); os.close(lock_fd)
                for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
                    signal.signal(signum, signal.SIG_DFL)
                os.setsid()
                os.dup2(stdout_fd, 1); os.dup2(stderr_fd, 2)
                os.close(stdout_fd); os.close(stderr_fd)
                os.chdir(CANDIDATE)
                handshake = {'pid': os.getpid(), 'sid': os.getsid(0), 'pgid': os.getpgrp(),
                             'child_command_sha256': command_pin}
                os.write(write_fd, (json.dumps(handshake) + '\n').encode())
                os.close(write_fd)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
                os.execve(command[0], command, environment)
            except BaseException as error:
                os.write(2, ('Owned child setup/exec failed: ' + type(error).__name__ + ': ' + str(error) + '\n').encode())
                os._exit(125)
        os.close(write_fd); os.close(stdout_fd); os.close(stderr_fd)
        terminal['child_pid'] = child_pid
        if stopped_at is not None:
            held_child_signal(child_pid, terminal['stop_reason'], terminal['termination_actions'])
        signal.setitimer(signal.ITIMER_REAL, max(.001, CAPS['wall_seconds'] - (time.monotonic() - started)))
        physical = 'OWNED_CHILD_RUNNING'
        buffer = b''
        last_heartbeat = started
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= CAPS['wall_seconds']:
                stop('WALL')
            if stopped_at is not None and time.monotonic() - stopped_at >= CLEANUP_SECONDS:
                physical = 'INCOMPLETE_OWNERSHIP_OR_REAP_UNRESOLVED'
                break
            try:
                part = os.read(read_fd, 4096)
            except BlockingIOError:
                part = b''
            buffer += part
            require(len(buffer) <= 4096, 'Child handshake exceeded bound')
            if identity is None and b'\n' in buffer:
                identity = json.loads(buffer.split(b'\n', 1)[0])
                require(identity == {'pid': child_pid, 'sid': child_pid, 'pgid': child_pid, 'child_command_sha256': command_pin},
                        'Dedicated child session handshake differs')
                terminal['child_session_handshake'] = identity
                atomic_json(OUT / 'CHILD_IDENTITY.json', identity)
            try:
                rows = census(child_pid)
                observations += 1
                leader = next((row for row in rows if row['pid'] == child_pid), None)
                require(leader is not None, 'Held direct child missing from census')
                require(leader['ppid'] == os.getpid(), 'Direct child parent identity changed')
                if 'first_observed_child_birth' not in terminal:
                    terminal['first_observed_child_birth'] = leader['start_lstart']
                require(leader['start_lstart'] == terminal['first_observed_child_birth'], 'Direct child birth identity changed')
                if identity is not None:
                    require(leader['sid'] == child_pid and leader['pgid'] == child_pid, 'Held child changed session/group')
                seen = {(row['pid'], row['start_lstart']) for row in terminal['owned_identity_ledger']}
                for row in rows:
                    if (row['pid'], row['start_lstart']) not in seen:
                        terminal['owned_identity_ledger'].append(row)
                aggregate = sum(row['rss_bytes'] for row in rows if row['sid'] == child_pid)
                terminal['observed_aggregate_session_peak_RSS_bytes'] = max(aggregate, terminal['observed_aggregate_session_peak_RSS_bytes'])
                require(not any(row['sid'] != child_pid or row['pgid'] != child_pid for row in rows)
                        if identity is not None else True,
                        'Unauthorized new child session/group; ownership closure unresolved')
                if aggregate > CAPS['peak_RSS_bytes']:
                    stop('AGGREGATE_SESSION_RSS')
                live = [row for row in rows if row['sid'] == child_pid and not row['state'].startswith('Z')]
                if leader['state'].startswith('Z'):
                    if live:
                        stop('LEADER_EXITED_WITH_LIVE_SESSION_MEMBERS')
                    else:
                        # Never poll/reap earlier: holding this leader pins PID,
                        # session and process-group ownership through cleanup.
                        waited_pid, status = os.waitpid(child_pid, 0)
                        require(waited_pid == child_pid, 'Exact child waitpid differs')
                        terminal['waitpid_status'] = status
                        terminal['child_exit_code'] = os.waitstatus_to_exitcode(status)
                        terminal['physical_child_wall_seconds'] = time.monotonic() - started
                        physical = 'REAPED_OWNED_SESSION_EMPTY'
                        break
                if time.monotonic() - last_heartbeat >= 1:
                    atomic_json(OUT / 'PHYSICAL_PROGRESS.json', {'UTC': utc(), 'child_pid': child_pid,
                                'elapsed_wall_seconds': time.monotonic() - started, 'owned_session_members': rows,
                                'aggregate_session_RSS_bytes': aggregate, 'stop_reason': terminal['stop_reason']})
                    last_heartbeat = time.monotonic()
            except BaseException as error:
                errors.append(type(error).__name__ + ': ' + str(error))
                stop('OBSERVER_OR_OWNERSHIP_ERROR')
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        errors.append(type(error).__name__ + ': ' + str(error))
        stop('SUPERVISOR_ERROR')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child_pid is not None and terminal['waitpid_status'] is None:
            held_child_signal(child_pid, 'FINAL_OWNED_CHILD_CLEANUP', terminal['termination_actions'])
            deadline = (stopped_at if stopped_at is not None else time.monotonic()) + CLEANUP_SECONDS
            while time.monotonic() < deadline:
                try:
                    rows = census(child_pid)
                    leader = next((row for row in rows if row['pid'] == child_pid), None)
                    live = [row for row in rows if row['sid'] == child_pid and not row['state'].startswith('Z')]
                    escaped_groups = [row for row in rows if row['sid'] != child_pid or row['pgid'] != child_pid]
                    if leader and leader['state'].startswith('Z') and not live and not escaped_groups:
                        waited_pid, status = os.waitpid(child_pid, 0)
                        require(waited_pid == child_pid, 'Cleanup waitpid differs')
                        terminal['waitpid_status'] = status
                        terminal['child_exit_code'] = os.waitstatus_to_exitcode(status)
                        terminal['physical_child_wall_seconds'] = time.monotonic() - started
                        physical = 'REAPED_OWNED_SESSION_EMPTY'
                        break
                except BaseException as error:
                    errors.append(type(error).__name__ + ': ' + str(error))
                time.sleep(POLL_SECONDS)
            if terminal['waitpid_status'] is None:
                physical = 'INCOMPLETE_OWNERSHIP_OR_REAP_UNRESOLVED'
                # A failed observer must not leave an already exited direct
                # child unreaped. After this last nonblocking wait there are no
                # more signals, since the held-PID guarantee may have ended.
                try:
                    waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
                    if waited_pid == child_pid:
                        terminal['waitpid_status'] = status
                        terminal['child_exit_code'] = os.waitstatus_to_exitcode(status)
                        terminal['physical_child_wall_seconds'] = time.monotonic() - started
                        physical = 'REAPED_OWNERSHIP_UNRESOLVED'
                except BaseException as error:
                    errors.append(type(error).__name__ + ': ' + str(error))
        for signum, handler in ((signal.SIGALRM, previous_alarm), (signal.SIGTERM, previous_term), (signal.SIGINT, previous_int)):
            signal.signal(signum, handler)
        if read_fd is not None:
            os.close(read_fd)
        terminal.update(status='PHYSICAL_TERMINAL_RECORDED', completed_UTC=utc(), physical_status=physical,
                        census_observations=observations, observer_errors=errors,
                        total_supervisor_wall_seconds=time.monotonic() - started)
        try:
            source_gate(release['supervisor_manifest_sha256'])
            require(sha(ROOT_RELEASE) == release_pin and sha(INNER_RELEASE) == release['inner_cpu_release_sha256'],
                    'Release changed during child attempt')
            source_review(Path(release['independent_supervisor_review_path']), release['independent_supervisor_review_sha256'],
                          release['supervisor_manifest_sha256'])
            source_review(Path(inner['independent_source_review_path']), inner['independent_source_review_sha256'], CANDIDATE_MANIFEST)
            profile = release['runtime_profile']
            require(profile['platform'] == sys.platform
                    and Path(profile['python_executable']).resolve() == Path(sys.executable).resolve()
                    and sha(Path(profile['python_executable'])) == profile['python_executable_sha256']
                    and profile['ps_executable'] == PS_ARGV[0]
                    and sha(Path(profile['ps_executable'])) == profile['ps_executable_sha256'],
                    'Runtime interpreter/observer changed during child attempt')
            terminal['final_runtime_closure_matches'] = True
            terminal['final_source_closure_matches'] = True
        except BaseException as error:
            terminal['final_source_closure_matches'] = False
            terminal['stop_reason'] = terminal['stop_reason'] or 'FINAL_SOURCE_CHANGED'
            errors.append(type(error).__name__ + ': ' + str(error))
        try:
            terminal['collected_oracle_result'] = collected_result(inner, physical, terminal)
        except BaseException as error:
            terminal['collected_oracle_result'] = {'status': 'UNQUALIFIED_COLLECTED_OUTPUT', 'reason': type(error).__name__ + ': ' + str(error)}
        terminal['overall_status'] = ('PASS_FABRICATED_CPU_ONLY' if terminal['collected_oracle_result']['status'] == 'COLLECTED_PASS'
                                     else 'FAILED_OR_INCOMPLETE_NO_QUALIFICATION')
        atomic_json(OUT / 'SUPERVISOR_TERMINAL.json', terminal)
        try:
            custody = {'schema': 'graph-count-conditioned-pattern-physical-cpu-supervisor-custody-v1', 'UTC': utc(),
                       'physical_status': physical, 'overall_status': terminal['overall_status'],
                       'terminal_path': str(OUT / 'SUPERVISOR_TERMINAL.json'),
                       'terminal_sha256': sha(OUT / 'SUPERVISOR_TERMINAL.json'),
                       'child_output_root': str(CHILD_OUT), 'child_output_files': snapshot_files(CHILD_OUT),
                       'supervisor_output_files': snapshot_files(OUT), 'root_lock_path': str(ROOT_LOCK),
                       'root_lock_sha256': sha(ROOT_LOCK), 'automatic_retry': False, 'scientific_fit_admitted': False}
        except BaseException as error:
            terminal['overall_status'] = 'CUSTODY_COLLECTION_FAILED_NO_QUALIFICATION'
            atomic_json(OUT / 'SUPERVISOR_TERMINAL.json', terminal)
            custody = {'schema': 'graph-count-conditioned-pattern-physical-cpu-supervisor-custody-v1', 'UTC': utc(),
                       'overall_status': 'CUSTODY_COLLECTION_FAILED_NO_QUALIFICATION', 'physical_status': physical,
                       'terminal_path': str(OUT / 'SUPERVISOR_TERMINAL.json'),
                       'terminal_sha256': sha(OUT / 'SUPERVISOR_TERMINAL.json'),
                       'error': type(error).__name__ + ': ' + str(error), 'automatic_retry': False, 'scientific_fit_admitted': False}
        atomic_json(OUT / 'SUPERVISOR_CUSTODY.json', custody)
        os.close(lock_fd)
    return 0 if terminal['overall_status'] == 'PASS_FABRICATED_CPU_ONLY' and custody['overall_status'] == 'PASS_FABRICATED_CPU_ONLY' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-release', required=True, type=Path)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    release, inner, command, command_pin = preflight(args.root_release, args.release_sha256)
    return supervise(release, inner, command, command_pin, args.release_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
