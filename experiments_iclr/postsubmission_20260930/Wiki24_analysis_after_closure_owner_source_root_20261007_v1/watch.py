"""Wait for the unchanged Wiki24 to end, then run the approved CPU scalar analysis once."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
F = P / 'learnable_internal_be_WikiCS_scientific_family_execution_root_20261007_v1'
D = P / 'internal_BE_Wiki24_closed_family_reader_source_20261007_v3'
E = P / 'Wiki24_analysis_after_closure_execution_root_20261007_v1'
PY = P / 'native_ncn_runtime_20261005_v1/.venv/bin/python'
EXPECTED_MANIFEST = '476a59b56bd3377c6c93d0048a979f2fe6f08795e77b68e1452ebc874ec96042'
WAIT_SECONDS = 86400
STAGE_ACTIVE_SECONDS, STAGE_CLEANUP_SECONDS = 1800, 10


def write(name, value):
    path = E / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def read(path):
    return json.loads(path.read_text())


def binding(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return dict(path=str(path.relative_to(P)), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def identity(pid):
    try:
        fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()
        return dict(PID=pid, start_ticks=int(fields[19]), state=fields[0])
    except FileNotFoundError:
        return None


def same_live(pid, ticks):
    value = identity(pid)
    return value is not None and value['start_ticks'] == ticks and value['state'] not in ('Z', 'X')


def terminal_handles(closure):
    """Inspect only recorded owned scientific supervisor/worker identities."""
    handles = {(510850, 6015502511)}
    for row in closure['cells']:
        for kind in ('scientific_supervisor', 'new_resource_attempt'):
            supervisor = row.get(kind, {}).get('supervisor_owner')
            if supervisor:
                handles.add((supervisor['pid'], supervisor['start_ticks']))
        bound_terminal = row.get('fit_custody', {}).get('terminal')
        if bound_terminal:
            path = P / bound_terminal['path']
            assert binding(path)['sha256'] == bound_terminal['sha256']
            terminal = read(path)
            if type(terminal.get('child_pid')) is int and type(terminal.get('child_start_ticks')) is int:
                handles.add((terminal['child_pid'], terminal['child_start_ticks']))
        if 'new_resource_attempt' in row:
            job_path = F / 'resource_jobs' / (row['cell'] + '.json')
            job = read(job_path)
            assert binding(job_path)['sha256'] == row['new_resource_attempt']['supervisor_owner']['job_sha256']
            live_path = P / job['supervisor_receipt_path']
            terminal_path = live_path.with_name(live_path.stem + '_TERMINAL.json')
            terminal = read(terminal_path)
            if type(terminal.get('child_pid')) is int and type(terminal.get('child_start_ticks')) is int:
                handles.add((terminal['child_pid'], terminal['child_start_ticks']))
    return [dict(PID=pid, start_ticks=ticks, current_identity=identity(pid), owned_identity_live=same_live(pid, ticks))
            for pid, ticks in sorted(handles)]


def run_once(name, program, activation, env):
    """Ordinary owned subprocess with a finite cap; no retry or other-job signals."""
    config = E / (name + '_ACTIVATION.json')
    assert not config.exists()
    config.write_text(json.dumps(activation, indent=2, sort_keys=True) + '\n')
    command = [str(PY), '-B', str(D / program), '--activation', str(config),
               '--activation-sha256', binding(config)['sha256']]
    began = time.monotonic()
    with (E / (name + '.log')).open('x') as log:
        child = subprocess.Popen(command, cwd=R, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        handle = identity(child.pid)
        write(name + '_OWNER.json', dict(command=command, identity=handle,
              active_seconds=STAGE_ACTIVE_SECONDS, cleanup_seconds=STAGE_CLEANUP_SECONDS))
        timed_out = False
        try:
            code = child.wait(timeout=STAGE_ACTIVE_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            assert same_live(child.pid, handle['start_ticks']) and os.getpgid(child.pid) == child.pid
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                assert same_live(child.pid, handle['start_ticks']) and os.getpgid(child.pid) == child.pid
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=5)
    result = dict(exit_code=code, timed_out=timed_out, reaped=True,
                  elapsed_seconds=time.monotonic() - began, identity=handle, automatic_retry=False)
    write(name + '_TERMINAL.json', result)
    assert code == 0 and not timed_out, name + ' failed; inspect retained log and costs'


def main():
    started = time.monotonic()
    os.chdir(R); os.umask(0o077)
    assert socket.gethostname() == 'anogena-2-0'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    assert binding(D / 'MANIFEST.json')['sha256'] == EXPECTED_MANIFEST
    E.mkdir(exist_ok=False)
    write('OWNER.json', dict(identity=identity(os.getpid()), watched_owner=dict(PID=510850, start_ticks=6015502511),
                            wait_seconds=WAIT_SECONDS, stages=['EXTRACT', 'READ'], GPU_work=False, automatic_retry=False))
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        closure_path = F / 'FAMILY_CLOSURE.json'
        while True:
            assert time.monotonic() - started < WAIT_SECONDS, 'Analysis wait expired; no science restarted'
            owner_live = same_live(510850, 6015502511)
            if closure_path.is_file():
                closure = read(closure_path)
                assert len(closure['cells']) == 24 and closure['closed'] is True
                handles = terminal_handles(closure)
                if not owner_live and not any(x['owned_identity_live'] for x in handles):
                    break
            elif not owner_live:
                raise RuntimeError('Scientific owner ended without full family closure; no opening or restart')
            write('WAIT.json', dict(watched_owner_live=owner_live, full_closure_present=closure_path.is_file(),
                                   elapsed_seconds=time.monotonic() - started, predictive_values_opened=False))
            time.sleep(60)
        approval = P / 'Wiki24_analysis_source_root_review_20261007_v1/READER_APPROVAL.json'
        family_release = P / 'learnable_internal_be_WikiCS_scientific_family_activation_20261007_v1/RELEASE.json'
        evidence = dict(owner_and_children_terminal=True, owner=binding(F / 'OWNER.json'),
                        closure=binding(closure_path), observed_owned_handles=handles,
                        observed_epoch_seconds=time.time(), no_scientific_process_signalled=True)
        write('TERMINAL_EVIDENCE.json', evidence)
        common = dict(schema='internal-be-Wiki24-reader-activation-v1', stage_enabled=True,
                      root_comparative_opening_authorized=True, source_review_approved=True,
                      reader_manifest_sha256=EXPECTED_MANIFEST, root_reader_inspection=binding(approval),
                      family_release=binding(family_release), family_closure=binding(closure_path), TEST_access=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                   PYTHONPATH=str(P / 'native_ncn_dependency_overlay_20261005_v1') + os.pathsep + str(R / '.venv/lib/python3.11/site-packages'))
        env.pop('PYTHONHOME', None)
        extract = dict(common, stage='extract_selected_metadata', trusted_checkpoint_metadata_deserialization_authorized=True,
                       metadata_extraction_cost_charged=True, output_directory=str((E / 'extraction').relative_to(P)))
        run_once('EXTRACT', 'extract_selected_metadata.py', extract, env)
        reader = dict(common, stage='read_selected_metadata', metadata_export=binding(E / 'extraction/METADATA_EXPORT.json'),
                      metadata_extraction_cost=binding(E / 'extraction/EXTRACTION_COST.json'),
                      output_directory=str((E / 'readout').relative_to(P)))
        run_once('READ', 'read_wiki24.py', reader, env)
        write('COMPLETE.json', dict(complete=True, report=binding(E / 'readout/REPORT.json'),
                                    report_markdown=binding(E / 'readout/REPORT.md'), GPU_work=False, TEST_access=False))
    except Exception as error:
        write('FAILURE.json', dict(complete=False, error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        write('OWNER_COST.json', dict(inclusive_wait_and_stage_seconds=time.monotonic() - started,
              owned_child_CPU_user_seconds=after.ru_utime - child_usage.ru_utime,
              owned_child_CPU_system_seconds=after.ru_stime - child_usage.ru_stime,
              cumulative_owned_child_peak_RSS_bytes=after.ru_maxrss * 1024,
              GPU_work=False, scientific_jobs_changed=False))


if __name__ == '__main__':
    main()
