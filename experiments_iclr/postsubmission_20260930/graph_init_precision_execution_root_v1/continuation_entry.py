"""Root-authorized exact finite continuation under existing whole supervision."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from continuation_support import (HERE, PHASE, REMOTE_PHASE, REMOTE_REPO, LOGIN, GPU_UUID, PHASES,
    R17_REL, require, confined, sha, read, bound, write, decision_guard, expected_admission,
    source_descriptors, completed_phase)
from build_continuation import next_row


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--supervisor', required=True)
    args = parser.parse_args()
    require(PHASE == REMOTE_PHASE and Path.cwd().resolve() == REMOTE_REPO and
            Path(os.path.abspath(sys.executable)) == REMOTE_REPO/'.venv/bin/python', 'Exact authorized repository/venv required')
    require(os.environ.get('GNNM_PHASE_ROOT') == str(PHASE) and os.environ.get('GNNM_SSH_DESTINATION') == LOGIN and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Exact root route/cache environment required')
    request_path = confined(args.request)
    request_hash = sha(request_path)
    request = read(request_path)
    require(request['schema'] == 'graph-init-continuation-launch-request-v1' and request['root_admitted'] is True and
            request['action'] in PHASES and request['heldout_scoring_admitted'] is False and
            request['compare_admitted'] is False and request['final_labels_accessible'] is False and
            request['automatic_retry'] is False and request['full_fit_admitted'] is (request['action'] == 'fit'),
            'Only separately signed finite source phases are allowed')
    require(request['entry_script'] == str(Path(__file__)) and
            request['driver_script'] == str(PHASE/R17_REL/'prototype/graph_init_driver.py') and
            request['source_packet'] == str(PHASE/R17_REL) and request['source_seals'] == source_descriptors(),
            'Exact source/entry/driver paths required')
    supervisor = confined(args.supervisor)
    require(supervisor == Path(request['supervisor_directory']) and supervisor.is_dir(), 'Declared active inner supervisor required')
    env = read(supervisor/'environment.json')
    command = read(supervisor/'command.json')
    require(env['repository'] == str(REMOTE_REPO) and env['phase'] == str(PHASE) and command['shell'] is False and
            command['cwd'] == str(REMOTE_REPO) and command['argv'] == [str(REMOTE_REPO/'.venv/bin/python'),
            str(Path(__file__)), '--request', str(request_path), '--supervisor', str(supervisor)], 'Exact supervisor invocation required')
    route = read(bound(request['allocation_route']))
    require(route['ssh_destination'] == LOGIN and route['repository'] == str(REMOTE_REPO) and
            route['visible_gpu_count'] == 1 and route['repository_pwd_matches'] is True, 'Exact route evidence required')
    rows = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15).stdout.strip().splitlines()
    require(len(rows) == 1 and [x.strip() for x in rows[0].split(',')] ==
            ['0', GPU_UUID, 'NVIDIA A100-SXM4-80GB', '81920'], 'Exact one-GPU UUID required')
    decision_path = bound(request['root_decision'])
    decision, registry, plan = decision_guard(decision_path)
    row = next_row(decision, registry, plan)
    require(row is not None and request['attempt'] == row and request['action'] == row['phase'] and
            request['output'] == row['output'] and request['whole_cap_seconds'] == row['whole_cap_seconds'],
            'Exact next registered attempt and forecast whole cap required')
    admission_path = bound(request['phase_admission'])
    require(read(admission_path) == expected_admission(decision, registry, row), 'Admission must match completed canonical dependencies')
    run_root = Path(decision['coordinator_run_root'])
    key = row['key']
    require(request_path == run_root/'requests'/key/'REQUEST.json' and
            admission_path == run_root/'requests'/key/'ADMISSION.json' and
            supervisor == run_root/'supervision'/(key+'_inner') and
            request['outer_supervisor_directory'] == str(run_root/'supervision'/(key+'_outer')),
            'Fixed single-use request/supervision paths required')
    output = confined(row['output'])
    receipt = confined(request['receipt_directory'])
    require(receipt == run_root/'root_receipts'/key and not receipt.is_relative_to(Path(registry['anchor_directory'])) and
            not output.exists() and not receipt.exists(), 'Fresh separate root receipt and canonical driver output required')
    # Confirm the active bounded process covers this exact forecast cap.
    outer_path = confined(os.environ.get('GNNM_BOUND_START_JSON', ''))
    outer = read(outer_path)
    require(outer_path == run_root/'supervision'/(key+'_outer')/'START.json' and
            outer['whole_cap_seconds'] == row['whole_cap_seconds'] and
            outer['inner_supervisor_directory'] == str(supervisor) and
            outer['root_request']['sha256'] == request_hash, 'Exact active whole-process bound required')
    receipt.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write(receipt/'START.json', {'UTC': utc(), 'request_sha256': request_hash, 'root_decision': request['root_decision'],
        'attempt': row, 'gpu_uuid': GPU_UUID, 'whole_cap_seconds': row['whole_cap_seconds'],
        'outer_start': {'path': str(outer_path), 'sha256': sha(outer_path)}, 'automatic_retry': False})
    try:
        child_env = os.environ.copy()
        child_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        child = subprocess.run([sys.executable, request['driver_script'], row['phase'], '--admission',
            str(admission_path), '--output', str(output)], env=child_env, check=False)
        # Recheck immutable sources/registry/decision after all scientific work.
        decision_guard(decision_path)
        bound(request['phase_admission'])
        require(sha(request_path) == request_hash, 'Root request changed during execution')
        canonical = [r for r in registry['attempts'] if r['key'] == key][0]
        phase_terminal = None
        if child.returncode == 0:
            _, phase_terminal, _ = completed_phase(decision['attempt_registry'], registry, canonical)
        write(receipt/'TERMINAL.json', {'UTC': utc(), 'completed': child.returncode == 0,
            'child_exit_code': child.returncode, 'seconds': time.monotonic()-start,
            'request_sha256': request_hash, 'phase_terminal': phase_terminal,
            'final_labels_read': False, 'compare_or_report_executed': False, 'automatic_retry': False})
        return child.returncode
    except Exception as error:
        write(receipt/'FAILED_ATTEMPT.json', {'UTC': utc(), 'error_type': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(), 'seconds': time.monotonic()-start,
            'automatic_retry': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
