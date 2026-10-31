"""Confined stdlib root entry: only fresh registry metadata."""
import argparse
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
from admission_support import (PHASE, REMOTE_PHASE, REMOTE_REPO, LOGIN, GPU_UUID, HERE,
    R17_REL, R17_MANIFEST_SHA, require, confined, sha, read, bound, write,
    source_descriptors, verify_sources, expected_attempts, object_hash, bindings)


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--supervisor', required=True)
    args = parser.parse_args()
    require(PHASE == REMOTE_PHASE and Path.cwd().resolve() == REMOTE_REPO,
            'Exact authorized remote repository required')
    require(Path(os.path.abspath(sys.executable)) == REMOTE_REPO/'.venv/bin/python',
            'Exact repository venv entry point required')
    require(os.environ.get('GNNM_PHASE_ROOT') == str(PHASE) and
            os.environ.get('GNNM_SSH_DESTINATION') == LOGIN and
            os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'Exact route/environment required')
    request_path = confined(args.request)
    request_hash = sha(request_path)
    request = read(request_path)
    require(request['schema'] == 'graph-init-root-launch-request-v1' and request['root_admitted'] is True and
            request['full_fit_admitted'] is False and request['heldout_scoring_admitted'] is False and
            request['action'] == 'register', 'Only explicitly admitted initial actions allowed')
    require(request['source_seals'] == source_descriptors() and
            request['entry_script'] == str(HERE/'graph_init_root_entry.py') and
            request['driver_script'] == str(PHASE/R17_REL/'prototype/graph_init_driver.py'),
            'Exact source/entry/driver identities required')
    require(request['source_packet'] == str(PHASE/R17_REL) and
            request['source_manifest'] == source_descriptors()[0], 'R17 source packet differs')
    supervisor = confined(args.supervisor)
    require(supervisor == Path(request['supervisor_directory']) and supervisor.is_dir(),
            'Exact active inner supervisor required')
    environment = read(supervisor/'environment.json')
    command = read(supervisor/'command.json')
    require(environment['repository'] == str(REMOTE_REPO) and environment['phase'] == str(PHASE) and
            command['cwd'] == str(REMOTE_REPO) and command['shell'] is False and
            command['argv'] == [str(REMOTE_REPO/'.venv/bin/python'), str(HERE/'graph_init_root_entry.py'),
                                '--request', str(request_path), '--supervisor', str(supervisor)],
            'Entry must run inside the declared repository supervisor')
    route = read(bound(request['allocation_route']))
    require(route['ssh_destination'] == LOGIN and route['repository'] == str(REMOTE_REPO) and
            route['visible_gpu_count'] == 1 and route['repository_pwd_matches'] is True,
            'Route evidence differs')
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15)
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    require(len(rows) == 1 and [x.strip() for x in rows[0].split(',')] ==
            ['0', GPU_UUID, 'NVIDIA A100-SXM4-80GB', '81920'], 'Exact one-GPU UUID required')
    verify_sources()
    from continuation_support import own_sources, verify_own_sources
    own = own_sources()
    verify_own_sources(*own)
    require(isinstance(request['protected_files'], list) and bool(request['protected_files']),
            'Protected source descriptors required')
    require(all(record in request['protected_files'] for record in own),
            'Registration must protect the exact sealed wrapper manifest and seal')
    for record in request['protected_files']:
        bound(record)
    decision_path = bound(request['root_decision'])
    decision = read(decision_path)
    require(decision['approved'] is True and isinstance(decision['approved_by'], str) and
            decision['approved_by'].strip() and decision['approved_utc'] and
            decision['independent_source_audit_accepted'] is True and
            decision['r17_source_manifest'] == source_descriptors()[0] and
            decision['full_fit_authorized'] is False and decision['heldout_scoring_authorized'] is False,
            'Separate signed root decision required')
    bound(decision['independent_source_audit'])
    bound(decision['entry_source']); bound(decision['support_source'])
    require(decision['entry_source']['path'] == str(Path(__file__)) and
            decision['support_source']['path'] == str(HERE/'admission_support.py'),
            'Signed entry sources differ')
    payload_path = bound(request['phase_payload'])
    payload = read(payload_path)
    output = confined(request['output'])
    require(not output.exists(), 'Prospectively declared child output already exists')
    action = request['action']
    if action == 'register':
        require(decision['schema'] == 'graph-init-root-registration-decision-v1' and
                decision['registration_authorized'] is True and
                decision['resource_forecasts_accepted'] is True and
                decision['source_copy_memory_review_accepted'] is True and
                decision['predecessor_history_complete'] is True and
                decision['cold_qualification_authorized'] is False, 'Only registry admission accepted')
        require(payload['schema'] == 'graph-init-registry-request-v1' and
                payload['registration_authorized'] is True and
                payload['study_id'] == decision['study_id'] and
                payload['anchor_directory'] == decision['anchor_directory'] and
                output == Path(payload['anchor_directory'])/'GRAPH_INIT_ATTEMPT_REGISTRY.json' and
                not output.parent.exists(), 'Exact new registry anchor required')
        from lineage_support import verify_lineage, prior_registries
        verify_lineage()
        require(payload['prior_attempt_registries'] == prior_registries(), 'Complete failed v2 registry lineage required')
        draft = read(bound(decision['prepared_contexts']))
        require(object_hash(draft) == decision['draft_contexts_sha256'], 'Signed draft contexts differ')
        approved = {(r['graph'], r['seed']): r['forecast'] for r in decision['accepted_forecasts']}
        require(len(decision['accepted_forecasts']) == len(approved) == 6,
                'Signed exact six forecast rows required')
        final = copy.deepcopy(draft)
        for context in final:
            require(context['resource_forecast']['admitted'] is False, 'Original draft context required')
            context['resource_forecast']['admitted'] = True
            context['resource_forecast']['forecast'] = approved[(context['graph'], context['seed'])]
            context['resource_forecast']['evidence'] += [decision['independent_source_audit'], request['root_decision']]
        require(payload['contexts'] == final, 'Final contexts must exactly implement signed root forecast decision')
        lineage = read(bound(payload['lineage_authorization']))
        require(lineage['approved'] is True and lineage['predecessor_history_complete'] is True and
                lineage['contexts_sha256'] == object_hash(payload['contexts']) and
                lineage['study_id'] == payload['study_id'] and
                lineage['anchor_directory'] == payload['anchor_directory'] and
                lineage['prior_attempt_registries'] == prior_registries() and
                lineage['source_bindings'] == bindings()[0] and lineage['protocol'] == bindings()[1],
                'Exact finalized lineage required')
        require(request['whole_cap_seconds'] == 600, 'Metadata registration cap is600 seconds')
        argument = '--request'
    # Root receipt belongs outside the new driver phase output and registry anchor.
    receipt = confined(request['receipt_directory'])
    anchor = Path(payload['anchor_directory'])
    require(not receipt.is_relative_to(anchor) and not output.is_relative_to(receipt) and
            receipt != supervisor and not receipt.exists(), 'Separate fresh root receipt directory required')
    receipt.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write(receipt/'START.json', {'request_sha256': request_hash, 'root_decision': request['root_decision'],
        'action': action, 'output': str(output), 'supervisor_directory': str(supervisor), 'UTC': utc(),
        'gpu_uuid': GPU_UUID, 'entry_source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try:
        child_env = os.environ.copy()
        child_env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Existing bounded_run_v1 owns the whole process-group cap. No recipe changes.
        argv = [sys.executable, request['driver_script'], action, argument, str(payload_path), '--output', str(output)]
        child = subprocess.run(argv, env=child_env, check=False)
        verify_sources()
        for record in request['protected_files']:
            bound(record)
        bound(request['root_decision']); bound(request['phase_payload'])
        bound(decision['independent_source_audit'])
        require(sha(request_path) == request_hash, 'Root request changed during execution')
        write(receipt/'TERMINAL.json', {'completed': child.returncode == 0, 'child_exit_code': child.returncode,
            'seconds': time.monotonic()-start, 'UTC': utc(), 'request_sha256': request_hash,
            'action': action, 'output': str(output), 'final_labels_read': False,
            'report_eligible': False, 'automatic_retry': False})
        return child.returncode
    except Exception as error:
        write(receipt/'FAILED_ATTEMPT.json', {'error_type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'seconds': time.monotonic()-start, 'UTC': utc(),
            'automatic_retry': False})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
