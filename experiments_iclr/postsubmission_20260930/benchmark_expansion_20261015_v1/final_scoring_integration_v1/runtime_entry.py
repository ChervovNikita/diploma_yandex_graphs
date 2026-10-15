"""Fixed reservation-owned scoring invocation; bounded scalar child checks.

Campaign-wide validation runs once in the supervisor. A byte-pinned capsule
on the actual committed reservation binds each child to its own request.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    expansion = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(expansion / 'portable_dispatcher_v3'))
    from host_binding import guard, load_config
    config = guard()  # before project metadata/runtime imports
    _, config_pin = load_config()
    sys.path.insert(0, str(expansion))
    from final_scoring_integration_v1.common import exclusive, pin, require, source_module, verify, verify_source
    from final_scoring_integration_v1.preparation import validate_capsule, verify_own_scalar_chain
    import backend as portable_backend
    import custody
    root = Path(config['project_root'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True); parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--capsule', required=True); parser.add_argument('--capsule-sha256', required=True)
    args = parser.parse_args()
    request_pin, capsule_pin = pin(args.request, root), pin(args.capsule, root)
    require(request_pin['sha256'] == args.request_sha256 and capsule_pin['sha256'] == args.capsule_sha256, 'Exact admitted request/capsule required')
    request = verify(request_pin, root, json_metadata=True)
    capsule = verify(capsule_pin, root, json_metadata=True)
    preparation_pin = capsule['preparation']
    preparation = verify(preparation_pin, root, json_metadata=True)
    task = capsule['task']
    require(preparation['host_config'] == config_pin and preparation['worker'] == config['worker'], 'Prepared worker config differs')
    go = verify(preparation['GO_view'], root, json_metadata=True)
    require(go['schema'] == 'portable-human-GO-v2' and go['user_explicit_GO'] is True
            and request['worker'] in go['allowed_workers'], 'Actual same-decision campaign GO required')
    verify(go['human_instruction_evidence'], root)
    own_check = validate_capsule(preparation, capsule, request, go)
    # Shared immutable metadata/source hashes may be checked. No unrelated
    # task descriptor, outcome, checkpoint or qualification is traversed here.
    for field in ('run_spec', 'release', 'training_release', 'training_seal', 'global_decision', 'global_contract'):
        verify(preparation[field], root)
    verify_source(root, preparation['source_manifest'])
    observer = source_module(expansion / 'benchmark_reporting_integration_v2/observe_exports.py', '_accepted_final_scalar_helpers')
    ledger = root / config['phase_relative'] / '.portable_campaign_host_v2'
    def admit_runtime():
        guard(config)
        verify(preparation_pin, root, json_metadata=True)
        verify(capsule_pin, root, json_metadata=True)
        verify(request_pin, root, json_metadata=True)
        verify_own_scalar_chain(capsule, root)
        verify(capsule['profile'], root, json_metadata=True)
        verify(task['role_contract'], root, json_metadata=True)
        with custody.locked(ledger / 'HOST.lock'):
            envelope = custody.read(ledger / 'STATE.json')
            require(envelope['sha256'] == custody.digest(envelope['state'])
                    and envelope['state']['ownership_sha256'] == preparation['final_ownership_sha256'], 'Final reservation ledger changed')
            rows = [row for row in envelope['state']['active'].values()
                    if row['assignment']['id'] == task['id'] and (row.get('child_identity') or {}).get('pid') == os.getpid()]
            require(len(rows) == 1 and rows[0]['state'] == 'running'
                    and rows[0]['final_admission'] == {'preparation': preparation_pin, 'capsule': capsule_pin, 'request': request_pin}
                    and rows[0]['assignment']['worker'] == task['worker']
                    and rows[0]['assignment']['science_contract_hash'] == task['science_contract_hash']
                    and portable_backend.identity_status(rows[0]['child_identity']) == 'same'
                    and portable_backend.identity_status(rows[0]['supervisor_identity']) == 'same', 'Only the original committed resource-owned child may score')
    result = None
    started = time.perf_counter()
    try:
        from final_selected_scoring_v2.backend import score_fit, aggregate_i4
        function = score_fit if task['kind'] == 'selected_fit_TEST' else aggregate_i4
        result = function(request, preparation['GO_view'], root, admit_runtime=admit_runtime)
    finally:
        elapsed = time.perf_counter() - started
        record = {'schema': 'observed-final-scoring-invocation-v1', 'task_id': task['id'],
            'request': request_pin, 'admission_capsule': capsule_pin, 'supervisor_preparation': preparation_pin,
            'run_spec': preparation['run_spec'], 'producer': pin(__file__, root),
            'accepted_scalar_helper_source': pin(observer.__file__, root),
            'worker_wall_seconds': elapsed, 'scalar_RESULT': None if result is None else observer.scalar_copy(result),
            'RESULT': pin(Path(request['output']) / 'RESULT.json', root) if (Path(request['output']) / 'RESULT.json').is_file() else None,
            'native_cost_may_overlap_observed_wall': True, 'RNG_calls': 0, 'tensor_deserializations': 0,
            'extra_model_forwards_or_updates': 0, 'TEST_accessed': None if result is None else result['TEST_accessed'],
            'bounded_child_metadata_checks': own_check}
        exclusive(Path(args.request).parent / 'SCORING_OBSERVATION.json', record)
    print(json.dumps({'task_id': task['id'], 'status': result['status'], 'TEST_accessed': result['TEST_accessed']}))
    raise SystemExit(2 if result['status'] == 'failed' else 0)


if __name__ == '__main__':
    main()
