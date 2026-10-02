"""Check retained numerical receipts and lineage, without recomputing tensors."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
REMOTE = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930/'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def descriptor(path):
    return {'path': REMOTE + str(path.relative_to(PHASE)), 'sha256': digest(path)}

def main():
    output = PHASE/'graph_init_root_decisions_v1/FIRST_QUALIFICATION_METADATA_AUDIT.json'
    assert not output.exists()
    directory = PHASE/'graph_init_execution_root_v1/study_v1/qualify/Squirrel_seed17'
    freeze = read(directory/'FREEZE.json')
    completed = read(directory/'COMPLETED.json')
    result = read(directory/'QUALIFICATION.json')
    registry_path = PHASE/'graph_init_execution_root_v1/study_v1/GRAPH_INIT_ATTEMPT_REGISTRY.json'
    registry = read(registry_path)
    assert completed['completed'] and freeze['completed'] and freeze['passed'] and result['passed']
    assert freeze['phase'] == 'qualify' and freeze['qualification_only'] and not freeze['final_labels_read']
    assert result['label_scope'] == ['train'] and not result['scientific_result'] and result['disposable_only']
    assert freeze['context'] in registry['contexts'] and freeze['attempt_registry'] == descriptor(registry_path)
    rows = [r for r in registry['attempts'] if r['context_sha256'] == freeze['context_sha256']
            and r['phase'] == 'qualify' and r['arm'] is None]
    assert len(rows) == 1
    for item in freeze['payload']:
        path = directory/item['path']
        assert digest(path) == item['sha256'] and path.stat().st_size == item['bytes']
    claim_path = PHASE/freeze['attempt_claim']['path'][len(REMOTE):]
    assert digest(claim_path) == freeze['attempt_claim']['sha256']
    claim = read(claim_path)
    assert claim['attempt'] == rows[0] and claim['attempt_registry'] == descriptor(registry_path)
    assert result['installation']['max_absolute'] == 0 and result['installation']['passed']
    assert result['equivalence']['passed']
    for key in ('gradient_checks', 'parameter_checks', 'state_checks', 'logit_checks'):
        assert all(row['passed'] for row in result['equivalence'][key])
    directions = result['ad']['records']
    assert result['ad']['passed'] and len(directions) == 3
    for row in directions:
        assert row['dual_relative_error'] <= 2e-4
        assert [fd['epsilon'] for fd in row['finite_differences']] == [1e-3, 3e-4]
        assert all(fd['logits_relative_error'] <= .05 and fd['ce_directional_error'] <= .05
                   for fd in row['finite_differences'])
    initializer = result['initializer']
    assert initializer['status'] == 'graph_band' and initializer['reason'] == 'accepted'
    assert all(math.isfinite(float(x)) for x in initializer['attempts'][0]['route_train_ce'])
    outer_path = PHASE/'graph_init_execution_root_v1/supervision/qualify_Squirrel_seed17_v1_outer/TERMINAL.json'
    outer = read(outer_path)
    assert outer['complete'] and not outer['timed_out'] and outer['root_request_unchanged']
    terminal = PHASE/'graph_init_execution_root_v1/root_receipts/qualify_Squirrel_seed17_v1/TERMINAL.json'
    assert read(terminal)['completed']
    record = dict(
        schema='graph-init-first-qualification-root-metadata-audit-v1',
        UTC=datetime.now(timezone.utc).isoformat(),
        status='PASS_RETAINED_NUMERICAL_GATES_AND_METADATA_LINEAGE',
        scope='Root rechecked retained qualification claims, payload hashes, registry/context linkage, recorded numerical errors and outer/root terminals. Numerical tensors were not recomputed.',
        qualification_freeze=descriptor(directory/'FREEZE.json'),
        qualification_receipt=descriptor(directory/'QUALIFICATION.json'),
        registry=descriptor(registry_path), root_terminal=descriptor(terminal),
        outer_terminal=descriptor(outer_path),
        whole_supervised_seconds=outer['whole_supervised_seconds'],
        max_operation_allocated_bytes=max(r['peak_allocated_bytes'] for r in completed['costs']),
        max_operation_reserved_bytes=max(r['peak_reserved_bytes'] for r in completed['costs']),
        closure_calls=result['closure_count'], initializer_status=initializer['status'],
        installation_max_absolute=0,
        dual_error_max=max(r['dual_relative_error'] for r in directions),
        fd_logit_relative_max=max(fd['logits_relative_error'] for r in directions for fd in r['finite_differences']),
        source_label_scope=['train'], heldout_scoring=False, scientific_performance_claim=False,
        actual_warm_state_gate_still_required=True, other_target_qualifications_pending=5,
        full_schedule_feasibility_asserted=False, metadata_registration_prior_failure_preserved=True,
        audit_source_sha256=digest(Path(__file__)))
    with output.open('x') as stream:
        json.dump(record, stream, indent=2); stream.write('\n')
    print(json.dumps({k: record[k] for k in ('status', 'whole_supervised_seconds',
        'max_operation_allocated_bytes', 'max_operation_reserved_bytes', 'scientific_performance_claim')}))

if __name__ == '__main__':
    main()
