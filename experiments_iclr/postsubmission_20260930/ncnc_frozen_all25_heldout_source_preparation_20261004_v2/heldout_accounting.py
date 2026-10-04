"""Stdlib durable work ledger and all25 failed public closure; no numerical imports."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import tempfile

ARMS = ('native_single_64', 'independent_native_4', 'factorized_private_4',
        'factorized_pooled_after_clamp_4', 'native_single_70')
COUNTERS = ('attempted', 'entered_original_scorer', 'returned', 'completed_validated',
            'cells_completed', 'official_metric_calls', 'official_metric_attempted',
            'official_metric_returned', 'official_metric_validated', 'metric_helper_attempted', 'metric_helper_returned')


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def descriptor(path):
    path = Path(path)
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def private_json(value, *, seen=None, depth=0):
    """Preserve receipt fields while explicitly marking malformed/nonfinite nodes."""
    if value is None or type(value) in (str, bool, int): return value
    if type(value) is float:
        return value if math.isfinite(value) else dict(nonfinite_float=repr(value))
    if depth >= 16: return dict(unavailable_node='receipt_depth_limit',type=type(value).__name__)
    if type(value) not in (list,tuple,dict): return dict(unsupported_type=type(value).__name__,tensor_or_object_body_exported=False)
    seen=set() if seen is None else seen
    if id(value) in seen: return dict(unavailable_node='cyclic_receipt',type=type(value).__name__)
    seen.add(id(value))
    try:
        if type(value) is dict:
            if all(type(k) is str for k in value): return {k:private_json(v,seen=seen,depth=depth+1) for k,v in value.items()}
            return dict(dictionary_with_non_string_keys=[dict(key=private_json(k,seen=seen,depth=depth+1),value=private_json(v,seen=seen,depth=depth+1)) for k,v in value.items()])
        return [private_json(v,seen=seen,depth=depth+1) for v in value]
    finally: seen.remove(id(value))


def new_work():
    return dict(scorer_calls_planned=40, **{k: 0 for k in COUNTERS}, cells_planned=25,
                official_metric_calls_planned=25, training_updates=0, automatic_retry=False,
                official_metric_calls_semantics='alias_of_official_metric_attempted')


def new_cells():
    return [dict(arm=arm, base_seed=seed, status='NOT_ATTEMPTED', phase='not_started',
                 TEST_hits50=None, official_metric_attempted=False,
                 official_metric_returned=False, official_metric_validated=False)
            for arm in ARMS for seed in range(5)]


def public_cells(cells):
    return [{k: c[k] for k in ('arm', 'base_seed', 'status', 'phase',
                             'official_metric_attempted', 'official_metric_returned',
                             'official_metric_validated')} | dict(TEST_hits50=None) | ({'failed_phase':c['failed_phase']} if 'failed_phase' in c else {}) for c in cells]


def progress(output, result, work, cells, phase, current_cell, current_slot):
    # The private copy precedes the count-only public ledger. Both are atomically fsynced.
    atomic_json(output / 'PRIVATE_CELL_RECEIPTS.json', private_json(dict(
        identity=result['identity'], root_release_sha256=result['root_release_sha256'],
        heldout_source_manifest_sha256=result['heldout_source_manifest_sha256'], cells=cells)))
    atomic_json(output / 'STATUS.json', dict(schema='ncnc-frozen-all25-heldout-progress-v2',
        status='IN_PROGRESS', identity=result['identity'],
        root_release_sha256=result['root_release_sha256'],
        heldout_source_manifest_sha256=result['heldout_source_manifest_sha256'],
        phase=phase, work=work, cells=public_cells(cells),
        current_cell=current_cell, current_slot=private_json(current_slot),
        count_semantics='observed_events_exact_if_child_closes_normally_else_lower_bounds',
        interruption_uncertainty='entry_event_is_persisted_before_call; event_can_precede_actual_entry; return_may_occur_before_next_persist',
        predictive_values_exposed=False, authoritative_terminal_file='HELDOUT_RESULT.json'))


def preserve_bytes(source, target):
    with Path(source).open('rb') as src, Path(target).open('xb') as dst:
        for block in iter(lambda: src.read(1024 * 1024), b''): dst.write(block)
        dst.flush(); os.fsync(dst.fileno())
    return descriptor(target)


def private_inventory(output):
    # Includes any partially written private numerical temporary left by SIGKILL.
    rows = []
    for p in sorted(output.rglob('*')):
        if p.is_file() and (p.name.startswith('PRIVATE_') or p.parent.name == 'PRIVATE_NUMERICAL_EVIDENCE'):
            rows.append(descriptor(p))
    return rows


def supervisor_closure(output, release, release_sha, source_sha, physical, child_spawned):
    """After wait4/cleanup, close missing or physically failed logical endpoints."""
    output = Path(output); output.mkdir(parents=True, mode=0o700, exist_ok=True)
    expected = {(a, s) for a in ARMS for s in range(5)}
    logical_path = output / 'HELDOUT_RESULT.json'
    original = status = None
    evidence = []
    read_errors = []
    for name in ('HELDOUT_RESULT.json', 'STATUS.json'):
        p = output / name
        if not p.exists(): continue
        try:
            value = json.loads(p.read_text())
            assert value.get('schema') in ('ncnc-frozen-all25-heldout-result-v2','ncnc-frozen-all25-heldout-progress-v2')
            assert value['root_release_sha256'] == release_sha
            assert value['heldout_source_manifest_sha256'] == source_sha
            assert value['identity'] == release['identity']
            assert {(c['arm'], c['base_seed']) for c in value['cells']} == expected and len(value['cells']) == 25
            if name == 'HELDOUT_RESULT.json': original = value
            else: status = value
        except Exception as error:
            read_errors.append(dict(file=name, type=type(error).__name__, condition=str(error)))
    # A complete physical terminal alone cannot adopt a missing/malformed logical terminal.
    logical_complete = original is not None and original['status'] == 'ALL25_FROZEN_HELDOUT_CONFIRMATION_COMPLETE'
    logical_failed = original is not None and original.get('status') == 'FAILED_HELDOUT_CONFIRMATION' and original.get('summary','MISSING') is None and all(c['TEST_hits50'] is None for c in original['cells'])
    if physical['status'] == 'PHYSICALLY_COMPLETE' and logical_complete:
        return dict(action='CHILD_SUCCESS_RETAINED', logical_terminal=descriptor(logical_path))
    if logical_failed:
        return dict(action='CHILD_NULL_METRIC_FAILURE_RETAINED', logical_terminal=descriptor(logical_path))
    for name in ('HELDOUT_RESULT.json', 'STATUS.json'):
        p = output / name
        if p.exists(): evidence.append(preserve_bytes(p, output / ('PRIVATE_PRE_SUPERVISOR_' + name)))
    observed = original or status
    cells = public_cells(observed['cells']) if observed else new_cells()
    work = dict(observed['work']) if observed else new_work()
    in_flight = None
    if original is not None:
        count_semantics = 'literal_child_terminal_observed_events; physical_failure_prevents_adoption'
    elif child_spawned:
        count_semantics = 'persisted_observed_event_lower_bounds; entry_events_are_not_proof_calls_began; actual_entries_returns_and_unflushed_work_unknown'
        if observed:
            in_flight = dict(phase=observed.get('phase'), current_cell=observed.get('current_cell'),
                            current_slot=observed.get('current_slot'), actual_call_completion='UNKNOWN', recorded_entry_event_is_not_proof_call_began=True)
        else:
            for k in COUNTERS: work[k] = None
            in_flight = dict(phase='before_first_authenticated_progress_or_progress_unavailable', actual_work='UNKNOWN')
        for c in cells:
            if c['status'] == 'NOT_ATTEMPTED': c['status'] = 'NOT_OBSERVED_ATTEMPTED_BEFORE_INTERRUPTION'
            elif c['status'] != 'PASS': c['status'] = 'INTERRUPTED_WORK_DISPOSITION_UNKNOWN'
    else:
        count_semantics = 'exact_zero_numerical_child_not_spawned'
    closure = dict(schema='ncnc-frozen-all25-heldout-result-v2', status='FAILED_HELDOUT_CONFIRMATION',
        identity=release['identity'], root_release_sha256=release_sha,
        heldout_source_manifest_sha256=source_sha, family_lock=release['family_lock'],
        v4_audit_result=release['v4_audit_result'], policy=release['policy'],
        frozen_unique_training_fits=35, frozen_served_cells=25, training_updates=0,
        cells=cells, summary=None, work=work, count_semantics=count_semantics,
        in_flight_work=in_flight, TEST_opened=original.get('TEST_opened') if original else 'UNKNOWN' if child_spawned else False,
        failures=[dict(category='SUPERVISOR_SYNTHESIZED_ALL25_FAILURE', physical_status=physical['status'],
                       exit_code=physical['exit_code'], bounded_stop_reason=physical['bounded_stop_reason'],
                       error=physical['error'], progress_read_errors=read_errors)],
        private_preclosure_evidence=evidence, private_evidence=private_inventory(output),
        unavailable_partial_scorer_values='A scorer that did not return has no caller-visible tensor; unseen or unflushed values remain unknown.',
        automatic_retry=False, no_success_only_subset_summary=True,
        supervisor_synthesized=True, UTC=datetime.now(timezone.utc).isoformat())
    atomic_json(logical_path, closure)
    atomic_json(output / 'STATUS.json', dict(schema='ncnc-frozen-all25-heldout-progress-v2',
        status=closure['status'], identity=release['identity'], root_release_sha256=release_sha,
        heldout_source_manifest_sha256=source_sha, cells=cells, work=work,
        count_semantics=count_semantics, in_flight_work=in_flight, predictive_values_exposed=False,
        authoritative_result_receipt=descriptor(logical_path)))
    return dict(action='ALL25_NULL_METRIC_FAILURE_SYNTHESIZED', logical_terminal=descriptor(logical_path))
