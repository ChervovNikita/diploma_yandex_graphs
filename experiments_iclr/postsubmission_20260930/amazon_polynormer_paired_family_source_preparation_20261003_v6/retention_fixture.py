"""Stdlib-only fabricated metadata/placeholder custody tests; no model execution."""
import copy
import json
import sys
import tempfile
from pathlib import Path
sys.dont_write_bytecode = True
import common as c
import evaluate
import retention


def case(root, name, selected_updates):
    output = root / name
    (output / 'checkpoints').mkdir(parents=True)
    bindings = {'fixture_only': True, 'fit_id': name}
    records = []
    selected = local = None
    rows = []
    best = -1
    replay = {'bitwise_full_next_step': True, 'fixture_only_not_numerical_evidence': True}
    with (output / 'RETIREMENTS.jsonl').open('x') as retired:
        for update in range(1, 2701):
            improved = update in selected_updates
            count = best + 1 if improved else best
            event = {'actual_update': update, 'author_display_epoch': update - 1,
                     'stage': 'global' if update > 200 else 'local',
                     'stage_epoch': update - 200 if update > 200 else update,
                     'actual_local_updates': min(update, 200), 'actual_global_updates': max(update - 200, 0),
                     'global': update > 200, 'fit_CE': 1.0, 'val_correct': count, 'val_count': 6123,
                     'strict_selected': improved, 'checkpoint': None, 'portable_replay': None}
            if improved:
                path = output / 'checkpoints' / ('selected_%04d.pt' % update)
                path.write_bytes(('FABRICATED_NOT_A_MODEL|' + name + '|' + str(update)).encode())
                checkpoint = c.record(path)
                selection = {k: event[k] for k in ('actual_update', 'author_display_epoch', 'stage', 'stage_epoch',
                    'actual_local_updates', 'actual_global_updates', 'global', 'val_correct', 'val_count')}
                records.append({'checkpoint': checkpoint, 'selection': selection,
                    'checkpoint_io': {'serialized_bytes': checkpoint['bytes']}, 'binary_retention': {'status': 'retained'}})
                selected = checkpoint
                if update <= 200:
                    local = checkpoint
                event['checkpoint'] = checkpoint
                retention.retire(output, records, (selected, local), selected, retired, bindings)
                best = count
            rows.append(event)
    trace = output / 'TRACE.jsonl'
    trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    transition = output / 'TRANSITION.json'
    transition.write_text(json.dumps({'selected_checkpoint': local, 'portable_replay': replay,
        'restored_model_Adam_bitwise': True, 'live_end_local_RNG_preserved_bitwise': True}))
    protected = list({c.object_sha(row): row for row in (selected, local)}.values())
    result = {'bindings': bindings, 'trace': c.record(trace), 'selected_checkpoint': selected,
        'selected_local_checkpoint': local, 'selection': records[-1]['selection'], 'checkpoints': records,
        'transition': c.record(transition), 'local_transition_portable_replay': replay,
        'actual_local_updates': 200, 'actual_global_updates': 2500, 'actual_optimizer_updates': 2700,
        'final_portable_replay': replay,
        'checkpoint_retention': {'policy': retention.POLICY, 'retained_checkpoints': protected,
            'retirement_journal': c.record(output / 'RETIREMENTS.jsonl'),
            'all_epoch_and_selection_records_preserved': True, 'all_superseded_images_reopenable': False,
            'all_selected_candidate_next_step_replays_promised': False}}
    assert evaluate.check_trace(result) == 2700
    validated = retention.validate(result)
    assert validated['retained_binary_count'] == len(protected)
    assert len(list((output / 'checkpoints').glob('*.pt'))) == len(protected)
    altered = copy.deepcopy(result)
    altered['checkpoint_retention']['all_selected_candidate_next_step_replays_promised'] = True
    try:
        retention.validate(altered)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed audit guarantee accepted')
    if validated['retired_binary_count']:
        altered = copy.deepcopy(result)
        next(r for r in altered['checkpoints'] if r['binary_retention']['status'] == 'retired')['binary_retention']['receipt']['intent_sha256'] = '0' * 64
        try:
            retention.validate(altered)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed retirement receipt accepted')
    return {'case': name, 'selected_events': len(selected_updates), 'trace_updates': 2700,
            'retained_images': len(protected), 'retired_images': validated['retired_binary_count'],
            'altered_scope_and_retirement_receipt_rejected': True}


def main():
    with tempfile.TemporaryDirectory(prefix='amazon_retention_metadata_fixture_', dir=c.PHASE) as directory:
        root = Path(directory)
        cases = [case(root, 'global_final', {1, 2, 199, 201, 202, 1001, 2700}),
                 case(root, 'local_final', {1, 2, 199}),
                 case(root, 'first_update_only', {1}),
                 case(root, 'all_2700_strict_improvements', set(range(1, 2701)))]
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    result = {'schema': 'amazon_polynormer_retention_metadata_fixture_v1', 'status': 'passed_metadata_only',
        'cases': cases, 'project_execution_scope': 'stdlib metadata/placeholder custody only',
        'numerical_model_or_data_execution': False, 'real_checkpoint_or_array_payloads_opened': False,
        'real_existing_outputs_or_prior_history_deleted': False}
    c.write(c.PACKET / 'RETENTION_FIXTURE_RESULT.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
