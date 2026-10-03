"""Retire only newly created superseded fit images; keep exact audit metadata."""
import json
import os
import time
import common as c

POLICY = 'current_best_and_selected_local_v1'


def append(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def retire(output, checkpoints, keep, promoted, stream, bindings):
    """Hash first, fsync intent, unlink owned image, fsync completion; no RNG."""
    protected = list({c.object_sha(row): row for row in keep}.values())
    retired = []
    for record in checkpoints:
        checkpoint = record['checkpoint']
        if checkpoint in protected or record['binary_retention']['status'] == 'retired':
            continue
        started = time.perf_counter()
        path = c.confined(checkpoint['path'])
        directory = c.confined(output) / 'checkpoints'
        c.require(path.parent == directory and path.suffix == '.pt' and
                  checkpoint != promoted and checkpoint['bytes'] == record['checkpoint_io']['serialized_bytes'],
                  'Retirement escaped this attempt or protected image')
        c.verify(checkpoint)
        intent = {'schema': 'amazon_polynormer_retirement_intent_v1',
                  'index': len([r for r in checkpoints if r['binary_retention']['status'] == 'retired']) + 1,
                  'UTC': c.utc(), 'bindings': bindings, 'checkpoint': checkpoint,
                  'promoted_checkpoint': promoted, 'protected_checkpoints': protected,
                  'original_selection': record['selection'], 'checkpoint_io': record['checkpoint_io'],
                  'hash_size_rechecked_before_retirement': True,
                  'superseded_image_portable_replay_promised': False}
        append(stream, intent)
        path.unlink()
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        completed = {'schema': 'amazon_polynormer_retirement_complete_v1',
                     'index': intent['index'], 'intent_sha256': c.object_sha(intent),
                     'checkpoint': checkpoint, 'UTC': c.utc(),
                     'retirement_wall_seconds': time.perf_counter() - started,
                     'binary_absent_after_retirement': not path.exists()}
        c.require(completed['binary_absent_after_retirement'], 'Retired binary remains')
        append(stream, completed)
        record['binary_retention'] = {'status': 'retired', 'receipt': completed}
        retired.append(completed)
    return retired


def validate(result):
    value = result['checkpoint_retention']
    c.require(value['policy'] == POLICY and value['all_epoch_and_selection_records_preserved'] is True and
              value['all_superseded_images_reopenable'] is False and
              value['all_selected_candidate_next_step_replays_promised'] is False,
              'Explicit bounded retention audit scope differs')
    protected = list({c.object_sha(row): row for row in
                     (result['selected_checkpoint'], result['selected_local_checkpoint'])}.values())
    c.require(value['retained_checkpoints'] == protected, 'Final/local protected images differ')
    path = c.verify(value['retirement_journal'])
    events = [json.loads(line) for line in path.read_text().splitlines()]
    retired = [r for r in result['checkpoints'] if r['checkpoint'] not in protected]
    c.require(len(events) == 2 * len(retired), 'Incomplete retirement intent/completion custody')
    by_checkpoint = {}
    selections = {c.object_sha(r['checkpoint']): r['selection'] for r in result['checkpoints']}
    for index in range(0, len(events), 2):
        intent, completed = events[index:index + 2]
        c.require(intent['schema'] == 'amazon_polynormer_retirement_intent_v1' and
                  completed['schema'] == 'amazon_polynormer_retirement_complete_v1' and
                  intent['index'] == completed['index'] == index // 2 + 1 and
                  completed['intent_sha256'] == c.object_sha(intent) and
                  completed['checkpoint'] == intent['checkpoint'] and
                  intent['bindings'] == result['bindings'] and
                  intent['hash_size_rechecked_before_retirement'] is True and
                  intent['superseded_image_portable_replay_promised'] is False and
                  completed['binary_absent_after_retirement'] is True and
                  completed['retirement_wall_seconds'] >= 0,
                  'Retired-image custody differs')
        key = c.object_sha(intent['checkpoint'])
        promoted_selection = selections.get(c.object_sha(intent['promoted_checkpoint']))
        c.require(key not in by_checkpoint and intent['checkpoint'] not in intent['protected_checkpoints'] and
                  intent['promoted_checkpoint'] in intent['protected_checkpoints'], 'Repeated/protected retirement')
        c.require(promoted_selection is not None and
                  promoted_selection['actual_update'] > intent['original_selection']['actual_update'] and
                  promoted_selection['val_correct'] > intent['original_selection']['val_correct'],
                  'Retirement is not a later strict improvement')
        c.require(not c.confined(intent['checkpoint']['path']).exists(), 'Retired binary reappeared')
        by_checkpoint[key] = (intent, completed)
    for record in result['checkpoints']:
        if record['checkpoint'] in protected:
            c.require(record['binary_retention'] == {'status': 'retained'}, 'Protected image marked retired')
            c.verify(record['checkpoint'])
        else:
            pair = by_checkpoint.get(c.object_sha(record['checkpoint']))
            c.require(pair is not None and record['binary_retention'] == {'status': 'retired', 'receipt': pair[1]} and
                      record['selection'] == pair[0]['original_selection'] and
                      record['checkpoint_io'] == pair[0]['checkpoint_io'], 'Retired selector/image metadata differs')
    return {'retained_binary_count': len(protected), 'retired_binary_count': len(retired),
            'retirement_wall_seconds': sum(pair[1]['retirement_wall_seconds'] for pair in by_checkpoint.values())}
