"""Prospective complete-family paired receipts and descriptive comparisons.

Stdlib only. No fallback to an incomplete or convenient subset of cells.
"""
import itertools
import json
from math import sqrt
from pathlib import Path
import statistics


ARMS = ('target_only', 'joint', 'separate')
COMMON = ('seed', 'epoch', 'batch', 'native_records', 'native_record_ids_sha256',
          'positive_positions', 'negative_positions', 'selected_query_sha256',
          'before_target_rng', 'after_target_rng', 'native_negative_draw_sha256',
          'initial_parameter_sha256')
AUXILIARY_COMMON = ('auxiliary_seed', 'hidden_keys_sha256', 'visible_keys_sha256',
                    'ordered_support_teacher_sha256', 'support_counts', 'teacher_counts',
                    'candidate_slots', 'native_margin_terms')


def compare_family(output, summaries, epochs=500, train_records=1067911, batch_size=65536):
    output = Path(output)
    cells = {(row['seed'], row['arm']): row for row in summaries}
    if len(summaries) != 9 or set(cells) != set(itertools.product((0, 1, 2), ARMS)):
        raise RuntimeError('The full declared nine-cell family is required.')
    if any(row['completed_epochs'] != epochs or row['selected_replay'] != 'PASS'
           or row['actual_loss_branch'] != 'AUC' for row in summaries):
        raise RuntimeError('A complete native AUC fit and selected replay are required for every cell.')
    mismatches, complete_rows = [], 0
    batches = (train_records + batch_size - 1) // batch_size
    for seed in (0, 1, 2):
        paths = [output / arm / f'seed_{seed}' / 'paired_stream.jsonl' for arm in ARMS]
        rows_for_seed = 0
        with paths[0].open() as a, paths[1].open() as b, paths[2].open() as c:
            for index, lines in enumerate(itertools.zip_longest(a, b, c)):
                if any(line is None for line in lines):
                    mismatches.append({'seed': seed, 'row': index, 'field': 'stream_length'})
                    break
                rows = [json.loads(line) for line in lines]
                expected_epoch, expected_batch = index // batches + 1, index % batches
                expected_records = min(batch_size, train_records - expected_batch * batch_size)
                for row in rows:
                    if ((row['epoch'], row['batch'], row['native_records']) !=
                            (expected_epoch, expected_batch, expected_records)):
                        mismatches.append({'seed': seed, 'row': index, 'field': 'native_complete_schedule'})
                if index == 0 and not rows[0].get('initial_parameter_sha256'):
                    mismatches.append({'seed': seed, 'row': index, 'field': 'initial_state_missing'})
                for field in COMMON:
                    if any(row.get(field) != rows[0].get(field) for row in rows[1:]):
                        mismatches.append({'seed': seed, 'row': index, 'field': field})
                for field in AUXILIARY_COMMON:
                    if rows[1].get(field) != rows[2].get(field):
                        mismatches.append({'seed': seed, 'row': index, 'field': field})
                complete_rows += 1
                rows_for_seed += 1
        if rows_for_seed != epochs * batches:
            mismatches.append({'seed': seed, 'field': 'complete_update_count', 'observed': rows_for_seed,
                               'required': epochs * batches})
    contrasts = []
    for first, second, role in (('joint', 'separate', 'primary'),
                                ('joint', 'target_only', 'secondary'),
                                ('separate', 'target_only', 'secondary')):
        differences = [cells[seed, first]['best_valid_hits20'] - cells[seed, second]['best_valid_hits20']
                       for seed in (0, 1, 2)]
        average, sd = statistics.mean(differences), statistics.stdev(differences)
        radius = 4.302652729911275 * sd / sqrt(3)
        sign_means = [statistics.mean(sign * value for sign, value in zip(signs, differences))
                      for signs in itertools.product((-1, 1), repeat=3)]
        contrasts.append({'contrast': f'{first}-{second}', 'role': role,
                          'seed_differences': differences, 'mean': average, 'sample_sd': sd,
                          'range': [min(differences), max(differences)],
                          'descriptive_t95': [average - radius, average + radius],
                          'exact_two_sided_sign_flip_p': sum(abs(value) >= abs(average) for value in sign_means) / 8,
                          'units': 'Hits@20_fraction'})
    return {'schema': 'hlgnn-ddi-cb-complete-paired-comparison-v1', 'cells': summaries,
            'paired_eligible': not mismatches, 'stream_mismatches': mismatches,
            'compared_stream_rows': complete_rows, 'contrasts': contrasts,
            'interpretation': 'descriptive seed variation on one development graph; no graph replication'}
