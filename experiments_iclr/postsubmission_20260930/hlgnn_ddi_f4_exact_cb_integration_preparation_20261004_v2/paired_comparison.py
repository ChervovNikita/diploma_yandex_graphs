"""Prospective complete-family paired receipts and descriptive comparisons.

Stdlib only. No fallback to an incomplete or convenient subset of cells.
"""
import itertools
import json
from math import isfinite, sqrt
from pathlib import Path
import statistics
import re


ARMS = ('target_only', 'joint', 'separate')
COMMON = ('seed', 'epoch', 'batch', 'native_records', 'native_record_ids_sha256',
          'positive_positions', 'negative_positions', 'selected_query_sha256',
          'before_target_rng', 'after_target_rng', 'native_negative_draw_sha256',
          'initial_parameter_sha256')
AUXILIARY_COMMON = ('auxiliary_seed', 'hidden_keys_sha256', 'visible_keys_sha256',
                    'ordered_support_teacher_sha256', 'support_counts', 'teacher_counts',
                    'candidate_slots', 'native_margin_terms')


SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _hash(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _strict_object(pairs):
    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError(f'Duplicate JSON field: {key}')
        row[key] = value
    return row


def _invalid_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def validate_receipt(row, arm, seed, epoch, batch, records):
    """Return schema errors; omitted fields are never treated as equal evidence."""
    if not isinstance(row, dict):
        return [('receipt', 'expected JSON object')]
    required = ('arm', 'auxiliary_parameter_count', *COMMON)
    if arm != 'target_only':
        required += AUXILIARY_COMMON
    errors = [(field, 'missing required field') for field in required if field not in row]
    for field, expected in (('arm', arm), ('seed', seed), ('epoch', epoch),
                            ('batch', batch), ('native_records', records)):
        if field in row and (type(row[field]) is not type(expected) or row[field] != expected):
            errors.append((field, 'incorrect type or native complete schedule'))
    if 'auxiliary_parameter_count' in row and (type(row['auxiliary_parameter_count']) is not int
                                              or row['auxiliary_parameter_count'] != 0):
        errors.append(('auxiliary_parameter_count', 'must be integer zero'))
    for field in COMMON + (AUXILIARY_COMMON if arm != 'target_only' else ()):
        if field.endswith('_sha256') and field in row and not _hash(row[field]):
            errors.append((field, 'must be a canonical SHA256 hex digest'))
    for field in ('before_target_rng', 'after_target_rng'):
        if field in row:
            value = row[field]
            if (not isinstance(value, dict) or set(value) not in ({'cpu'}, {'cpu', 'cuda'})
                    or not all(_hash(digest) for digest in value.values())):
                errors.append((field, 'expected CPU and optional CUDA SHA256 state digests'))
    for field, population in (('positive_positions', records), ('negative_positions', 3 * records)):
        if field in row:
            value = row[field]
            if (not isinstance(value, list) or len(value) != min(32, population)
                    or not all(_integer(position) and position < population for position in value)
                    or len(set(value)) != len(value)):
                errors.append((field, 'expected complete unique selected positions in range'))
    if arm != 'target_only':
        if 'auxiliary_seed' in row and (not _integer(row['auxiliary_seed'])
                                       or row['auxiliary_seed'] >= 2**63 - 1):
            errors.append(('auxiliary_seed', 'expected owned nonnegative int64 seed'))
        queries = min(32, records) + min(32, 3 * records)
        counts_valid = {}
        for field in ('support_counts', 'teacher_counts'):
            if field in row:
                value = row[field]
                valid = (isinstance(value, list) and len(value) == 2
                         and all(isinstance(side, list) and len(side) == queries
                                 and all(_integer(count) for count in side) for side in value))
                counts_valid[field] = valid
                if not valid:
                    errors.append((field, 'expected two nonnegative integer counts per selected query'))
        if all(counts_valid.get(field, False) for field in ('support_counts', 'teacher_counts')):
            if any(k > n for side_n, side_k in zip(row['support_counts'], row['teacher_counts'])
                   for n, k in zip(side_n, side_k)):
                errors.append(('teacher_counts', 'teacher count exceeds support'))
        if 'candidate_slots' in row:
            if (not _integer(row['candidate_slots']) or
                    (counts_valid.get('support_counts', False)
                     and row['candidate_slots'] != sum(map(sum, row['support_counts'])))):
                errors.append(('candidate_slots', 'must equal the sum of complete support counts'))
        if 'native_margin_terms' in row and (type(row['native_margin_terms']) is not int
                                            or row['native_margin_terms'] != 3 * records):
            errors.append(('native_margin_terms', 'must equal three times native records'))
    return errors


def compare_family(output, summaries, epochs=500, train_records=1067911, batch_size=65536):
    output = Path(output)
    if (not isinstance(summaries, list) or len(summaries) != 9
            or any(not isinstance(row, dict) for row in summaries)):
        raise RuntimeError('The full declared nine-cell family is required.')
    required = ('seed', 'arm', 'completed_epochs', 'selected_replay', 'actual_loss_branch',
                'best_epoch', 'best_valid_hits20')
    for row in summaries:
        if any(field not in row for field in required):
            raise RuntimeError('Complete family summary fields are required.')
        if (type(row['seed']) is not int or row['seed'] not in (0, 1, 2)
                or row['arm'] not in ARMS or type(row['completed_epochs']) is not int
                or row['completed_epochs'] != epochs or row['selected_replay'] != 'PASS'
                or row['actual_loss_branch'] != 'AUC'
                or not _integer(row['best_epoch'], 1) or row['best_epoch'] > epochs
                or type(row['best_valid_hits20']) not in (int, float)
                or not isfinite(row['best_valid_hits20']) or not 0 <= row['best_valid_hits20'] <= 1):
            raise RuntimeError('A complete native AUC fit and selected replay are required for every cell.')
    cells = {(row['seed'], row['arm']): row for row in summaries}
    if set(cells) != set(itertools.product((0, 1, 2), ARMS)):
        raise RuntimeError('The full declared nine-cell family is required.')
    if not all(_integer(value, 1) for value in (epochs, train_records, batch_size)):
        raise ValueError('Complete schedule parameters must be positive integers.')
    mismatches, complete_rows = [], 0
    batches = (train_records + batch_size - 1) // batch_size
    for seed in (0, 1, 2):
        paths = [output / arm / f'seed_{seed}' / 'paired_stream.jsonl' for arm in ARMS]
        rows_for_seed, initial_hashes, negative_hashes = 0, {}, {}
        with paths[0].open() as a, paths[1].open() as b, paths[2].open() as c:
            for index, lines in enumerate(itertools.zip_longest(a, b, c)):
                if any(line is None for line in lines):
                    mismatches.append({'seed': seed, 'row': index, 'field': 'stream_length'})
                    break
                rows = []
                expected_epoch, expected_batch = index // batches + 1, index % batches
                expected_records = min(batch_size, train_records - expected_batch * batch_size)
                for arm, line in zip(ARMS, lines):
                    try:
                        row = json.loads(line, object_pairs_hook=_strict_object,
                                         parse_constant=_invalid_constant)
                    except (ValueError, TypeError) as error:
                        row = None
                        mismatches.append({'seed': seed, 'row': index, 'arm': arm,
                                           'field': 'receipt_json', 'reason': str(error)})
                    errors = validate_receipt(row, arm, seed, expected_epoch, expected_batch, expected_records)
                    for field, reason in errors:
                        mismatches.append({'seed': seed, 'row': index, 'arm': arm,
                                           'field': field, 'reason': reason})
                    if isinstance(row, dict):
                        initial = row.get('initial_parameter_sha256')
                        if _hash(initial):
                            if arm in initial_hashes and initial_hashes[arm] != initial:
                                mismatches.append({'seed': seed, 'row': index, 'arm': arm,
                                                   'field': 'initial_parameter_sha256', 'reason': 'changed within cell'})
                            initial_hashes[arm] = initial
                        negative = row.get('native_negative_draw_sha256')
                        if _hash(negative):
                            key = (arm, expected_epoch)
                            if key in negative_hashes and negative_hashes[key] != negative:
                                mismatches.append({'seed': seed, 'row': index, 'arm': arm,
                                                   'field': 'native_negative_draw_sha256', 'reason': 'changed within epoch'})
                            negative_hashes[key] = negative
                    rows.append(row)
                for field in COMMON:
                    if (all(isinstance(row, dict) and field in row for row in rows)
                            and any(row[field] != rows[0][field] for row in rows[1:])):
                        mismatches.append({'seed': seed, 'row': index, 'field': field})
                for field in AUXILIARY_COMMON:
                    if (all(isinstance(row, dict) and field in row for row in rows[1:])
                            and rows[1][field] != rows[2][field]):
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
    return {'schema': 'hlgnn-ddi-cb-complete-paired-comparison-v2', 'cells': summaries,
            'paired_eligible': not mismatches, 'stream_mismatches': mismatches,
            'compared_stream_rows': complete_rows, 'contrasts': contrasts,
            'interpretation': 'descriptive seed variation on one development graph; no graph replication'}
