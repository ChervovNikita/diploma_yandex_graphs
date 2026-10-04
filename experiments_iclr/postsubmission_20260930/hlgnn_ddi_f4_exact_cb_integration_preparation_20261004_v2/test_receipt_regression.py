"""Focused stdlib receipt/path regression fixtures; no numerical/model imports.

Fixtures exercise schema admission only. Their constant summary values are
test inputs and are never written as scientific/performance results.
"""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from paired_comparison import ARMS, COMMON, AUXILIARY_COMMON, compare_family
from runtime_paths import require_project_output


PACKET = Path(__file__).resolve().parent
PROJECT = PACKET.parent


class ReceiptRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.ddi_receipt_regression_', dir=PROJECT)
        self.output = Path(self.temp.name)
        self.rows = {}
        self.summaries = []
        for seed, arm in itertools.product((0, 1, 2), ARMS):
            self.summaries.append({'seed': seed, 'arm': arm, 'completed_epochs': 2,
                                   'selected_replay': 'PASS', 'actual_loss_branch': 'AUC',
                                   'best_epoch': 1, 'best_valid_hits20': 0.5})
            rows = []
            for epoch, batch in itertools.product((1, 2), (0, 1)):
                records = 2 if batch == 0 else 1
                queries = records + 3 * records
                row = {'seed': seed, 'arm': arm, 'epoch': epoch, 'batch': batch,
                       'native_records': records, 'native_record_ids_sha256': 'a' * 64,
                       'positive_positions': list(range(records)),
                       'negative_positions': list(range(3 * records)),
                       'selected_query_sha256': 'b' * 64,
                       'before_target_rng': {'cpu': 'c' * 64, 'cuda': 'd' * 64},
                       'after_target_rng': {'cpu': 'e' * 64, 'cuda': 'f' * 64},
                       'native_negative_draw_sha256': str(epoch) * 64,
                       'initial_parameter_sha256': '0' * 64, 'auxiliary_parameter_count': 0}
                if arm != 'target_only':
                    row.update(auxiliary_seed=123, hidden_keys_sha256='1' * 64,
                               visible_keys_sha256='2' * 64, ordered_support_teacher_sha256='3' * 64,
                               support_counts=[[1] * queries, [0] * queries],
                               teacher_counts=[[0] * queries, [0] * queries],
                               candidate_slots=queries, native_margin_terms=3 * records)
                rows.append(row)
            self.rows[seed, arm] = rows

    def tearDown(self):
        self.temp.cleanup()

    def compare(self):
        for (seed, arm), rows in self.rows.items():
            directory = self.output / arm / f'seed_{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'paired_stream.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return compare_family(self.output, self.summaries, epochs=2, train_records=3, batch_size=2)

    def test_complete_schema_fixture_is_admitted(self):
        result = self.compare()
        self.assertTrue(result['paired_eligible'])
        self.assertEqual(result['compared_stream_rows'], 12)

    def test_every_common_field_jointly_missing_is_ineligible(self):
        baseline = copy.deepcopy(self.rows)
        for field in COMMON:
            with self.subTest(field=field):
                self.rows = copy.deepcopy(baseline)
                for arm in ARMS:
                    del self.rows[0, arm][1][field]
                result = self.compare()
                self.assertFalse(result['paired_eligible'])
                self.assertTrue(any(m['field'] == field and m.get('reason') == 'missing required field'
                                    for m in result['stream_mismatches']))

    def test_every_auxiliary_field_jointly_missing_is_ineligible(self):
        baseline = copy.deepcopy(self.rows)
        for field in AUXILIARY_COMMON:
            with self.subTest(field=field):
                self.rows = copy.deepcopy(baseline)
                for arm in ('joint', 'separate'):
                    del self.rows[0, arm][1][field]
                self.assertFalse(self.compare()['paired_eligible'])

    def test_single_arm_missing_common_is_ineligible(self):
        del self.rows[0, 'target_only'][1]['before_target_rng']
        self.assertFalse(self.compare()['paired_eligible'])

    def test_jointly_malformed_fields_are_ineligible(self):
        cases = (('seed', True), ('initial_parameter_sha256', None),
                 ('native_negative_draw_sha256', 'bad-hash'),
                 ('native_record_ids_sha256', 'A' * 64), ('before_target_rng', {}),
                 ('after_target_rng', {'cpu': 1}), ('positive_positions', [0, 0]),
                 ('negative_positions', [True] * 6), ('auxiliary_parameter_count', False))
        baseline = copy.deepcopy(self.rows)
        for field, value in cases:
            with self.subTest(field=field):
                self.rows = copy.deepcopy(baseline)
                for arm in ARMS:
                    self.rows[0, arm][0][field] = value
                self.assertFalse(self.compare()['paired_eligible'])

    def test_auxiliary_type_counts_and_denominators(self):
        cases = (('auxiliary_seed', True), ('hidden_keys_sha256', None),
                 ('support_counts', [[True] * 8, [0] * 8]), ('teacher_counts', [[2] * 8, [0] * 8]),
                 ('candidate_slots', 9), ('native_margin_terms', 5))
        baseline = copy.deepcopy(self.rows)
        for field, value in cases:
            with self.subTest(field=field):
                self.rows = copy.deepcopy(baseline)
                for arm in ('joint', 'separate'):
                    self.rows[0, arm][0][field] = value
                self.assertFalse(self.compare()['paired_eligible'])

    def test_repeated_initial_and_epoch_negative_hashes_cannot_change_jointly(self):
        baseline = copy.deepcopy(self.rows)
        for field, index in (('initial_parameter_sha256', 2), ('native_negative_draw_sha256', 1)):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(baseline)
                for arm in ARMS:
                    self.rows[0, arm][index][field] = '9' * 64
                self.assertFalse(self.compare()['paired_eligible'])

    def test_invalid_summary_cannot_admit_family(self):
        for field, value in (('seed', True), ('best_valid_hits20', float('nan')),
                             ('best_epoch', 0), ('completed_epochs', True)):
            with self.subTest(field=field):
                previous = self.summaries[0][field]
                self.summaries[0][field] = value
                with self.assertRaises(RuntimeError):
                    self.compare()
                self.summaries[0][field] = previous

    def test_missing_cell_cannot_admit_family(self):
        self.summaries.pop()
        with self.assertRaises(RuntimeError):
            self.compare()

    def test_invalid_or_duplicate_json_fields_are_ineligible(self):
        for bad in ('{"seed":0,"seed":0}', '{"seed":NaN}', '{broken'):
            with self.subTest(json=bad):
                self.compare()
                path = self.output / 'joint' / 'seed_0' / 'paired_stream.jsonl'
                lines = path.read_text().splitlines()
                lines[0] = bad
                path.write_text('\n'.join(lines) + '\n')
                result = compare_family(self.output, self.summaries, epochs=2, train_records=3, batch_size=2)
                self.assertFalse(result['paired_eligible'])
                self.assertTrue(any(m['field'] == 'receipt_json' for m in result['stream_mismatches']))


class OutputPathRegression(unittest.TestCase):
    def test_explicit_new_project_output_is_accepted(self):
        with tempfile.TemporaryDirectory(prefix='.ddi_path_regression_', dir=PROJECT) as temp:
            output = Path(temp) / 'new_run'
            self.assertEqual(require_project_output(output, PACKET), output)
            self.assertFalse(output.exists())

    def test_relative_outside_existing_and_sealed_outputs_are_rejected(self):
        for path in (Path('relative_run'), PROJECT.parent / 'outside_run', PROJECT,
                     PACKET / 'runs', PROJECT / 'hlgnn_ddi_f4_exact_cb_integration_preparation_20261004_v1' / 'runs'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                require_project_output(path, PACKET)

    def test_readonly_parent_and_symlink_to_sealed_source_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='.ddi_path_regression_', dir=PROJECT) as temp:
            directory = Path(temp)
            readonly = directory / 'readonly'
            readonly.mkdir()
            readonly.chmod(0o555)
            try:
                with self.assertRaises(ValueError):
                    require_project_output(readonly / 'run', PACKET)
            finally:
                readonly.chmod(0o755)
            alias = directory / 'sealed_alias'
            alias.symlink_to(PACKET, target_is_directory=True)
            with self.assertRaises(ValueError):
                require_project_output(alias / 'run', PACKET)


if __name__ == '__main__':
    unittest.main()
