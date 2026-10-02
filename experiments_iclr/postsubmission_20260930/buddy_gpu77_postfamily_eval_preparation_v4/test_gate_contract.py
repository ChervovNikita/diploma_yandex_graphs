"""Synthetic receipt-only failure fixtures. No Torch/data/archive/GPU access."""
import copy
import hashlib
import json
from pathlib import Path
import re
import unittest

from gate_contract import terminal_gate, complete_cells_gate, evaluation_admission_gate
from gate_contract import archive_path_gate, audit_admission_gate


ARMS = ['native1024', 'single256', 'factorized4', 'independent4', 'matched_single']
CONTEXT = dict(source_manifest_sha256='a' * 64, cache_manifest_sha256='b' * 64,
               CPU_qualification_sha256='c' * 64, torch_version='synthetic-fixture', preflight_seconds=1.0,
               runtime_environment_sha256='7' * 64, operational_repair_admission_sha256='e' * 64)
ARGV = ['synthetic-python', 'synthetic-launch-family']


def terminal():
    return dict(schema='buddy77-family-launch-receipt-v2', exit_code=0, argv=ARGV,
                resource_receipt_sha256='d' * 64, admission_sha256='e' * 64,
                family_cells=15, optimizer_fits=24, other_jobs_stopped=False, test_access=False, context=CONTEXT,
                runtime_environment_sha256=CONTEXT['runtime_environment_sha256'], prior_partial_fits_excluded=True,
                predecessor_resource_is_runtime_observation_not_confinement_proof=True)


def cells():
    return [dict(arm=arm, seed=seed, status='training_complete', epochs_completed=100,
                 optimizer_fits=4 if arm == 'independent4' else 1, test_loaded_or_scored=False)
            for arm in ARMS for seed in [0, 1, 2]]


def identity():
    return dict(family_lock_sha256='f' * 64, lock_audit_sha256='1' * 64, archive_sha256='2' * 64,
                prior_partial_fits_excluded=True, family_runtime_environment_sha256='7' * 64,
                family_external_namespace_proof_sha256='8' * 64)


def admission():
    return dict(schema='buddy77-postfamily-evaluation-admission-v2', decision='admitted', **identity(),
                family_cells=15, optimizer_fits=24, epochs_per_cell=100,
                all_selected_checkpoints_and_ledgers_reviewed=True, production_family_lock_reviewed=True,
                official_test_access_authorized=True, evaluation_once=True, other_jobs_stopped=False,
                execution_guarded=True, runtime_boundary_mode='external_namespace_profile_bound',
                root_evaluation_cost_decision='Synthetic gate fixture only; not an actual admission.')


class ReceiptGates(unittest.TestCase):
    def test_actual77_archive_path_is_bound_to_real_staging_command_metadata(self):
        # Text/JSON source metadata only. No archive is opened or executed.
        import evaluate77
        here = Path(__file__).resolve().parent
        phase = here.parent
        binding = json.loads((here / 'ARCHIVE_PATH_BINDING.json').read_text())
        command = phase / binding['staging_command_path']
        transfer = phase / binding['transfer_source_path']
        self.assertEqual(hashlib.sha256(command.read_bytes()).hexdigest(), binding['staging_command_sha256'])
        self.assertEqual(hashlib.sha256(transfer.read_bytes()).hexdigest(), binding['transfer_source_sha256'])
        archive_names = re.findall(r'buddy_official_archive_staging_v1/root_transfer_v2/[A-Za-z0-9_]+\.zip', command.read_text())
        self.assertEqual(archive_names, [binding['actual77_archive_relative_path']])
        target = re.findall(r"cat > (/[^'\s]+\.zip)", transfer.read_text())
        self.assertEqual(target, [str(evaluate77.PHASE / binding['actual77_archive_relative_path'])])
        self.assertEqual(str(evaluate77.ARCHIVE), target[0])
        declared = json.loads((here / 'CONTRACT.json').read_text())['archive_relative_path']
        archive_path_gate(declared, archive_names[0])
        with self.assertRaises(RuntimeError):
            archive_path_gate('buddy_official_archive_staging_v1/root_transfer_v2/collab.zip', archive_names[0])

    def terminal_gate(self, value):
        terminal_gate(value, CONTEXT, ARGV, 'd' * 64, 'e' * 64)

    def test_terminal_failure_and_detached_start_cannot_unlock(self):
        for change in [dict(exit_code=1), dict(exit_code=False), dict(schema='buddy77-detached-family-start-v1'),
                       dict(schema='buddy77-family-launch-receipt-v1'), dict(test_access=True), dict(optimizer_fits=23),
                       dict(prior_partial_fits_excluded=False), dict(runtime_environment_sha256='0' * 64)]:
            value = terminal(); value.update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.terminal_gate(value)

    def test_terminal_identity_changes_rejected_but_preflight_timing_ignored(self):
        value = terminal(); value['context'] = dict(CONTEXT, preflight_seconds=99.0)
        self.terminal_gate(value)
        for key in ['source_manifest_sha256', 'cache_manifest_sha256', 'CPU_qualification_sha256']:
            altered = copy.deepcopy(value); altered['context'][key] = '0' * 64
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.terminal_gate(altered)

    def test_missing_duplicate_foreign_and_boolean_seed_cells_rejected(self):
        good = cells(); complete_cells_gate(good, ARMS, [0, 1, 2], 100)
        duplicate = cells(); duplicate[-1] = dict(duplicate[0])
        foreign = cells(); foreign[-1]['seed'] = 99
        boolean = cells(); boolean[1]['seed'] = True
        for rows in [good[:-1], duplicate, foreign, boolean]:
            with self.subTest(rows=rows[-1]), self.assertRaises(RuntimeError):
                complete_cells_gate(rows, ARMS, [0, 1, 2], 100)

    def test_incomplete_epochs_optimizer_fits_and_test_contamination_rejected(self):
        for change in [dict(epochs_completed=99), dict(epochs_completed=True), dict(status='resource_only_complete'),
                       dict(optimizer_fits=3), dict(test_loaded_or_scored=True)]:
            rows = cells(); rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                complete_cells_gate(rows, ARMS, [0, 1, 2], 100)

    def test_root_review_and_explicit_test_authorization_are_separate_gates(self):
        evaluation_admission_gate(admission(), identity())
        for key in ['all_selected_checkpoints_and_ledgers_reviewed', 'production_family_lock_reviewed',
                    'official_test_access_authorized', 'evaluation_once']:
            value = admission(); value[key] = False
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                evaluation_admission_gate(value, identity())

    def test_absent_rejected_or_unbound_root_admission_cannot_unlock(self):
        for value in [{}, dict(admission(), decision='deferred'), dict(admission(), family_lock_sha256='0' * 64),
                      dict(admission(), lock_audit_sha256='0' * 64), dict(admission(), root_evaluation_cost_decision='')]:
            with self.subTest(value=value.get('decision')), self.assertRaises(RuntimeError):
                evaluation_admission_gate(value, identity())

    def test_separate_prospective_lock_audit_admission(self):
        value = dict(identity(), schema='buddy77-postfamily-lock-audit-admission-v1', decision='admitted',
            family_cells=15, optimizer_fits=24, epochs_per_cell=100,
            full_terminal_and_all_training_ledgers_reviewed=True, audit_once=True,
            official_test_access_authorized=False, other_jobs_stopped=False, execution_guarded=True,
            runtime_boundary_mode='external_namespace_profile_bound', root_audit_cost_decision='Fixture only')
        audit_admission_gate(value, identity())
        for changed in [dict(value, decision='deferred'), dict(value, audit_once=False),
                        dict(value, full_terminal_and_all_training_ledgers_reviewed=False),
                        dict(value, official_test_access_authorized=True), dict(value, execution_guarded=False),
                        dict(value, prior_partial_fits_excluded=False)]:
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                audit_admission_gate(changed, identity())


if __name__ == '__main__':
    unittest.main(verbosity=2)
