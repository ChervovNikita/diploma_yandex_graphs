"""Finite source/JSON and synthetic arithmetic checks; no scientific payloads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
E, P = HERE.parent, HERE.parent.parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


launch = module(HERE / 'run.py', '_structural_pilot_launcher_checks')
analysis = module(HERE / 'selected_VALID.py', '_structural_pilot_selected_checks')
helper = module(E / 'pilot_20261015_bn_geometry_correction_v1/complete_route_analysis_v2/analyze.py', '_structural_pilot_analysis_helpers_checks')
SPEC = json.loads((HERE / 'SPEC.json').read_text())
PLAN = json.loads((HERE / 'COLLECTION_PLAN.json').read_text())


def local(path):
    remote_phase = SPEC['project_root'] + '/experiments_iclr/postsubmission_20260930/'
    if path.startswith(remote_phase): return P / path[len(remote_phase):]
    return E / path


class Checks(unittest.TestCase):
    def test_plan_and_source_pins(self):
        self.assertEqual(analysis.sha(HERE / 'COLLECTION_PLAN.json'), analysis.PLAN_SHA)
        for row in PLAN['source_pins']:
            path = local(row['path'])
            if 'bytes' in row: self.assertEqual(path.stat().st_size, row['bytes'])
            self.assertEqual(analysis.sha(path), row['sha256'])
        for row in SPEC['source_pins']:
            if '/.venv/' in row['path']: continue  # actual installed OGB is an allocation preflight gate
            path = local(row['path'])
            if 'bytes' in row: self.assertEqual(path.stat().st_size, row['bytes'])
            self.assertEqual(analysis.sha(path), row['sha256'])
        self.assertEqual((PLAN['reused_banks'], PLAN['new_banks'], PLAN['result_banks']), (33, 6, 39))
        self.assertEqual(PLAN['unique_historical_plus_new_training_fits'], 45)
        self.assertEqual(PLAN['timeout_seconds_per_collection'], 600)

    def test_exact_remote_jobs_and_native_policy(self):
        extension = module(E / 'research/structural_residual_composition_v2/extension.py', '_structural_pilot_v2_checks')
        jobs = []
        for row in SPEC['jobs']:
            path = HERE / row['file']; job = json.loads(path.read_text())
            self.assertEqual(launch.pin(path)['sha256'], row['file_sha256'])
            self.assertEqual(path.stat().st_size, row['file_bytes'])
            parent = json.loads((extension.PARENT / ('JOB_seed' + str(row['seed']) + '.json')).read_text())
            exact = extension.descriptor(parent, row['procedure'], SPEC['producer_manifest']['sha256'])
            exact['session_transform']['path'] = SPEC['project_root'] + '/' + str(SPEC['expansion_relative']) + '/research/structural_residual_composition_v2/extension.py'
            self.assertEqual(job, exact)
            self.assertEqual(job['recipe'], parent['recipe'])
            self.assertEqual(job['policy'], SPEC['native_protocol']['policy'])
            self.assertEqual(job['seed'], row['seed'])
            self.assertEqual(job['device'], 'cuda:0')
            self.assertFalse(job['TEST_access'])
            self.assertEqual(analysis.digest(job), row['job_hash'])
            self.assertEqual(PLAN['jobs'][row['index']]['job_hash'], row['job_hash'])
            jobs.append(job)
        launch.validate_structure(SPEC, jobs)
        sys.path.insert(0, str(E / 'runtime'))
        runner = module(E / 'runtime/run_fit.py', '_structural_pilot_original_runner_checks')
        for job in jobs: runner.validate_job(job)

    def test_actual_qualifications_and_preserved_failure(self):
        for row in SPEC['actual_v2_qualifications']:
            path = local(row['authority']['path'])
            self.assertEqual(analysis.sha(path), row['authority']['sha256'])
            value = json.loads(path.read_text()); result = value['result']
            self.assertEqual((value['status'], value['exit_code'], result['status']), ('complete', 0, 'passed'))
            self.assertEqual(result['variant'], row['variant'])
            self.assertEqual(result['optimizer_calls'], {'attempts': [2], 'completed': [2]})
            self.assertEqual((result['native_constructions_completed'], result['native_forwards_completed'],
                result['reverse_completed_including_checks'], result['VALID_completed']), (2, 73, 8, 3))
            for key in ('fresh_restore_VALID_bit_exact', 'changed_state_rejected',
                        'correct_shared_F_private_W_same_old_state', 'initial_native_eval_bit_exact'):
                self.assertIs(result[key], True)
            for key in ('actual_parameter_IDs_and_group_boundaries_unchanged', 'every_group_option_unchanged', 'optimizer_defaults_unchanged'):
                self.assertIs(result['fresh_optimizer_restore_contract'][key], True)
            self.assertIs(result['TEST_accessed'], False)
        failed = SPEC['retained_failed_v1_qualification']
        self.assertEqual(analysis.sha(local(failed['path'])), failed['sha256'])

    def test_completed_authority_and_no_new_control_work(self):
        path = E / 'pilot_20261015_residual_only_private_bn_v1/complete_results_v1/RESULT.json'
        self.assertEqual(analysis.sha(path), SPEC['completed_uniform_and_33_bank_authority']['sha256'])
        prior = json.loads(path.read_text())
        self.assertEqual((len(prior['rows']), len(prior['training_fit_outcomes'])), (33, 39))
        self.assertEqual(len({r['job_id'] for r in prior['training_fit_outcomes']}), 39)
        self.assertEqual(prior['training_fit_outcomes'], SPEC['reused_training_outcomes'])
        self.assertEqual(prior['route_collection_outcomes'], SPEC['reused_route_collection_outcomes'])
        self.assertEqual([r['seed'] for r in prior['rows'] if r['bank'] == 'residual_only_private_BN'], launch.SEEDS)
        self.assertEqual((SPEC['new_fits'], SPEC['new_control_fits'], SPEC['campaign_roster_additions']), (6, 0, 0))
        self.assertEqual(SPEC['cost_bounds']['maximum_wall_seconds_per_fit'], 14400)
        self.assertEqual(SPEC['cost_bounds']['maximum_concurrent_new_GPU_workers'], 1)

    def test_reject_partial_failed_reordered_or_unreceipted_terminal(self):
        jobs = PLAN['jobs']
        outcomes = [dict(j, status='complete', returncode=0, RESULT={'sha256': 'a'*64}) for j in jobs]
        def gate(function, rows, status='complete', TEST=False):
            function({'status': status, 'TEST_accessed': TEST, 'outcomes': rows}, rows, jobs)
        for function in (analysis.terminal_gate, launch.terminal_gate):
            gate(function, outcomes)
            for bad in (outcomes[:-1], list(reversed(outcomes)),
                [dict(outcomes[0], status='started')] + outcomes[1:],
                outcomes[:-1] + [dict(outcomes[-1], returncode=1)],
                outcomes[:-1] + [dict(outcomes[-1], timed_out=True)],
                outcomes[:-1] + [dict(outcomes[-1], RESULT={})]):
                with self.assertRaises(ValueError): gate(function, bad)
            with self.assertRaises(ValueError): gate(function, outcomes, 'running')
            with self.assertRaises(ValueError): gate(function, outcomes, TEST=True)

    def test_actual_future_hashes_required(self):
        for bad in (None, '', 'deferred', 'a'*63, 'g'*64, 3):
            with self.assertRaises(ValueError): analysis.required_hash(bad)
        self.assertEqual(analysis.required_hash('a'*64), 'a'*64)

    def test_collection_exact_selection_and_zero_TRAIN(self):
        pin = {'path': 'synthetic-result', 'bytes': 7, 'sha256': 'synthetic'}
        fit = {'job_id': 'synthetic', 'job_hash': 'job-hash', 'selected_epoch': 17,
            'selected_checkpoint': {'sha256': 'selected'}, 'selected_VALID_export': {'sha256': 'export'}}
        value = {'status': 'complete_with_sanity_warning', 'role': 'VALID', 'TEST_accessed': False,
            'job_id': 'synthetic', 'job_hash': 'job-hash', 'selected_epoch': 17, 'fit_receipt': pin,
            'members': 4, 'evaluated_VALID_graphs': 151, 'selected_checkpoint': fit['selected_checkpoint'],
            'saved_VALID_export': fit['selected_VALID_export'], 'actual_work': {'TRAIN_callbacks': 0, 'optimizer_steps': 0}}
        analysis.collection_gate(value, fit, pin)
        for key, bad in (('selected_epoch', 16), ('TEST_accessed', True), ('role', 'TEST'),
            ('members', 3), ('evaluated_VALID_graphs', 150), ('close_error', 'restore failed')):
            with self.assertRaises(ValueError): analysis.collection_gate(dict(value, **{key: bad}), fit, pin)
        for key in ('TRAIN_callbacks', 'optimizer_steps'):
            bad = copy.deepcopy(value); bad['actual_work'][key] = 1
            with self.assertRaises(ValueError): analysis.collection_gate(bad, fit, pin)

    def test_signed_quality_and_BCE_reductions(self):
        ref = {'bank': 'I4', 'pool_AUC': .75, 'mean_member_AUC': .72, 'worst_member_AUC': .68,
            'uniform_graph_BCE': .7, 'mean_member_BCE': .8, 'saved_native_VALID_BCE': None,
            'pool_minus_mean_member_AUC_pp': 3., 'all_members_reversed': 10, 'pool_reversed_despite_some_member_correct': 4}
        new = dict(ref, bank='structural_private_risk', pool_AUC=.73, mean_member_AUC=.70,
            worst_member_AUC=.69, uniform_graph_BCE=.6, mean_member_BCE=.9,
            saved_native_VALID_BCE=.61, pool_minus_mean_member_AUC_pp=2., all_members_reversed=8)
        result = analysis.contrast(101, new, ref)
        self.assertAlmostEqual(result['delta_pool_AUC_pp'], -2)
        self.assertAlmostEqual(result['delta_uniform_graph_BCE'], -.1)
        self.assertAlmostEqual(result['delta_mean_member_BCE'], .1)
        self.assertAlmostEqual(result['delta_worst_member_AUC_pp'], 1)
        self.assertEqual(result['delta_common_member_reversals'], -2)
        self.assertEqual(result['delta_pool_minus_mean_member_AUC_pp'], -1)
        self.assertIsNone(result['delta_saved_native_VALID_BCE'])
        ref['saved_native_VALID_BCE'] = .71
        self.assertAlmostEqual(analysis.contrast(101, new, ref)['delta_saved_native_VALID_BCE'], -.1)

    def test_pair_repairs_harms_acquisition_and_ties(self):
        labels = [1, 1, 0]
        reference = {'pool': [-2, 2, 0], 'members': [[-2, 2, 0]]*4}
        candidate = {'pool': [1, -1, 0], 'members': [[1, -1, 0]]*4}
        value = analysis.matched_pairs(helper, reference, candidate, labels)
        for key in ('strict_repairs', 'strict_harms', 'reference_all_members_reversed',
            'reference_common_with_any_candidate_member_correct', 'reference_common_pool_strict_repairs',
            'new_common_member_reversals_outside_reference_common'):
            self.assertEqual(value[key], 1)
        self.assertEqual(value['delta_AUC_pp'], 0)
        candidate = {'pool': [-1, 0, 0], 'members': [[2, 0, 0]] + [[-2, 0, 0]]*3}
        value = analysis.matched_pairs(helper, reference, candidate, labels)
        self.assertEqual(value['reference_common_acquired_correct_member_but_pool_reversed'], 1)
        self.assertEqual(value['new_ties'], 1)
        self.assertEqual(value['candidate_tied_pairs'], 1)
        self.assertEqual(value['strict_repairs'], 0)
        self.assertEqual(value['delta_AUC_pp'], -25)

    def test_no_partial_three_seed_summary(self):
        good = [{'seed': s, 'comparison': 'one', 'value': s} for s in launch.SEEDS]
        self.assertEqual(helper.summarize(good, 'comparison', ['value'])[0]['summary']['value']['values'], launch.SEEDS)
        with self.assertRaises(ValueError): helper.summarize(good[:-1], 'comparison', ['value'])

    def test_stdlib_import_and_prediction_only_analysis(self):
        for filename in ('run.py', 'selected_VALID.py'):
            text = (HERE / filename).read_text(); tree = ast.parse(text)
            imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            self.assertNotIn('torch', [a.name for n in imports if isinstance(n, ast.Import) for a in n.names])
        text = (HERE / 'selected_VALID.py').read_text()
        tree = ast.parse(text)
        analyzer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze')
        body = ast.get_source_segment(text, analyzer)
        self.assertNotIn('selected.pt', body)
        self.assertNotIn('create_adapter', body)
        self.assertEqual(body.count('torch.load('), 2)
        self.assertLess(body.index('All six admitted collections must finish'), body.index('import torch'))
        self.assertLess(body.index('uniform_inputs.append'), body.index('import torch'))

    def test_manifest_when_present(self):
        path = HERE / 'MANIFEST.json'
        if not path.exists(): return
        for row in json.loads(path.read_text())['files']:
            actual = launch.pin(HERE / row['path'])
            self.assertEqual((actual['bytes'], actual['sha256']), (row['bytes'], row['sha256']))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'schema': 'structural-private-risk-six-fit-source-checks-v1',
        'status': 'passed' if result.wasSuccessful() else 'failed', 'checks': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'new_scientific_fits': 0,
        'local_tensor_or_data_payload_reads': 0, 'new_native_forwards': 0, 'new_optimizer_steps': 0,
        'TEST_accessed': False, 'scope': 'Source, JSON and synthetic pair arithmetic only; actual data/installed OGB pins, allocation preflight and resource admission remain root gates.'}
    (HERE / 'SOURCE_CHECKS.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
