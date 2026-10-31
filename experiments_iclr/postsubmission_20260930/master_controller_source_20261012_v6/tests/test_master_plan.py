"""Pure metadata/accounting/gate tests; no Torch, graph arrays or approval files."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from compile_master_plan import (CONTROLLER_FILES, assemble, bank_roster, controller_bindings,
    apply_resources, fit_semantics, graph_check, readiness, receipt, typed_counts, write_master)
from partition import DEFAULT_WORKERS
from phase_closure import expected_keys
from spec import canonical, fingerprint

PACKAGE = Path(__file__).resolve().parents[1]


def run(identifier, kind, method, config, parents=(), body=0, budget_unit='optimizer_updates', maximum=2):
    return {'run_id': identifier, 'kind': kind, 'method': method, 'split': 0, 'seed': 10, 'body': body,
        'adapter': 'packet/adapters/node.py:make_adapter', 'config': {'reservation_mib': 4, 'estimated_seconds': 2., **config},
        'parents': list(parents), 'budget': {'unit': budget_unit, 'policy': 'full_budget', 'maximum': maximum} if kind == 'model_fit' else {},
        'selection': {'metric': 'accuracy', 'direction': 'max', 'ties': 'first', 'evaluation_interval': 1,
                      'checkpoint0': False} if kind == 'model_fit' else {}}


def study(identifier, *, coordinate=False, molecular=False):
    def fit(arm, members, stage):
        if coordinate:
            return {'source_arm': arm, 'source_optimizer_count': 4 if arm == 'untied' else 1,
                    'members': members, 'stage': stage}
        if molecular:
            return {'molecular_inputs': {'train': {'path': 'sealed/unopened_train.pt'}, 'select': {'path': 'sealed/unopened_select.pt'}},
                    'arm': 'single' if members == 1 else 'be_unit', 'stage': stage}
        return {'architecture': arm, 'members': members, 'stage': stage}
    unit, maximum = ('epochs', 2000) if coordinate else (('epochs', 100) if molecular else ('optimizer_updates', 2))
    single = 'single'
    candidate = 'untied' if coordinate else 'joint_untied'
    rows = [run('search', 'model_fit', 'reference', fit(single, 1, 'search'), budget_unit=unit, maximum=maximum),
        run('select', 'selection', 'reference', {'stage': 'search_selection'}, ['search']),
        run('reference', 'model_fit', 'reference', fit(single, 1, 'comparative'), ['select'], budget_unit=unit, maximum=maximum),
        run('candidate', 'model_fit', 'candidate', fit(candidate, 4, 'comparative'), ['select'], budget_unit=unit, maximum=maximum)]
    for name in ('reference', 'candidate'):
        rows += [run('bank-' + name, 'ensemble_assembly', name, {'stage': 'bank_export', 'fit_parents': [name]}, [name]),
            run('cal-' + name, 'calibration', name, {'stage': 'calibration', 'steps': 500}, ['bank-' + name]),
            run('eval-' + name, 'evaluation', name, {'stage': 'evaluation'}, ['bank-' + name, 'cal-' + name])]
    return {'schema_version': 1, 'study_id': identifier, 'stage': 'engineering', 'dataset': 'synthetic_engineering_fixture',
        'hypothesis': 'Metadata fixture only', 'data': {'task': 'fixture', 'preprocessing': 'fixture',
            'roles': {role: 'sealed/unopened_' + role + '.pt' for role in ('train', 'select', 'cal', 'evaluate')},
            'prior_exposure': 'synthetic', 'evidence': {'fixture': True}},
        'source': {'git_commit': 'fixture', 'files': [{'path': 'sealed/unopened_evaluate.pt', 'bytes': 42, 'sha256': 'a' * 64}]},
        'runs': rows, 'comparisons': [{'references': ['reference'], 'candidates': ['candidate'],
            'unimplemented_author_reference': 'preserved_blocker'}],
        'release': {'policy': 'whole_roster', 'primary_endpoint': 'fixture', 'global_plan_gates_required': True},
        'resources': {'wall_seconds': 10, 'profile': {'evidence': ['provisional fixture']}, 'status': 'DISABLED_QA_PENDING'}}


class MasterPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.controller = 'frozen_controller'
        target = self.root / self.controller
        target.mkdir()
        for name in CONTROLLER_FILES:
            shutil.copyfile(PACKAGE / name, target / name)
        self.control_files = controller_bindings(self.root, self.controller)

    def tearDown(self):
        self.temporary.cleanup()

    def packet(self, packet_id, spec):
        folder = self.root / packet_id
        folder.mkdir()
        path = folder / 'study.json.gz'
        path.write_bytes(gzip.compress((canonical(spec) + '\n').encode(), mtime=0))
        plan = {'schema_version': 1, 'slots_per_gpu': 2, 'scientific_launch_enabled': False, 'human_GO_required': True,
            'studies': [{'path': str(path.relative_to(self.root)), 'spec_hash': fingerprint(spec)}],
            'groups': [{'id': spec['study_id'], 'run_keys': [spec['study_id'] + '/' + r['run_id'] for r in spec['runs']],
                'estimated_seconds': 20, 'maximum_cuda_bytes': 4 * 1024**2, 'gpu_assignment': 'allocation/gpu0'}],
            'barriers': {}, 'global_gates': {'old_packet_gate': {'path': packet_id + '/old_gate.json'}}}
        summary = {'plan_hash': fingerprint(plan), 'status': 'DISABLED_QA_PENDING', 'per_study': [{'study_id': spec['study_id']}],
            'author_replication': 'unmaterialized', 'data_blockers': ['fixture retained blocker'],
            'GO_receipt_created': False, 'benchmark_training_started': False}
        for name, value in [('PLAN.json', plan), ('SUMMARY.json', summary)]:
            (folder / name).write_text(json.dumps(value))
        return {'id': packet_id, 'plan': receipt(self.root, packet_id + '/PLAN.json'),
                'summary': receipt(self.root, packet_id + '/SUMMARY.json')}

    def prepare(self, entries):
        index = {'schema': 'disabled-master-input-index-v1', 'packets': entries}
        packets, ready = readiness(self.root, index)
        resources = {'worker_capacities_cuda_bytes': {worker: 100 * 1024**2 for worker in DEFAULT_WORKERS},
                     'study_profiles': {}}
        return assemble(self.root, packets, ready, self.controller, self.control_files, resources, 'master', [])

    def test_native_coupled_and_coordinate_four_objects_are_distinct(self):
        native, coordinate, molecule = study('native'), study('coordinate', coordinate=True), study('molecule', molecular=True)
        self.assertEqual(fit_semantics(native['runs'][3])['optimizer_objects'], 1)
        self.assertEqual(fit_semantics(native['runs'][3])['physical_slow_bodies'], 4)
        self.assertEqual(fit_semantics(coordinate['runs'][3])['optimizer_objects'], 4)
        self.assertEqual(fit_semantics(coordinate['runs'][3])['known_optimizer_step_calls'], 8000)
        counts = typed_counts([native, coordinate, molecule])
        self.assertEqual(counts['totals']['atomic_fit_run_groups'], 9)
        self.assertEqual(counts['totals']['paid_optimizer_starts'], 12)
        self.assertEqual(counts['totals']['physical_slow_body_instances'], 15)
        self.assertEqual(counts['totals']['known_registered_optimizer_step_calls'], 12006)
        self.assertIsNone(counts['model_optimizer_step_call_total'])
        self.assertEqual(counts['epoch_fit_groups_with_unresolved_minibatch_update_count'], 3)
        broken = copy.deepcopy(coordinate['runs'][3]); broken['budget']['unit'] = 'optimizer_updates'
        with self.assertRaises(ValueError):
            fit_semantics(broken)

    def test_all_packet_gates_bind_all_cohorts_and_new_controller(self):
        entries = [self.packet('native_packet', study('native')),
                   self.packet('coordinate_packet', study('coordinate', coordinate=True))]
        plan, specs, summary = self.prepare(entries)
        self.assertEqual(set(plan['global_gates']), {'whole_search', 'whole_comparative', 'whole_calibration'})
        self.assertTrue(all('master/gate_receipts' in gate['path'] for gate in plan['global_gates'].values()))
        self.assertEqual(len(plan['global_gates']['whole_comparative']['all_packet_source_run_keys']), 4)
        self.assertFalse(plan['scientific_launch_enabled'])
        self.assertFalse(summary['GO_receipt_created'])
        self.assertFalse(summary['all_249_predictive_candidates_covered'])
        for spec in specs:
            self.assertTrue(spec['release']['global_plan_gates_required'])
            self.assertTrue(set(r['path'] for r in self.control_files) <= set(r['path'] for r in spec['source']['files']))
            self.assertTrue(all(r['adapter'] == 'packet/adapters/node.py:make_adapter' for r in spec['runs']))
            self.assertEqual(spec['comparisons'][0]['unimplemented_author_reference'], 'preserved_blocker')
        records, _ = graph_check(specs, plan['barriers'])
        placement = {key: {'worker': group['gpu_assignment']} for group in plan['groups'] for key in group['run_keys']}
        proof_roster = set().union(*(set(expected_keys(plan, records, placement, 'whole_comparative', host))
                                  for host in plan['global_gates']['whole_comparative']['required_hosts']))
        self.assertEqual(proof_roster, set(plan['global_gates']['whole_comparative']['all_packet_source_run_keys']))
        self.assertTrue({'native/reference', 'native/candidate', 'coordinate/reference', 'coordinate/candidate'} <= proof_roster)
        self.assertEqual(sum(v['counts']['totals'].get('paid_optimizer_starts', 0) for v in summary['per_worker'].values()), 9)

    def test_changed_metadata_duplicate_study_and_duplicate_run_fail(self):
        entry = self.packet('packet', study('one'))
        (self.root / 'packet/PLAN.json').write_text((self.root / 'packet/PLAN.json').read_text() + ' ')
        _, ready = readiness(self.root, {'schema': 'disabled-master-input-index-v1', 'packets': [entry]})
        self.assertEqual(ready['invalid_inputs'], ['packet'])
        duplicate_entries = [self.packet('same1', study('duplicate')), self.packet('same2', study('duplicate'))]
        with self.assertRaisesRegex(ValueError, 'Duplicate study'):
            readiness(self.root, {'schema': 'disabled-master-input-index-v1', 'packets': duplicate_entries})
        bad = study('duplicate_run'); bad['runs'].append(copy.deepcopy(bad['runs'][0]))
        duplicate = self.packet('run_packet', bad)
        _, ready = readiness(self.root, {'schema': 'disabled-master-input-index-v1', 'packets': [duplicate]})
        self.assertEqual(ready['invalid_inputs'], ['run_packet'])

    def test_missing_reference_and_missing_dependency_block_input(self):
        bad = study('unmatched')
        for row in bad['runs']:
            if row['method'] == 'reference':
                row['method'] = 'renamed_reference'
        entry = self.packet('unmatched_packet', bad)
        _, ready = readiness(self.root, {'schema': 'disabled-master-input-index-v1', 'packets': [entry]})
        self.assertEqual(ready['invalid_inputs'], ['unmatched_packet'])
        bad = study('missing_parent'); bad['runs'][3]['parents'] = ['unregistered']
        entry = self.packet('missing_parent_packet', bad)
        _, ready = readiness(self.root, {'schema': 'disabled-master-input-index-v1', 'packets': [entry]})
        self.assertEqual(ready['invalid_inputs'], ['missing_parent_packet'])

    def test_pending_link_custody_is_visible_and_not_added_as_fits(self):
        pending = {'id': 'link_pending', 'status': 'pending', 'blockers': ['sealed_TEST_custody', 'final_endpoints'],
                   'prospective_unmaterialized_fits': 1234}
        entries = [self.packet('native_packet', study('native')), pending]
        plan, specs, summary = self.prepare(entries)
        self.assertEqual(summary['pending_unmaterialized_packets'], ['link_pending'])
        self.assertEqual(summary['counts']['totals']['atomic_fit_run_groups'], 3)
        self.assertEqual(len(specs), 1)
        self.assertFalse((self.root / 'sealed/unopened_evaluate.pt').exists())
        digest = write_master(self.root, 'master', plan, specs, summary)
        self.assertEqual(digest, fingerprint(json.loads((self.root / 'master/PLAN.json').read_text())))
        self.assertFalse((self.root / 'master/gate_receipts').exists())
        self.assertFalse(any('approval' in p.name.lower() for p in (self.root / 'master').rglob('*')))

    def test_independent_bank_counts_body0_once_and_rejects_duplicate_seed(self):
        spec = study('I4')
        fit0 = next(row for row in spec['runs'] if row['run_id'] == 'reference')
        for body in (1, 2, 3):
            row = copy.deepcopy(fit0); row.update(run_id='member' + str(body), body=body)
            spec['runs'].append(row)
        spec['runs'].append(run('bank-I4', 'ensemble_assembly', 'ordinary_I4',
            {'stage': 'bank_export', 'fit_parents': ['reference', 'member1', 'member2', 'member3']},
            ['reference', 'member1', 'member2', 'member3']))
        banks = bank_roster(spec)
        self.assertEqual(sum(b['genuine_I4'] for b in banks), 1)
        self.assertEqual(sum(b['member0_reused'] for b in banks), 1)
        self.assertEqual(typed_counts([spec])['totals']['atomic_fit_run_groups'], 6)
        for row in spec['runs']:
            if row['kind'] == 'model_fit':
                row['config']['initialization_seed'] = 123
        with self.assertRaises(ValueError):
            bank_roster(spec)

    def test_resource_scope_changes_only_reservation_and_estimate(self):
        spec = study('resource')
        profile = {'memory': {'kind': 'provisional', 'scope': 'explicit future qualification required'},
                   'time': {'kind': 'provisional', 'scope': 'training and serializer unmeasured'},
                   'runs': {'candidate': {'reservation_mib': 12, 'estimated_seconds': 50.}}}
        changed = apply_resources(spec, {'resource': profile})
        self.assertEqual(changed['runs'][3]['config']['reservation_mib'], 12)
        self.assertEqual(changed['runs'][3]['budget'], spec['runs'][3]['budget'])
        self.assertEqual(changed['runs'][3]['selection'], spec['runs'][3]['selection'])
        self.assertEqual(changed['runs'][3]['adapter'], spec['runs'][3]['adapter'])
        self.assertEqual(spec['runs'][3]['config']['reservation_mib'], 4)
        invalid = copy.deepcopy(profile); invalid['runs']['candidate']['learning_rate'] = .5
        with self.assertRaises(ValueError):
            apply_resources(spec, {'resource': invalid})
        invalid = copy.deepcopy(profile); invalid['memory'] = {'kind': 'measured', 'scope': 'claim', 'evidence': ['unbound claim']}
        with self.assertRaises(ValueError):
            apply_resources(spec, {'resource': invalid})


if __name__ == '__main__':
    unittest.main()
