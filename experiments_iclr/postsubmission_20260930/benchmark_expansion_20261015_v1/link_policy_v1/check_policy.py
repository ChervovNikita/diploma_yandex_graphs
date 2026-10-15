"""Pure policy/source-custody/bookkeeping checks; no model or data imports."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys


def run():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('_link_policy_static', here/'policy.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    def rejects(function):
        try: function()
        except (ValueError, KeyError): return
        raise AssertionError('Incompatible link policy or state was admitted')
    def declaration(track):
        return {'family': 'link', 'track': track, 'maximum_epochs': 100 if track == 'source_reference' else 1000,
                'minimum_epochs': 100, 'patience': None if track == 'source_reference' else 50,
                'metric': 'hits@50', 'direction': 'max', 'ties': 'first',
                'evaluate_TRAIN_each_epoch': False, 'evaluation_mode': 'VALID_only_disclosed_adaptation'}
    binding = module.source_bindings()
    assert binding['roster_cells'] == 27
    assert {row['native_family'] for row in binding['cells']} == {'NCN', 'NCNC', 'BUDDY'}
    for family in ('NCN', 'NCNC', 'BUDDY'):
        for track in ('source_reference', 'matched_extended'):
            policy = module.LinkPolicy.from_dict(declaration(track), family)
            assert policy.family == 'link' and policy.validation_interval_epochs == 1
            assert policy.checkpoint0 is False and policy.warmup_epochs == 0
            assert policy.source_protocol_equivalent is False and policy.source_RNG_trajectory_equivalent is False
            assert module.LinkPolicy.from_dict(json.loads(json.dumps(policy.__dict__)), family).__dict__ == policy.__dict__
            assert policy.initial_best_metric == (0.0 if family == 'BUDDY' and track == 'source_reference' else None)
    source_ncn = module.LinkPolicy.from_dict(declaration('source_reference'), 'NCN')
    selector = module.create_selector(source_ncn)
    assert selector.observe(1, {'hits@50': 0.0})['save_selected']
    assert not selector.observe(2, {'hits@50': 0.0})['save_selected'] and selector.selected_epoch == 1
    for epoch in range(3, 101):
        row = selector.observe(epoch, {'hits@50': 0.0})
        assert row['stop'] is (epoch == 100)
    assert row['stop_reason'] == 'fixed_source_horizon'
    rejects(lambda: selector.observe(101, {'hits@50': 0.5}))
    buddy = module.create_selector(module.LinkPolicy.from_dict(declaration('source_reference'), 'BUDDY'))
    for epoch in range(1, 101):
        assert buddy.observe(epoch, {'hits@50': 0.0})['save_selected'] is False
    assert buddy.selected_epoch is None and buddy.stopping_status()['selected_checkpoint_available'] is False
    restored = module.create_selector(buddy.policy); restored.load_state(json.loads(json.dumps(buddy.state())))
    assert restored.state() == buddy.state() and restored.stopping_status()['stop']
    buddy = module.create_selector(buddy.policy)
    buddy.observe(1, {'hits@50': 0.0})
    assert buddy.observe(2, {'hits@50': 0.01})['save_selected'] and buddy.selected_epoch == 2
    extended = module.create_selector(module.LinkPolicy.from_dict(declaration('matched_extended'), 'BUDDY'))
    assert extended.observe(1, {'hits@50': 0.0})['save_selected']
    for epoch in range(2, 100): extended.observe(epoch, {'hits@50': 0.0})
    assert extended.stale == 0
    assert not extended.observe(100, {'hits@50': 0.0})['stop'] and extended.stale == 1
    saved = json.loads(json.dumps(extended.state()))
    resumed = module.create_selector(extended.policy); resumed.load_state(saved)
    assert resumed.state() == extended.state()
    for epoch in range(101, 150):
        row = resumed.observe(epoch, {'hits@50': 0.0})
        assert row['stop'] is (epoch == 149)
    assert row['stop_reason'] == 'patience' and resumed.selected_epoch == 1
    before = resumed.state(); resumed.stopping_status(); assert resumed.state() == before
    cap = module.create_selector(extended.policy)
    for epoch in range(1, 1001): row = cap.observe(epoch, {'hits@50': epoch/1001})
    assert row['stop'] and row['stop_reason'] == 'cap_without_patience_plateau' and cap.selected_epoch == 1000
    fresh = module.create_selector(source_ncn)
    rejects(lambda: fresh.observe(0, {'hits@50': 0.5}))
    rejects(lambda: fresh.observe(2, {'hits@50': 0.5}))
    rejects(lambda: fresh.observe(1, {'hits@50': float('nan')}))
    rejects(lambda: fresh.observe(1, {'hits@50': 50.0}))
    rejects(lambda: fresh.observe_validation(1, {'role': 'TRAIN', 'metrics': {'hits@50': 0.5}}))
    valid = {'role': 'VALID', 'complete_source_VALID': True,
             'complete_VALID_positive_records': 60084, 'complete_VALID_negative_records': 100000,
             'metrics': {'hits@50': 0.5}}
    assert fresh.observe_validation(1, valid)['save_selected']
    rejects(lambda: fresh.observe(1, {'hits@50': 0.5}))
    corrupted = copy.deepcopy(fresh.state()); corrupted['selected_epoch'] = 0
    rejects(lambda: fresh.load_state(corrupted))
    corrupted = copy.deepcopy(fresh.state()); corrupted['policy']['evaluation_mode'] = 'undisclosed'
    rejects(lambda: fresh.load_state(corrupted))
    for field, value in (('maximum_epochs', 101), ('metric', 'rocauc'), ('family', 'ogb'),
                         ('evaluate_TRAIN_each_epoch', True), ('ties', 'last'), ('evaluation_mode', 'source_exact')):
        rejects(lambda field=field, value=value: module.LinkPolicy.from_dict({**declaration('source_reference'), field: value}, 'NCN'))
    missing_mode = declaration('source_reference'); missing_mode.pop('evaluation_mode')
    rejects(lambda: module.LinkPolicy.from_dict(missing_mode, 'NCN'))
    rejects(lambda: module.LinkPolicy.from_dict(declaration('source_reference'), 'unknown'))
    rejects(lambda: module._inside(module.PHASE, here))
    cell = next(row for row in binding['cells'] if row['native_family'] == 'NCN')
    job = {'family': 'link', 'TEST_access': False, 'recipe': {'id': cell['cell_id'], 'architecture': 'NCN',
           'dataset': 'ogbl-collab', 'native_fixed_config': copy.deepcopy(cell['native_fixed_config'])},
           'policy': declaration('source_reference'), 'procedure': 'native_single', 'seed': 0, 'body': 0, 'split': 0}
    assert module.create_policy(job).cell_id == 'collab-ncn'
    for key, value in (('TEST_access', True), ('seed', 42), ('body', 1), ('procedure', 'gnnm_boundary_native')):
        rejects(lambda key=key, value=value: module.create_policy({**job, key: value}))
    changed = copy.deepcopy(job); changed['recipe']['native_fixed_config']['batch_size'] = 1
    rejects(lambda: module.create_policy(changed))
    for name in ('policy.py', 'check_policy.py'):
        compile((here/name).read_bytes(), str(here/name), 'exec')
    for path in ('ncn/ncn__NeighborOverlap.py', 'buddy/buddy_src_runners_run.py'):
        tree = ast.parse((module.PHASE/'reference_wrapper_redesign_20261013_v1/native_sources'/path).read_text())
        assert not any(isinstance(node, ast.Break) for node in ast.walk(tree))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    return {'frozen27_roster_and_saved_source_hashes': 'passed', 'three_cells_two_declared_tracks': 'passed',
            'source100_no_early_stop': 'passed', 'source_BUDDY_zero_sentinel_and_first_ties': 'passed',
            'extended1000_min100_patience50_and_cap': 'passed', 'positive_complete_epoch_VALID_only_selection': 'passed',
            'raw_Hits50_units_and_nonfinite_rejection': 'passed', 'strict_resume_and_stop_on_entry_status': 'passed',
            'explicit_evaluation_adaptation_and_source_role_guards': 'passed', 'exact_source_config_and_seed_guards': 'passed',
            'JSON_safe_policy_state': 'passed', 'python_syntax_and_import_without_Torch_NumPy': 'passed',
            'saved_native_programs_executed': False, 'model_or_dataset_imports': 0, 'server_operations': 0,
            'fits_launched': 0, 'TEST_opened': False,
            'numerical_native_and_full_runner_qualification': 'pending'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
