"""Disabled F-only successor: reuse original collector, restore helper and counts."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def entry(release_path, release_sha256):
    started = time.monotonic(); usage0 = resource.getrusage(resource.RUSAGE_SELF)
    cfg = read(release_path)
    if hashlib.sha256(Path(release_path).read_bytes()).hexdigest() != release_sha256:
        raise ValueError('Exact root release')
    if not all(cfg.get(k) is True for k in ('enabled', 'root_readout_authorized', 'source_review_approved',
            'trusted_checkpoint_deserialization_authorized', 'external_owned_bound_confirmed')):
        raise ValueError('Disabled until reviewed root readout release and existing finite owner')
    if any(cfg.get(k) is not False for k in ('training', 'TEST_access', 'automatic_retry', 'reselection', 'calibration')):
        raise ValueError('Readout only, original decisions and TEST boundary')
    pins0 = read(HERE / 'INPUTS.json')
    reader_file = PHASE / pins0['reader_gate']['path']
    if reader_file.stat().st_size != pins0['reader_gate']['bytes'] or hashlib.sha256(reader_file.read_bytes()).hexdigest() != pins0['reader_gate']['sha256']:
        raise ValueError('Exact existing stdlib reader source before import')
    spec = importlib.util.spec_from_file_location('_alphaF3_existing_reader_gate', reader_file)
    reader_gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader_gate)
    pins = read(PHASE / pins0['legacy_source_bindings']['path']); g = reader_gate.helpers(pins)
    g.bound(pins0['reader_gate']); g.bound(pins0['legacy_source_bindings'])
    manifest = read(g.bound(cfg['source_manifest']))
    for row in manifest['files']:
        g.bound(row)
    read_bindings = read(HERE / 'READ_BINDINGS.json')
    custody_descriptor = next(r for r in read_bindings['files']
        if r['path'] == 'three_research_roles_continuation_20261010_v2/GPU77_CLOSED6_CUSTODY.json')
    authenticated_custody = read(g.bound(custody_descriptor))
    g.require(authenticated_custody == pins0['whole6_original_custody'] and authenticated_custody['whole6_closed']
        and all(o['complete'] for o in authenticated_custody['owner_closures']), 'Authenticated original whole-six custody')
    authenticated_rows = {r['cell_id']: (o['lane'], r) for o in authenticated_custody['owner_closures'] for r in o['records']}
    g.require(len(authenticated_rows) == 6, 'Exact original six custody slots')
    g.require(pins0['whole6_closed'] and len(pins0['whole6_original_selected_records']) == 6
        and [r['seed'] for r in pins0['records']] == [6101, 6203, 6307]
        and cfg['maximum_member_forwards'] == 12, 'Already closed six; exact F-only three')
    old_collection = read(g.bound(pins0['old_collection']))
    g.require(old_collection['status'] == 'complete' and old_collection['analysis_complete']
        and old_collection['all_three_alphaF_cohorts_frozen_before_candidates'], 'Original completed old collector')
    old_by_seed = {r['seed']: r for r in old_collection['cells'] if r['condition'] == 'alphaF'}
    old_cohorts = read(g.bound(pins0['old_cohorts']))
    for pair in pins0['records']:
        old, new = pair['old'], pair['new']; seed = pair['seed']
        g.require(old['raw_archive'] == old_by_seed[seed]['raw_archive']
            and old['selected_checkpoint'] == old_by_seed[seed]['selected_checkpoint']
            and old['selected_metadata'] == {k: v for k, v in old_by_seed[seed]['selected_metadata'].items() if k != 'graph_relation_credit'}
            and old_cohorts['cohorts'][str(seed)] == pair['old_frozen_cohort'], 'Exact existing old F bytes/cohorts')
        for key in ('complete', 'run', 'exit_receipt'):
            g.bound(old[key])
        completed = read(g.bound(new['complete']))
        g.require(completed['complete'] and completed['epochs'] == completed['steps'] == 1100
            and completed['selected_sha256'] == new['selected_checkpoint']['sha256'], 'Original completed new F')
        cell = new['original_selected_scalars']['cell_id']; lane, authenticated = authenticated_rows[cell]
        g.require(new['original_closed_custody'] == authenticated and authenticated['status'] == 'complete'
            and authenticated['automatic_retry'] is False and new['complete'] == authenticated['completion']
            and new['selected_checkpoint'] == authenticated['selected_checkpoint'], 'Exact authenticated inline cell and descriptors')
        for key in ('release', 'completion', 'selected_checkpoint'):
            g.bound(authenticated[key])
        actual_exit = authenticated['actual_exit']; physical_terminal = authenticated['physical_terminal']
        identity = actual_exit['child_identity']
        expected_argv = [pins['runtime']['python'], '-B', str(g.bound(pins0['native_helper'])), '--release',
            str(g.bound(authenticated['release'])), '--release-sha256', authenticated['release']['sha256']]
        g.require(actual_exit['cell_id'] == cell and actual_exit['exit_code'] == 0
            and actual_exit['terminal_wait_observed'] is True and actual_exit['exit_code_authority'] == 'subprocess.Popen.wait/poll'
            and actual_exit['identity_admitted_for_signals'] is True and actual_exit['attempts'] == 1
            and actual_exit['retry'] is False and actual_exit['reason'] is None and not actual_exit['signals_sent']
            and actual_exit['signal_refusal'] is None and actual_exit['job_sha256'] == authenticated['release']['sha256']
            and actual_exit['cwd'] == actual_exit['observed_cwd'] == pins['runtime']['repository']
            and identity == actual_exit['raw_identity_observation'] == physical_terminal['identity']
            and identity['observation_complete'] is True and identity['argv'] == expected_argv
            and type(identity['PID']) is int and identity['PID'] > 0 and identity['pgid'] == identity['sid'] == identity['PID']
            and type(identity['start_ticks']) is int and identity['start_ticks'] > 0
            and physical_terminal['actual_exit_code'] == 0 and physical_terminal['direct_wait_observed'] is True
            and physical_terminal['child_absent'] is True and physical_terminal['child_no_CUDA_rows'] is True,
            'Original actual wait, child identity and process/CUDA absence')
        # Original run_existing_owner.py writes these under lane.owner_output/logs, not release.parent/../logs.
        original_logs = g.inside('graph_relation_independent_native_local_scorer_training_supervision_root_20261010_v1/' + lane) / 'logs'
        g.require(read(original_logs / (cell + '.EXIT.json')) == actual_exit
            and read(original_logs / (cell + '.PHYSICAL_TERMINAL.json')) == physical_terminal, 'Actual raw lane receipts equal authenticated observations')
    runtime = pins['runtime']; source = pins['data_source_pins']
    g.require(socket.gethostname() == runtime['hostname'] and Path.cwd().resolve() == Path(runtime['repository']).resolve()
        and str(Path(sys.executable).absolute()) == runtime['python'] and os.environ.get('PYTHONPATH', '') == ''
        and cfg['physical_gpu_uuid'] in runtime['physical_gpu_inventory']
        and os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['physical_gpu_uuid'], 'Original normal77 runtime and selected physical GPU')
    output = g.inside(cfg['output_directory'])
    g.require(not output.exists() and output.name == 'WikiCS_initializer_alphaF3_readout_execution_root_20261010_v1', 'Separate fresh fixed output')
    output.mkdir(mode=0o700); (output / 'raw').mkdir(); (output / 'compact').mkdir()
    legacy = g.module(g.bound(pins['reuse']['legacy_collect']), '_alphaF3_original_collector'); write = legacy.write
    records = [copy.deepcopy(p['new']) for p in pins0['records']]
    cost = dict(schema='WikiCS-alphaF3-readout-cost-v1', status='running', maximum_member_forwards=12,
        attempted_member_forwards=0, completed_member_forwards=0, checkpoint_deserialization_attempts=0,
        checkpoint_bytes_submitted_to_deserializer=0, optimizer_construction_attempts=0, optimizer_constructions=0,
        optimizer_construction_seconds=0., Session_construction_attempts=0, Session_constructions=0,
        Session_construction_seconds=0., peak_CUDA_allocated_bytes=0, peak_CUDA_reserved_bytes=0,
        TRAIN_updates=0, backward_calls=0, Adam_steps=0, old_member_forwards=0,
        Session_counter_scope='Exact native base.make_session call, including initializer and partition setup.',
        old_training_and_reader_costs_not_readded=True, external_wait_cleanup_transfer_root_owned=True)
    try:
        import numpy as np
        import torch
        import importlib.metadata as metadata
        g.require({k: metadata.version(k) for k in ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')}
            == {k: runtime[k] for k in ('torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')}, 'Original providers')
        torch.set_num_threads(2)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(cfg['owned_GPU_cap_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        public = g.module(g.bound(pins['relation_source_pins']['public_portable']), '_alphaF3_public'); sys.modules['portable'] = public
        data = g.module(g.bound(pins['reuse']['public_data']), 'data_interface'); sys.modules['data_interface'] = data
        new_root = g.bound(pins0['native_helper']).parent; sys.path.insert(0, str(new_root))
        helper = g.module(g.bound(pins0['native_helper']), '_alphaF3_exact_native_helper')
        helper.source_gate()
        hooks = g.module(g.bound(pins0['collection_hooks']), '_alphaF3_existing_hooks')
        readout = g.module(g.bound(pins0['readout']), '_alphaF3_existing_readout')
        analysis, contract, _ = readout.configure(g, pins)
        # Same existing restore gate, with the actual native source identity.
        restore_pins = copy.deepcopy(pins); restore_pins['reuse']['relation_manifest'] = pins0['native_manifest']
        restore = hooks.restore_factory(g, restore_pins, helper)
        original_base = helper.base_module
        def observed_base():
            base = original_base(); original_make = base.make_session
            def observed_make(*args, **kwargs):
                cost['Session_construction_attempts'] += 1; began = time.monotonic()
                try:
                    session = original_make(*args, **kwargs); cost['Session_constructions'] += 1; return session
                finally:
                    cost['Session_construction_seconds'] += time.monotonic() - began
            base.make_session = observed_make
            return base
        helper.base_module = observed_base
        tree = hooks.adapt_tree(g.bound(pins['reuse']['legacy_collect']).read_text())
        namespace = dict(vars(legacy), MAX_FORWARDS=12, _relation_restore=restore)
        exec(compile(tree, str(g.bound(pins['reuse']['legacy_collect'])) + ':native-alphaF3', 'exec'), namespace)
        collect_cell = namespace['collect_cell']
        train, valid, origin = data.load_train_valid('wikics', g.bound(source['train']), g.bound(source['development']))
        ordered = dict(x=train['x'], edge_index=train['edge_index'], train_ids=train['ids'], train_y=train['y'], valid_ids=valid['ids'], valid_y=valid['y'])
        for name, tensor in ordered.items():
            expected = source['ordered_raw_tensor_fingerprints'][name]
            g.require(str(tensor.dtype) == expected['dtype'] and list(tensor.shape) == expected['shape']
                and hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() == expected['contiguous_raw_bytes_sha256'], 'Original ordered role: ' + name)
        iterator = data.validation_batches('wikics', train, valid, 'cuda:0'); batch, truth = next(iterator)
        g.require(next(iterator, None) is None and len(truth) == 5274, 'Original complete VALID')
        native, _ = public.native_sources('wikics', g.bound(source['polynormer']))
        for row in records:
            collect_cell(torch, np, public, None, analysis, contract, native, batch, truth, row, source, output, cost)
            write(output / 'compact/COLLECTION.json', dict(status='running', cells=records))
            g.require(row['collection_status'] == 'complete', 'Retain collection failure; no continuation or retry')
        g.require(cost['attempted_member_forwards'] == cost['completed_member_forwards'] == 12
            and cost['Session_constructions'] == cost['optimizer_constructions'] == 3, 'Exactly twelve calls and three fresh constructors')
        paired = []; summaries = []
        for pair, row in zip(pins0['records'], records):
            old = pair['old']; after = readout.arrays_for(np, g, row, contract); before = readout.arrays_for(np, g, old, contract)
            frozen = contract.load(np, g.bound(pair['old_frozen_cohort']['archive']), pair['old_frozen_cohort']['archive'])
            g.require(np.array_equal(after['valid_ids'], batch['ids'].cpu().numpy()) and np.array_equal(after['truth'], truth.numpy()), 'Original new IDs/truth')
            fresh = row['selected_metadata']; original = row['original_selected_scalars']
            g.require(fresh['epoch'] == original['selected_epoch'] and fresh['global_mode'] == original['selected_global'], 'Original selected endpoint and mode')
            for arrays, selected in [(after, fresh), (before, old['selected_metadata'])]:
                actual = [int((arrays['pool_prediction'] == arrays['truth']).sum())] + [int((r == arrays['truth']).sum()) for r in arrays['member_prediction']]
                expected = [selected['stored_accuracy']] + selected['stored_member_accuracy']
                g.require(actual == [round(x * 5274) for x in expected], 'Exact original selected integer correct counts')
            paired.append(readout.pair(np, before, after, frozen, pair['seed'], 'native_alphaF-copied_alphaF', analysis, contract))
            summaries.append(dict(seed=pair['seed'], old_selected=old['selected_metadata'], new_selected=fresh,
                old=analysis.summary(np, before, frozen['full_population'], contract), new=analysis.summary(np, after, frozen['full_population'], contract)))
            del before, after, frozen
        metrics = ('served_accuracy', 'served_nll', 'mean_member_accuracy', 'worst_member_accuracy')
        deltas = {key: [r['new'][key] - r['old'][key] for r in summaries] for key in metrics}
        protocol = read(g.bound(pins0['native_protocol']))['unchanged_quality_screen']
        checks = dict(positive_primary_accuracy_all3seeds=all(x > 0 for x in deltas['served_accuracy']),
            minimum_mean_primary_gain_pp=sum(deltas['served_accuracy']) / 3 * 100 >= protocol['minimum_mean_primary_gain_pp'],
            mean_pool_NLL_not_worse=sum(deltas['served_nll']) / 3 <= 0,
            mean_member_accuracy_degradation_limit=sum(deltas['mean_member_accuracy']) / 3 * 100 >= -protocol['maximum_mean_member_accuracy_degradation_pp'],
            worst_member_accuracy_degradation_limit=sum(deltas['worst_member_accuracy']) / 3 * 100 >= -protocol['maximum_worst_member_accuracy_degradation_pp'])
        write(output / 'compact/COMPARISON.json', dict(schema='WikiCS-native-initializer-F-only-comparison-v1', summaries=summaries,
            paired=paired, scalar_deltas=deltas, original_screen=protocol, checks=checks, passes=all(checks.values()),
            TEST_access=False, original_scores_changed=False, confirmation=False, novelty='none', execution_extension_authorized=False,
            parity_scope='Original common source/recipe/selector/data/providers and own selected endpoints; original modes differ; no bitwise or clean population-level causal claim.'))
        write(output / 'compact/COLLECTION.json', dict(status='complete', cells=records, old_archive_forward_calls=0, TEST_access=False))
        cost['status'] = 'complete'
    except BaseException as error:
        cost.update(status='retained_failure', failure=dict(type=type(error).__name__, message=str(error), automatic_retry=False))
        write(output / 'compact/FAILURE.json', cost['failure']); raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cost.update(inclusive_wall_seconds=time.monotonic() - started, CPU_user_seconds=usage.ru_utime - usage0.ru_utime,
            CPU_system_seconds=usage.ru_stime - usage0.ru_stime, peak_RSS_bytes=int(usage.ru_maxrss * 1024),
            cells=records, output_bytes=sum(f.stat().st_size for f in output.rglob('*') if f.is_file()))
        write(output / 'compact/COST.json', cost)
    return str(output / 'compact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True); args = parser.parse_args()
    print(entry(args.release, args.release_sha256))
