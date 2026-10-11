"""Inactive ten-bank after-fit consumer; existing statistical/acquisition plan.

Root supplies new admission after complete original E support. No numerical
imports at module import, new GNN fitting, fabricated original roster, or launch.
"""
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
ARM = 'common_builder_E4'
PORTABLE_MANIFEST = 'd52b185077d81692472a0bedc002df864c1c7a34f6de55b4e398126c2ce80781'
ANCESTOR_MANIFEST = '5cf66efd955348f1c0fbdc80c48ec60523364fd574803da172b98ed3fefa81b1'
ANCESTOR_SOURCE = '4fa885e8e2f0e6ddc5de861be72628a3853eb3c843590246a5f24d4ff02c2943'
ANCESTOR_PLAN = 'fc2170940178ac5444898845f0872a3d56c7b79e0475d4d9f3a716700fd1f328'
ORIGINAL_CONTRACT = 'aea7f5dc47c1d3870bf6e8598ecdd2a3a3e6931c20c6a490cf75e5561bb390e0'
ORIGINAL_PROTOCOL = 'ed30ed3c3ce340431c945f0ffc35a719630f80b2f7e18f4c88f2d33fb12dcab1'
CAPS = dict(export=172800, calibrate=3600, confirm=3600)
RSS_CAP, CUDA_CAP = 64 * 1024 ** 3, 48 * 1024 ** 3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def checked(item):
    require(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}
            and receipt(item['path']) == item, 'Exact immutable receipt differs')
    return Path(item['path'])


def read(item):
    return json.loads(checked(item).read_text())


def write(path, value):
    require(not path.exists(), 'One fresh terminal record; retain prior attempts')
    temp = path.with_suffix(path.suffix + '.tmp')
    require(not temp.exists(), 'Partial write retained')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same_bytes(a, b):
    return a['bytes'] == b['bytes'] and a['sha256'] == b['sha256']


def admit(spec, root_guard, stage, context):
    """Callback first verifies applicable host/GPU, then exact future custody."""
    source = Path(__file__).resolve()
    interface = source.with_name('INTERFACE.json')
    proof = root_guard(copy.deepcopy(spec), action='common_builder_E4_after_fit_' + stage,
                       source_path=str(source), interface_path=str(interface),
                       stage_cap_seconds=CAPS[stage], context=copy.deepcopy(context))
    require(isinstance(proof, dict) and proof.get('admitted') is True
            and proof.get('action') == 'common_builder_E4_after_fit_' + stage
            and proof.get('whole_original_confirmation_complete') is True
            and proof.get('complete_E_support') is True
            and proof.get('actual_portable_runtime_check_complete') is True
            and proof.get('control_acquisition_closed_and_reaped') is True
            and proof.get('paired_role_graph_config_verified') is True
            and proof.get('finite_external_owner_bound') is True
            and type(proof.get('external_owner_wall_seconds')) is int
            and 0 < proof['external_owner_wall_seconds'] <= sum(CAPS.values())
            and proof.get('stage_cap_seconds') == CAPS[stage]
            and proof.get('source_sha256') == sha(source)
            and proof.get('interface_sha256') == sha(interface),
            'Explicit new root stage admission/full support/custody/finite owner required')
    return proof


def dependencies(spec):
    package = Path(spec['portable_directory']).resolve()
    ancestor = Path(spec['attribution_directory']).resolve()
    require(sha(package / 'MANIFEST.json') == PORTABLE_MANIFEST
            and sha(ancestor / 'MANIFEST.json') == ANCESTOR_MANIFEST
            and sha(ancestor / 'common_builder.py') == ANCESTOR_SOURCE
            and sha(ancestor / 'PLAN.json') == ANCESTOR_PLAN, 'Preserve exact sealed ancestor and portable package')
    common = load('common_builder_consumer_local_common', package / 'common.py')
    common.bundle()  # Actual package bytes, no numerical import or PASS.
    engine = load('common_builder_consumer_engine', package / 'engine.py')
    adapter = load('common_builder_consumer_ancestor', ancestor / 'common_builder.py')
    load('contracts', package / 'deps/readout/contracts.py')
    load('inference', package / 'deps/readout/inference.py')
    reader = load('common_builder_consumer_math', package / 'deps/readout/readout.py')
    original_contract = checked(spec['original_prediction_contract'])
    require(sha(original_contract) == ORIGINAL_CONTRACT, 'Exact sealed original native closure validator')
    contracts = load('common_builder_consumer_original_contracts', original_contract)
    return common, engine, adapter, contracts, reader


def original_reference_join(spec, contracts):
    """Validate real original60/90 closures; hash only, never rescore originals."""
    o = spec['original']
    require(set(o) == {'cohort', 'predictions', 'calibration', 'test'}, 'Exact full original reference closures')
    cohort, prediction, calibration, test = (read(o[name]) for name in ('cohort', 'predictions', 'calibration', 'test'))
    require(cohort.get('complete') is True and cohort.get('banks') == 60 and cohort.get('fits') == 90
            and cohort.get('status') == 'BOTH_COMPLETE_ZERO_EXIT'
            and set(cohort['children']) == {'core', 'author'}
            and all(r['exit_code'] == 0 for r in cohort['children'].values()),
            'Actual complete original cohort, not ten-control metadata')
    contracts.validate_predictions(prediction, ORIGINAL_PROTOCOL)
    require(all(r['status'] == 'selected_finite' and all(f['status'] == 'selected_finite' for f in r['fits'])
                for r in prediction['records']), 'All real original60 banks/90 selected states finite')
    core, author = (read(cohort[name]) for name in ('core_complete_receipt', 'author_complete_receipt'))
    require(core.get('complete') is True and core.get('full_same_runtime_stage_complete') is True
            and core.get('groups') == 50 and core.get('fit_units') == 80
            and len(core['results']) == 50 and author.get('complete') is True
            and author.get('banks') == author.get('fits') == 10 and len(author['records']) == 10,
            'Real original50/80 core and10/10 author acquisition receipts')
    acquisition = {(r['seed'], r['arm']): r for r in core['results']}
    require(all(r['arm'] == 'author_meanSAGE_M1' for r in author['records']), 'Actual original author arm')
    acquisition.update({(r['seed'], contracts.ARMS[-1]): r for r in author['records']})
    require(set(acquisition) == {(s, a) for s in SEEDS for a in contracts.ARMS}
            and all([f['selected_state_receipt'] for f in r['fits']]
                    == [f['selected_state_receipt'] for f in acquisition[r['block'], r['arm']]['fits']]
                    for r in prediction['records']), 'Join actual original acquisition and prediction selected states')
    states = [f['selected_state_receipt'] for r in prediction['records'] for f in r['fits']]
    require(len(states) == 90 and len({r['path'] for r in states}) == 90, 'Ninety distinct actual original selected states')
    require(calibration.get('terminal') is True and calibration.get('scalar_endpoints') == 60
            and calibration.get('all_finite') is True and len(calibration['records']) == 60
            and calibration['prediction_closure'] == o['predictions']
            and {(r['block'], r['arm']) for r in calibration['records']} == {(s, a) for s in SEEDS for a in contracts.ARMS}
            and all(r['status'] == 'finite_fixed_endpoint' and r['updates'] == 500
                    and r['endpoint'] == 'fixed_final_update' and math.isfinite(r['log_T'])
                    and math.isfinite(r['temperature']) and r['temperature'] > 0
                    for r in calibration['records']), 'All original60 fixed500 CAL endpoints closed')
    banks = {(r['block'], r['arm']): r for r in prediction['records']}
    require(all(r['selected_CAL_archive'] == banks[r['block'], r['arm']]['selected_CAL_archive']
                for r in calibration['records']), 'Actual original CAL endpoint/archive join')
    require(test.get('terminal') is True and test.get('one_complete_TEST_truth_release') is True
            and test.get('all_raw_finite') is True and test.get('all_calibrated_finite') is True
            and test['blocks'] == list(SEEDS) and test['arms'] == list(contracts.ARMS)
            and test['prediction_closure'] == o['predictions'] and test['calibration_terminal'] == o['calibration']
            and len(test['rows']) == 60, 'Actual all60 original raw/calibrated TEST closure')
    rows = {(r['block'], r['arm']): r for r in test['rows']}
    require(set(rows) == {(s, a) for s in SEEDS for a in contracts.ARMS}, 'No original survivor reference projection')
    endpoints = {(r['block'], r['arm']): r for r in calibration['records']}
    require(calibration['custody']['protocol_sha256'] == test['custody']['protocol_sha256'] == ORIGINAL_PROTOCOL
            and all(r['selected_predictions'] == banks[key]['selected_CONFIRM_archive']
                    and r['fits'] == banks[key]['fits'] and r['calibration_endpoint'] == endpoints[key]
                    and r['same_selected_native_state'] is True for key, r in rows.items()),
            'Actual original prediction/CAL/TEST scalar-record join under frozen protocol')
    for s in SEEDS:
        require(same_bytes(calibration['CAL_truth_receipts'][str(s)], spec['roles'][str(s)]['cal_truth'])
                and same_bytes(test['TEST_truth_receipts'][str(s)], spec['roles'][str(s)]['confirm_truth']),
                'Reuse exact original paired CAL/CONFIRM target bytes')
    def finite(value):
        if isinstance(value, dict): return all(finite(v) for v in value.values())
        if isinstance(value, list): return all(finite(v) for v in value)
        return isinstance(value, (int, float)) and math.isfinite(value)
    for r in rows.values():
        for kind in ('raw', 'calibrated'):
            require('quality' in r[kind] and 'counts' in r[kind]
                    and finite(r[kind]['quality']) and finite(r[kind]['counts'])
                    and len(r[kind]['classes']) == 5
                    and all(c['class_id'] == k and finite(c['quality']) and finite(c['counts'])
                            for k, c in enumerate(r[kind]['classes'])),
                    'Finite original scalar records reused verbatim')
    return dict(rows=rows, receipts=copy.deepcopy(o), original_banks=60, original_selected_states=90,
                original_CAL_endpoints=60, original_scores_recalculated=False)


def control_fit(spec):
    fit = read(spec['fit_closure'])
    require(fit.get('complete') is True and fit.get('banks') == 10
            and fit.get('optimizer_acquisition_bundles') == 10 and fit.get('native_body_fit_records') == 10
            and fit['fixed_seed_roster'] == list(SEEDS) and fit['fixed_arm_roster'] == [ARM]
            and len(fit['records']) == 10 and {r['seed'] for r in fit['records']} == set(SEEDS)
            and fit['source']['sha256'] == ANCESTOR_SOURCE and fit['plan']['sha256'] == ANCESTOR_PLAN,
            'Actual ancestor ten-bank acquisition closure')
    for name in ('cohort', 'calibration', 'test'):
        require(same_bytes(fit['original_closure_receipts'][name], spec['original'][name]), 'Same original confirmation reused')
    states = []
    for r in fit['records']:
        require(r['arm'] == ARM and r['members'] == 4 and len(r['fits']) == 1
                and r['builder_tying']['parameter_reduction'] == 3264
                and r['fits'][0]['value_parameter_roles']['private_builder_factor_parameters'] == 0
                and r['fits'][0]['value_parameter_roles']['common_builder_factor_parameters'] == 1088,
                'Actual ten selected common-builder states')
        checked(r['selected_VALID_archive'])
        for f in r['fits']:
            checked(f['selected_state_receipt']); states.append(f['selected_state_receipt'])
    require(len(states) == 10 and len({r['path'] for r in states}) == 10, 'Ten distinct selected files; no substitutes')
    return fit


def resource_guard(start, seconds, torch=None, device=None):
    require(time.monotonic() - start < seconds, 'Fixed stage wall cap; retain failure without retry')
    multiplier = 1 if sys.platform == 'darwin' else 1024
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier <= RSS_CAP, 'Fixed64GiB RSS cap')
    if torch is not None and device is not None and device.type == 'cuda':
        require(torch.cuda.max_memory_allocated(device) <= CUDA_CAP, 'Fixed48GiB CUDA cap')


def export_ten(spec, engine, common, adapter, reader, output):
    start = time.monotonic(); fit = control_fit(spec)
    cfg = copy.deepcopy(spec['cfg'])
    require(cfg['seeds'] == list(SEEDS) and cfg['family_limits'] == adapter.LIMITS
            and cfg['device'] == 'cuda:0' and cfg['profile']['dataset'] == 'CoauthorPhysics'
            and cfg['protocol'].get('schema') == 'Physics_E_common_builder_attribution_v1', 'Unchanged ancestor acquisition config policy')
    truth_paths = {str(Path(r[k]['path']).resolve()) for r in spec['roles'].values() for k in ('cal_truth', 'confirm_truth')}
    require(not any(str(Path(r['path']).resolve()) in truth_paths for r in cfg.get('file_bindings', {}).values()),
            'Engine local validation binds public/safe files; no premature target read')
    family = adapter.bind_common_builders(engine.construct(cfg, output / 'constructor', common.local_validate))
    t, np, records = family.torch, family.np, []
    index = {r['seed']: r for r in fit['records']}
    for seed in SEEDS:
        resource_guard(start, CAPS['export'], t, family.device)
        family.activate_role_block(seed); family._operation = 'endpoint'
        role = spec['roles'][str(seed)]
        cal = reader.ids_only(np, role['cal_ids'], 150)
        confirm = reader.ids_only(np, role['confirm_ids'], 34093)
        all_ids = np.concatenate((family.train_ids.cpu().numpy(), family.valid_ids.cpu().numpy(), cal, confirm))
        require(np.array_equal(np.sort(all_ids), np.arange(34493, dtype=np.int64)), 'Complete unique original paired role membership')
        ids = t.from_numpy(np.concatenate((cal, confirm))).to(family.device)
        selected = index[seed]['fits'][0]
        # Native constructor plus real ancestor adapter; no optimizer creation.
        before_restore = time.monotonic()
        body, model = family.native_body(seed, 0, 0, True, 4)
        state = t.load(checked(selected['selected_state_receipt']), map_location='cpu', weights_only=True)
        require(state['step'] == selected['selected_step'] and state['selection'] == selected['selection'], 'Actual original control selector custody')
        model.load_state_dict(state['model'], strict=True)
        for p in model.parameters(): p.requires_grad_(False)
        model.to(family.device).eval()
        family.synchronize(); before = time.monotonic()
        restore_seconds = before - before_restore
        with t.no_grad(): logits = family.logits(model, state['streams'], True, ids)
        resource_guard(start, CAPS['export'], t, family.device)
        require(logits.shape == (4, 34243, 5) and bool(t.isfinite(logits).all()), 'Complete label-free selected CAL/CONFIRM forward')
        logits = logits.cpu(); logp = logits.log_softmax(-1)
        member = logits.argmax(-1); pool = logits.softmax(-1).mean(0).argmax(-1)
        folder = output / ('common_builder_E4_seed' + str(seed)); folder.mkdir()
        archives = {}
        for name, membership, subset in (('CAL', cal, slice(0, 150)), ('CONFIRM', confirm, slice(150, None))):
            path = folder / ('selected_' + name + '.npz')
            np.savez(path, ids=membership, member_log_probs=logp[:, subset].numpy(),
                     native_member_argmax=member[:, subset].numpy(), native_pool_argmax=pool[subset].numpy())
            archives[name] = receipt(path)
        family.synchronize()
        records.append(dict(block=seed, arm=ARM, status='selected_finite', members=4,
                            fits=[dict(status='selected_finite', selected_state_receipt=selected['selected_state_receipt'],
                                       selected_step=selected['selected_step'], costs=selected['costs'])],
                            selected_CAL_archive=archives['CAL'], selected_CONFIRM_archive=archives['CONFIRM'],
                            selected_constructor_restore_seconds=restore_seconds,
                            export_seconds=time.monotonic() - before, native_selected_trajectories=4,
                            CAL_CONFIRM_truth_access=False, model_updates=0))
        family._fit_model = family._single_adapter = None
        del model, body, state, logits, logp, member, pool, ids
        if family.device.type == 'cuda': t.cuda.empty_cache()
        resource_guard(start, CAPS['export'], t, family.device)
    control_fit(spec)
    value = dict(schema='Physics_common_builder_predictions_v1', terminal=True, complete=True,
                 banks=10, selected_states=10, native_selected_trajectories=40, records=records,
                 fit_closure=spec['fit_closure'], CAL_truth_access=False, CONFIRM_truth_access=False,
                 model_updates=0, seconds=time.monotonic() - start,
                 peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                 peak_CUDA_allocated_bytes=t.cuda.max_memory_allocated(family.device), isolated_speed_claim=False)
    write(output / 'COMPLETE_COMMON_BUILDER_PREDICTIONS.json', value)
    return value


def check_predictions(spec, value):
    require(value.get('schema') == 'Physics_common_builder_predictions_v1' and value.get('complete') is True
            and value.get('terminal') is True and value.get('fit_closure') == spec['fit_closure']
            and value.get('CAL_truth_access') is False and value.get('CONFIRM_truth_access') is False
            and value.get('model_updates') == 0 and value.get('native_selected_trajectories') == 40
            and value.get('banks') == value.get('selected_states') == 10 and len(value['records']) == 10
            and {(r['block'], r['arm']) for r in value['records']} == {(s, ARM) for s in SEEDS}, 'Exact ten-bank prediction closure')
    require(read(receipt(Path(spec['output']) / 'export/COMPLETE_COMMON_BUILDER_PREDICTIONS.json')) == value,
            'Actual saved prediction closure equals the completed controller return')
    fit = control_fit(spec); selected = {r['seed']: r['fits'][0] for r in fit['records']}
    for r in value['records']:
        require(r['status'] == 'selected_finite' and r['members'] == 4 and len(r['fits']) == 1, 'Complete new control bank')
        require(r['fits'][0]['selected_state_receipt'] == selected[r['block']]['selected_state_receipt']
                and r['fits'][0]['selected_step'] == selected[r['block']]['selected_step'], 'Actual control fit/export selected-state join')
        checked(r['fits'][0]['selected_state_receipt']); checked(r['selected_CAL_archive']); checked(r['selected_CONFIRM_archive'])


def load_control_bank(np, record, role, expected_ids):
    # Same original loader contract, with actual common-control member count4;
    # never relabel the new bank E4 or use a fabricated original60-bank closure.
    with np.load(checked(record['selected_' + role + '_archive']), allow_pickle=False) as a:
        require(set(a.files) == {'ids', 'member_log_probs', 'native_member_argmax', 'native_pool_argmax'}, 'Native label-free archive schema')
        ids, logp, member, pool = (a[k].copy() for k in ('ids', 'member_log_probs', 'native_member_argmax', 'native_pool_argmax'))
    count = 150 if role == 'CAL' else 34093
    require(ids.dtype == np.int64 and np.array_equal(ids, expected_ids)
            and logp.dtype == np.float32 and logp.shape == (4, count, 5) and np.isfinite(logp).all()
            and np.allclose(np.exp(logp.astype(np.float64)).sum(-1), 1., rtol=1e-5, atol=1e-7), 'Existing finite native encoding contract')
    require(member.dtype == pool.dtype == np.int64 and member.shape == (4, count) and pool.shape == (count,)
            and np.isin(member, np.arange(5)).all() and np.isin(pool, np.arange(5)).all(), 'Native integer decision authority')
    return logp, member, pool


def calibrate_ten(spec, reader, predictions, output):
    check_predictions(spec, predictions)  # All ten states/twenty archives before CAL targets.
    require(set(spec['provider_modules']) == {'numpy', 'torch'}, 'Exact existing numeric providers')
    np, t = reader.providers({'provider_modules': spec['provider_modules']})
    cfg = dict(roles=spec['roles']); start = time.monotonic()
    roles = {s: reader.truth(np, cfg, s, 'CAL') for s in SEEDS}
    index = {r['block']: r for r in predictions['records']}
    rows = [dict(block=s, arm=ARM, status='unstarted_retained', requested_updates=500, updates=0,
                 role='CAL', parameters=1, CONFIRM_truth_access=False, retry=False) for s in SEEDS]
    for record in rows:
        before = time.monotonic()
        try:
            resource_guard(start, CAPS['calibrate'])
            ids, targets = roles[record['block']]; y = t.from_numpy(targets)
            raw, _, _ = load_control_bank(np, index[record['block']], 'CAL', ids)
            logp = t.from_numpy(raw).to(t.float64)
            log_T = t.nn.Parameter(t.zeros((), dtype=t.float64, device='cpu'))
            optimizer = t.optim.Adam([log_T], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0)
            trace = []
            for update in range(1, 501):
                resource_guard(start, CAPS['calibrate'])
                optimizer.zero_grad(); pooled, _ = reader.log_predictions(t, logp, log_T)
                objective = -pooled[t.arange(len(y)), y].mean()
                require(bool(t.isfinite(objective)), 'Finite fixed CAL objective')
                objective.backward(); require(bool(t.isfinite(log_T.grad)), 'Finite scalar gradient')
                optimizer.step(); record['updates'] = update
                if update in (1, 100, 200, 300, 400, 500):
                    trace.append(dict(update=update, pre_update_CAL_NLL=float(objective.detach())))
            reader.log_predictions(t, logp, log_T)
            record.update(status='finite_fixed_endpoint', log_T=float(log_T.detach()),
                          temperature=float(log_T.detach().exp()), endpoint='fixed_final_update', trace=trace,
                          selected_CAL_archive=index[record['block']]['selected_CAL_archive'])
        except Exception as error:
            record.update(status='failed_retained', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        record['seconds'] = time.monotonic() - before
    check_predictions(spec, predictions)
    value = dict(schema='Physics_common_builder_CAL_v1', terminal=True, scalar_endpoints=10, records=rows,
                 all_finite=all(r['status'] == 'finite_fixed_endpoint' and r['updates'] == 500 for r in rows),
                 requested_updates=5000, actual_updates=sum(r['updates'] for r in rows),
                 prediction_receipt=receipt(output.parent / 'export/COMPLETE_COMMON_BUILDER_PREDICTIONS.json'),
                 CAL_truth_receipts={str(s): spec['roles'][str(s)]['cal_truth'] for s in SEEDS},
                 CONFIRM_truth_access=False, comparisons=[], interpretation_eligible=False, seconds=time.monotonic() - start)
    write(output / 'COMPLETE_COMMON_BUILDER_CAL.json', value)
    return value


def confirm_ten(spec, adapter, reader, predictions, calibration, original, output):
    require(calibration['schema'] == 'Physics_common_builder_CAL_v1' and calibration['terminal'] is True
            and calibration['scalar_endpoints'] == 10 and calibration['all_finite'] is True
            and calibration['CONFIRM_truth_access'] is False
            and len(calibration['records']) == 10 and {r['block'] for r in calibration['records']} == set(SEEDS)
            and all(r['arm'] == ARM and r['status'] == 'finite_fixed_endpoint' and r['updates'] == 500
                    and r['endpoint'] == 'fixed_final_update' and math.isfinite(r['log_T'])
                    and math.isfinite(r['temperature']) and r['temperature'] > 0
                    for r in calibration['records']), 'All ten finite fixed500 CAL endpoints before CONFIRM')
    check_predictions(spec, predictions)
    require(read(receipt(output.parent / 'cal/COMPLETE_COMMON_BUILDER_CAL.json')) == calibration
            and calibration['prediction_receipt'] == receipt(output.parent / 'export/COMPLETE_COMMON_BUILDER_PREDICTIONS.json')
            and calibration['CAL_truth_receipts'] == {str(s): spec['roles'][str(s)]['cal_truth'] for s in SEEDS},
            'Actual saved CAL closure and prediction/target receipts joined before release')
    banks = {r['block']: r for r in predictions['records']}
    require(all(r['selected_CAL_archive'] == banks[r['block']]['selected_CAL_archive'] for r in calibration['records']),
            'Actual ten CAL endpoint/archive join before release')
    require(original['original_banks'] == 60 and original['original_selected_states'] == 90, 'Real full joined original-reference closure')
    np, t = reader.providers({'provider_modules': spec['provider_modules']})
    start = time.monotonic(); cfg = dict(roles=spec['roles'])
    # One marker before any new CONFIRM target decoding; a failed attempt cannot
    # silently re-release truth. Original scores have already closed separately.
    write(output / 'CONFIRM_RELEASE.json', dict(one_complete_release=True, new_control_banks=10,
              original_reference_banks=60, original_reference_selected_states=90,
              calibration=receipt(output.parent / 'cal/COMPLETE_COMMON_BUILDER_CAL.json'),
              same_graph_attribution_not_unused_confirmation=True))
    roles = {s: reader.truth(np, cfg, s, 'CONFIRM') for s in SEEDS}
    index = {r['block']: r for r in predictions['records']}
    endpoint = {r['block']: r for r in calibration['records']}
    rows, paired_rows, reference_deltas = [], [], []
    for seed in SEEDS:
        resource_guard(start, CAPS['confirm'])
        ids, target = roles[seed]; y = t.from_numpy(target)
        logp, member, pool = load_control_bank(np, index[seed], 'CONFIRM', ids)
        logp, member, pool = (t.from_numpy(a) for a in (logp, member, pool))
        row = dict(block=seed, arm=ARM, calibration_endpoint=endpoint[seed], original_scores_recalculated=False)
        for kind in ('raw', 'calibrated'):
            log_T = t.zeros((), dtype=t.float64) if kind == 'raw' else t.tensor(endpoint[seed]['log_T'], dtype=t.float64)
            value = reader.score(t, logp, y, member, pool, log_T, kind == 'calibrated')
            row[kind] = {k: value[k] for k in ('quality', 'counts', 'classes', 'float64_member_argmax_disagreements_with_native_archive')}
        originals = {ref: original['rows'][seed, ref] for ref in ('E4', 'BASE', 'equipped_I4')}
        paired_rows.append(dict(seed=seed, common_raw_accuracy_pct=row['raw']['quality']['accuracy_pct'],
                               E4_raw_accuracy_pct=originals['E4']['raw']['quality']['accuracy_pct'],
                               common_calibrated_NLL=row['calibrated']['quality']['NLL'],
                               BASE_calibrated_NLL=originals['BASE']['calibrated']['quality']['NLL'],
                               I4_calibrated_NLL=originals['equipped_I4']['calibrated']['quality']['NLL']))
        for ref, prior in originals.items():
            reference_deltas.append(dict(block=seed, reference=ref,
                raw={k: row['raw']['quality'][k] - prior['raw']['quality'][k] for k in reader.METRICS},
                calibrated={k: row['calibrated']['quality'][k] - prior['calibrated']['quality'][k] for k in reader.METRICS},
                class_deltas=[dict(class_id=cl, **{kind: {k: row[kind]['classes'][cl]['quality'][k] - prior[kind]['classes'][cl]['quality'][k]
                              for k in reader.METRICS} for kind in ('raw', 'calibrated')}) for cl in range(5)],
                original_metric_record_reused_verbatim=True))
        rows.append(row)
    require(len(rows) == 10 and len(paired_rows) == 10, 'Complete joined paired roster before inherited criteria')
    effect = adapter.paired_effect(paired_rows)  # Exact predecessor margin/CI/NLL rules.
    resource_guard(start, CAPS['confirm']); check_predictions(spec, predictions)
    for role in spec['roles'].values():
        for item in role.values(): checked(item)
    require(read(receipt(output.parent / 'cal/COMPLETE_COMMON_BUILDER_CAL.json')) == calibration,
            'Fixed CAL closure remains unchanged through CONFIRM')
    result = dict(schema='Physics_common_builder_attribution_v1', terminal=True, complete=True,
                  new_control_banks=10, new_selected_states=10, new_CAL_endpoints=10, original_reference_closure=original['receipts'],
                  original_banks=60, original_selected_states=90, original_CAL_endpoints=60,
                  rows=rows, paired_scalar_rows=paired_rows, paired_effect=effect, reference_metric_deltas=reference_deltas,
                  one_complete_CONFIRM_truth_release=True, CONFIRM_truth_access=True,
                  original_scores_recalculated=False, no_new_GNN_or_head_fit=True,
                  same_graph_attribution_not_unused_confirmation=True, no_new_methodology_or_automatic_followup=True,
                  seconds=time.monotonic() - start)
    write(output / 'COMPLETE_COMMON_BUILDER_ATTRIBUTION.json', result)
    with (output / 'PAIRED_SCALARS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0])); writer.writeheader(); writer.writerows(paired_rows)
    return result


def run_after_fit(spec, root_guard):
    """Complete finite consumer of real ten fits; never invokes acquisition."""
    require(callable(root_guard), 'New root admission callback required')
    proof = admit(spec, root_guard, 'export', dict(fit_closure=spec['fit_closure'], original=spec['original']))
    output = Path(spec['output']).resolve()
    require(not output.exists(), 'Fresh whole after-fit attempt only; no retry/release reuse')
    require(set(spec['roles']) == {str(s) for s in SEEDS}
            and all(set(r) == {'cal_ids', 'confirm_ids', 'cal_truth', 'confirm_truth'} for r in spec['roles'].values()), 'Exact ten paired roles')
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Dedicated finite root-owned consumer process')
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame): raise TimeoutError('Finite after-fit stage cap; no retry or release promotion')
    signal.signal(signal.SIGALRM, timeout)
    output.mkdir(); started = time.monotonic(); stage = 'export'; admissions = dict(export=proof)
    try:
        signal.alarm(CAPS['export'])
        common, engine, adapter, contracts, reader = dependencies(spec)
        original = original_reference_join(spec, contracts); control_fit(spec)
        export = output / 'export'; export.mkdir()
        predictions = export_ten(spec, engine, common, adapter, reader, export)
        stage = 'calibrate'
        admissions[stage] = admit(spec, root_guard, stage, dict(predictions=receipt(export / 'COMPLETE_COMMON_BUILDER_PREDICTIONS.json'), original=original['receipts']))
        signal.alarm(CAPS[stage]); original_reference_join(spec, contracts)
        cal = output / 'cal'; cal.mkdir()
        calibration = calibrate_ten(spec, reader, predictions, cal)
        require(calibration['all_finite'] is True, 'Failed/nonfinite CAL blocks stop before CONFIRM; no survivor promotion')
        stage = 'confirm'
        admissions[stage] = admit(spec, root_guard, stage, dict(predictions=receipt(export / 'COMPLETE_COMMON_BUILDER_PREDICTIONS.json'),
            calibration=receipt(cal / 'COMPLETE_COMMON_BUILDER_CAL.json'), original=original['receipts']))
        signal.alarm(CAPS[stage]); original = original_reference_join(spec, contracts)
        confirm = output / 'confirm'; confirm.mkdir()
        result = confirm_ten(spec, adapter, reader, predictions, calibration, original, confirm)
        for item in spec['original'].values(): checked(item)
        checked(spec['fit_closure'])
        checked(spec['original_prediction_contract'])
        require(all(p['source_sha256'] == sha(__file__) and p['interface_sha256'] == sha(Path(__file__).with_name('INTERFACE.json'))
                    for p in admissions.values()), 'All new stage bindings remain fixed')
        complete = dict(complete=True, schema='Physics_common_builder_after_fit_complete_v1',
                        result=receipt(confirm / 'COMPLETE_COMMON_BUILDER_ATTRIBUTION.json'),
                        predictions=receipt(export / 'COMPLETE_COMMON_BUILDER_PREDICTIONS.json'),
                        calibration=receipt(cal / 'COMPLETE_COMMON_BUILDER_CAL.json'), original=spec['original'],
                        new_control_banks=10, new_selected_states=10, new_CAL_endpoints=10,
                        admissions=admissions, seconds=time.monotonic() - started,
                        peak_RSS_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        acquisition_reused_without_new_fit=True, original_scores_recalculated=False,
                        same_graph_attribution_not_unused_confirmation=True, automatic_followup=False)
        write(output / 'COMPLETE_COMMON_BUILDER_AFTER_FIT.json', complete)
        return complete
    except BaseException as error:
        write(output / 'AFTER_FIT_FAILURE.json', dict(complete=False, status='FAILED_RETAINED', stage=stage,
              type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
              retry_or_survivor_means=False, original_scores_recalculated=False,
              no_automatic_CONFIRM_after_failure=True, seconds=time.monotonic() - started))
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    raise SystemExit('Inactive source: root supplies actual completed fits/spec/new admission callback and finite external owner.')
