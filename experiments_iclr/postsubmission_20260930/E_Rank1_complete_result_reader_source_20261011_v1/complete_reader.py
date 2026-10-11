"""Complete prospective Rank-1 reader; stdlib custody before numerical imports.

Original boundary raw-logit fusion has its own post-pool temperature contract.
No model/state deserialization, graph forwards, retry, refit or schema relabeling.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time
import traceback

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS = (7301, 7403, 7507)
SPECS = (('E_point4', 'point', 4, 1), ('E_fixed_noise4', 'fixed_noise', 4, 1),
         ('E_Rank1_q4', 'learned_q', 4, 1), ('E_point_M1', 'point', 1, 1),
         ('E_point_I4', 'point', 4, 4), ('E_Rank1_q_M1', 'learned_q', 1, 1),
         ('E_Rank1_q_I4', 'learned_q', 4, 4))
ORIGINAL = 'original_boundary_shared4'
NATIVE_OPERATOR = 'member_temperature_before_probability_mean'
ORIGINAL_OPERATOR = 'temperature_of_archived_native_mean_logits'
FIXED = dict(depth=2, hidden=128, dropout=.2, learning_rate=.001, max_updates=1000,
    patience_updates=300, rank=16, member_seed_stride=1000003, factor_seed_offset=2000003,
    dropout_seed_offset=3000007, block_seed_offset=4000037, stalk_dim=4,
    baseline_policy='fresh_required', stage='F', K=None, k_selection_receipt=None)


def require(value, message):
    if not value: raise ValueError(message)
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()
def current(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
def receipt(row, under=R):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'Exact portable receipt')
    p = Path(row['path']).resolve()
    require(p.is_relative_to(under) and p.is_file() and current(p) == row, 'Changed/out-of-scope file: ' + str(p))
    return p
def write(path, value, replace=False):
    p = Path(path); require(replace or not p.exists(), 'Fresh output, retain previous attempt')
    temp = p.with_suffix(p.suffix + '.tmp'); require(not temp.exists(), 'Retain partial output')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temp.replace(p)
def finite(value): return type(value) in (int, float) and math.isfinite(value)
def route():
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R, 'Literal allocation/repository before project reads')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines() == [UUID], 'Sole authorized GPU')
def package(rows, pinned):
    require(set(rows) == set(pinned), 'Exact immutable source package')
    parents = set()
    for name, row in rows.items():
        p = receipt(row); parents.add(p.parent)
        require(p.name == name and (row['bytes'], row['sha256']) == (pinned[name]['bytes'], pinned[name]['sha256']), 'Pinned source changed: ' + name)
    require(len(parents) == 1, 'One source package directory')
def owner(panel, action, runtime):
    spec_path = receipt(panel['spec']); complete_path = receipt(panel['complete']); end = read(receipt(panel['owner_end']))
    expected = dict(status='COMPLETE_EXIT_0', exit_code=0, direct_child_wait=True,
        child_pid_absent=True, owned_cuda_pid_absent=True, action=action, TEST_access=False,
        current_Physics_Penn_access=False, spec_receipt=panel['spec'], complete_result_receipt=panel['complete'],
        runtime_fingerprint=runtime)
    require(all(end.get(k) == v for k, v in expected.items()), 'Actual entire external-owner exit/reap/spec/runtime closure required')
    root = Path(panel['root']).resolve()
    require(root.is_relative_to(R) and complete_path.parent == root and read(spec_path)['output'] == str(root), 'Actual action output/spec mapping')
    return root, read(spec_path), read(complete_path), end
def proof(value, action, source_rows, limits):
    expected = dict(admitted=True, action=action, source_sha256={r['path']: r['sha256'] for r in source_rows.values()},
        limits=limits, finite_external_owner_bound=True, TEST_access=False, current_Physics_Penn_access=False)
    require(all(value.get(k) == v for k, v in expected.items()) and type(value.get('external_owner_wall_seconds')) is int
            and value['external_owner_wall_seconds'] > 0, 'Exact root source/action/finite admission')
def selector(folder, fit, law, routes):
    rows = [json.loads(line) for line in (folder / 'trace.jsonl').read_text().splitlines()]
    require(len(rows) == fit['completed_updates'], 'Complete native acquisition trace')
    best, stale, selected = -1., 0, 0
    for step, row in enumerate(rows, 1):
        metric = row['valid'] if law == 'point' else row['mean_factor_VALID']
        require(row['step'] == step and finite(metric['accuracy']), 'Ordered finite native selector trace')
        improved = metric['accuracy'] > best
        if improved: best, stale, selected = metric['accuracy'], 0, step
        else: stale += 1
        require(row['strict_improvement'] == improved and row['stale_updates'] == stale
                and (stale < 300 or step == len(rows)), 'Original earliest strict mean-factor selector/patience')
        if law != 'point':
            u = row['update']; require(u['factor_draws'] == u['TRAIN_routes'] == routes
                and u['backward_calls'] == u['optimizer_steps'] == 1
                and all(finite(u[k]) for k in ('objective', 'mean_sampled_own_CE', 'mean_KL_per580'))
                and len(u['conditional_KL']) == routes and all(finite(v) for v in u['conditional_KL']), 'Actual sampled own CE/KL trace')
    require(selected == fit['selected_step'] and fit['selection'] == fit['restored'], 'Selected native endpoint/strict restore metadata')
    return current(folder / 'trace.jsonl')


def preflight(spec):
    here = Path(__file__).resolve().parent; binding = read(here / 'SOURCE_BINDINGS.json')
    require(spec.get('schema_version') == 1 and spec.get('plan_id') in ('existing21', 'amended24'), 'One explicit prospective plan')
    plan_path = here / ('PLAN_EXISTING21.json' if spec['plan_id'] == 'existing21' else 'PLAN_AMENDED24.json')
    plan = read(plan_path); admission = read(receipt(spec['reader_admission']))
    sources = {str(here / n): sha(here / n) for n in ('complete_reader.py', 'reader_math.py', 'SOURCE_BINDINGS.json', 'INTERFACE.json', plan_path.name)}
    limits = spec['limits']; require(set(limits) == {'wall_seconds', 'rss_bytes'} and all(type(v) is int and v > 0 for v in limits.values()), 'Finite actual readout caps')
    expected = dict(admitted=True, action='E_Rank1_complete_result_reader', source_sha256=sources,
        plan_id=spec['plan_id'], plan_sha256=sha(plan_path), finite_external_owner_bound=True,
        full_closed_inputs_verified=True, original_runtime_checked=True, limits=limits,
        TEST_access=False, current_Physics_Penn_access=False)
    require(all(admission.get(k) == v for k, v in expected.items()), 'New actual reader admission; no source-only PASS')
    runtime = spec['runtime_fingerprint']; fingerprint = read(receipt(runtime))
    require(fingerprint['hostname'] == socket.gethostname() and fingerprint['gpu_uuid'] == UUID, 'Same physical/runtime custody')
    for row in fingerprint['python_sources'].values(): receipt(row)
    roster = read(receipt(spec['roster_admission']))
    require(roster.get('status') == 'ROOT_FIXED_ROSTER_ADOPTED' and roster.get('plan_id') == spec['plan_id']
        and roster.get('plan_sha256') == sha(plan_path) and roster.get('calibration_contracts') == plan['calibration_contracts']
        and roster.get('counts') == plan['counts'] and roster.get('seeds') == list(SEEDS), 'Actual root adoption of exact roster/operators required')
    panel = spec['rank1']; package(panel['package'], binding['rank1_package'])
    root, action_spec, full, end = owner(panel, 'E_Rank1_BNN_native_acquisition', runtime)
    cfg = action_spec['native_cfg']; dependencies = cfg['source_receipts']
    require(set(dependencies) == set(binding['native_dependencies']) and cfg['seeds'] == list(SEEDS)
        and cfg['native_repo'] == str(R) and cfg['device'] == 'cuda:0'
        and all(cfg.get(k) == v for k, v in FIXED.items()), 'Exact native source/recipe')
    for key, row in dependencies.items():
        receipt(row); require((row['bytes'], row['sha256']) == (binding['native_dependencies'][key]['bytes'], binding['native_dependencies'][key]['sha256']), 'Native dependency bytes changed')
    require(set(cfg['safe_roles']) == {'train', 'valid'}, 'Only original TRAIN/VALID roles')
    for key, row in cfg['safe_roles'].items():
        receipt(row); require((row['bytes'], row['sha256']) == (binding['safe_roles'][key]['bytes'], binding['safe_roles'][key]['sha256'])
            and row['path'] == cfg[key + '_npz'], 'Exact original WikiCS role bytes/path')
    require(read(root / 'native_constructor/CONFIG.json') == cfg, 'Saved native constructor config')
    proposal = Path(action_spec['proposal_directory']).resolve()
    for name, row in binding['proposal_files'].items():
        require(proposal.is_relative_to(R) and (proposal / name).stat().st_size == row['bytes'] and sha(proposal / name) == row['sha256'], 'Frozen proposal changed')
    proof(full['admission'], 'E_Rank1_BNN_native_acquisition', panel['package'], cfg['family_limits'])
    require(full['admission'].get('new_full_input_readiness_complete') is True, 'Actual root-verified full-input readiness')
    expected_full = dict(complete=True, action='E_Rank1_BNN_native_acquisition', banks=21, optimizer_fits=39,
        selected_states=39, primary_archives=21, selected_primary_routes=66, secondary_sampled_routes=144,
        required_future_calibration_attempts=105, requested_future_scalar_updates=52500,
        calibration_performed=False, TEST_access=False, current_Physics_Penn_access=False)
    require(all(full.get(k) == v for k, v in expected_full.items()) and end['source_sha256'] == full['admission']['source_sha256'], 'Entire native21/39/144 closure')
    records = full['records']; require(len(records) == 21 and {(x['seed'], x['procedure']) for x in records} == {(s, a) for s in SEEDS for a, _, _, _ in SPECS}, 'All native banks')
    primary, secondary, states, traces, costs = [], [], [], [], []
    totals = dict(updates=0, TRAIN_routes=0, selector_routes=0)
    lookup = {a: (law, members, bundles) for a, law, members, bundles in SPECS}
    for bank in records:
        arm, seed = bank['procedure'], bank['seed']; law, members, bundles = lookup[arm]
        folder = root / (arm + '_seed' + str(seed))
        require(read(folder / 'RESULT.json') == bank and bank['status'] == 'SELECTED_COMPLETE'
            and bank['factor_law'] == law and bank['members'] == members and bank['optimizer_fits'] == bundles
            and bank['primary_selected_native_routes'] == members and bank['primary_serving'] == 'mean_factor_probability_mean'
            and len(bank['fits']) == bundles, 'Exact complete native bank/ownership/schema')
        p = receipt(bank['primary_archive'], root); require(p == folder / 'selected_VALID.npz', 'Native primary archive path')
        primary.append(dict(bank['primary_archive'], arm=arm, seed=seed, members=members, operator=NATIVE_OPERATOR))
        for member, fit in enumerate(bank['fits']):
            routes = 1 if bundles == 4 else members; f = folder / ('member' + str(member))
            require(read(f / 'RESULT.json') == fit and fit['member'] == member and fit['factorized'] is True
                and fit['native_bodies_fitted'] == fit['optimizer_acquisition_bundles'] == 1
                and fit['architecture'] == ('shared' if routes == 4 else 'single')
                and type(fit['completed_updates']) is int and 1 <= fit['selected_step'] <= fit['completed_updates'] <= 1000, 'All distinct complete native optimizer bodies')
            state = receipt(fit['selected_state_receipt'], root); require(state == f / 'selected.pt', 'Exact full selected state path')
            traces.append(selector(f, fit, law, routes)); states.append(dict(fit['selected_state_receipt'], arm=arm, seed=seed, member=member))
            op = fit['operation_counts']; steps = fit['completed_updates']
            if law == 'point':
                require(fit['learning_objective'] == 'own_only' and fit['within_route_supcon'] is False and fit['value_operation'] == 'endpoint'
                    and len(fit['body_fit_records']) == 1 and op['backwards'] == op['adam_steps'] == steps
                    and op['train_fullgraph_native_trajectories'] == op['valid_selection_fullgraph_native_trajectories'] == routes * steps
                    and op['selected_valid_fullgraph_native_trajectories'] == routes
                    and fit['costs']['parameters'] == 221642 + 3958 * routes, 'Original point own-CE/capacity/work')
                body = fit['body_fit_records'][0]
                require(body['native_initialization_seed'] == seed + 1000003 * (0 if routes == 4 else member)
                    and body['body_index'] == member and body['selected_step'] == fit['selected_step'], 'Genuine native initialization/selection')
            else:
                q = fit['posterior']; require(fit['factor_law'] == q['law'] == law and q['trainable_scales'] == (law == 'learned_q')
                    and q['scale_coordinates'] == 3958 * routes and 0 < q['std_min'] <= q['std_mean'] <= q['std_max']
                    and all(finite(q[k]) for k in ('std_min', 'std_mean', 'std_max'))
                    and fit['mean_selector'] is True and fit['sampled_selector'] is False
                    and fit['factor_means_mutated_to_install_draw'] is False
                    and len(fit['factor_rng_selected']) == routes
                    and op['updates'] == op['backward_calls'] == op['optimizer_steps'] == steps
                    and op['TRAIN_routes'] == op['mean_selector_VALID_routes'] == op['factor_draws'] == routes * steps
                    and op['selected_primary_routes'] == routes
                    and fit['costs']['parameters'] == 221642 + 2 * 3958 * routes
                    and fit['costs']['trainable_parameters'] == 221642 + (2 if law == 'learned_q' else 1) * 3958 * routes,
                    'Gaussian law, actual selected full scales, ownership/work/capacity')
            totals['updates'] += steps; totals['TRAIN_routes'] += routes * steps; totals['selector_routes'] += routes * steps
            costs.append(dict(arm=arm, seed=seed, member=member, costs=fit['costs'], operation_counts=op))
        if law != 'point':
            sample = bank['secondary']; receipt(sample['archive'], root)
            require(sample['archive']['path'] == str(folder / 'selected_secondary_sampled_VALID.npz')
                and sample['primary_component_means_archive'] == bank['primary_archive']
                and sample['native_selected_routes'] == 12 and sample['RNG_seed'] == seed + 6000011
                and sample['component_sequence'] == ([0] * 12 if members == 1 else list(range(4)) * 3)
                and sample['raw_only'] is True and sample['blocks'] == 3 and sample['best_block_selection'] is False
                and sample['primary'] is False and sample['calibration'] is False and sample['model_updates'] == 0,
                'All fixed144 sampled diagnostics, no sampled selection/calibration')
            secondary.append(dict(sample['archive'], arm=arm, seed=seed, members=members, component_sequence=sample['component_sequence']))
        else: require('secondary' not in bank, 'Point banks acquire no sampled panel')
    require(len(states) == 39 and len({r['path'] for r in states}) == 39 and len(secondary) == 12
        and totals == dict(updates=full['actual_optimizer_updates'], TRAIN_routes=full['actual_TRAIN_routes'], selector_routes=full['actual_mean_selector_VALID_routes'])
        and totals['updates'] <= 39000 and totals['TRAIN_routes'] <= 66000 and totals['selector_routes'] <= 66000, 'Complete actual native work closure')
    original_full = None
    if spec['plan_id'] == 'amended24':
        panel_o = spec['original']; package(panel_o['package'], binding['original_package'])
        root_o, spec_o, original_full, end_o = owner(panel_o, 'WikiCS_original_boundary_GNNM_comparator', runtime)
        require(spec_o['roster_amendment'] == spec['roster_admission'], 'Comparator must use the actual amended24 adoption')
        receipt(spec_o['roster_amendment']); require(spec_o['roles'] == cfg['safe_roles'], 'Same paired original role files')
        require(set(spec_o['source_receipts']) == set(binding['original_runtime']), 'Original native runtime source set')
        for key, row in spec_o['source_receipts'].items():
            receipt(row); require((row['bytes'], row['sha256']) == (binding['original_runtime'][key]['bytes'], binding['original_runtime'][key]['sha256']), 'Pinned original runtime unchanged')
        proof(original_full['admission'], 'WikiCS_original_boundary_GNNM_comparator', panel_o['package'], spec_o['limits'])
        require(original_full['admission']['future_roster_amendment_verified'] is True and original_full['admission']['native_runtime_and_full_inputs_verified'] is True
            and original_full['complete'] is True and original_full['arm'] == ORIGINAL
            and original_full['banks'] == original_full['optimizer_fits'] == original_full['selected_states'] == 3
            and original_full['selected_native_routes'] == 12 and original_full['TEST_access'] is False
            and original_full['current_Physics_Penn_access'] is False
            and end_o['source_sha256'] == original_full['admission']['source_sha256'], 'Separate original3/3 terminal closure')
        original_rows = original_full['records']; require(len(original_rows) == 3 and {r['seed'] for r in original_rows} == set(SEEDS), 'All original paired seeds')
        for bank in original_rows:
            seed = bank['seed']; folder = root_o / (ORIGINAL + '_seed' + str(seed))
            require(read(folder / 'RESULT.json') == bank and bank['status'] == 'selected_complete'
                and bank['members'] == 4 and bank['optimizer_fits'] == bank['selected_states'] == 1
                and bank['selector_and_serving'] == 'raw_logit_mean_argmax' and bank['dropout'] == 'original_global_sequential'
                and 10 <= bank['selected_step'] <= bank['completed_updates'] <= 5000 and bank['selected_step'] % 10 == 0
                and bank['TRAIN_native_routes'] == bank['VALID_selector_native_routes'] == bank['backward_calls'] == 4 * bank['completed_updates']
                and bank['optimizer_steps'] == bank['completed_updates'] and bank['selected_native_routes'] == 4, 'Original cadence/global dropout/work')
            rows = [json.loads(line) for line in (folder / 'trace.jsonl').read_text().splitlines()]
            require(len(rows) == bank['completed_updates'], 'Complete original cadence trace')
            best, stale, selected, saves = -1., 0, 0, 0
            for step, row in enumerate(rows, 1):
                correct = row['raw_logit_VALID_correct']; require(row['update'] == step and type(correct) is int and 0 <= correct <= 5274, 'Original integer selector trace')
                improving = correct / 5274 > best; stale = 0 if improving else stale + 1; saved = improving and step % 10 == 0
                if saved: best, selected, saves = correct / 5274, step, saves + 1
                require(row['strict_above_saved_best'] == improving and row['divisible_by_ten'] == (step % 10 == 0)
                    and row['checkpoint_written'] == saved and row['selected_step'] == selected and row['stale'] == stale
                    and (stale < 300 or step == len(rows)), 'Exact original improving-off-cadence reset, eligible saves and ties')
            require(selected == bank['selected_step'] and saves == bank['checkpoint_writes'], 'Original saved selected cadence endpoint')
            state = receipt(bank['selected_state_receipt'], root_o); archive = receipt(bank['selected_VALID_archive'], root_o)
            require(state == folder / 'selected.pt' and archive == folder / 'selected_VALID.npz', 'Separate original state/archive')
            states.append(dict(bank['selected_state_receipt'], arm=ORIGINAL, seed=seed, member=0)); traces.append(current(folder / 'trace.jsonl'))
            primary.append(dict(bank['selected_VALID_archive'], arm=ORIGINAL, seed=seed, members=4, operator=ORIGINAL_OPERATOR))
            costs.append(dict(arm=ORIGINAL, seed=seed, member=0, actual_record=bank))
    else: require('original' not in spec, 'Existing21 plan must not silently include original3')
    require(len(primary) == plan['counts']['banks'] and len(states) == plan['counts']['selected_states'], 'Exact chosen full primary roster')
    return dict(plan=plan, cfg=cfg, primary=primary, secondary=secondary, states=states, traces=traces,
        costs=costs, native_complete=spec['rank1']['complete'], original_complete=None if original_full is None else spec['original']['complete'],
        runtime=runtime, native_actual_counts=totals, limits=limits)


def log_probability(np, core, logits):
    value = logits.astype(np.float64)
    return value - core.logsumexp(np, value, -1)[..., None]
def normalized(np, probability):
    return np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all() and np.allclose(probability.sum(-1), 1, atol=1e-5, rtol=0)
def arrays(np, context):
    with np.load(receipt(context['cfg']['safe_roles']['valid']), allow_pickle=False) as f:
        require(set(f.files) == {'ids', 'y'}, 'Exact safe VALID schema'); ids, labels = f['ids'].copy(), f['y'].copy()
    require(ids.dtype == labels.dtype == np.int64 and ids.shape == labels.shape == (5274,)
        and len(np.unique(ids)) == 5274 and set(labels.tolist()) == set(range(10)), 'All ordered development nodes/classes')
    result = {}
    for row in context['primary']:
        receipt({k: row[k] for k in ('path', 'bytes', 'sha256')})
        with np.load(row['path'], allow_pickle=False) as f: a = {k: f[k].copy() for k in f.files}
        native = row['operator'] == NATIVE_OPERATOR
        fields = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'} if native else {
            'ids', 'y', 'raw_logits', 'raw_logit_mean', 'raw_logit_mean_probabilities', 'raw_logit_decision', 'member_decision'}
        require(set(a) == fields and np.array_equal(a['ids'], ids) and np.array_equal(a['y'], labels), 'Operator-specific archive keys and ordered role identity')
        require(a['raw_logits'].dtype == np.float32 and a['raw_logits'].shape == (row['members'], 5274, 10) and np.isfinite(a['raw_logits']).all(), 'Full original raw member logits')
        if native:
            require(a['probability_mean'].dtype == np.float32 and a['probability_mean'].shape == (5274, 10) and normalized(np, a['probability_mean'])
                and a['member_errors'].dtype == a['pooled_errors'].dtype == np.bool_
                and a['member_errors'].shape == (row['members'], 5274) and a['pooled_errors'].shape == (5274,)
                and np.array_equal(a['member_errors'], a['raw_logits'].argmax(-1) != labels)
                and np.array_equal(a['pooled_errors'], a['probability_mean'].argmax(-1) != labels), 'Native probability-pool integer authority')
        else:
            require(a['raw_logit_mean'].dtype == a['raw_logit_mean_probabilities'].dtype == np.float32
                and a['raw_logit_mean'].shape == a['raw_logit_mean_probabilities'].shape == (5274, 10)
                and np.isfinite(a['raw_logit_mean']).all() and normalized(np, a['raw_logit_mean_probabilities'])
                and a['raw_logit_decision'].dtype == a['member_decision'].dtype == np.int64
                and a['raw_logit_decision'].shape == (5274,) and a['member_decision'].shape == (4, 5274)
                and np.array_equal(a['raw_logit_decision'], a['raw_logit_mean'].argmax(-1))
                and np.array_equal(a['member_decision'], a['raw_logits'].argmax(-1)), 'Original raw-logit mean/decision authority; never probability-mean relabeling')
        result[row['seed'], row['arm']] = dict(a, operator=row['operator'])
    return result, ids, labels


def calibrate(np, torch, core, values, ids, labels, context, output, guard):
    began = time.perf_counter(); permutation = torch.randperm(len(labels), generator=torch.Generator(device='cpu').manual_seed(11709))
    folds = torch.empty(len(labels), dtype=torch.long); folds[permutation] = torch.arange(len(labels)) % 5
    y = torch.from_numpy(labels); endpoints, records = {}, []
    progress = [dict(arm=row['arm'], seed=row['seed'], operator=row['operator'], fold=fold, status='unstarted_retained') for row in context['primary'] for fold in range(5)]
    write(output / 'CALIBRATION_PROGRESS.json', progress)
    for row in context['primary']:
        key = row['seed'], row['arm']; a = values[key]; native = row['operator'] == NATIVE_OPERATOR
        log_p = torch.from_numpy(a['raw_logits']).to(torch.float64).log_softmax(-1)
        # Original primary target is the archived native float32 mean, not a
        # mean-probability bank or a recomputed higher-precision mean.
        fit_base = log_p if native else torch.from_numpy(a['raw_logit_mean']).to(torch.float64).log_softmax(-1).unsqueeze(0)
        log_members = torch.full(log_p.shape, float('nan'), dtype=torch.float64)
        log_pool = torch.full((5274, 10), float('nan'), dtype=torch.float64); fits = []
        for fold in range(5):
            guard(); before, completed, phase = time.perf_counter(), 0, 'initialization'; trace = []
            record = next(r for r in progress if (r['seed'], r['arm'], r['fold']) == (key[0], key[1], fold))
            record['status'] = 'running'; write(output / 'CALIBRATION_PROGRESS.json', progress, replace=True)
            fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
            try:
                log_T = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device='cpu'))
                optimizer = torch.optim.Adam([log_T], lr=.01, betas=(.9, .999), eps=1e-8, weight_decay=0)
                fit_log, fit_y = fit_base[:, fit_ids], y[fit_ids]
                for update in range(1, 501):
                    guard(); phase = 'objective'; optimizer.zero_grad(set_to_none=True)
                    scaled = (fit_log / log_T.exp().reshape(1, 1, 1)).log_softmax(-1)
                    pooled = torch.logsumexp(scaled, dim=0) - math.log(len(fit_base))
                    loss = torch.nn.functional.nll_loss(pooled, fit_y)
                    require(torch.isfinite(loss).item(), 'Nonfinite fixed temperature objective')
                    phase = 'backward'; loss.backward()
                    require(log_T.grad is not None and torch.isfinite(log_T.grad).all().item(), 'Missing/nonfinite scalar gradient')
                    phase = 'optimizer_step'; optimizer.step(); completed = update
                    if update in (1, 100, 200, 300, 400, 500): trace.append(dict(update=update, fit_objective_before_update=loss.item()))
                phase = 'endpoint'
                with torch.no_grad():
                    temperature = log_T.exp(); require(torch.isfinite(temperature).all().item() and (temperature > 0).all().item(), 'Finite positive fixed endpoint')
                    held_members = (log_p[:, held_ids] / temperature.reshape(1, 1, 1)).log_softmax(-1)
                    held_base = (fit_base[:, held_ids] / temperature.reshape(1, 1, 1)).log_softmax(-1)
                    held_pool = torch.logsumexp(held_base, dim=0) - math.log(len(fit_base))
                    require(torch.isfinite(held_members).all().item() and torch.isfinite(held_pool).all().item(), 'Finite complete calibration endpoint')
                    log_members[:, held_ids], log_pool[held_ids] = held_members, held_pool
                fit = dict(status='finite_fixed_endpoint', fold=fold, updates=500, parameters=1,
                    fit_nodes=len(fit_ids), held_nodes=len(held_ids), log_T=log_T.detach().tolist(), temperature=temperature.item(), trace=trace,
                    selected_endpoint='fixed_final_update_no_heldout_selector')
            except (FloatingPointError, RuntimeError, ValueError, MemoryError) as error:
                fit = dict(status='failed_retained', fold=fold, error=str(error), failure_phase=phase,
                    requested_updates=500, actual_updates_before_failure=None if phase == 'optimizer_step' else completed,
                    known_completed_updates_lower_bound=completed, trace=trace)
            fit['seconds'] = time.perf_counter() - before; fit['operator'] = row['operator']
            fit['fit_channels'] = len(fit_base); fit['reported_members'] = len(log_p)
            record.update(fit); write(output / 'CALIBRATION_PROGRESS.json', progress, replace=True); fits.append(fit)
        valid = all(f['status'] == 'finite_fixed_endpoint' for f in fits) and torch.isfinite(log_members).all().item() and torch.isfinite(log_pool).all().item()
        stream = io.BytesIO(); np.savez_compressed(stream, ids=ids, folds=folds.numpy(), member_log_probability=log_members.numpy(),
            pool_log_probability=log_pool.numpy(), operator=np.asarray(row['operator']))
        path = output / ('CALIBRATED_' + row['arm'] + '_seed' + str(row['seed']) + '_OOF.npz'); path.write_bytes(stream.getvalue())
        records.append(dict(arm=row['arm'], seed=row['seed'], operator=row['operator'], status='finite_complete_five_fold' if valid else 'failed_retained',
            fits=fits, attempted_scalar_fits=5, OOF_archive=current(path),
            member_confidence_scope='same served scalar applied to original members for diagnostics; not four separately fitted temperatures' if not native else 'original member-before-pool calibration'))
        endpoints[key] = (log_members.numpy(), log_pool.numpy()) if valid else None
    require(len(progress) == context['plan']['counts']['calibration_attempts'] and all(r['status'] in ('finite_fixed_endpoint', 'failed_retained') for r in progress), 'Every planned scalar attempt terminal before contrasts')
    success = sum(r['status'] == 'finite_fixed_endpoint' for r in progress)
    costs = dict(attempted_scalar_fits=len(progress), successful_scalar_fits=success, failed_scalar_fits=len(progress)-success,
        operator_attempts={op: sum(r['operator'] == op for r in progress) for op in context['plan']['calibration_contracts'].values()},
        requested_updates_per_fit=500, requested_scalar_updates=500*len(progress), known_successful_scalar_updates=500*success,
        fit_seconds_sum=sum(r['seconds'] for r in progress), wall_seconds=time.perf_counter()-began,
        no_retry_or_grid_or_final_refit=True, native_models_or_forwards_added=0)
    return endpoints, records, costs, dict(seed=11709, counts=[int((folds == k).sum()) for k in range(5)],
        assignment_sha256=hashlib.sha256(folds.numpy().tobytes()).hexdigest(), ordered_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())


def raw_scores(np, core, values, labels):
    scores = {}
    for key, a in values.items():
        log_members = log_probability(np, core, a['raw_logits']); native = a['operator'] == NATIVE_OPERATOR
        probability = a['probability_mean'] if native else a['raw_logit_mean_probabilities']
        member_errors = a['member_errors'] if native else a['member_decision'] != labels
        pooled_errors = a['pooled_errors'] if native else a['raw_logit_decision'] != labels
        explicit_pool = None if native else log_probability(np, core, a['raw_logit_mean'])
        scores[key] = core.score(np, log_members, probability, member_errors, pooled_errors, labels, pool_log_probability=explicit_pool)
    return scores
def sampled_readouts(np, core, context, values, labels):
    results = []
    for row in context['secondary']:
        receipt({k: row[k] for k in ('path', 'bytes', 'sha256')})
        with np.load(row['path'], allow_pickle=False) as f: a = {k: f[k].copy() for k in f.files}
        require(set(a) == {'ids', 'y', 'draw_raw_logits', 'draw_probability', 'draw_component', 'component_integrated_probabilities',
            'all12_probability_mean', 'fixed_four_forward_block_probability_means'} and np.array_equal(a['ids'], values[row['seed'], row['arm']]['ids'])
            and np.array_equal(a['y'], labels), 'Separate complete sampled schema/roles')
        shapes = dict(draw_raw_logits=(12,5274,10), draw_probability=(12,5274,10), component_integrated_probabilities=(row['members'],5274,10),
            all12_probability_mean=(5274,10), fixed_four_forward_block_probability_means=(3,5274,10))
        require(all(a[k].shape == v and a[k].dtype == np.float32 and np.isfinite(a[k]).all() for k,v in shapes.items())
            and a['draw_component'].dtype == np.int64 and a['draw_component'].shape == (12,)
            and a['draw_component'].tolist() == row['component_sequence'], 'All12 raw draw/component identities')
        require(all(normalized(np,a[k]) for k in shapes if k != 'draw_raw_logits'), 'Finite normalized sampled probabilities')
        log_draw = log_probability(np, core, a['draw_raw_logits']); p = a['all12_probability_mean']
        score = core.score(np, log_draw, p, a['draw_raw_logits'].argmax(-1) != labels, p.argmax(-1) != labels, labels)
        block_scores = [core.public_score(core.score(np, log_draw[4*b:4*b+4], a['fixed_four_forward_block_probability_means'][b],
            a['draw_raw_logits'][4*b:4*b+4].argmax(-1) != labels, a['fixed_four_forward_block_probability_means'][b].argmax(-1) != labels, labels)) for b in range(3)]
        integrated = a['component_integrated_probabilities']; component_log = np.stack([core.logsumexp(np,log_draw[a['draw_component']==m],0)-math.log(int((a['draw_component']==m).sum())) for m in range(row['members'])])
        integrated_score = core.score(np,component_log,p,integrated.argmax(-1)!=labels,p.argmax(-1)!=labels,labels)
        results.append(dict(arm=row['arm'],seed=row['seed'],all12_sampled_raw=core.public_score(score),
            individual_draw_members_not_component_means=True,component_integrated_raw=core.public_score(integrated_score),
            fixed_blocks=block_scores,primary_component_means=core.public_score(raw_scores(np,core,{(row['seed'],row['arm']):values[row['seed'],row['arm']]},labels)[row['seed'],row['arm']]),
            raw_only=True,primary=False,best_block_selection=False,calibration=False))
    return results
def assessment(raw, calibrated, contrasts, plan):
    candidate = plan['primary_candidate']; references = plan['primary_references']; rows = []
    for ref in references:
        delta = contrasts['raw'][ref]['quality_deltas']; cal = contrasts['calibrated'][ref]
        complete = cal['status'] == 'finite_complete'; cd = cal['quality_deltas'] if complete else None
        passed = delta['pooled_accuracy_pct']['mean'] > 0 and all(v >= 0 for v in delta['pooled_accuracy_pct']['seed_deltas'])
        passed = passed and all(delta[k]['mean'] <= 0 for k in ('pooled_nll','pooled_brier'))
        passed = passed and complete and all(cd[k]['mean'] <= 0 for k in ('pooled_nll','pooled_brier'))
        rows.append(dict(reference=ref,full_raw_and_calibrated_quality_supported=bool(passed),complete_calibration=complete))
    competence = []
    for ref in ('E_point4','E_fixed_noise4'):
        d = contrasts['raw'][ref]['quality_deltas']
        passed = d['mean_member_accuracy_pct']['mean'] >= .2 and d['worst_member_accuracy_pct']['mean'] >= 0
        passed = passed and all(d[k]['mean'] <= 0 for k in ('mean_member_nll','worst_member_nll'))
        competence.append(dict(reference=ref,prospective_competence_target_supported=passed))
    return dict(primary_candidate=candidate,practical_prespecified_rules=plan['practical_rules'],references=rows,competence=competence,
        native_acquisition_hypothesis_supported=all(r['full_raw_and_calibrated_quality_supported'] for r in rows) and all(r['prospective_competence_target_supported'] for r in competence),
        comparison_with_original_GNNM_complete=ORIGINAL in references,missing_original_scope=ORIGINAL not in references,
        sharing_specific_or_novelty_claim=False,uncertainty_scope='Three paired optimizer seeds on one encountered graph; descriptive intervals, no independent-node inference.')


def run(spec_path, output):
    route(); spec_path = Path(spec_path).resolve(); require(spec_path.is_relative_to(R), 'Root-owned readout spec')
    spec = read(spec_path); context = preflight(spec)
    output = Path(output).resolve(); require(output.is_relative_to(R) and not output.exists(), 'One fresh full readout')
    require(signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'Dedicated finite external readout owner')
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame): raise TimeoutError('Finite whole-reader cap; no retry or survivor report')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(context['limits']['wall_seconds']); output.mkdir(); began=time.monotonic(); phase='numerical_imports'
    def guard(): require(time.monotonic()-began < context['limits']['wall_seconds'] and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= context['limits']['rss_bytes'],'Finite readout wall/RSS')
    try:
        import numpy as np
        import torch
        require(np.__version__=='1.26.4' and torch.__version__=='2.1.2+cu118','Original numerical runtime'); torch.set_num_threads(2)
        module_spec=importlib.util.spec_from_file_location('rank1_complete_math',Path(__file__).with_name('reader_math.py'))
        core=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(core)
        phase='all_archive_schemas'; values,ids,labels=arrays(np,context); guard()
        phase='all_fixed_calibration_attempts'; endpoints,fit_records,calibration_costs,folds=calibrate(np,torch,core,values,ids,labels,context,output,guard)
        phase='complete_raw_calibrated_contrasts'; raw=raw_scores(np,core,values,labels); calibrated={}
        for key, endpoint in endpoints.items():
            if endpoint is None: calibrated[key]=None; continue
            member_log,pool_log=endpoint; a=values[key]; native=a['operator']==NATIVE_OPERATOR
            member_error=member_log.argmax(-1)!=labels; pool_error=pool_log.argmax(-1)!=labels
            if not native: require(np.array_equal(pool_error,a['raw_logit_decision']!=labels),'Positive pooled-logit temperature preserves original decisions')
            calibrated[key]=core.score(np,member_log,np.exp(pool_log),member_error,pool_error,labels,pool_log_probability=pool_log)
        references=context['plan']['primary_references']; candidate=context['plan']['primary_candidate']
        contrasts={regime:{ref:core.contrast(np,scores,candidate,ref,labels,SEEDS,scope='complete_native_Rank1_or_original_boundary_procedures') for ref in references} for regime,scores in [('raw',raw),('calibrated',calibrated)]}
        control_refs=('E_point_M1','E_Rank1_q_M1','E_Rank1_q_I4')
        controls={regime:{ref:core.contrast(np,scores,candidate,ref,labels,SEEDS) for ref in control_refs} for regime,scores in [('raw',raw),('calibrated',calibrated)]}
        same_state={arm:core.contrast(np,{(s,'raw'):raw[s,arm] for s in SEEDS}|{(s,'calibrated'):calibrated[s,arm] for s in SEEDS},'calibrated','raw',labels,SEEDS,scope='same_selected_state_calibration') for arm in context['plan']['arms']}
        details=[dict(seed=s,arm=a,operator=values[s,a]['operator'],raw=core.public_score(raw[s,a]),calibrated=core.public_score(calibrated[s,a]),
            strict_common_wrong_rival_nodes=int(core.strict_mask(np,values[s,a]['raw_logits'],labels).sum())) for a in context['plan']['arms'] for s in SEEDS]
        phase='all_raw_secondary_diagnostics'; samples=sampled_readouts(np,core,context,values,labels); guard()
        result=dict(complete=True,plan_id=spec['plan_id'],counts=context['plan']['counts'],primary_candidate=candidate,
            calibration_costs=calibration_costs,calibration_records=fit_records,folds=folds,arm_seed_readouts=details,
            primary_contrasts=contrasts,capable_control_contrasts=controls,calibration_minus_raw_same_selected_state=same_state,
            sampled_raw_diagnostics=samples,assessment=assessment(raw,calibrated,contrasts,context['plan']),
            custody={k:context[k] for k in ('primary','secondary','states','traces','costs','native_complete','original_complete','runtime','native_actual_counts')},
            input_spec=current(spec_path),selected_states_deserialized=0,native_models_or_forward_calls_added=0,TEST_access=False,
            current_Physics_Penn_access=False,previous_paper_scores_replaced=False,
            source_only_plans_auto_adopted=False,all_scalar_attempts_terminal_before_interpretation=True,
            elapsed_seconds=time.monotonic()-began,limits=['Fixed-noise/posterior utility remains unknown before this actual comparison.',
            'Rank-1 BNN, BatchEnsemble and existing E conditional-geometry ancestry; no novelty certification.',
            'Original boundary architecture/init/fusion/dropout/cadence/5000-update recipe differ; whole-procedure comparison, not isolated placement.',
            'Repeated development nodes and three optimizer seeds; no unused whole-pipeline confirmation.',
            'Calibration failures, harms, acquired alternatives lost in pooling and all reference gaps remain visible.'])
        write(output/'COMPLETE_REPORT.json',result)
        partitions=[]
        for category,items in [('ARM_READOUTS',details),('CALIBRATION_FITS',fit_records),('SECONDARY_RAW',samples),('SELECTED_CUSTODY',context['states']),('ACTUAL_COSTS',context['costs'])]:
            for name,value in core.chunk_documents(category,items): write(output/name,value); partitions.append(current(output/name))
        summary=dict(complete=True,plan_id=spec['plan_id'],counts=context['plan']['counts'],calibration_costs=calibration_costs,
            assessment=result['assessment'],full_report=current(output/'COMPLETE_REPORT.json'),partitions=partitions,
            all_scalar_attempts_terminal_before_interpretation=True,native_models_or_forwards_added=0,TEST_access=False,current_Physics_Penn_access=False)
        write(output/'COMPLETE_ANALYSIS_SUMMARY.json',summary); return summary
    except BaseException as error:
        write(output/'FAILURE.json',dict(complete=False,status='FAILED_RETAINED',phase=phase,error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),retry=False,survivor_readout=False,automatic_continuation=False,elapsed_seconds=time.monotonic()-began)); raise
    finally: signal.alarm(0); signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--input-spec',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(args.input_spec,args.output)
