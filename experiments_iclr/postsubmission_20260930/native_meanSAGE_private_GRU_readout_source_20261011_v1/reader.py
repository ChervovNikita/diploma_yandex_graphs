"""V2 whole-cohort CPU CAL/scorer; routes, views and independent bodies stay explicit."""
import json
import math
from pathlib import Path
import resource
import statistics
import time
from runtime import (ACCOUNTING, BLOCKS, ENDPOINTS, RAW_REFERENCES, STAGE2,
    HERE, checked, require, write)
from arithmetic import (calibrated, delta, descriptive, guard, logprob,
    owner_closed, pair, truth)

METRICS = ('accuracy_pct', 'calibrated_accuracy_pct', 'mean_member_accuracy_pct',
    'worst_member_accuracy_pct', 'raw_NLL', 'raw_Brier', 'calibrated_NLL',
    'calibrated_Brier', 'mean_member_NLL', 'worst_member_NLL',
    'mean_member_Brier', 'worst_member_Brier')
LEGACY_FIELDS = {'ids', 'member_log_probs', 'native_member_argmax',
    'native_pool_argmax', 'native_pool_probabilities'}
VIEW_FIELDS = LEGACY_FIELDS | {'view_body_ids', 'native_body_probabilities', 'native_body_argmax'}

def layout(endpoint):
    if endpoint.startswith('P1_b'):
        return 1, 1, [0]
    if endpoint in ('GI16', 'GFI16'):
        return 16, 4, [b for b in range(4) for _ in range(4)]
    if endpoint in ('GA4', 'inherited_WI4'):
        return 4, 4, list(range(4))
    require(endpoint in STAGE2, 'Unknown fixed endpoint')
    return 4, 1, [0] * 4

def endpoint_temperature(ep, archive, cal_truth):
    require(ep['status'] == 'fixed_endpoint_complete' and type(ep['updates']) is int and ep['updates'] == 500
        and ep['endpoint'] == 'latest_tied_minimum_step0_to500'
        and type(ep['selected_step']) is int and 0 <= ep['selected_step'] <= 500 and -5 <= ep['logT'] <= 5
        and math.isfinite(ep['temperature']) and ep['temperature'] == math.exp(ep['logT'])
        and math.isfinite(ep['CAL_minimum_NLL']) and ep['objective_forward_evaluations'] == 1001
        and ep['operator'] == 'mean_m softmax(member_log_probs_m/T)'
        and ep['cal_archive'] == archive and ep['cal_truth'] == cal_truth,
        'Exact per-view-before-mean scalar CAL endpoint required')

def inherited_sources(config, custodian):
    result = {}
    for family, source in config['inherited_sources'].items():
        require(set(source) == {'panel', 'temperatures', 'acquisition_owner_end',
            'calibration_owner_end', 'source_manifest'}, 'Exact inherited source interface')
        owner_closed(source['acquisition_owner_end'])
        owner_closed(source['calibration_owner_end'])
        manifest_path = checked(source['source_manifest'])
        manifest = json.loads(manifest_path.read_text())
        for rel, expected in manifest['files'].items():
            from runtime import receipt
            got = receipt(manifest_path.parent / rel)
            require(got['bytes'] == expected['bytes'] and got['sha256'] == expected['sha256'],
                'Inherited native/reader source changed: ' + rel)
        panel = json.loads(checked(source['panel']).read_text())
        temperatures = json.loads(checked(source['temperatures']).read_text())
        require(panel['complete'] and panel['terminal'] and panel['CAL_DEV_truth_read'] is False
            and panel['role_custody'] == config['custody'] and temperatures['complete']
            and temperatures['DEV_truth_read'] is False, 'Whole inherited source/CAL closure')
        old_rows = {(r['block'], r['arm']): r for r in panel['records']}
        old_eps = {(r['block'], r['arm']): r for r in temperatures['records']}
        require(len(old_rows) == len(panel['records']) and len(old_eps) == len(temperatures['records']),
            'Unique inherited panel and CAL keys')
        selected = {}
        for b in BLOCKS:
            row, ep = old_rows[b, family], old_eps[b, family]
            require(row['status'] == 'exported_complete', 'Inherited selected export complete')
            endpoint_temperature(ep, row['exports']['cal'],
                custodian['files'][f'calibration_only/block_{b}_cal.npz'])
            selected[b] = dict(row=row, endpoint=ep)
        result[family] = dict(receipts=source, panel=panel, selected=selected,
            temperatures=temperatures)
    return result

def close_panel(config):
    owner_closed(config['acquisition_owner_end'])
    custodian = json.loads(checked(config['custody']).read_text())
    require(custodian['complete'] and custodian['blocks'] == list(BLOCKS)
        and custodian['role_rng'] == 'RandomState(block+5000011+223000019)', 'Original role custody')
    panel = json.loads(checked(config['panel']).read_text())
    binding = json.loads((HERE / 'BINDING.json').read_text())
    expected = {(b, e) for b in BLOCKS for e in ENDPOINTS}
    index = {(r['block'], r['endpoint_id']): r for r in panel['records']}
    require(panel['complete'] and panel['terminal'] and panel['retry'] is False
        and panel['CAL_DEV_truth_read'] is False and panel['serving'] == 'mean_class_probabilities'
        and panel['role_custody'] == config['custody'] and len(panel['records']) == 39
        and panel['inherited_sources'] == config['inherited_sources']
        and panel['source_manifest'] == binding['acquisition_source_manifest']
        and len(index) == 39 and set(index) == expected, 'Whole exact39-row acquisition before CAL/DEV')
    require(all(panel['accounting'].get(k) == v for k, v in ACCOUNTING.items()),
        'Distinct cohort/new/inherited/external/ancestor accounting')
    prior = inherited_sources(config, custodian)
    for (b, e), r in index.items():
        V, B, body_ids = layout(e)
        legacy = e in ('P1_b0', 'inherited_WI4')
        require(r['status'] == 'exported_complete' and r['probability_pool_authority'] is True
            and r['actual_served_views'] == V and r['actual_native_bodies'] == B
            and r['view_body_ids'] == body_ids and isinstance(r['construction'], dict)
            and isinstance(r['costs'], dict) and isinstance(r['provenance'], dict),
            'Fixed native view/body ownership and cost/provenance records')
        require(r['export_schema'] == ('legacy_native_probability_members_v1' if legacy
            else 'native_probability_views_v1'), 'Fixed original/new export schema')
        if e.startswith('P1_b'):
            stage, origin, arm = 'stage1', ('inherited_P1' if e == 'P1_b0' else 'new_stage1_P1'), 'P1'
            require(r['body'] == int(e[-1]), 'Exact ordinary ancestor body index')
        elif e in STAGE2:
            stage, origin, arm = 'stage2', 'new_GRU_stage2', e
        elif e == 'GA4':
            stage, origin, arm = 'derived', 'derived_GA4', e
        else:
            stage, origin, arm = 'external', 'inherited_WI4', 'WI4'
        require((r['stage'], r['origin'], r['arm']) == (stage, origin, arm), 'Endpoint alias/provenance')
        nf = 0 if e == 'GA4' else (4 if e in ('GI16', 'GFI16', 'inherited_WI4') else 1)
        require(len(r['fits']) == nf, 'All independent selected fit trajectories required')
        require(len({f['checkpoint']['path'] for f in r['fits']}) == nf, 'Distinct selected checkpoint paths')
        for f in r['fits']:
            lower = 0 if stage == 'stage2' else 1
            require(f['status'] == 'selected_complete' and f['selected_restore_verified'] is True
                and lower <= f['selected_step'] <= f['completed_updates'] <= 100000
                and f['completed_updates'] >= 1 and isinstance(f['costs'], dict)
                and isinstance(f['operation_counts'], dict), 'Every selected state restored, including stage2 checkpoint0')
            checked(f['checkpoint'])
        require(set(r['exports']) == {'cal', 'dev'}, 'Both complete label-free exports required')
        for item in r['exports'].values():
            checked(item)
        if legacy:
            original = prior['P1' if e == 'P1_b0' else 'WI4']['selected'][b]['row']
            require(r['fits'] == original['fits'] and r['exports'] == original['exports'],
                'Original inherited selected states/exports/costs must remain unchanged')
    require(sum(len(r['fits']) for r in index.values() if r['stage'] != 'external') == 51
        and sum(len(r['fits']) for r in index.values() if r['origin'].startswith('new_')) == 48
        and sum(len(r['fits']) for r in index.values() if r['stage'] == 'external') == 12,
        '51cohort/48new/12external optimizer trajectories')
    for b in BLOCKS:
        ancestors = [index[b, f'P1_b{k}']['fits'][0]['checkpoint'] for k in range(4)]
        require(len({a['path'] for a in ancestors}) == 4, 'Four distinct ordinary P1 ancestors')
        for e in STAGE2 + ('GA4',):
            wanted = ancestors if e in ('GI16', 'GFI16', 'GA4') else ancestors[:1]
            require(index[b, e]['ancestors'] == wanted, 'Exact single/shared versus independent ancestor ownership')
    return panel, index, custodian, prior

def validate_arrays(np, v, V, n):
    require(v['ids'].dtype == np.int64 and v['ids'].shape == (n,)
        and len(np.unique(v['ids'])) == n and v['ids'].min() >= 0 and v['ids'].max() < 34493
        and v['member_log_probs'].dtype == np.float32
        and v['member_log_probs'].shape == (V, n, 5), 'Complete role/view log-probabilities')
    require(v['native_member_argmax'].dtype == v['native_pool_argmax'].dtype == np.int64
        and v['native_member_argmax'].shape == (V, n) and v['native_pool_argmax'].shape == (n,),
        'Native decision shape/type')
    p = v['native_pool_probabilities']
    require(p.dtype == np.float32 and p.shape == (n, 5) and np.isfinite(p).all()
        and (p > 0).all() and np.max(np.abs(p.astype(np.float64).sum(-1) - 1)) <= 1e-6,
        'Native positive FP32 pool distribution')
    require(np.isfinite(v['member_log_probs']).all()
        and np.array_equal(v['native_member_argmax'], v['member_log_probs'].argmax(-1))
        and np.array_equal(v['native_pool_argmax'], p.argmax(-1)), 'Stored native decision authority')
    vp = np.exp(logprob(np, v['member_log_probs'].astype(np.float64)))
    reconstructed = vp.mean(0)
    require(np.max(np.abs(reconstructed - p)) <= 1e-6, 'Inherited numeric export custody tolerance')
    v['raw_float64_decision_sensitivity_count'] = int((reconstructed.argmax(-1) != v['native_pool_argmax']).sum())
    return vp

def bank_arrays(np, r, role):
    with np.load(checked(r['exports'][role]), allow_pickle=False) as archive:
        fields = LEGACY_FIELDS if r['export_schema'] == 'legacy_native_probability_members_v1' else VIEW_FIELDS
        require(set(archive.files) == fields, 'Exact agreed native probability export schema')
        v = {k: archive[k].copy() for k in archive.files}
    V, B, body_ids = layout(r['endpoint_id'])
    n = 150 if role == 'cal' else 34093
    vp = validate_arrays(np, v, V, n)
    if r['export_schema'] == 'legacy_native_probability_members_v1':
        # Adapter does not rewrite original files or replace their native decisions.
        v['view_body_ids'] = np.asarray(body_ids, dtype=np.int64)
        v['native_body_argmax'] = v['native_member_argmax'].copy()
        v['native_body_probabilities'] = vp
        v['body_probability_origin'] = 'legacy_log_probs_float64_reconstitution; native member argmax retained'
    else:
        require(v['view_body_ids'].dtype == np.int64 and v['view_body_ids'].shape == (V,)
            and np.array_equal(v['view_body_ids'], np.asarray(body_ids, dtype=np.int64)), 'Exact body-major mapping')
        bp, ba = v['native_body_probabilities'], v['native_body_argmax']
        require(bp.dtype == np.float32 and bp.shape == (B, n, 5) and np.isfinite(bp).all()
            and (bp > 0).all() and np.max(np.abs(bp.astype(np.float64).sum(-1) - 1)) <= 1e-6
            and ba.dtype == np.int64 and ba.shape == (B, n)
            and np.array_equal(ba, bp.argmax(-1)), 'Authoritative native body-average distribution/decisions')
        v['body_probability_origin'] = 'native_Torch_FP32_body_view_mean'
    reconstructed_bodies = np.stack([vp[v['view_body_ids'] == b].mean(0) for b in range(B)])
    require(np.max(np.abs(reconstructed_bodies - v['native_body_probabilities'])) <= 1e-6,
        'Body-view numeric custody tolerance')
    v['body_float64_decision_sensitivity_counts'] = [int((reconstructed_bodies[b].argmax(-1)
        != v['native_body_argmax'][b]).sum()) for b in range(B)]
    return v

def calibrate(config, custody, output):
    start = time.monotonic()
    panel, index, custodian, prior = close_panel(config)
    import numpy as np
    import torch
    require(config['expected_runtime_versions'] == {'numpy': np.__version__, 'torch': torch.__version__}, 'Bound CAL versions')
    torch.set_num_threads(2)
    rows = [dict(block=b, endpoint_id=e, arm=index[b, e]['arm'], status='unstarted_retained',
        updates=0) for b in BLOCKS for e in ENDPOINTS]
    result = dict(complete=False, records=rows, requested_endpoints=39, nominal_cohort_endpoints=36,
        requested_new_endpoints=33, requested_scalar_updates=16500, inherited_endpoints=6,
        inherited_P1_endpoints=3, external_WI4_endpoints=3, inherited_scalar_updates=3000,
        inherited_P1_scalar_updates=1500, external_WI4_scalar_updates=1500,
        DEV_truth_read=False, panel=config['panel'], role_custody=config['custody'], custody=custody,
        retry=False, inherited_sources={k: v['receipts'] for k, v in prior.items()})
    try:
        for row in rows:
            b, e = row['block'], row['endpoint_id']
            if e in ('P1_b0', 'inherited_WI4'):
                family = 'P1' if e == 'P1_b0' else 'WI4'
                original = prior[family]['selected'][b]['endpoint']
                row.update(original)
                row.update(endpoint_id=e, origin='inherited_' + family, inherited=True,
                    original_endpoint=original, original_temperature_source=prior[family]['receipts']['temperatures'])
                continue
            guard(config, start)
            row.update(status='running', origin='new_private_GRU_CAL', inherited=False)
            ids, y = truth(np, custodian, b, 'cal')
            raw = bank_arrays(np, index[b, e], 'cal')
            require(np.array_equal(ids, raw['ids']), 'Exact CAL role identity')
            lp = torch.from_numpy(raw['member_log_probs']).to(torch.float64)
            targets, idx, M = torch.from_numpy(y), torch.arange(len(y)), len(lp)
            logT = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
            optimizer = torch.optim.Adam([logT], lr=.05, betas=(.9, .999), eps=1e-8, weight_decay=0)
            def objective():
                pooled = torch.logsumexp(torch.nn.functional.log_softmax(lp / logT.exp(), dim=-1), dim=0) - math.log(M)
                return -pooled[idx, targets].mean()
            with torch.no_grad():
                best, best_logT, best_step = float(objective()), 0., 0
            require(math.isfinite(best), 'Finite initial CAL NLL')
            for step in range(1, 501):
                guard(config, start)
                optimizer.zero_grad()
                loss = objective()
                require(torch.isfinite(loss).item(), 'Finite fixed CAL loss')
                loss.backward()
                require(torch.isfinite(logT.grad).item(), 'Finite scalar CAL gradient')
                optimizer.step()
                with torch.no_grad():
                    logT.clamp_(-5., 5.)
                    value = float(objective())
                    require(math.isfinite(value), 'Finite post-update CAL NLL')
                    if value <= best:
                        best, best_logT, best_step = value, float(logT), step
                row['updates'] = step
            row.update(status='fixed_endpoint_complete', endpoint='latest_tied_minimum_step0_to500',
                selected_step=best_step, logT=best_logT, temperature=math.exp(best_logT), CAL_minimum_NLL=best,
                objective_forward_evaluations=1001, operator='mean_m softmax(member_log_probs_m/T)',
                temperature_axis='all exported views before averaging; one bank scalar',
                cal_archive=index[b, e]['exports']['cal'],
                cal_truth=custodian['files'][f'calibration_only/block_{b}_cal.npz'])
            write(output / 'CAL_PROGRESS.json', dict(closed_endpoints=sum(r['status'] == 'fixed_endpoint_complete' for r in rows),
                actual_new_scalar_updates=sum(r['updates'] for r in rows if r.get('origin') == 'new_private_GRU_CAL'),
                inherited_scalar_updates=3000, expected_bound_endpoints=39))
        require(all(r['status'] == 'fixed_endpoint_complete' and r['updates'] == 500 for r in rows)
            and sum(not r['inherited'] for r in rows) == 33
            and sum(r['updates'] for r in rows if not r['inherited']) == 16500,
            'All33new+6inherited endpoints complete without refits')
        result.update(complete=True, endpoints=39, actual_scalar_updates=16500,
            total_bound_scalar_updates=19500, seconds=time.monotonic() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            fitted_CAL_quality_interpretation=False)
    except BaseException as error:
        for row in rows:
            if row['status'] == 'running':
                row['status'] = 'failed_retained'
        result.update(failure=dict(type=type(error).__name__, message=str(error)),
            actual_scalar_updates=sum(r['updates'] for r in rows if r.get('origin') == 'new_private_GRU_CAL'))
        raise
    finally:
        write(output / 'TEMPERATURES.json', result)
    return result

def unit_summary(np, probabilities, argmax, calibrated_probabilities, y, mask):
    n = int(mask.sum())
    correct = argmax == y[None]
    counts = correct[:, mask].sum(1)
    nll = -np.log(probabilities[np.arange(len(probabilities))[:, None], np.arange(len(y))[None], y[None]])
    cnll = -np.log(calibrated_probabilities[np.arange(len(probabilities))[:, None], np.arange(len(y))[None], y[None]])
    onehot = np.eye(5, dtype=np.float64)[y]
    brier = ((probabilities - onehot[None]) ** 2).sum(-1)
    cbrier = ((calibrated_probabilities - onehot[None]) ** 2).sum(-1)
    return dict(units=len(probabilities), correct=counts.tolist(), accuracy_pct=(100 * counts / n).tolist(),
        mean_accuracy_pct=float(100 * counts.mean() / n), worst_accuracy_pct=float(100 * counts.min() / n),
        raw_NLL=nll[:, mask].mean(1).tolist(), raw_Brier=brier[:, mask].mean(1).tolist(),
        calibrated_accuracy_pct=(100 * (calibrated_probabilities.argmax(-1)[:, mask] == y[None, mask]).sum(1) / n).tolist(),
        calibrated_NLL=cnll[:, mask].mean(1).tolist(), calibrated_Brier=cbrier[:, mask].mean(1).tolist())

def score_bank(np, r, raw, y, T):
    vp = np.exp(logprob(np, raw['member_log_probs'].astype(np.float64)))
    cvp = np.exp(logprob(np, raw['member_log_probs'].astype(np.float64) / T))
    bp = raw['native_body_probabilities'].astype(np.float64)
    cbp = np.stack([cvp[raw['view_body_ids'] == b].mean(0) for b in range(len(bp))])
    p, cp = raw['native_pool_probabilities'].astype(np.float64), calibrated(np, raw['member_log_probs'], T)
    # GM4/GN4 use FOUR private route outputs despite one physical body; GI/GFI use FOUR body averages.
    body_primary = r['endpoint_id'] in ('GI16', 'GFI16', 'GA4', 'inherited_WI4')
    decisions = raw['native_body_argmax'] if body_primary else raw['native_member_argmax']
    primary_p, primary_cp = (bp, cbp) if body_primary else (vp, cvp)
    unit = ('independent_body_average' if body_primary else
        ('private_route' if r['endpoint_id'] in ('GM4', 'GN4') else
         ('single' if r['endpoint_id'].startswith('P1_b') else 'tied_view')))
    P, CP = raw['native_pool_argmax'] == y, cp.argmax(-1) == y
    member = decisions == y[None]
    V = member.any(0)
    Q = (decisions == decisions[0:1]).all(0) & ~member[0]
    onehot = np.eye(5, dtype=np.float64)[y]
    rnll, cnll = -np.log(p[np.arange(len(y)), y]), -np.log(cp[np.arange(len(y)), y])
    rbrier, cbrier = ((p - onehot) ** 2).sum(-1), ((cp - onehot) ** 2).sum(-1)
    def subset(mask):
        n = int(mask.sum())
        require(n > 0, 'Full class population')
        units = unit_summary(np, primary_p, decisions, primary_cp, y, mask)
        c = dict(nodes=n, served_correct=int(P[mask].sum()), calibrated_served_correct=int(CP[mask].sum()),
            member_correct=units['correct'], coverage=int(V[mask].sum()),
            pooling_losses=int((V & ~P & mask).sum()), aggregation_only_correct=int((~V & P & mask).sum()),
            common_wrong_rivals=int(Q[mask].sum()))
        require(c['served_correct'] == c['coverage'] - c['pooling_losses'] + c['aggregation_only_correct'], 'Exact primary-unit coverage identity')
        q = dict(accuracy_pct=100 * c['served_correct'] / n, calibrated_accuracy_pct=100 * c['calibrated_served_correct'] / n,
            member_accuracy_pct=units['accuracy_pct'], mean_member_accuracy_pct=units['mean_accuracy_pct'],
            worst_member_accuracy_pct=units['worst_accuracy_pct'], raw_NLL=float(rnll[mask].mean()),
            calibrated_NLL=float(cnll[mask].mean()), raw_Brier=float(rbrier[mask].mean()), calibrated_Brier=float(cbrier[mask].mean()),
            member_NLL=units['raw_NLL'], mean_member_NLL=float(np.mean(units['raw_NLL'])), worst_member_NLL=max(units['raw_NLL']),
            member_Brier=units['raw_Brier'], mean_member_Brier=float(np.mean(units['raw_Brier'])), worst_member_Brier=max(units['raw_Brier']),
            raw_ECE=None, calibrated_ECE=None, ECE_unavailable_reason='No prospectively frozen bin rule')
        require(all(math.isfinite(q[k]) for k in METRICS), 'Finite whole-bank metrics')
        return dict(counts=c, quality=q, primary_metric_unit=unit,
            views=unit_summary(np, vp, raw['native_member_argmax'], cvp, y, mask),
            bodies=unit_summary(np, bp, raw['native_body_argmax'], cbp, y, mask))
    return dict(**subset(np.ones(len(y), dtype=bool)),
        classes=[dict(class_id=c, **subset(y == c)) for c in range(5)], P=P, V=V, Q=Q, member=member,
        raw_float64_decision_sensitivity_count=raw['raw_float64_decision_sensitivity_count'],
        body_float64_decision_sensitivity_counts=raw['body_float64_decision_sensitivity_counts'],
        body_probability_origin=raw['body_probability_origin'])

def gates(values):
    accuracy, confidence = [], []
    def test(rows, name, reference, metric, mean_min=None, each_min=None, each_max=None, strict_positive=False):
        v = [delta(values[b, 'GM4'], values[b, reference], metric) for b in BLOCKS]
        passed = ((mean_min is None or statistics.mean(v) >= mean_min)
            and (each_min is None or all(z >= each_min for z in v))
            and (each_max is None or all(z <= each_max for z in v))
            and (not strict_positive or all(z > 0 for z in v)))
        rows.append(dict(name=name, candidate='GM4', reference=reference, metric=metric,
            minimum_mean=mean_min, minimum_each=each_min, maximum_each=each_max,
            all_three_strict_positive=strict_positive, passed=passed, contrast=descriptive(v, True)))
    for ref in RAW_REFERENCES:
        test(accuracy, 'worthwhile_raw_served_gain', ref, 'accuracy_pct', mean_min=.20, strict_positive=True)
    for ref in ('GI16', 'GFI16'):
        require(all(values[b, 'GM4']['views']['units'] == 4
            and values[b, 'GM4']['primary_metric_unit'] == 'private_route' and values[b, ref]['bodies']['units'] == 4
            and values[b, ref]['primary_metric_unit'] == 'independent_body_average' for b in BLOCKS),
            'Competence compares FOUR GM4routes to FOUR independent body averages')
        test(accuracy, 'route_vs_equipped_independent_body_retention', ref,
            'mean_member_accuracy_pct', mean_min=-.25, each_min=-.50)
    for ref in RAW_REFERENCES:
        test(confidence, 'calibrated_NLL_block_protection', ref, 'calibrated_NLL', each_max=.005)
    means = {ref: statistics.mean(values[b, ref]['quality']['calibrated_NLL'] for b in BLOCKS) for ref in RAW_REFERENCES}
    cm = statistics.mean(values[b, 'GM4']['quality']['calibrated_NLL'] for b in BLOCKS)
    confidence.append(dict(name='calibrated_NLL_best_reference_mean_protection', candidate='GM4',
        metric='calibrated_NLL', candidate_mean=cm, reference_means=means,
        best_reference_mean=min(means.values()), passed=cm <= min(means.values())))
    ap, cp = all(r['passed'] for r in accuracy), all(r['passed'] for r in confidence)
    return dict(raw_and_member_screen_pass=ap, calibrated_confidence_protection_pass=cp,
        combined_accuracy_and_confidence_pass=ap and cp, accuracy_screens=accuracy,
        confidence_screens=confidence, confidence_failure_kept_separate=True,
        topology_private_order_causal_attribution_tested=False,
        continuation='Raw/member failure closes exact package. Passing requires separately frozen same-GRU topology/order controls before causal attribution/unused confirmation/promotion.',
        inference='Three encountered-Physics development protocols; no confirmation or primitive novelty')

def legacy_diagnostics(np, r, ids, y):
    with np.load(checked(r['exports']['dev']), allow_pickle=False) as archive:
        require(set(archive.files) == LEGACY_FIELDS, 'Unchanged fixed-cohort legacy export schema')
        raw = {k: archive[k].copy() for k in archive.files}
    validate_arrays(np, raw, 4, 34093)
    require(np.array_equal(ids, raw['ids']), 'Exact inherited hard-cohort role identity')
    member = raw['native_member_argmax'] == y[None]
    return dict(P=raw['native_pool_argmax'] == y, V=member.any(0),
        Q=(raw['native_member_argmax'] == raw['native_member_argmax'][0:1]).all(0) & ~member[0])

def evaluate(config, custody, output):
    start = time.monotonic()
    panel, index, custodian, prior = close_panel(config)
    owner_closed(config['calibration_owner_end'])
    temperatures = json.loads(checked(config['temperatures']).read_text())
    endpoints = {(r['block'], r['endpoint_id']): r for r in temperatures['records']}
    require(temperatures['complete'] and temperatures['endpoints'] == 39
        and temperatures['actual_scalar_updates'] == 16500 and temperatures['inherited_scalar_updates'] == 3000
        and temperatures['total_bound_scalar_updates'] == 19500 and len(temperatures['records']) == 39
        and set(endpoints) == set(index) and temperatures['panel'] == config['panel']
        and temperatures['role_custody'] == config['custody'] and temperatures['DEV_truth_read'] is False,
        'All39 bound CAL endpoints and owner closure before any DEV truth')
    require(sum(r['inherited'] for r in endpoints.values()) == 6, 'Six inherited endpoints, no silent refits')
    for (b, e), ep in endpoints.items():
        endpoint_temperature(ep, index[b, e]['exports']['cal'], custodian['files'][f'calibration_only/block_{b}_cal.npz'])
        if e in ('P1_b0', 'inherited_WI4'):
            family = 'P1' if e == 'P1_b0' else 'WI4'
            original = prior[family]['selected'][b]['endpoint']
            require(ep['inherited'] is True and ep['original_endpoint'] == original
                and ep['original_temperature_source'] == prior[family]['receipts']['temperatures']
                and all(ep[k] == original[k] for k in ('updates', 'endpoint', 'selected_step', 'logT', 'temperature',
                    'CAL_minimum_NLL', 'cal_archive', 'cal_truth')), 'Inherited scalar and every original endpoint receipt preserved')
        else:
            require(ep['inherited'] is False and ep['origin'] == 'new_private_GRU_CAL', 'Exactly33 new CAL fits')
    import numpy as np
    require(np.__version__ == config['expected_runtime_versions']['numpy'], 'Bound score runtime version')
    release = dict(one_whole_release=True, blocks=list(BLOCKS), after_cohort_optimizer_trajectories=51,
        after_new_optimizer_trajectories=48, additional_external_optimizer_trajectories=12,
        after_exports=39, after_CAL_endpoints=39, nominal_cohort_CAL_endpoints=36,
        new_CAL_endpoints=33, inherited_P1_CAL_endpoints=3, additional_external_WI4_CAL_endpoints=3,
        temperatures=config['temperatures'], custody=custody,
        DEV_truth={str(b): custodian['files'][f'scorer_only/block_{b}_dev.npz'] for b in BLOCKS})
    write(output / 'DEV_RELEASE.json', release)
    result = dict(complete=False, records=[], release=release, panel=config['panel'],
        temperatures=config['temperatures'], retry=False, survivor_subset=False)
    try:
        roles = {b: truth(np, custodian, b, 'dev') for b in BLOCKS}
        values = {}
        for b in BLOCKS:
            ids, y = roles[b]
            for e in ENDPOINTS:
                guard(config, start)
                raw = bank_arrays(np, index[b, e], 'dev')
                require(np.array_equal(ids, raw['ids']), 'Exact complete DEV export identity')
                value = score_bank(np, index[b, e], raw, y, endpoints[b, e]['temperature'])
                values[b, e] = value
                result['records'].append(dict(block=b, endpoint_id=e, arm=index[b, e]['arm'],
                    **{k: v for k, v in value.items() if k not in ('P', 'V', 'Q', 'member')},
                    temperature=endpoints[b, e]['temperature'], stage=index[b, e]['stage'], origin=index[b, e]['origin']))
                write(output / 'DEV_PROGRESS.json', dict(scored_rows=len(result['records']), expected_rows=39))
        comparisons = []
        for a in ENDPOINTS:
            for ref in ENDPOINTS:
                if a == ref:
                    continue
                guard(config, start)
                comparisons.append(dict(candidate=a, reference=ref,
                    metric_deltas={k: descriptive([delta(values[b, a], values[b, ref], k) for b in BLOCKS], True) for k in METRICS},
                    per_block=[dict(block=b, **pair(values[b, a], values[b, ref], np.ones(34093, dtype=bool)),
                        classes=[dict(class_id=c, **pair(values[b, a], values[b, ref], roles[b][1] == c)) for c in range(5)]) for b in BLOCKS]))
        old_index = {(r['block'], r['arm']): r for r in prior['P1']['panel']['records']}
        regions, diagnostic_provenance = [], []
        for b in BLOCKS:
            guard(config, start)
            ids, y = roles[b]
            base = legacy_diagnostics(np, old_index[b, 'S4'], ids, y)
            old_i4 = legacy_diagnostics(np, old_index[b, 'I4'], ids, y)
            candidate = values[b, 'GM4']
            masks = dict(S4_common_wrong_rival=base['Q'],
                S4_common_wrong_rival_outside_I4_coverage=base['Q'] & ~old_i4['V'],
                S4_common_wrong_rival_outside_WI4_coverage=base['Q'] & ~values[b, 'inherited_WI4']['V'])
            for name, mask in masks.items():
                regions.append(dict(block=b, region=name, nodes=int(mask.sum()),
                    new_correct_member_supply=int((~base['V'] & candidate['V'] & mask).sum()),
                    actual_served_repairs=int((~base['P'] & candidate['P'] & mask).sum()),
                    diagnostics=pair(candidate, base, mask), outside_diagnostics=pair(candidate, base, ~mask)))
            for a in ('S4', 'I4'):
                diagnostic_provenance.append(dict(block=b, arm=a, diagnostic_only=True,
                    additional_CAL_endpoints=0, source_panel=config['inherited_sources']['P1']['panel'],
                    export=old_index[b, a]['exports']['dev'], fits=old_index[b, a]['fits']))
        paired_members = [dict(block=b, candidate='GM4', reference=a,
            candidate_unit='four private routes', reference_unit='four independently selected four-view bodies',
            candidate_route_accuracy_pct=values[b, 'GM4']['views']['accuracy_pct'],
            reference_body_accuracy_pct=values[b, a]['bodies']['accuracy_pct'],
            indexwise_descriptive_deltas_pp=[z - w for z, w in zip(values[b, 'GM4']['views']['accuracy_pct'], values[b, a]['bodies']['accuracy_pct'])],
            independent_selected_horizons=[f['selected_step'] for f in index[b, a]['fits']])
            for b in BLOCKS for a in ('GI16', 'GFI16')]
        require(len(result['records']) == 39 and len(comparisons) == 156, 'Complete39-row/156ordered-contrast release')
        result.update(complete=True, comparisons=comparisons, paired_member_diagnostics=paired_members,
            encountered_common_rival_regions=regions, diagnostic_only_reference_provenance=diagnostic_provenance,
            gates=gates(values), arm_summaries=[dict(endpoint_id=e,
                metrics={k: descriptive([values[b, e]['quality'][k] for b in BLOCKS]) for k in METRICS}) for e in ENDPOINTS],
            all_reference_paid_and_serving_records=panel['records'], paid_acquisition_accounting=panel['accounting'],
            paid_acquisition_costs=panel.get('costs'), paid_acquisition_owner=config['acquisition_owner_end'],
            complete_acquisition_metadata={k: v for k, v in panel.items() if k != 'records'},
            paid_calibration_owner=config['calibration_owner_end'], new_calibration_scalar_updates=16500,
            inherited_P1_scalar_updates=1500, additional_external_WI4_scalar_updates=1500,
            inherited_calibration_scalar_updates=3000, total_bound_calibration_scalar_updates=19500,
            inherited_sources={k: v['receipts'] for k, v in prior.items()},
            reader_costs=dict(seconds=time.monotonic() - start, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            numeric_reconstitution='Custody sensitivity only; native bank/view/body decisions remain authoritative',
            scope='Complete fixed encountered development; no automatic next fit/confirmation/causal/novelty/paper claim')
    except BaseException as error:
        result['failure'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        write(output / 'DEVELOPMENT_REPORT.json', result)
    return result

def run(config, custody):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    if config['stage'] == 'calibrate':
        return calibrate(config, custody, output)
    if config['stage'] == 'score':
        return evaluate(config, custody, output)
    raise ValueError('Fixed reader stage required')
