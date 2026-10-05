"""Complete retrospective development only. No final refit or TEST interface."""
import contextlib
import io
import os
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
from custody import require, sha_object, record, verify, load_bank, load_visible, load_validation, SEEDS, FAMILIES


def write_json(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == 'darwin' else raw*1024)


def phase_event(output, value):
    with (output/'PROGRESS.jsonl').open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()


def single_context(n, t, anchors, forbidden):
    require(not n.np.intersect1d(anchors, forbidden).size, 'Single whole-fold exclusion failed')
    seed = n.np.zeros((n.N, 1))
    seed[anchors, 0] = 1
    mass = n.diffuse(t, seed, purpose='single_mass_context')[:, 0]
    return {'mass': mass, 'anchor_ids': anchors.copy(), 'forbidden_ids': forbidden.copy(),
            'zero_mass_count': int((mass == 0).sum())}


def honest_training(n, a, t, neighbour, distances, degree, ids, y_by_node, split, outer, scored, single):
    """Two stitched inner folds; no head target ever seeds its own training feature."""
    d = 17 if single else 69
    xs = n.np.empty((len(ids), d))
    xf = None if single else n.np.empty((len(ids), 94))
    skip = {} if single else {j: n.np.empty((len(ids), n.C)) for j in range(2)}
    receipts = []
    checks = []
    for inner, held in enumerate(n.folds(ids, split, outer=outer, count=2)):
        anchors = n.np.setdiff1d(ids, held)
        forbidden = n.np.union1d(scored, held)
        start = time.perf_counter()
        ctx = (single_context(n, t, anchors, forbidden) if single else
               n.context(t, a['p'], distances, anchors, y_by_node[anchors], forbidden))
        positions = n.np.searchsorted(ids, held)
        xs[positions] = n.features(a, neighbour, distances, degree, ctx, held, single=single)
        if not single:
            xf[positions] = n.features(a, neighbour, distances, degree, ctx, held, full=True)
            for j, rho in enumerate(n.RHO):
                _, logq, _, check = n.moment(ctx, a['p'], a['logp'], rho, 'full', rows=held)
                skip[j][positions] = logq
                checks.append(check)
        receipts.append({'inner': inner, 'fit_feature_ids': held.tolist(),
                         'label_seed_ids': anchors.tolist(), 'forbidden_ids_sha256': sha_object(forbidden.tolist()),
                         'wall_seconds': time.perf_counter()-start})
    return xs, xf, skip, receipts, checks


def save_model(n, output, tag, model, diagnostics, standardization=None):
    start = time.perf_counter()
    path = output/'models'/f'{tag}.pt'
    state = {'diagnostics': diagnostics}
    if isinstance(model, dict):
        state['parameters'] = model
    else:
        state['parameters'] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if standardization is not None:
        mu, sd = standardization
        state['feature_mean'] = n.torch.from_numpy(mu)
        state['feature_std'] = n.torch.from_numpy(sd)
    n.torch.save(state, path)
    return {'path': str(path.relative_to(output)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size, 'write_and_hash_wall_seconds': time.perf_counter()-start}


def group(n, output, z, provenance, edge, ids, labels, split, bank):
    started = time.perf_counter()
    single = bank == 'single_author'
    a = n.arrays(z)
    t, degree, graph_record = n.graph(edge, a['native_class'])
    neighbour = n.diffuse(t, a['native'], purpose='label_free_neighbour')
    distances = None if single else n.pairwise(a['p'])
    outer_folds = n.folds(ids, split)
    graph_record.update(bank=bank, split=split,
                        precision_argmax_discrepancies=a['precision_argmax_discrepancies'],
                        native_scoring_vs_graph_discrepancies=a['native_scoring_vs_graph_discrepancies'])
    write_json(output/f'{bank}_split{split}_GRAPH.json', graph_record)
    write_json(output/f'{bank}_split{split}_FOLDS.json',
               {'salt': f'amazon-moment-v1|split={split}|outer|node={{v}}',
                'folds': [{'fold': j, 'ids': d.tolist(), 'sha256': sha_object(d.tolist())}
                          for j, d in enumerate(outer_folds)]})
    y_by_node = n.np.full(n.N, -1, dtype=n.np.int64)
    y_by_node[ids] = labels
    if single:
        configs = [(0, 0), (10, 0), (10, 1)]
    else:
        configs = [(op, setting) for op in range(1, 10)
                   for setting in range(1 if op in (1, 2, 9) else 2)]
    raw_oof = {f'{op}_{j}': n.np.empty((len(ids), n.C)) for op, j in configs}
    cs_oof = {k: n.np.empty((len(ids), n.C)) for k in raw_oof}
    fold_reports, fitted_models = [], []
    mlp_fits = calibration_fits = 0
    propagation_seconds = 0.0
    for f, scored in enumerate(outer_folds):
        anchors = n.np.setdiff1d(ids, scored)
        require(len(anchors) == 4082 and len(scored) == 2041 and
                not n.np.intersect1d(anchors, scored).size, 'Complete outer fold cardinality differs')
        anchor_y, score_y = y_by_node[anchors], y_by_node[scored]
        checkpoint = time.perf_counter()
        ctx = (single_context(n, t, anchors, scored) if single else
               n.context(t, a['p'], distances, anchors, anchor_y, scored))
        xs, xf, train_skip, inner_receipts, qp_checks = honest_training(
            n, a, t, neighbour, distances, degree, anchors, y_by_node, split, f, scored, single)
        propagation_seconds += time.perf_counter()-checkpoint
        all_rows = n.np.arange(n.N)
        serve_xs = n.features(a, neighbour, distances, degree, ctx, all_rows, single=single)
        serve_xf = None if single else n.features(a, neighbour, distances, degree, ctx, all_rows, full=True)
        raw = {'0_0' if single else '1_0': a['native']}
        weights = {}
        fit_records = []
        if not single:
            raw['2_0'] = n.torch.softmax(n.torch.from_numpy(a['z']).mean(0), dim=-1).numpy()
            candidates = {}
            for j, rho in enumerate(n.RHO):
                for op, kind in ((4, 'global'), (5, 'diagonal'), (6, 'full')):
                    q, logq, w, checks = n.moment(ctx, a['p'], a['logp'], rho, kind)
                    raw[f'{op}_{j}'] = q
                    qp_checks.append(checks)
                    weights[f'{op}_{j}'] = {'mean_max_weight': float(w.max(-1).mean()),
                                           'mean_squared_distance_from_uniform': float(((w-0.25)**2).sum(-1).mean())}
                    if op == 6:
                        candidates[j] = (q, logq)
            q, _, w, checks = n.projection(ctx, a['p'], a['logp'])
            raw['9_0'] = q
            qp_checks.append(checks)
            for j, regularizer in enumerate(n.REG):
                q, model, diagnostics = n.fit_calibrator(a['z'], anchors, anchor_y, regularizer)
                calibration_fits += 1
                raw[f'3_{j}'] = q
                tag = f'{bank}_split{split}_fold{f}_op3_setting{j}'
                descriptor = save_model(n, output, tag, model, diagnostics)
                fit_records.append({'operator': 3, 'setting': j, 'model': descriptor, 'fit': diagnostics})
                fitted_models.append(descriptor)
            head_specs = ((7, 44, xs, serve_xs), (8, 32, xf, serve_xf))
        else:
            head_specs = ((10, 156, xs, serve_xs),)
            candidates = {}
        for op, width, train_x, serve_x in head_specs:
            for j, regularizer in enumerate(n.REG):
                fit_skip = train_skip[j] if op == 8 else a['native_log'][anchors]
                serve_skip = candidates[j][1] if op == 8 else a['native_log']
                head, mu, sd, diagnostics = n.fit_head(
                    train_x, fit_skip, anchor_y, width=width,
                    seed=n.seed(bank, split, f, op, j), regularizer=regularizer)
                expected_parameters = {7: 3650, 8: 3675, 10: 3678}[op]
                require(diagnostics['parameters'] == expected_parameters, 'Matched head parameter budget differs')
                mlp_fits += 1
                serving_start = time.perf_counter()
                raw[f'{op}_{j}'] = n.serve_head(head, mu, sd, serve_x, serve_skip)
                diagnostics['saved_score_map_serving_seconds'] = time.perf_counter()-serving_start
                diagnostics['saved_score_map_serving_seconds_per_node'] = diagnostics['saved_score_map_serving_seconds']/n.N
                tag = f'{bank}_split{split}_fold{f}_op{op}_setting{j}'
                descriptor = save_model(n, output, tag, head, diagnostics, (mu, sd))
                fit_records.append({'operator': op, 'setting': j, 'model': descriptor, 'fit': diagnostics})
                fitted_models.append(descriptor)
        require(set(raw) == set(raw_oof), 'Every predeclared configuration required')
        cs_start = time.perf_counter()
        corrected = n.correct_smooth_many(t, raw, anchors, anchor_y)
        propagation_seconds += time.perf_counter()-cs_start
        positions = n.np.searchsorted(ids, scored)
        rows = []
        for op, j in configs:
            key = f'{op}_{j}'
            raw_oof[key][positions], cs_oof[key][positions] = raw[key][scored], corrected[key][scored]
            native = op == (0 if single else 1)
            raw_metrics = n.metrics(raw[key][scored], score_y,
                                   native_class=a['native_scoring_class'][scored] if native else None,
                                   native_log=a['native_log'][scored] if native else None)
            rows.append({'operator': op, 'setting': j, 'raw': raw_metrics,
                         'corrected': n.metrics(corrected[key][scored], score_y)})
        fold_report = {'fold': f, 'scored_ids': scored.tolist(), 'fusion_fit_ids': anchors.tolist(),
                       'scored_ids_sha256': sha_object(scored.tolist()),
                       'fusion_fit_ids_sha256': sha_object(anchors.tolist()),
                       'inner_feature_receipts': inner_receipts, 'outcomes': rows,
                       'fits': fit_records, 'QP_checks': qp_checks, 'weights': weights,
                       'zero_mass_count_all_nodes': ctx['zero_mass_count'],
                       'anchor_mass_on_scored': {'minimum': float(ctx['mass'][scored].min()),
                                                 'mean': float(ctx['mass'][scored].mean())}}
        fold_reports.append(fold_report)
        write_json(output/f'{bank}_split{split}_fold{f}.json', fold_report)
        phase_event(output, {'event': 'outer_fold_complete', 'bank': bank, 'split': split, 'fold': f,
                             'MLP_fits_group_so_far': mlp_fits,
                             'calibration_fits_group_so_far': calibration_fits, 'peak_rss_bytes': peak_rss()})
    outcomes = []
    for op, j in configs:
        key = f'{op}_{j}'
        native = op == (0 if single else 1)
        outcomes.append({'operator': op, 'setting': j,
                         'raw': n.metrics(raw_oof[key], labels,
                                          native_class=a['native_scoring_class'][ids] if native else None,
                                          native_log=a['native_log'][ids] if native else None),
                         'corrected': n.metrics(cs_oof[key], labels)})
    selected = []
    for op in sorted(set(op for op, _ in configs)):
        settings = [r for r in outcomes if r['operator'] == op]
        selected.append(min(settings, key=lambda r: (r['corrected']['Brier'], r['setting'])))
    finalist = min(selected, key=lambda r: (r['corrected']['Brier'], r['operator']))
    oof_path = output/f'{bank}_split{split}_OOF.npz'
    n.np.savez_compressed(oof_path, ids=ids, configuration_keys=n.np.array(list(raw_oof)),
                         raw=n.np.stack(list(raw_oof.values())), corrected=n.np.stack(list(cs_oof.values())))
    report = {'bank': bank, 'split': split, 'provenance': provenance, 'graph': graph_record,
              'all_configurations': outcomes, 'selected_settings': selected, 'finalist': finalist,
              'native_uncorrected': next(r['raw'] for r in outcomes if r['operator'] == (0 if single else 1)),
              'OOF_predictions': {'path': oof_path.name, 'sha256': hashlib.sha256(oof_path.read_bytes()).hexdigest(),
                                  'bytes': oof_path.stat().st_size},
              'cost': {'wall_seconds': time.perf_counter()-started, 'propagation_and_context_seconds': propagation_seconds,
                       'MLP_fits': mlp_fits, 'calibration_fits': calibration_fits,
                       'optimizer_updates': 150*(mlp_fits+calibration_fits),
                       'peak_rss_bytes': peak_rss()},
              'development_biased': True, 'base_checkpoint_not_cross_fitted': True,
              'final_refits_performed': 0, 'folds_are_not_independent_replicates': True}
    write_json(output/f'{bank}_split{split}_SUMMARY.json', report)
    return report


def pair_summary(rows):
    out = {}
    for metric in ('Brier', 'NLL', 'accuracy'):
        values = [r[metric] for r in rows]
        mean = sum(values)/3
        sd = math.sqrt(sum((v-mean)**2 for v in values)/2)
        half = 4.303*sd/math.sqrt(3)
        out[metric] = {'split_differences': values, 'mean': mean, 'range': [min(values), max(values)],
                       'descriptive_t_df2_interval': [mean-half, mean+half]}
    return out


def differences(candidate, reference):
    # Positive Brier/NLL means candidate harm; positive accuracy means candidate gain.
    return {k: candidate[k]-reference[k] for k in ('Brier', 'NLL', 'accuracy')}


def practical_screen(rows):
    return {'passed': (sum(r['Brier'] for r in rows)/3 <= -0.002 and
                       sum(r['Brier'] < 0 for r in rows) >= 2 and
                       sum(r['accuracy'] for r in rows)/3 >= 0.0025 and
                       min(r['accuracy'] for r in rows) >= -0.005 and
                       sum(r['NLL'] for r in rows)/3 <= 0.01),
            'mean_Brier_improvement': -sum(r['Brier'] for r in rows)/3,
            'mean_accuracy_gain_pp': 100*sum(r['accuracy'] for r in rows)/3,
            'mean_NLL_harm': sum(r['NLL'] for r in rows)/3,
            'paired_descriptive_summary': pair_summary(rows)}


def synthesize(groups):
    by = {(r['bank'], r['split']): r for r in groups}
    comparisons = {}
    chosen = []
    for bank in FAMILIES:
        for split, _, _ in SEEDS:
            r, single = by[bank, split], by['single_author', split]
            cheap_rows = [x for x in r['selected_settings'] if x['operator'] <= 5]
            # Exact ties use own-bank ID order before single, as fixed in V2.
            tagged = [(x, 0, x['operator']) for x in cheap_rows]
            tagged.extend((x, 1, x['operator']) for x in single['selected_settings'])
            best_cheap = min(tagged, key=lambda z: (z[0]['corrected']['Brier'], z[1], z[2], z[0]['setting']))[0]
            chosen.append({'bank': bank, 'split': split, 'operator': r['finalist']['operator'],
                           'setting': r['finalist']['setting'], 'strongest_cheap': best_cheap})
        own = [by[bank, split] for split, _, _ in SEEDS]
        native = [differences(r['finalist']['corrected'], r['native_uncorrected']) for r in own]
        cheap = [differences(by[bank, q['split']]['finalist']['corrected'], q['strongest_cheap']['corrected'])
                 for q in chosen if q['bank'] == bank]
        comparisons[bank] = {'against_native': practical_screen(native), 'against_strongest_cheap': practical_screen(cheap)}
    shared_independent = []
    shared_single = []
    for split, _, _ in SEEDS:
        shared_independent.append(differences(by[FAMILIES[0], split]['finalist']['corrected'],
                                             by[FAMILIES[1], split]['finalist']['corrected']))
        shared_single.append(differences(by[FAMILIES[0], split]['finalist']['corrected'],
                                       by['single_author', split]['finalist']['corrected']))
    cross_bank = {'shared_vs_best_processed_independent': practical_screen(shared_independent),
                  'shared_vs_best_processed_single': practical_screen(shared_single),
                  'attribution':'whole per-split processing pipelines; prediction graphs differ; no pure causal sharing claim'}
    moment_attribution = {}
    for bank in FAMILIES:
        contrast = {}
        for control in (4, 5, 8, 9):
            differences_by_split = []
            for split, _, _ in SEEDS:
                selected = {r['operator']: r for r in by[bank, split]['selected_settings']}
                differences_by_split.append(differences(selected[6]['corrected'], selected[control]['corrected']))
            contrast[str(control)] = pair_summary(differences_by_split)
        global_diagonal_pass = all(contrast[str(op)]['Brier']['mean'] <= -0.001 and
                                  max(contrast[str(op)]['Brier']['split_differences']) <= 0.002 for op in (4, 5))
        matched_rho = {}
        for j in range(2):
            for control in (4, 5):
                rows = []
                for split, _, _ in SEEDS:
                    outcomes = {(r['operator'], r['setting']): r for r in by[bank, split]['all_configurations']}
                    rows.append(differences(outcomes[6, j]['corrected'], outcomes[control, j]['corrected']))
                matched_rho[f'rho_setting{j}_control{control}'] = pair_summary(rows)
        moment_attribution[bank] = {'contrasts_candidate_minus_control': contrast,
                                  'matched_rho_contrasts': matched_rho,
                                  'global_and_diagonal_gate_passed': global_diagonal_pass,
                                  'same_info_matches_or_wins': contrast['8']['Brier']['mean'] >= -0.001,
                                  'posterior_projection_matches_or_wins': contrast['9']['Brier']['mean'] >= -0.001}
    shared_go = (comparisons[FAMILIES[0]]['against_native']['passed'] and
                 comparisons[FAMILIES[0]]['against_strongest_cheap']['passed'] and
                 all(cross_bank[k]['passed'] for k in ('shared_vs_best_processed_independent', 'shared_vs_best_processed_single')))
    return {'selected_per_split_pipelines': chosen, 'quality_screens': comparisons,
            'complete_cross_bank_comparison': cross_bank, 'moment_attribution': moment_attribution,
            'shared_pipeline_confirmation_screen': 'GO_PROPOSAL_ONLY' if shared_go else 'NO_GO',
            'execution_or_TEST_authorization': False, 'finalist_selection_is_development_biased': True,
            'uncertainty':'descriptive only; common graph/overlapping blocks and tuning are not iid confirmation'}


def run(state, output):
    # Imports are delayed until custody.prepare has bound all original inputs.
    import numerical as n
    import scipy
    n.setup()
    output = Path(output)
    output.mkdir(exist_ok=False)
    (output/'models').mkdir()
    start = time.perf_counter()
    write_json(output/'INPUTS.json', {'closure': state['closure'], 'protocol': state['protocol'],
                                     'payload_descriptors': state['payloads'],
                                     'decoded_members_only':['edge_index', 'val_mask', 'compact VALID ids/labels', 'selected raw logits'],
                                     'no_features_TRAIN_control_TEST_or_checkpoints': True})
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        n.np.show_config()
    write_json(output/'ENVIRONMENT.json', {'BLAS_config': capture.getvalue(),
                                          'thread_environment': {key: os.environ.get(key) for key in
                                              ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')},
                                          'python': sys.version, 'platform': platform.platform(),
                                          'numpy': n.np.__version__, 'torch': str(n.torch.__version__),
                                          'scipy': scipy.__version__, 'CPU_threads': 1, 'FP64_postprocessing': True,
                                          'native_accuracy_dtype':'FP32', 'base_model_execution': False})
    edge, val_mask = load_visible(state)
    groups = []
    for split, _, _ in SEEDS:
        ids, labels = load_validation(state, val_mask, split)
        independent_alias = None
        alias_provenance = None
        for bank in FAMILIES:
            family = next(f for f in state['registry']['families'] if f['split'] == split and f['family'] == bank)
            z, provenance = load_bank(state, family)
            if bank == FAMILIES[1]:
                independent_alias = z[:1].clone()
                alias_provenance = provenance[:1]
            groups.append(group(n, output, z, provenance, edge, ids, labels, split, bank))
        groups.append(group(n, output, independent_alias, alias_provenance, edge, ids, labels, split, 'single_author'))
    mlp = sum(g['cost']['MLP_fits'] for g in groups)
    calibration = sum(g['cost']['calibration_fits'] for g in groups)
    require(n.COUNTERS['H_calls'] == 144 and n.COUNTERS['sparse_steps'] == 2880 and
            n.COUNTERS['logical_H_applications'] == 684, 'Fixed propagation budget differs')
    require(mlp == 90 and calibration == 36 and len(groups) == 9,
            'Complete fixed 90/36 fitting budget differs')
    # No original input/source mutation may be hidden during the CPU development run.
    for row in state['payloads']:
        verify(state['phase'], row)
    result = {'status':'complete_retrospective_development', 'groups':groups, 'synthesis':synthesize(groups),
              'cost':{'whole_study_wall_seconds':time.perf_counter()-start, 'peak_rss_bytes':peak_rss(),
                      'MLP_fits':mlp, 'calibration_fits':calibration, 'optimizer_updates':150*(mlp+calibration),
                      'input_payload_bytes_bound':state['input_bytes'], 'historical_base_acquisition_cost':'separate original cohort',
                      'final_refits':0, 'base_model_fits_or_replays':0,
                      'H_and_QP_counters':n.COUNTERS,
                      'custody_before_decode':state.get('custody_cost'),
                      'output_files_bytes_before_result':sum(p.stat().st_size for p in output.rglob('*') if p.is_file())},
              'final_TEST_labels_used':False, 'refit_requires_separate_admission':True}
    write_json(output/'RESULT.json', result)
    with (output/'ALL_CONFIGURATION_METRICS.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['bank','split','operator','setting','view','count','Brier','NLL','accuracy'])
        for g in groups:
            for row in g['all_configurations']:
                for view in ('raw','corrected'):
                    m = row[view]
                    writer.writerow([g['bank'],g['split'],row['operator'],row['setting'],view,m['count'],m['Brier'],m['NLL'],m['accuracy']])
    phase_event(output, {'event':'complete', 'MLP_fits':mlp, 'calibration_fits':calibration,
                         'whole_study_wall_seconds':result['cost']['whole_study_wall_seconds']})
