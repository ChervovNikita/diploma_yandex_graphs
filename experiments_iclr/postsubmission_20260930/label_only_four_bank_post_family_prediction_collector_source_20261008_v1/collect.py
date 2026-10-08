"""Disabled, once-only selected-endpoint collector after root's full-family opening."""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SEEDS = (6101, 6203, 6307)
ARMS = ('C4', 'S_joint4head', 'U4_sharedB', 'S_one_path')
MEMBERS = dict(C4=4, S_joint4head=1, U4_sharedB=4, S_one_path=1)
COHORTS = ('whole', 'covered', 'no_visible_TRAIN_neighbor')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def bound(phase, row):
    path = (phase / row['path']).resolve(strict=True)
    require(path.is_relative_to(phase) and path.stat().st_size == row['bytes']
            and sha(path) == row['sha256'], 'Exact existing file binding required')
    return path


def seal(root, manifest_sha):
    require(sha(root / 'MANIFEST.json') == manifest_sha, 'Exact sealed manifest required')
    for row in read(root / 'MANIFEST.json')['files']:
        require((root / row['path']).stat().st_size == row['bytes']
                and sha(root / row['path']) == row['sha256'], 'Sealed payload changed')


def release(path, exact_sha, authorized):
    require(authorized is True, 'Disabled collector; separate root whole-family opening required')
    require(sha(path) == exact_sha, 'Exact root release hash required')
    cfg, fixed = read(path), read(HERE / 'ROOT_RELEASE_TEMPLATE_DISABLED.json')
    variable = {'enabled', 'source_review_approved', 'root_whole_family_opened',
                'collector_manifest_sha256', 'execution_source_commit',
                'family_closure_sha256', 'selected_states'}
    require(set(cfg) == set(fixed) and all(cfg[k] == fixed[k] for k in fixed if k not in variable)
            and all(cfg[k] is True for k in ('enabled', 'source_review_approved', 'root_whole_family_opened')),
            'Disabled or changed fixed collector contract')
    for key, length in (('collector_manifest_sha256', 64), ('family_closure_sha256', 64),
                        ('execution_source_commit', 40)):
        require(isinstance(cfg[key], str) and len(cfg[key]) == length
                and all(c in '0123456789abcdef' for c in cfg[key]), 'Exact root hash/commit required')
    require([(r['seed'], r['arm']) for r in cfg['selected_states']] ==
            [(seed, arm) for seed in SEEDS for arm in ARMS], 'All 12 ordered existing state hashes required')
    require(all(set(r) == {'seed', 'arm', 'sha256'} and len(r['sha256']) == 64
                and all(c in '0123456789abcdef' for c in r['sha256']) for r in cfg['selected_states']),
            'No selected-state substitution or unbound state')
    seal(HERE, cfg['collector_manifest_sha256'])
    return cfg


def family_and_sources(cfg):
    pins = read(HERE / 'SOURCE_BINDINGS.json')
    owner = read(bound(HERE.parent, pins['owner_bindings']))
    phase, runtime = Path(owner['runtime']['phase']), owner['runtime']
    require(HERE.parent == phase and socket.gethostname() == runtime['hostname']
            and sys.executable == runtime['python'], 'Original allocation and pinned runtime required')
    seal(phase / pins['owner_directory'], pins['owner_manifest_sha256'])
    seal(phase / pins['screen_directory'], pins['screen_manifest_sha256'])
    require(owner['screen_manifest_sha256'] == pins['screen_manifest_sha256']
            and owner['screen_program'] == pins['screen_program'], 'Exact v2 screen/owner join')
    for row in owner['source_files'] + pins['plan_sources']:
        bound(phase, row)
    for item in owner['source_manifests']:
        seal(phase / item['directory'], item['sha256'])
    for row in [owner['projection_manifest'], owner['polynormer'], *owner['roles'].values()]:
        bound(phase, row)
    projection = read(phase / owner['projection_manifest']['path'])
    require(projection['payloads'] == owner['roles'] and projection['train_count'] == 580
            and projection['valid_count'] == 5274 and projection['TEST_values_in_payload'] is False,
            'Unchanged complete official TRAIN/development roles; no TEST')
    family = phase / cfg['family_output_relative']
    closure_path = family / 'FAMILY_CLOSURE.json'
    require(sha(closure_path) == cfg['family_closure_sha256'], 'Existing full-family closure hash required')
    closure = read(closure_path)
    require(closure['complete'] is True and closure['fixed_seeds'] == list(SEEDS)
            and closure['required_correction_records'] == 12 and closure['full_epochs_per_block'] == 1100
            and closure['source_manifest_sha256'] == pins['owner_manifest_sha256']
            and closure['protocol'] == owner['protocol'] and closure['dataset_projection'] == owner['projection_manifest']
            and closure['scores_read'] is False and closure['comparative_opening_authorized'] is False,
            'Complete v2 family custody; root opening is separately required')
    require([r['seed'] for r in closure['completed']] == list(SEEDS), 'All three completed native blocks')
    terminal = read(family / 'PARENT_TERMINAL.json')
    require(terminal['family_complete'] is True and terminal['completed_seed_count'] == 3
            and terminal['error'] is None, 'Family parent terminal required')
    selected = {(r['seed'], r['arm']): r['sha256'] for r in cfg['selected_states']}
    state_paths = {}
    for record in closure['completed']:
        seed = record['seed']; cell = family / ('seed' + str(seed))
        exit_receipt = record['exit_receipt']
        require(record['required_arm_count'] == 4 and exit_receipt['reaped'] is True
                and exit_receipt['exit_code'] == 0 and exit_receipt['reason'] is None,
                'Every successful owned child closed')
        require(sha(cell / 'COMPLETE.json') == record['completion_sha256']
                and sha(cell / 'WORKER_COST.json') == record['worker_cost_sha256'], 'Completion/cost custody')
        done = read(cell / 'COMPLETE.json')
        require(done['complete'] is True and done['epochs'] == 1100 and done['native_trajectories'] == 1
                and done['run']['seed'] == seed
                and done['run']['source']['manifest_sha256'] == pins['screen_manifest_sha256']
                and [r['arm'] for r in done['required_bank_records']] == list(ARMS)
                and done['work']['native_update_completions'] == done['work']['all_bank_update_completions'] == 1100
                and done['work']['complete_VALID_events'] == 1100 and done['work']['native_local_restorations'] == 1,
                'All 12 full-1100 correction records before any prediction opening')
        for row in done['required_bank_records']:
            arm = row['arm']; state_path = cell / arm / 'selected.pt'
            require(row['complete'] is True and row['epochs'] == 1100
                    and row['selected_sha256'] == selected[seed, arm] == sha(state_path),
                    'Existing coherent selected-state bytes required')
            state_paths[seed, arm] = state_path
    subprocess.run(['git', 'merge-base', '--is-ancestor', closure['execution_source_commit'],
                    cfg['execution_source_commit']], cwd=runtime['repository'], check=True, timeout=10)
    committed = subprocess.check_output(['git', 'show', cfg['execution_source_commit'] +
        ':experiments_iclr/postsubmission_20260930/' + HERE.name + '/MANIFEST.json'],
        cwd=runtime['repository'], timeout=10)
    require(hashlib.sha256(committed).hexdigest() == cfg['collector_manifest_sha256'], 'Published collector source commit')
    return pins, owner, state_paths


def metrics(torch, logits, probabilities, truth, mask):
    count = int(mask.sum()); members = len(logits)
    if count == 0:
        return dict(correctcount=0, accuracy=None, NLL=None, Brier=None,
                    member_correctcount=[0] * members, member_accuracy=[None] * members,
                    member_NLL=[None] * members)
    bank, pool, labels = logits[:, mask], probabilities[mask], truth[mask]
    true_lp = torch.nn.functional.log_softmax(bank, -1).gather(
        -1, labels[None, :, None].expand(members, -1, 1)).squeeze(-1)
    member_counts = [int((row.argmax(-1) == labels).sum()) for row in bank]
    correct = int((pool.argmax(-1) == labels).sum())
    result = dict(correctcount=correct, accuracy=correct / count,
        NLL=float(-(torch.logsumexp(true_lp, 0) - math.log(members)).mean()),
        Brier=float(((pool - torch.nn.functional.one_hot(labels, 10)) ** 2).sum(-1).mean()),
        member_correctcount=member_counts, member_accuracy=[v / count for v in member_counts],
        member_NLL=[float(-row.mean()) for row in true_lp])
    require(all(math.isfinite(v) for v in [result['NLL'], result['Brier'], *result['member_NLL']]),
            'Finite selected-endpoint readouts required')
    return result


def flows(before, after, truth, mask):
    old, new, labels = before[mask], after[mask], truth[mask]
    old_ok, new_ok = old == labels, new == labels
    result = dict(repairs=int((~old_ok & new_ok).sum()), introduced_errors=int((old_ok & ~new_ok).sum()),
        wrong_to_different_wrong=int((~old_ok & ~new_ok & (old != new)).sum()),
        persistent_wrong_same_label=int((~old_ok & ~new_ok & (old == new)).sum()),
        persistent_errors=int((~old_ok & ~new_ok).sum()), correct_to_correct=int((old_ok & new_ok).sum()))
    require(result['repairs'] - result['introduced_errors'] == int(new_ok.sum()) - int(old_ok.sum()),
            'Exact repair/harm correctcount identity')
    return result


def complementarity(torch, bank, pool, truth, mask, metric):
    require(len(bank) == 4, 'Only four served predictors have route complementarity')
    count = int(mask.sum()); logits, labels = bank[:, mask], truth[mask]
    predicted = logits.argmax(-1); correct = predicted == labels[None]
    pooled_correct = pool[mask].argmax(-1) == labels
    pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    any_correct, all_wrong = correct.any(0), ~correct.any(0)
    true_logits = logits.gather(-1, labels[None, :, None].expand(4, -1, 1))
    common_rival = (logits > true_logits).all(0).any(-1) & all_wrong
    require(not bool((common_rival & pooled_correct).any()), 'Strict common wrong rival cannot be pooled-rescued')
    result = dict(members=4, support=count,
        any_top1_disagreement_count=int((predicted != predicted[0:1]).any(0).sum()),
        any_correctness_disagreement_count=int((correct.any(0) & ~correct.all(0)).sum()),
        any_member_correct_count=int(any_correct.sum()),
        all_member_wrong_count=int(all_wrong.sum()),
        pooled_only_rescue_count=int((all_wrong & pooled_correct).sum()),
        pool_harm_count=int((any_correct & ~pooled_correct).sum()),
        strict_common_wrong_rival_count=int(common_rival.sum()))
    result.update(pairwise_top1_disagreement=(sum(int((predicted[a] != predicted[b]).sum()) for a, b in pairs) / (6 * count)) if count else None,
        pairwise_correctness_disagreement=(sum(int((correct[a] != correct[b]).sum()) for a, b in pairs) / (6 * count)) if count else None,
        any_member_correct_rate=int(any_correct.sum()) / count if count else None,
        pooled_minus_mean_member_accuracy_pp=100 * (metric['accuracy'] - sum(metric['member_accuracy']) / 4) if count else None,
        mean_member_minus_pooled_NLL=sum(metric['member_NLL']) / 4 - metric['NLL'] if count else None)
    return result


def csv_write(path, rows):
    require(bool(rows), 'Required complete summary table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def one_state(screen, torch, state_path, state_sha, train, valid, covered, polynormer,
              seed, arm, raw_path, work):
    work['reconstruction_attempts'] += 1
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    require(state['kind'] == 'arm_final_selected' and state['schema'] == 'label-only-four-bank-coherent-state-v1'
            and state['arm'] == arm and state['run']['seed'] == seed and tuple(state['banks']) == (arm,)
            and 1 <= state['epoch'] <= 1100 and state['native']['epoch'] == state['epoch']
            and state['native']['global'] == state['global_mode']
            and state['run']['source'] == screen.source_identity(), 'Own coherent exact-v2 selected arm only')
    session = None
    try:
        work['reconstruction_API_calls_attempted'] += 1
        session = screen.reconstruct_for_serving(state, train_data=train, polynormer=polynormer,
                    device='cuda:0', later_execution_authorized=True)
        work['reconstructions_completed'] += 1
        work['native_constructor_model_restore_calls_completed'] += 1
        work['corrector_bank_constructor_parameter_restore_calls_completed'] += 1
        native_optimizers = len(session.native.optimizers)
        bank_optimizers = len(getattr(session.banks[arm], 'optimizers', [session.banks[arm].optimizer]))
        work['native_Adam_objects_constructed_and_restored'] += native_optimizers
        work['corrector_Adam_objects_constructed_and_restored'] += bank_optimizers
        require(tuple(session.banks) == (arm,) and session.banks[arm].members == MEMBERS[arm],
                'Singles are one served predictor; C4/U4 have four')
        require(session.native.np.__version__ == '1.26.4', 'Original qualified NumPy provider')
        native = session.native; native.model.eval()
        with torch.no_grad():
            ids = valid['ids'].to(native.device)
            batch = dict(x=train['x'].to(native.device), edge_index=train['edge_index'].to(native.device), ids=ids)
            session.capture.begin('SERVE', 1)
            work['native_forward_attempts'] += 1
            native.forward(batch)
            H, base = session.capture.finish()
            work['native_forwards_completed'] += 1
            prediction = session.serve_banks(H, base, ids, native_capture=session.metadata())[arm]
            work['corrector_serving_calls_completed'] += session.banks[arm].counters['serving_calls']
            work['corrector_attention_branches_completed'] += session.banks[arm].counters['serving_route_forwards']
            native_logits = base[ids]; native_probability = native_logits.softmax(-1)
            bank, pool = prediction['member_logits'], prediction['served_probabilities']
            require(native_logits.shape == (5274, 10) and bank.shape == (MEMBERS[arm], 5274, 10)
                    and pool.shape == (5274, 10) and prediction['value_scale'] == 1., 'Full unchanged unmasked serving')
            require(bool(torch.isfinite(native_logits).all()) and bool(torch.isfinite(bank).all())
                    and bool(torch.isfinite(pool).all()), 'Finite complete predictions')
            truth = valid['y'].to(native.device)
            cohort_masks = dict(whole=torch.ones(5274, dtype=torch.bool, device=native.device),
                                covered=covered.to(native.device), no_visible_TRAIN_neighbor=(~covered).to(native.device))
            missing = cohort_masks['no_visible_TRAIN_neighbor']
            logits_max = float((bank[:, missing] - native_logits[missing][None]).abs().max()) if bool(missing.any()) else None
            probability_max = float((pool[missing] - native_probability[missing]).abs().max()) if bool(missing.any()) else None
            require(logits_max in (None, 0.) and (probability_max is None or probability_max <= 2e-7)
                    and torch.equal(pool[missing].argmax(-1), native_probability[missing].argmax(-1)),
                    'No-neighbor correction zero; own native prediction equality')
            endpoint = {}; state_rows = []; complement_rows = []
            native_pred, corrected_pred = native_probability.argmax(-1).cpu(), pool.argmax(-1).cpu()
            for cohort in COHORTS:
                mask = cohort_masks[cohort]; support = int(mask.sum())
                plain = metrics(torch, native_logits[None], native_probability, truth, mask)
                corrected = metrics(torch, bank, pool, truth, mask)
                endpoint[cohort] = dict(native=plain, corrected=corrected)
                row = dict(seed=seed, arm=arm, selected_epoch=state['epoch'], global_mode=state['global_mode'],
                    selected_sha256=state_sha, members=MEMBERS[arm], cohort=cohort, support=support)
                for prefix, result in (('native', plain), ('corrected', corrected)):
                    for name, value in result.items():
                        row[prefix + '_' + name] = value
                row.update(correctcount_change=corrected['correctcount'] - plain['correctcount'],
                    accuracy_change_pp=100 * (corrected['accuracy'] - plain['accuracy']) if support else None,
                    NLL_change=corrected['NLL'] - plain['NLL'] if support else None,
                    Brier_change=corrected['Brier'] - plain['Brier'] if support else None,
                    no_neighbor_logits_max_abs=logits_max, no_neighbor_probability_max_abs=probability_max)
                row.update(flows(native_pred, corrected_pred, valid['y'], mask.cpu()))
                state_rows.append(row)
                if arm in ('C4', 'U4_sharedB'):
                    complement_rows.append(dict(seed=seed, arm=arm, selected_epoch=state['epoch'], cohort=cohort,
                        **complementarity(torch, bank, pool, truth, mask, corrected)))
            require(endpoint['whole']['corrected']['correctcount'] == state['stats']['correctcount'],
                    'Existing selected endpoint reproduced; no new selection')
            require(session.capture.counts['SERVE_head_captures'] == 1
                    and session.capture.counts['TRAIN_head_captures'] == session.capture.counts['VALID_head_captures'] == 0
                    and session.work['common_Q_checks'] == session.work['native_update_completions'] == 0
                    and session.banks[arm].counters['corrector_Adam_steps'] == 0,
                    'Exactly one native forward; no training, evaluation selector or mask draw')
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(native._cpu_tree(dict(seed=seed, arm=arm, selected_epoch=state['epoch'],
                selected_sha256=state_sha, valid_ids=ids, native_logits=native_logits,
                native_probabilities=native_probability, member_logits=bank, served_probabilities=pool)), raw_path)
            raw_path.chmod(0o600)
            work['raw_files_completed'] += 1; work['raw_bytes'] += raw_path.stat().st_size
            return dict(seed=seed, arm=arm, epoch=state['epoch'], endpoint=endpoint,
                        prediction=corrected_pred, state_rows=state_rows, complement_rows=complement_rows,
                        raw_sha256=sha(raw_path))
    finally:
        if session is not None:
            session.close()


def decompositions(states, truth, covered):
    rows = []; masks = dict(whole=covered | ~covered, covered=covered, no_visible_TRAIN_neighbor=~covered)
    for seed in SEEDS:
        candidate = states[seed, 'C4']
        for arm in ('S_joint4head', 'U4_sharedB'):
            reference = states[seed, arm]
            for cohort in COHORTS:
                mask = masks[cohort]; count = int(mask.sum())
                c, r = candidate['endpoint'][cohort], reference['endpoint'][cohort]
                row = dict(seed=seed, reference=arm, cohort=cohort, support=count,
                    C4_selected_epoch=candidate['epoch'], reference_selected_epoch=reference['epoch'])
                for metric in ('correctcount', 'accuracy', 'NLL', 'Brier'):
                    scale = 100 if metric == 'accuracy' else 1
                    name = 'accuracy_pp' if metric == 'accuracy' else metric
                    pipeline = scale * (c['corrected'][metric] - r['corrected'][metric]) if count or metric == 'correctcount' else None
                    native_epoch = scale * (c['native'][metric] - r['native'][metric]) if count or metric == 'correctcount' else None
                    correction = scale * ((c['corrected'][metric] - c['native'][metric]) -
                        (r['corrected'][metric] - r['native'][metric])) if count or metric == 'correctcount' else None
                    residual = pipeline - native_epoch - correction if pipeline is not None else None
                    require(residual is None or (residual == 0 if metric == 'correctcount' else abs(residual) < 1e-10),
                            'Selected native-epoch/correction decomposition identity')
                    for suffix, value in (('pipeline_delta', pipeline), ('native_epoch_component', native_epoch),
                                          ('correction_component', correction), ('identity_residual', residual)):
                        row[name + '_' + suffix] = value
                row.update(flows(reference['prediction'], candidate['prediction'], truth, mask))
                rows.append(row)
    return rows


def run(*, release_path, release_sha256, later_execution_authorized=False):
    cfg = release(release_path, release_sha256, later_execution_authorized)
    output = HERE.parent / cfg['output_relative']
    require(not output.exists(), 'Fresh once-only collector output; no retry/resume/overwrite')
    output.mkdir(mode=0o700)
    started = time.monotonic(); initial = resource.getrusage(resource.RUSAGE_SELF)
    work = dict(reconstruction_attempts=0, reconstructions_completed=0, native_forward_attempts=0,
                native_forwards_completed=0, raw_files_completed=0, raw_bytes=0,
                reconstruction_API_calls_attempted=0, native_constructor_model_restore_calls_completed=0,
                corrector_bank_constructor_parameter_restore_calls_completed=0,
                native_Adam_objects_constructed_and_restored=0, corrector_Adam_objects_constructed_and_restored=0,
                corrector_serving_calls_completed=0, corrector_attention_branches_completed=0,
                optimizer_updates=0, backwards=0, TRAIN_updates=0, query_mask_draws=0)
    torch = None; complete = False; error = None
    def interrupted(number, frame):
        raise RuntimeError('Collector stop/deadline ' + str(number))
    old_handlers = {n: signal.signal(n, interrupted) for n in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.setitimer(signal.ITIMER_REAL, cfg['active_seconds'])
    try:
        pins, owner, paths = family_and_sources(cfg)
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=5).splitlines()
        require(gpu == [owner['runtime']['GPU_uuid']], 'Sole original GPU')
        free = int(subprocess.check_output(['nvidia-smi', '--id=' + gpu[0], '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=5).strip()) * 1024**2
        require(free >= cfg['minimum_fresh_GPU_bytes'], 'Fresh finite admission; Mol18 coexecution allowed')
        import torch as torch_module
        torch = torch_module
        require(str(torch.__version__) == '2.1.2+cu118' and torch.cuda.device_count() == 1, 'Original qualified provider')
        torch.cuda.set_per_process_memory_fraction(cfg['max_owned_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
        spec = importlib.util.spec_from_file_location('_opened_v2_four_bank_screen', HERE.parent / pins['screen_program']['path'])
        screen = importlib.util.module_from_spec(spec); sys.modules[spec.name] = screen; spec.loader.exec_module(screen)
        _, _, _, _, _, data, _, _, _, _, _ = screen._dependencies()
        train, valid, origin = data.load_train_valid('wikics', HERE.parent / owner['roles']['train']['path'],
                                                    HERE.parent / owner['roles']['valid']['path'])
        require(len(train['ids']) == 580 and len(valid['ids']) == 5274
                and tuple(train['x'].shape) == (11701, 300)
                and tuple(train['edge_index'].shape) == (2, 442907), 'Complete fixed graph and roles')
        edges = train['edge_index']; allowed = torch.zeros(11701, dtype=torch.bool)
        allowed[train['ids']] = True
        labeled_edges = (edges[0] != edges[1]) & allowed[edges[0]]
        reach = torch.zeros(11701, dtype=torch.bool); reach[edges[1, labeled_edges]] = True
        covered = reach[valid['ids']]
        write(output / 'RUN.json', dict(release_sha256=release_sha256, source_manifest_sha256=cfg['collector_manifest_sha256'],
            family_closure_sha256=cfg['family_closure_sha256'], selected_states=cfg['selected_states'], data=origin,
            cohorts='Existence of permitted TRAIN source on incoming nonself edges; no outcomes or learned thresholds.',
            supports=dict(whole=5274, covered=int(covered.sum()), no_visible_TRAIN_neighbor=int((~covered).sum())),
            raw_predictions_server_only=True, exploratory_consumed_development=True, no_new_gate_or_significance_claim=True))
        raw_root = output / 'SERVER_ONLY_RAW'; raw_root.mkdir(mode=0o700)
        states = {}; state_rows = []; complement_rows = []; raw_records = []
        for row in cfg['selected_states']:
            seed, arm = row['seed'], row['arm']
            raw_path = raw_root / ('seed' + str(seed)) / (arm + '.pt')
            result = one_state(screen, torch, paths[seed, arm], row['sha256'], train, valid, covered,
                HERE.parent / owner['polynormer']['path'], seed, arm, raw_path, work)
            state_rows.extend(result.pop('state_rows')); complement_rows.extend(result.pop('complement_rows'))
            raw_records.append(dict(seed=seed, arm=arm, path=str(raw_path.relative_to(output)), sha256=result.pop('raw_sha256')))
            states[seed, arm] = result
            gc.collect(); torch.cuda.empty_cache()
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= cfg['max_owned_RSS_bytes']
                    and torch.cuda.max_memory_reserved(0) <= cfg['max_owned_GPU_bytes'], 'Owned resource safety caps')
            write(output / 'PROGRESS.json', dict(work=work, complete=False, automatic_retry=False))
        require(work['reconstructions_completed'] == work['native_forwards_completed'] == work['raw_files_completed'] == 12
                and time.monotonic() - started < cfg['active_seconds'], 'One bounded complete 12-state collector')
        csv_write(output / 'SAME_STATE_CORRECTION.csv', state_rows)
        csv_write(output / 'CO_PRIMARY_DECOMPOSITION.csv', decompositions(states, valid['y'], covered))
        csv_write(output / 'ROUTE_COMPLEMENTARITY.csv', complement_rows)
        tables = {name: sha(output / name) for name in ('SAME_STATE_CORRECTION.csv', 'CO_PRIMARY_DECOMPOSITION.csv', 'ROUTE_COMPLEMENTARITY.csv')}
        write(output / 'COMPLETE.json', dict(complete=True, work=work, tables=tables, raw_predictions=raw_records,
            raw_predictions_server_only=True, family_closure_sha256=cfg['family_closure_sha256'],
            selected_states=cfg['selected_states'], new_selection_or_gate=False, exploratory_consumed_development=True))
        complete = True
    except BaseException as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
        write(output / 'FAILURE.json', dict(complete=False, error=error, work=work,
            release_sha256=release_sha256, family_closure_sha256=cfg['family_closure_sha256'],
            selected_states=cfg['selected_states'],
            all_partial_files_retained=True, automatic_retry=False, no_reselection=True))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        final = resource.getrusage(resource.RUSAGE_SELF)
        write(output / 'COST_TERMINAL.json', dict(complete=complete, error=error, work=work,
            release_sha256=release_sha256, source_manifest_sha256=cfg['collector_manifest_sha256'],
            family_closure_sha256=cfg['family_closure_sha256'], selected_states=cfg['selected_states'],
            wall_seconds=time.monotonic() - started, active_bound_seconds=cfg['active_seconds'],
            CPU_user_seconds=final.ru_utime - initial.ru_utime, CPU_system_seconds=final.ru_stime - initial.ru_stime,
            max_RSS_bytes=final.ru_maxrss * 1024,
            peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(0) if torch is not None and torch.cuda.is_initialized() else None,
            peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(0) if torch is not None and torch.cuda.is_initialized() else None,
            retained_bytes_before_cost_receipt=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            failed_reconstruction_internal_partial_constructor_counts='Not observable inside unchanged API; attempted calls and completed restorations retained.',
            automatic_retry=False, raw_predictions_server_only=True))
        for number, handler in old_handlers.items():
            signal.signal(number, handler)
    return dict(complete=True, output=str(output), reconstructed_states=12, native_forwards=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    parser.add_argument('--authorized', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(release_path=args.release, release_sha256=args.release_sha256,
                         later_execution_authorized=args.authorized), sort_keys=True))
