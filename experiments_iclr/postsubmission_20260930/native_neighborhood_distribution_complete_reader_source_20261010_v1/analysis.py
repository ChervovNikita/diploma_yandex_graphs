"""Inactive complete distribution pilot reader; stdlib until full custody.

No native model, forward, data loader, owner, retry, or training lifecycle.
The requested fixed scalar calibration runs only after all 24 fits close.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import time

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PILOT = 'native_neighborhood_distribution_pilot_root_20261010_v1'
LEARNING = 'native_neighborhood_quantile_native_source_20261010_v1'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
REFERENCE = 'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
REFERENCE_SHA = 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
CALIBRATOR = 'common_wrapper_graph_reliability_source_20261010_v2/operators.py'
SEEDS = (7301, 7403, 7507)
NEW = ('shared4_private_quartile', 'shared4_private_moments',
       'shared4_common_direction_quartile', 'factorized_M1_all_signature_quartile',
       'factorized_genuine_I4_all_signature_quartile')
REFS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
OLD = REFS + ('separable_equal_size', 'exchange')
ARMS = NEW + REFS
PRIMARY, I4 = NEW[0], NEW[4]
VALID_HASHES = {'ids': '48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801',
                'labels': '2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'}
TAIL = ('native_body_fits', 'declared_acquisition_arms',
        'distribution_operation_counts', 'descriptor_support_preparation_once')
BASE_KEYS = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}
NEW_KEYS = BASE_KEYS | {'member_probabilities', 'native_logits', 'native_member_probabilities',
                       'native_probability_mean', 'native_member_errors', 'native_pooled_errors'}
LIMIT, CHUNK_BUDGET = 2000000, 1800000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def scoped(path, parent):
    value = Path(path).resolve()
    require(value.is_relative_to(parent.resolve()), 'Out-of-scope artifact: ' + str(value))
    return value


def load_source(name, path, digest):
    require(sha(path) == digest, 'Immutable source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantics(arm):
    return {'training': 'four separately fitted/selected complete all-signature models' if arm == I4
            else 'one complete factorized all-signature M1' if arm == NEW[3] else 'shared native body, four private routes',
            'genuine_independent': arm == I4, 'native_factorization': 'unchanged all-map factors/Rademacher stem',
            'correction': 'all-signature H+four directions/12Q/degree joint nonlinear residual' if arm in NEW[3:]
            else 'private mean/std/min/max/log-degree maps' if arm == NEW[1] else 'private centered quartile/log-degree maps',
            'direction_ownership': 'one common raw direction for differing route H' if arm == NEW[2]
            else 'four own directions per native body' if arm in NEW[3:] else 'one private raw direction per route',
            'selection': 'own corrected VALID accuracy per independent member' if arm == I4 else 'whole-role corrected VALID pool accuracy',
            'archive_raw_logits': 'corrected scores; native_logits separate at same corrected-selected checkpoint',
            'stronger_I4_capacity': 'each of four members receives the complete capable four-direction M1 corrector' if arm == I4 else None}


def preflight(root, helper, inputs):
    """Owner/header gates precede complete records; all artifacts precede NumPy."""
    for name, digest in inputs['research_files'].items():
        require(sha(root / name) == digest, 'Pinned policy/source changed: ' + name)
    require(sha(root / REFERENCE) == REFERENCE_SHA, 'Pinned original SAGE reference binding')
    here = root / PILOT
    cfg, freeze = read(here / 'CONFIG.json'), read(here / 'FREEZE.json')
    start = read(here / 'OWNER_START.json')
    require(start['freeze_sha256'] == sha(here / 'FREEZE.json')
            and start['hostname'] == 'anogena-2-0' and start['gpu_uuid'] == UUID,
            'Execution freeze must match launch receipt')
    require(freeze.get('qualification_passed') is True and freeze.get('TEST_access') is False,
            'Qualified no-TEST execution freeze required')
    for row in freeze['bound_files']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Frozen bytes changed')
    for name in inputs['acquisition_files']:
        require(helper.freeze_hash(freeze, name) == inputs['research_files'][name], 'Acquisition source frozen at launch')
    for name in ('DECISION.md', 'PROSPECTIVE_PROTOCOL.json'):
        require(helper.freeze_hash(freeze, PILOT + '/' + name) == inputs['research_files'][PILOT + '/' + name],
                'Prospective policy frozen at launch')
    admission_path = here / 'SCIENTIFIC_ADMISSION.json'
    admission = read(admission_path)
    require(helper.freeze_hash(freeze, PILOT + '/SCIENTIFIC_ADMISSION.json') == sha(admission_path)
            and admission['admitted'] is True and admission['qualification_passed'] is True
            and admission['scientific_fits'] == admission['native_bodies'] == 24 and admission['banks'] == 15
            and admission['new_outcomes_seen'] is False and admission['TEST_access'] is False
            and admission['source_manifest_sha256'] == inputs['research_files'][LEARNING + '/MANIFEST.json'],
            'Prospective root scientific admission')
    qpath = here / 'ACTUAL_QUALIFICATION_V1.json'
    qualification = read(qpath)
    require(helper.freeze_hash(freeze, PILOT + '/ACTUAL_QUALIFICATION_V1.json') == sha(qpath)
            and qualification['qualified'] is True and qualification['VALID_quality_scored'] is False
            and qualification['TEST_access'] is False and qualification['scientific_quality_evidence'] is False
            and [r['arm'] for r in qualification['cases']] == list(NEW), 'Five-arm actual TRAIN qualification')
    binding = read(root / REFERENCE)
    require(cfg == dict(binding['configuration'], backbone='SAGE') and cfg['seeds'] == list(SEEDS)
            and cfg['hidden'] == 128 and cfg['depth'] == 2 and cfg['dropout'] == .2
            and cfg['learning_rate'] == .001 and cfg['max_updates'] == 1000 and cfg['patience_updates'] == 300,
            'Complete original native operating point')
    output = here / 'actual_family_v1'
    old_output = scoped(binding['family_root'], root)
    require(old_output == root / 'common_wrapper_SAGE_root_20261010_v1/actual_family_v1', 'Original family identity')
    # No complete outcome records before BOTH successful owner closures/headers.
    owner = helper.closed_owner(here / 'OWNER_END.json')
    old_owner = helper.closed_owner(old_output.parent / 'OWNER_END.json')
    require(sha(old_output.parent / 'OWNER_END.json') == binding['owner_end_sha256'], 'Original closure binding')
    header = helper.completion_header(output / 'COMPLETE_FAMILY.json')
    old_header = helper.completion_header(old_output / 'COMPLETE_FAMILY.json')
    for head, groups, fits in ((header, 15, 24), (old_header, 21, 39)):
        require(head['complete'] is True and head['groups'] == head['expected_groups'] == groups
                and head['fit_units'] == head['expected_fit_units'] == fits and head['TEST_access'] is False,
                'Whole declared roster must close before comparative records')
    complete_sha = sha(output / 'COMPLETE_FAMILY.json')
    require(complete_sha == owner['complete_sha256'] and sha(old_output / 'COMPLETE_FAMILY.json')
            == binding['complete_sha256'] == old_owner['complete_sha256'], 'Owner-pinned complete records')
    complete, old = read(output / 'COMPLETE_FAMILY.json'), read(old_output / 'COMPLETE_FAMILY.json')
    require({k: v for k, v in complete.items() if k not in {'results', 'comparisons', 'cost_scope', *TAIL}} == header
            and {k: v for k, v in old.items() if k not in ('results', 'comparisons', 'cost_scope')} == old_header,
            'Metadata/results boundary agrees with complete records')
    require(complete['native_body_fits'] == 24 and complete['declared_acquisition_arms'] == list(NEW)
            and complete['config_sha256'] == sha(here / 'CONFIG.json') and read(output / 'CONFIG.json') == cfg,
            'Full new count/config authority')
    for row in binding['frozen_inputs']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Original source/config/data binding')
    old_freeze = read(old_output.parent / 'FREEZE.json')
    require(sha(scoped(binding['config'], root)) == binding['config_sha256'] == old['config_sha256']
            and read(binding['config']) == binding['configuration'] and read(old_output / 'CONFIG.json') == binding['configuration'],
            'Original immutable configuration')
    for suffix, digest in inputs['native_suffixes'].items():
        require(helper.source_hash(complete, suffix) == helper.source_hash(old, suffix) == digest
                and helper.freeze_hash(freeze, suffix) == helper.freeze_hash(old_freeze, suffix) == digest,
                'Native source shared with original references: ' + suffix)
    for name in inputs['acquisition_runtime_files']:
        require(helper.source_hash(complete, name) == helper.freeze_hash(freeze, name)
                == inputs['research_files'][name], 'Exact acquired extension/helper/contract source')
    rows = {(r['seed'], r['arm']): r for r in complete['results']}
    old_rows = {(r['seed'], r['arm']): r for r in old['results']}
    require(len(rows) == len(complete['results']) == 15 and set(rows) == {(s, a) for s in SEEDS for a in NEW}
            and sum(len(r['fits']) for r in rows.values()) == 24
            and helper.sum_operations(complete['results']) == complete['operation_counts'], 'All unique new fits/operations')
    require(len(old_rows) == len(old['results']) == 21 and set(old_rows) == {(s, a) for s in SEEDS for a in OLD}
            and sum(len(r['fits']) for r in old_rows.values()) == 39
            and helper.sum_operations(old['results']) == old['operation_counts'], 'Original complete roster')
    support = complete['descriptor_support_preparation_once']
    require(support['support'] == 'all factual incoming nonself records' and support['duplicates'] == 'retained'
            and support['native_graph_unchanged'] is True and support['nonself_records'] == 431206
            and support['tie_order'] == 'stable original factual edge record order'
            and 0 < support['nonempty_recipients'] <= 11701 and 0 < support['maximum_degree'] <= 431206,
            'Complete nonself duplicate-preserving descriptor support')
    expected, checkpoints, records = {}, [], {}
    extra = dict(zip(NEW, (672, 712, 288, 19978, 19978)))
    for seed in SEEDS:
        for arm in NEW:
            row = rows[seed, arm]
            units, members, routes = (4 if arm == I4 else 1), (1 if arm == NEW[3] else 4), (4 if arm in NEW[:3] else 1)
            require(row['serving'] == 'probability_mean' and row['semantics'] == semantics(arm)
                    and [u['member'] for u in row['fits']] == list(range(units)), 'Exact source control/fit identity')
            for unit in row['fits']:
                body = unit['member'] if arm == I4 else 0
                dcount = 1 if arm == NEW[2] else 4
                starts = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[s, s + cfg['factor_seed_offset'], s + cfg['dropout_seed_offset']] for s in starts]
                        and unit['distribution_seeds'] == {'directions': [seed + 5000011 + 1000003 * (4 * body + k) for k in range(dcount)],
                                                         'joint_residual_head': seed + 6000017 + 1000003 * body if arm in NEW[3:] else None}
                        and unit['semantics'] == row['semantics'] and unit['native_bodies_fitted'] == 1
                        and unit['distribution_parameters'] == extra[arm] and unit['descriptor_support'] == support
                        and unit['optimizer_policy'] == 'one AdamW group/native+factors+correction; .001 lr, zero decay, no extra multiplier'
                        and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000, 'Fit seeds/ownership/selected state')
                updates, passes = unit['completed_updates'], 2 * unit['completed_updates'] + 1
                require(unit['operation_counts'] == {'updates': updates, 'backwards': updates, 'adam_steps': updates,
                        'train_fullgraph_native_trajectories': routes * updates, 'valid_selection_fullgraph_native_trajectories': routes * updates,
                        'selected_valid_fullgraph_native_trajectories': routes, 'native_graph_block_calls': 2 * routes * passes}, 'Native counters')
                head = 141 * 128 + 128 * 10 if arm in NEW[3:] else routes * (5 if arm == NEW[1] else 4) * 10
                require(unit['distribution_operation_counts'] == {'native_route_descriptor_passes': routes * passes,
                        'projected_fields': 4 * passes, 'projection_MACs': 4 * 11701 * 128 * passes,
                        'edge_scalar_gathers': 4 * 431206 * passes, 'complete_recipient_sort_rows': 4 * support['nonempty_recipients'] * passes,
                        'residual_dense_MACs': 11701 * head * passes}, 'Descriptor counters, including richer all-signature I4')
                checkpoint = scoped(unit['selected_state'], output)
                require(checkpoint == output / f'{arm}_seed{seed}/member{unit["member"]}/selected.pt'
                        and checkpoint.stat().st_size == unit['costs']['selected_checkpoint_bytes'], 'Canonical selected checkpoint path/size')
                checkpoints.append({'path': str(checkpoint), 'sha256': sha(checkpoint), 'arm': arm, 'seed': seed,
                                    'fit_member': unit['member'], 'selected_step': unit['selected_step'], 'selection': unit['selection']})
            path = output / f'{arm}_seed{seed}/selected_VALID.npz'
            expected[seed, arm] = {'path': str(path), 'sha256': sha(path), 'members': members, 'new': True}
            records[seed, arm] = row
    require(len(checkpoints) == len({r['path'] for r in checkpoints}) == 24
            and sum(u['native_bodies_fitted'] for r in rows.values() for u in r['fits']) == 24, 'All 24 fit/checkpoint identities before NumPy')
    keys = tuple(rows[SEEDS[0], NEW[0]]['fits'][0]['distribution_operation_counts'])
    require(all(set(u['distribution_operation_counts']) == set(keys) for r in rows.values() for u in r['fits'])
            and {k: sum(u['distribution_operation_counts'][k] for r in rows.values() for u in r['fits']) for k in keys}
            == complete['distribution_operation_counts'], 'Complete distribution counter sum')
    refs = {(r['seed'], r['arm']): r for r in binding['banks']}
    require(len(refs) == len(binding['banks']) == 15 and set(refs) == {(s, a) for s in SEEDS for a in REFS}, 'All original references')
    for seed in SEEDS:
        for arm in REFS:
            bound, row = refs[seed, arm], old_rows[seed, arm]
            members = 1 if arm in (REFS[0], REFS[2]) else 4
            units = 4 if 'genuine_I4' in arm else 1
            require(bound['members'] == members and row['serving'] == 'probability_mean'
                    and [u['member'] for u in row['fits']] == list(range(units))
                    and [u['selected_state'] for u in row['fits']] == bound['checkpoints'], 'Immutable selected reference fit identities')
            path = scoped(bound['archive'], old_output)
            require(path == old_output / f'{arm}_seed{seed}/selected_VALID.npz' and sha(path) == bound['archive_sha256'], 'Pinned reference archive')
            expected[seed, arm] = {'path': str(path), 'sha256': bound['archive_sha256'], 'members': members, 'new': False}
            records[seed, arm] = row
    require(len(expected) == len(records) == 30, 'All 15 new +15 original archives before numerical import')
    context = {'cfg': cfg, 'freeze': freeze, 'qualification': qualification, 'admission': admission, 'owner_start': start, 'owner': owner,
               'old_owner': old_owner, 'binding': binding, 'complete': complete, 'old_complete': old,
               'complete_sha256': complete_sha, 'freeze_sha256': sha(here / 'FREEZE.json'),
               'owner_end_sha256': sha(here / 'OWNER_END.json'), 'old_freeze_sha256': sha(old_output.parent / 'FREEZE.json')}
    return context, expected, checkpoints, records


def payloads(np, expected):
    arrays, ids, labels = {}, None, None
    for arm in ARMS:
        for seed in SEEDS:
            key, item = (seed, arm), expected[seed, arm]
            require(sha(item['path']) == item['sha256'], 'Admitted archive changed')
            with np.load(item['path'], allow_pickle=False) as source:
                require(set(source.files) == (NEW_KEYS if item['new'] else BASE_KEYS), 'Exact selected VALID schema')
                a = {k: source[k].copy() for k in source.files}
            require(sha(item['path']) == item['sha256'], 'Archive changed during read')
            require(a['ids'].shape == a['y'].shape == (5274,) and a['ids'].dtype == a['y'].dtype == np.int64, 'Exact ordered role')
            if ids is None:
                ids, labels = a['ids'], a['y']
                require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10))
                        and {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in [('ids', ids), ('labels', labels)]} == VALID_HASHES, 'Frozen VALID identity')
            require(np.array_equal(ids, a['ids']) and np.array_equal(labels, a['y']), 'All banks use identical rows/labels')
            for native in (False, True) if item['new'] else (False,):
                prefix, name = ('native_', 'native_logits') if native else ('', 'raw_logits')
                raw, pool = a[name], a[prefix + 'probability_mean']
                member, pooled = a[prefix + 'member_errors'], a[prefix + 'pooled_errors']
                require(raw.dtype == pool.dtype == np.float32 and raw.shape == (item['members'], 5274, 10) and pool.shape == (5274, 10)
                        and np.isfinite(raw).all() and np.isfinite(pool).all() and (pool >= 0).all() and (pool <= 1).all()
                        and member.dtype == pooled.dtype == np.bool_ and member.shape == (item['members'], 5274) and pooled.shape == (5274,)
                        and np.array_equal(pool.argmax(1) != labels, pooled), 'Archived float32 output/error authority')
                if item['new']:
                    probability = a[prefix + 'member_probabilities']
                    require(probability.dtype == np.float32 and probability.shape == raw.shape and np.isfinite(probability).all()
                            and (probability >= 0).all() and (probability <= 1).all()
                            and np.array_equal(probability.argmax(-1) != labels, member), 'Archived member probability/error authority')
            arrays[key] = a
    require(len(arrays) == 30, 'All schemas/identities before scoring')
    return arrays, ids, labels


def native_archive(a):
    return {'ids': a['ids'], 'y': a['y'], 'raw_logits': a['native_logits'],
            **{k: a['native_' + k] for k in ('probability_mean', 'member_errors', 'pooled_errors')}}


def strict_mask(np, logits, labels):
    truth = np.take_along_axis(logits, labels[None, :, None], axis=2)
    return (logits > truth).all(0).any(1)


def strict_counts(np, mask, served, labels):
    return {'nodes': int(mask.sum()), 'corrected': int((mask & served).sum()),
            'classes': [{'class': k, 'nodes': int((mask & (labels == k)).sum()),
                         'corrected': int((mask & served & (labels == k)).sum())} for k in range(10)]}


def score(helper, np, archive):
    value = helper.score(archive)
    truth = np.eye(10, dtype=np.float64)[archive['y']]
    brier = ((archive['probability_mean'].astype(np.float64) - truth) ** 2).sum(1)
    value['quality']['pooled_brier'] = float(brier.mean())
    for row in value['classes']:
        row['pooled_brier'] = float(brier[archive['y'] == row['class']].mean())
    return value


def contrast(helper, values, candidate, reference, labels):
    value = helper.contrast(values, candidate, reference, labels)
    value['quality_deltas']['pooled_brier'] = helper.paired(values[s, candidate]['quality']['pooled_brier'] - values[s, reference]['quality']['pooled_brier'] for s in SEEDS)
    for row in value['classes']:
        k = row['class']
        row['pooled_brier'] = helper.paired(values[s, candidate]['classes'][k]['pooled_brier'] - values[s, reference]['classes'][k]['pooled_brier'] for s in SEEDS)
    return value


def score_raw(helper, np, arrays, records, context, labels):
    raw, native, diagnostics = {}, {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            key, a, row = (seed, arm), arrays[seed, arm], records[seed, arm]
            value = score(helper, np, a)
            raw[key] = value
            require(value['counts']['pooled_correct'] == round(row['valid']['accuracy'] * 5274)
                    and value['counts']['coverage'] == row['valid']['correct_alternative_count']
                    and value['counts']['lost_correct_alternatives'] == row['valid']['coverage_lost_in_pooling'], 'Original corrected/reference counts agree')
            probability64 = np.exp(a['raw_logits'].astype(np.float64) - helper.logsumexp(a['raw_logits'].astype(np.float64), -1)[..., None])
            diagnostics[key] = {'raw_common_strict_rival': strict_counts(np, strict_mask(np, a['raw_logits'], labels), value['P'], labels),
                                'FP64_raw_pool_disagreements_with_archived_errors': int(((probability64.mean(0).argmax(1) != labels) != a['pooled_errors']).sum()),
                                'stable_NLL_minus_original_reported': value['quality']['pooled_nll'] - row['valid']['nll']}
            if arm in NEW:
                n = native_archive(a)
                component = score(helper, np, n)
                native[key] = component
                reported = row['native_at_corrected_selected']
                require(component['counts']['pooled_correct'] == round(reported['accuracy'] * 5274)
                        and component['counts']['coverage'] == reported['correct_alternative_count']
                        and component['counts']['lost_correct_alternatives'] == reported['coverage_lost_in_pooling'], 'Exact selected native counts agree')
                diagnostics[key].update(native_at_corrected_selected_common_strict_rival=strict_counts(np, strict_mask(np, n['raw_logits'], labels), component['P'], labels),
                                        corrected_on_native_strict_rivals=strict_counts(np, strict_mask(np, n['raw_logits'], labels), value['P'], labels),
                                        corrected_minus_native=helper.comparison(value, component, labels),
                                        original_strict_rival_cohorts={ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS})
    saved = {(r['seed'], r['arm']): r['versus'] for r in context['complete']['comparisons']}
    require(len(saved) == len(context['complete']['comparisons']) == 15 and set(saved) == {(s, a) for s in SEEDS for a in NEW}, 'Complete saved comparison roster')
    for (seed, arm), versus in saved.items():
        require(set(versus) == set(NEW), 'Every within-new saved contrast')
        for ref, original in versus.items():
            a, b = raw[seed, arm]['P'], raw[seed, ref]['P']
            require(original == {'repairs': int((~b & a).sum()), 'harms': int((b & ~a).sum())}, 'Integer archive comparisons agree')
    return raw, native, diagnostics


def calibrate(helper, np, torch, operators, arrays, ids, labels):
    """Fixed equal-policy scalar fits; no native forward or selection changes."""
    start = time.perf_counter()
    generator = torch.Generator(device='cpu').manual_seed(11709)
    permutation = torch.randperm(len(labels), generator=generator)
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    fold_meta = {'seed': 11709, 'folds': 5, 'counts': [(folds == k).sum().item() for k in range(5)],
                 'assignment_sha256': hashlib.sha256(folds.numpy().tobytes()).hexdigest(), 'ordered_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                 'scope': 'Encountered post-selection OOF temperature diagnostic; not fresh whole-pipeline validation.'}
    y = torch.from_numpy(labels)
    scores, records, binaries, attempted = {}, [], [], 0
    for arm in NEW:
        for seed in SEEDS:
            for kind in ('corrected', 'native'):
                a = arrays[seed, arm] if kind == 'corrected' else native_archive(arrays[seed, arm])
                log_p = torch.from_numpy(a['raw_logits']).to(torch.float64).log_softmax(-1)
                probability = log_p.exp()
                context = torch.empty((len(log_p), len(labels), 0), dtype=torch.float64)
                log_pool = torch.full((5274, 10), float('nan'), dtype=torch.float64)
                log_members = torch.full(log_p.shape, float('nan'), dtype=torch.float64)
                fits = []
                for fold in range(5):
                    fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                    attempted += 1
                    before = time.perf_counter()
                    returned_record = None
                    try:
                        held, record = operators.fit_fold(torch, 'temperature_global', log_p, probability, context, y, fit_ids, held_ids)
                        returned_record = record
                        log_T = torch.tensor(record['state']['log_T'], dtype=torch.float64)
                        temperature = log_T.exp()
                        require(log_T.shape == (1,) and record['updates'] == 500 and record['parameters'] == 1
                                and torch.isfinite(temperature).all().item() and (temperature > 0).all().item(), 'One finite positive final temperature')
                        scaled = (log_p[:, held_ids] / temperature.reshape(-1, 1, 1)).log_softmax(-1)
                        require(torch.isfinite(scaled).all().item() and torch.isfinite(held).all().item(), 'Finite fixed calibrated endpoint')
                        log_pool[held_ids], log_members[:, held_ids] = held, scaled
                        record.update(status='finite_fixed_endpoint', fold=fold, temperature=temperature.item(), seconds=time.perf_counter() - before)
                    except (FloatingPointError, RuntimeError, ValueError) as error:
                        record = {'status': 'failed_retained', 'fold': fold, 'error': str(error), 'fixed_budget_updates': 500,
                                  'actual_updates_before_failure': returned_record['updates'] if returned_record is not None else None,
                                  'returned_endpoint_record': returned_record, 'seconds': time.perf_counter() - before}
                    fits.append(record)
                finite = torch.isfinite(log_pool).all().item() and torch.isfinite(log_members).all().item()
                record = {'arm': arm, 'seed': seed, 'readout': kind, 'status': 'finite_complete_five_fold' if finite else 'failed_retained',
                          'fits': fits, 'fit_seconds_sum': sum(f['seconds'] for f in fits), 'attempted_scalar_fits': 5}
                if finite:
                    pool_array, member_array = log_pool.numpy(), log_members.numpy()
                    # Raw member top-class flags remain authoritative. A positive
                    # temperature acquires no new member alternatives; FP64/tie
                    # representation differences are reported separately.
                    compatible = {'raw_logits': member_array, 'y': labels, 'member_errors': a['member_errors'],
                                  'pooled_errors': pool_array.argmax(1) != labels, 'probability_mean': np.exp(pool_array)}
                    value = score(helper, np, compatible)
                    scores[seed, arm, kind] = value
                    record['quality'] = value['quality']
                    record['counts'] = value['counts']
                    record['classes'] = value['classes']
                    record['FP64_calibrated_member_argmax_disagreements_with_raw_archived_errors'] = int(((member_array.argmax(-1) != labels) != a['member_errors']).sum())
                    record['member_top_class_authority'] = 'Unchanged raw archived errors; FP64 calibrated argmax/tie discrepancies remain separate, never new-acquisition evidence.'
                else:
                    scores[seed, arm, kind] = None
                stream = io.BytesIO()
                np.savez_compressed(stream, ids=ids, folds=folds.numpy(), member_log_probability=log_members.numpy(), pool_log_probability=log_pool.numpy())
                name = f'CALIBRATED_{arm}_seed{seed}_{kind}_OOF.npz'
                data = stream.getvalue()
                record['OOF_archive'] = {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                binaries.append((name, data))
                records.append(record)
    require(attempted == 150 and len(records) == len(binaries) == 30, 'All fixed 30 readouts ×5 folds attempted')
    cost = {'attempted_scalar_fits': attempted, 'requested_updates_per_fit': 500,
            'requested_fullbatch_scalar_updates': attempted * 500,
            'successful_scalar_fits': sum(f['status'] == 'finite_fixed_endpoint' for r in records for f in r['fits']),
            'failed_scalar_fits': sum(f['status'] == 'failed_retained' for r in records for f in r['fits']),
            'fit_seconds_sum': sum(r['fit_seconds_sum'] for r in records), 'wall_seconds': time.perf_counter() - start,
            'failed_actual_update_counts_known': all(f.get('actual_updates_before_failure') is not None for r in records for f in r['fits'] if f['status'] == 'failed_retained'),
            'native_models_or_forwards_added': 0, 'no_retry_or_final_refit': True,
            'scope': 'Includes fold conversion/preparation and NPZ encoding; fit sums are nested, not additive to this wall interval.'}
    return scores, records, binaries, fold_meta, cost


def aggregate(helper, values, rows):
    return {'mean_quality': {k: statistics.mean(v['quality'][k] for v in values) for k in
                            ('pooled_accuracy_pct', 'pooled_nll', 'pooled_brier', 'mean_member_accuracy_pct', 'mean_member_nll', 'worst_member_accuracy_pct')},
            'summed_three_seed_readout_counts': {k: sum(v['counts'][k] for v in values) for k in values[0]['counts'] if k != 'member_correct'},
            'acquisition_costs': helper.cost_summary(rows)}


def analyze(helper, np, raw, native, calibrated, records, labels):
    within_new = [(a, b) for a in NEW for b in NEW if a != b]
    pairs = within_new + [(a, b) for a in NEW for b in REFS]
    raw_contrasts = {a + '_minus_' + b: contrast(helper, raw, a, b, labels) for a, b in pairs}
    native_context = dict(native)
    native_context.update({(s, a): raw[s, a] for s in SEEDS for a in REFS})
    native_contrasts = {a + '_minus_' + b: contrast(helper, native_context, a, b, labels) for a, b in pairs}
    same_state, calibrated_contrasts, calibration_effects = {}, {}, {}
    for arm in NEW:
        values = {(s, 'corrected'): raw[s, arm] for s in SEEDS}
        values.update({(s, 'native'): native[s, arm] for s in SEEDS})
        same_state[arm] = contrast(helper, values, 'corrected', 'native', labels)
        for kind in ('corrected', 'native'):
            if all(calibrated[s, arm, kind] is not None for s in SEEDS):
                v = {(s, 'calibrated'): calibrated[s, arm, kind] for s in SEEDS}
                v.update({(s, 'raw'): raw[s, arm] if kind == 'corrected' else native[s, arm] for s in SEEDS})
                calibration_effects[arm + '_' + kind] = contrast(helper, v, 'calibrated', 'raw', labels)
            else:
                calibration_effects[arm + '_' + kind] = {'failed_retained': True}
    for kind in ('corrected', 'native'):
        values = {(s, a): calibrated[s, a, kind] for s in SEEDS for a in NEW}
        calibrated_contrasts[kind] = {}
        for a, b in within_new:
            calibrated_contrasts[kind][a + '_minus_' + b] = contrast(helper, values, a, b, labels) if all(values[s, a] is not None and values[s, b] is not None for s in SEEDS) else {'failed_retained': True}
    required = {a: .2 if a == I4 else .1 for a in NEW if a != PRIMARY}
    required.update({a: .2 for a in (REFS[1], REFS[3], REFS[4])})
    accuracy, protection = {}, {}
    for ref, threshold in required.items():
        stat = raw_contrasts[PRIMARY + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        accuracy[ref] = {'required_mean_gain_pp': threshold, 'paired_accuracy_pp': stat,
                         'mean_at_least_threshold': stat['mean'] >= threshold,
                         'all_seeds_nonnegative': stat['nonnegative_seed_count'] == 3, 'at_least_two_positive': stat['positive_seed_count'] >= 2}
    for ref in NEW[1:]:
        calibrated_comparison = calibrated_contrasts['corrected'][PRIMARY + '_minus_' + ref]
        if calibrated_comparison.get('failed_retained'):
            protection[ref] = {'finite': False, 'mean_NLL_deterioration_at_most_0_02': False,
                               'each_seed_NLL_deterioration_at_most_0_05': False, 'calibrated_accuracy_each_seed_nonnegative': False}
        else:
            nll, acc = calibrated_comparison['quality_deltas']['pooled_nll'], calibrated_comparison['quality_deltas']['pooled_accuracy_pct']
            protection[ref] = {'finite': True, 'paired_calibrated_NLL_deterioration': nll, 'paired_calibrated_accuracy_pp': acc,
                               'mean_NLL_deterioration_at_most_0_02': nll['mean'] <= .02,
                               'each_seed_NLL_deterioration_at_most_0_05': nll['max'] <= .05,
                               'calibrated_accuracy_each_seed_nonnegative': acc['nonnegative_seed_count'] == 3}
    raw_pass = all(v['mean_at_least_threshold'] and v['all_seeds_nonnegative'] and v['at_least_two_positive'] for v in accuracy.values())
    protected = all(v['finite'] and v['mean_NLL_deterioration_at_most_0_02'] and v['each_seed_NLL_deterioration_at_most_0_05'] and v['calibrated_accuracy_each_seed_nonnegative'] for v in protection.values())
    policy = {'raw_accuracy_criteria': accuracy, 'raw_accuracy_screen_pass': raw_pass,
              'equal_policy_calibrated_protection_criteria': protection, 'equal_policy_calibrated_protection_pass': protected,
              'whole_quality_development_clue': raw_pass and protected, 'raw_NLL_tradeoffs_preserved': True,
              'scope': 'Prospective encountered development; calibration does not rescue any closed rule or change raw authority.'}
    return {'raw_corrected_or_original_contrasts': raw_contrasts, 'exact_native_context_contrasts': native_contrasts,
            'raw_corrected_minus_native_same_selected_state': same_state, 'calibrated_within_new_contrasts': calibrated_contrasts,
            'calibration_minus_raw_same_selected_state': calibration_effects}, policy


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode() + b'\n'


def chunk_documents(category, ordered_rows):
    """Deterministic roster order; split before a small document can overflow."""
    documents, current, offset = [], [], 0
    for row in ordered_rows:
        candidate = {'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current + [row]}
        if len(encode(candidate)) >= CHUNK_BUDGET and current:
            documents.append((category + '_part' + str(len(documents) + 1) + '.json',
                              {'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current}))
            offset += len(current)
            current = []
        current.append(row)
        require(len(encode({'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current})) < CHUNK_BUDGET,
                'An individual fixed row exceeds bounded export budget')
    if current:
        documents.append((category + '_part' + str(len(documents) + 1) + '.json',
                          {'category': category, 'offset': offset, 'total_rows': len(ordered_rows), 'rows': current}))
    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    require(socket.gethostname() == 'anogena-2-0', 'Literal authorized scientific hostname')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID], 'Sole authorized GPU UUID')
    root, report_path = args.research_root.resolve(), args.report.resolve()
    require(root == PHASE and Path(__file__).resolve().is_relative_to(root), 'Authorized research root')
    require(report_path.is_relative_to(root) and not report_path.exists(), 'Fresh in-scope report path')
    source = read(Path(__file__).with_name('SOURCE.json'))
    for name, digest in source['files'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Sealed reader source changed')
    inputs = read(Path(__file__).with_name('INPUT_BINDINGS.json'))
    helper = load_source('distribution_immutable_count_math', root / MATH, MATH_SHA)
    context, expected, checkpoints, records = preflight(root, helper, inputs)
    # No scientific module imports or array payloads until the complete gate.
    import numpy as np
    helper.np = np
    arrays, ids, labels = payloads(np, expected)
    raw, native, diagnostics = score_raw(helper, np, arrays, records, context, labels)
    import torch
    torch.set_num_threads(2)
    operators = load_source('fixed_temperature_operators', root / CALIBRATOR, inputs['research_files'][CALIBRATOR])
    calibrated, calibration_records, binaries, folds, calibration_costs = calibrate(helper, np, torch, operators, arrays, ids, labels)
    contrasts, policy = analyze(helper, np, raw, native, calibrated, records, labels)
    aggregates = {'raw_corrected_or_original': {a: aggregate(helper, [raw[s, a] for s in SEEDS], [records[s, a] for s in SEEDS]) for a in ARMS},
                  'exact_native_at_corrected_selected': {a: aggregate(helper, [native[s, a] for s in SEEDS], [records[s, a] for s in SEEDS]) for a in NEW}}
    aggregates['calibrated'] = {kind: {a: aggregate(helper, [calibrated[s, a, kind] for s in SEEDS], [records[s, a] for s in SEEDS])
                              if all(calibrated[s, a, kind] is not None for s in SEEDS) else {'failed_retained': True} for a in NEW} for kind in ('corrected', 'native')}
    groups = []
    for arm in ARMS:
        for seed in SEEDS:
            value = raw[seed, arm]
            row = {'arm': arm, 'seed': seed, 'raw': {k: value[k] for k in ('quality', 'counts', 'classes')},
                   'diagnostics': diagnostics[seed, arm], 'original_group_record': records[seed, arm]}
            if arm in NEW:
                row['native_at_corrected_selected'] = {k: native[seed, arm][k] for k in ('quality', 'counts', 'classes')}
            groups.append(row)
    costs = {'new24_native_fits': helper.cost_summary(context['complete']['results']),
             'reused33_selected_reference_fits': helper.cost_summary([records[s, a] for s in SEEDS for a in REFS]),
             'original39_parent_context_not_additive': helper.cost_summary(context['old_complete']['results']),
             'descriptor_support_once': context['complete']['descriptor_support_preparation_once'],
             'distribution_operation_counts': context['complete']['distribution_operation_counts'],
             'calibration_scalar_work': calibration_costs, 'new_owner': context['owner'], 'original_owner': context['old_owner']}
    report = {'complete': True, 'backbone': 'SAGE', 'new_banks': 15, 'new_native_fits': 24, 'new_native_bodies': 24,
              'new_route_trajectories_per_complete_roster_forward': 51, 'reused_reference_banks': 15, 'reused_reference_fits': 33,
              'seeds': SEEDS, 'roster': ARMS, 'TEST_access': False, 'frozen_policy': policy, 'aggregates': aggregates,
              'groups_detail': groups, 'contrasts': contrasts, 'calibration_folds': folds, 'calibration_records': calibration_records,
              'costs': costs, 'source_custody': context, 'selected_archive_custody': list(expected.values()), 'checkpoint_custody': checkpoints,
              'ordered_VALID_hashes': VALID_HASHES, 'analysis_source_sha256': sha(Path(__file__)),
              'immutable_count_math_sha256': MATH_SHA, 'input_bindings': inputs,
              'authority': 'Raw/native integer counts use archived float32 errors. Stable FP64 NLL is separate. Calibrated pooled decisions use fixed OOF FP64 outputs; raw member top-class flags remain unchanged and precision/tie differences are diagnostics.',
              'interpretation_limits': ['All24 new fits and15 new banks plus all original references close before arrays/calibration.',
                  'Raw accuracy and raw/native NLL remain visible. Prospective protected clue combines raw accuracy with equal-policy calibrated NLL and calibrated accuracy protection.',
                  'Post-selection calibration folds are encountered development, not fresh validation or whole-pipeline cross-fitting.',
                  'Native readouts share corrected-selected checkpoints; cross-state differences include learning, stopping and selection.',
                  'Positive temperature acquires no new member alternatives. FP64/raw tie disagreements are not new-acquisition evidence.',
                  'Three paired optimizer seeds on one graph are not graph/split replication; intervals are descriptive.',
                  'PNA/FSW/neighborhood statistics/quantiles/residual classification/calibration are prior ingredients; no novelty, injectivity or acceptance claim.',
                  'No native model/data loader/forward/TEST or checkpoint payload is loaded by this reader. Scalar calibration cost is separate from native fitting.',
                  'Checkpoint hashes are closure-time snapshots of canonical owner-bound fit paths/byte sizes, not earlier stored checkpoint SHA certificates.']}
    documents = [('SAGE_POLICY_SUMMARY.json', {'complete': True, 'backbone': 'SAGE', 'new_banks': 15, 'new_native_fits': 24,
                  'frozen_policy': policy, 'raw_NLL_visible': True, 'all_fixed_calibration_fits_attempted': 150, 'calibration_costs': calibration_costs})]
    documents += chunk_documents('SAGE_ARM_AGGREGATES', [{'kind': kind, 'arm': arm, 'aggregate': value} for kind, bank in aggregates.items()
                        for arm, value in (bank.items() if kind != 'calibrated' else ((k + '_' + a, v) for k, b in bank.items() for a, v in b.items()))])
    documents += chunk_documents('SAGE_ARM_DETAILS', groups)
    documents += chunk_documents('CALIBRATION_FIT_DETAILS', calibration_records)
    for category, values in contrasts.items():
        if category == 'calibrated_within_new_contrasts':
            items = [{'readout': k, 'key': name, 'contrast': value} for k, bank in values.items() for name, value in bank.items()]
        else:
            items = [{'key': name, 'contrast': value} for name, value in values.items()]
        documents += chunk_documents('SAGE_' + category, items)
    documents += chunk_documents('CHECKPOINT_CUSTODY', checkpoints)
    documents += chunk_documents('ARCHIVE_CUSTODY', [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS])
    prepared = [(report_path, encode(report))]
    prepared += [(report_path.parent / name, encode(value)) for name, value in documents]
    prepared += [(report_path.parent / name, data) for name, data in binaries]
    require(len({path for path, _ in prepared}) == len(prepared), 'Distinct deterministic output names')
    for path, data in prepared:
        require(not path.exists(), 'Every output path must be fresh')
        if path != report_path and path.suffix == '.json':
            require(len(data) < LIMIT, 'Bounded JSON partition required')
    for item in list(expected.values()) + checkpoints:
        require(sha(item['path']) == item['sha256'], 'Admitted artifact changed before output')
    receipts = [{'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in prepared]
    summary = {'complete': True, 'backbone': 'SAGE', 'new_banks': 15, 'new_native_fits': 24, 'new_native_bodies': 24,
               'original_reference_banks': 15, 'TEST_access': False, 'frozen_policy': policy, 'calibration_folds': folds,
               'calibration_costs': calibration_costs, 'raw_and_native_authorities_preserved': True,
               'post_selection_OOF_scope': 'Encountered development, not unused whole-pipeline confirmation.',
               'full_report': receipts[0], 'partitions': receipts[1:], 'input_binding_sha256': sha(Path(__file__).with_name('INPUT_BINDINGS.json')),
               'new_complete_sha256': context['complete_sha256'], 'new_freeze_sha256': context['freeze_sha256'],
               'new_owner_end_sha256': context['owner_end_sha256'], 'original_complete_sha256': context['binding']['complete_sha256']}
    summary_path = report_path.parent / 'COMPLETE_ANALYSIS_SUMMARY.json'
    require(summary_path not in {p for p, _ in prepared} and not summary_path.exists(), 'Fresh summary path')
    summary_bytes = encode(summary)
    require(len(summary_bytes) < LIMIT, 'Small complete summary')
    prepared.append((summary_path, summary_bytes))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for path, data in prepared:
        with path.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'complete': True, 'summary': str(summary_path), 'frozen_policy': policy,
                      'calibration_scalar_fits': calibration_costs, 'partitions': len(receipts) - 1}))


if __name__ == '__main__':
    main()
