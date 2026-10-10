"""Source-only complete SAGE GNCL/SupCon joined reader; stdlib until custody.

No native model, data loader, training forward, selector, owner or TEST reader.
Fixed scalar calibration starts after all27 fresh fits and all15 anchors close.
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
PILOT = 'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
LEARNING = 'native_SAGE_GNCL_SupCon_2x2_source_20261010_v1'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
REFERENCE = 'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
REFERENCE_SHA = 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
CALIBRATOR = 'common_wrapper_graph_reliability_source_20261010_v2/operators.py'
SEEDS = (7301, 7403, 7507)
NEW = ('shared4_GNCL', 'shared4_own_SupCon', 'shared4_GNCL_SupCon',
       'ordinary_M1_SupCon', 'factorized_M1_SupCon', 'genuine_factorized_I4_SupCon')
REFS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
OLD = REFS + ('separable_equal_size', 'exchange')
ARMS = NEW + REFS
A, B, AB, O1SC, F1SC, F4SC = NEW
O1, O4, F1, F4, BASE = REFS
VALID_HASHES = {'ids': '48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801',
                'labels': '2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'}
BASE_KEYS = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}
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

def strict_mask(np, logits, labels):
    truth = np.take_along_axis(logits, labels[None, :, None], axis=2)
    return (logits > truth).all(0).any(1)

def strict_counts(np, mask, served, labels):
    return {'nodes': int(mask.sum()), 'corrected': int((mask & served).sum()),
            'classes': [{'class': k, 'nodes': int((mask & (labels == k)).sum()),
                         'corrected': int((mask & served & (labels == k)).sum())} for k in range(10)]}

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


def preflight(root, helper, inputs):
    """All closure/metadata/custody checks precede NumPy and archive arrays."""
    for name, digest in inputs['research_files'].items():
        require(sha(root / name) == digest, 'Pinned prospective bytes changed: ' + name)
    require(sha(root / REFERENCE) == REFERENCE_SHA, 'Original SAGE binding unchanged')
    here = root / PILOT
    cfg, freeze = read(here / 'CONFIG.json'), read(here / 'FREEZE.json')
    start = read(here / 'OWNER_START.json')
    require(start['freeze_sha256'] == sha(here / 'FREEZE.json')
            and start['hostname'] == 'anogena-2-0' and start['gpu_uuid'] == UUID,
            'Actual launch freeze/host/UUID custody')
    require(freeze.get('qualification_passed') is True and freeze.get('TEST_access') is False,
            'Qualified no-TEST freeze')
    for row in freeze['bound_files']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Frozen file changed')
    for name in inputs['acquisition_files'] + [PILOT + '/' + n for n in
                                                ('CONFIG.json', 'DECISION.md', 'PROSPECTIVE_PROTOCOL.json')]:
        require(helper.freeze_hash(freeze, name) == inputs['research_files'][name], 'Prospective launch binding: ' + name)
    require(helper.freeze_hash(freeze, CALIBRATOR) == inputs['research_files'][CALIBRATOR],
            'The fixed calibration operator was frozen before fitting')
    admission_path, reference_path = here / 'SCIENTIFIC_ADMISSION.json', here / 'REFERENCE_ADMISSION.json'
    admission, reference_admission = read(admission_path), read(reference_path)
    for path in (admission_path, reference_path, here / 'ACTUAL_QUALIFICATION_V1.json'):
        require(helper.freeze_hash(freeze, PILOT + '/' + path.name) == sha(path), 'Root receipt frozen at launch')
    require(admission['admitted'] is True and admission['qualification_passed'] is True
            and admission['scientific_fits'] == admission['native_bodies'] == 27 and admission['banks'] == 18
            and admission['new_outcomes_seen'] is False and admission['TEST_access'] is False
            and admission['source_manifest_sha256'] == inputs['research_files'][LEARNING + '/MANIFEST.json'],
            'Whole prospective scientific admission')
    require(reference_admission['admitted'] is True and reference_admission['banks'] == 15
            and reference_admission['selected_fits'] == 33
            and reference_admission['reference_bindings_sha256'] == REFERENCE_SHA
            and reference_admission['exact_source_config_role_seed_selector_custody_compatible'] is True
            and reference_admission['TEST_access'] is False, 'Explicit exact root reference admission')
    qualification = read(here / 'ACTUAL_QUALIFICATION_V1.json')
    require(qualification['qualified'] is True and qualification['VALID_quality_scored'] is False
            and qualification['TEST_access'] is False, 'Actual TRAIN-only qualification receipt')
    binding = read(root / REFERENCE)
    require(cfg == binding['configuration'] and cfg['seeds'] == list(SEEDS)
            and cfg['hidden'] == 128 and cfg['depth'] == 2 and cfg['dropout'] == .2
            and cfg['learning_rate'] == .001 and cfg['max_updates'] == 1000 and cfg['patience_updates'] == 300,
            'Exact original configuration, without an added backbone key')
    output = here / 'actual_family_v1'
    old_output = scoped(binding['family_root'], root)
    require(old_output == root / 'common_wrapper_SAGE_root_20261010_v1/actual_family_v1', 'Original parent identity')
    owner = helper.closed_owner(here / 'OWNER_END.json')
    old_owner = helper.closed_owner(old_output.parent / 'OWNER_END.json')
    require(sha(old_output.parent / 'OWNER_END.json') == binding['owner_end_sha256'], 'Original owner custody')
    header = helper.completion_header(output / 'COMPLETE_FAMILY.json')
    old_header = helper.completion_header(old_output / 'COMPLETE_FAMILY.json')
    for head, groups, fits in ((header, 18, 27), (old_header, 21, 39)):
        require(head['complete'] is True and head['groups'] == head['expected_groups'] == groups
                and head['fit_units'] == head['expected_fit_units'] == fits and head['TEST_access'] is False,
                'Complete declared rosters before outcomes')
    require(header['completion_scope'] == 'fresh six-arm acquisition only'
            and header['full_comparative_family_complete'] is False
            and header['requires_root_admitted_own_only_anchors'] == {'banks': 15, 'selected_fit_records': 33},
            'Fresh completion cannot stand in for admitted anchors')
    complete_sha = sha(output / 'COMPLETE_FAMILY.json')
    require(complete_sha == owner['complete_sha256'] and sha(old_output / 'COMPLETE_FAMILY.json')
            == binding['complete_sha256'] == old_owner['complete_sha256'], 'Owner-bound complete records')
    # Complete metrics/records are read only after both owners and headers pass.
    complete, old = read(output / 'COMPLETE_FAMILY.json'), read(old_output / 'COMPLETE_FAMILY.json')
    require({k: v for k, v in complete.items() if k not in ('results', 'comparisons', 'cost_scope')} == header
            and {k: v for k, v in old.items() if k not in ('results', 'comparisons', 'cost_scope')} == old_header,
            'Closure header/results boundary')
    require(complete['config_sha256'] == sha(here / 'CONFIG.json') and read(output / 'CONFIG.json') == cfg,
            'Fresh exact configuration authority')
    for row in binding['frozen_inputs']:
        require(sha(scoped(REPO / row['path'], REPO)) == row['sha256'], 'Original frozen source/config/role bytes')
    old_freeze = read(old_output.parent / 'FREEZE.json')
    require(sha(scoped(binding['config'], root)) == binding['config_sha256'] == old['config_sha256']
            and read(binding['config']) == cfg and read(old_output / 'CONFIG.json') == cfg, 'Original same configuration')
    for suffix, digest in inputs['native_suffixes'].items():
        require(helper.source_hash(complete, suffix) == helper.source_hash(old, suffix) == digest
                and helper.freeze_hash(freeze, suffix) == helper.freeze_hash(old_freeze, suffix) == digest,
                'Native source/init/update correspondence: ' + suffix)
    require(helper.source_hash(complete, LEARNING + '/run_family.py')
            == inputs['research_files'][LEARNING + '/run_family.py'], 'Exact prospective acquired source')
    rows = {(r['seed'], r['arm']): r for r in complete['results']}
    old_rows = {(r['seed'], r['arm']): r for r in old['results']}
    require(len(rows) == len(complete['results']) == 18 and set(rows) == {(s, a) for s in SEEDS for a in NEW}
            and sum(len(r['fits']) for r in rows.values()) == 27
            and helper.sum_operations(complete['results']) == complete['operation_counts'], 'All27 fresh unique fits')
    require(len(old_rows) == len(old['results']) == 21 and set(old_rows) == {(s, a) for s in SEEDS for a in OLD}
            and sum(len(r['fits']) for r in old_rows.values()) == 39
            and helper.sum_operations(old['results']) == old['operation_counts'], 'Original full parent closure')
    expected, checkpoints, records = {}, [], {}

    def checkpoint(unit, arm, seed, folder, fresh):
        path = scoped(unit['selected_state'], folder)
        require(path == folder / f'{arm}_seed{seed}/member{unit["member"]}/selected.pt'
                and path.stat().st_size == unit['costs']['selected_checkpoint_bytes']
                and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000,
                'Canonical selected fit path/size/horizon')
        checkpoints.append({'path': str(path), 'sha256': sha(path), 'new': fresh, 'arm': arm, 'seed': seed,
                            'fit_member': unit['member'], 'selected_step': unit['selected_step']})

    for seed in SEEDS:
        for arm in NEW:
            row = rows[seed, arm]
            bank, use_sc = arm.startswith('shared4_'), arm != A
            units, members = (4 if arm == F4SC else 1), (1 if arm in (O1SC, F1SC) else 4)
            routes = 4 if bank else 1
            objective = 'gncl_half' if arm in (A, AB) else 'own_only'
            require(row['serving'] == 'probability_mean' and [u['member'] for u in row['fits']] == list(range(units)),
                    'Exact fresh bank/body identity')
            for unit in row['fits']:
                starts = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[s, s + cfg['factor_seed_offset'], s + cfg['dropout_seed_offset']] for s in starts]
                        and unit['learning_objective'] == objective and unit['within_route_supcon'] is use_sc
                        and unit['supcon_coefficient'] == (.05 if use_sc else 0.)
                        and unit['supcon_temperature'] == (.2 if use_sc else None)
                        and unit['preclassifier_width'] == 128, 'Loss/seed/representation identity')
                n = unit['completed_updates']
                require(unit['operation_counts'] == {'updates': n, 'backwards': n, 'adam_steps': n,
                        'train_probability_pool_loss_evaluations': n if bank else 0,
                        'train_probability_pool_backward_evaluations': n if arm in (A, AB) else 0,
                        'supcon_batched_gram_calls': n if use_sc else 0,
                        'supcon_route_grams': routes * n if use_sc else 0,
                        'supcon_normalized_coordinates': routes * 580 * 128 * n if use_sc else 0,
                        'supcon_cosine_gram_entries': routes * 580 * 580 * n if use_sc else 0,
                        'supcon_gram_multiply_accumulates': routes * 580 * 580 * 128 * n if use_sc else 0,
                        'supcon_anchor_rows': routes * 580 * n if use_sc else 0,
                        'train_fullgraph_native_trajectories': routes * n,
                        'valid_selection_fullgraph_native_trajectories': routes * n,
                        'selected_valid_fullgraph_native_trajectories': routes,
                        'native_graph_block_calls': 2 * routes * (2 * n + 1)}, 'Complete native/Gram work accounting')
                checkpoint(unit, arm, seed, output, True)
            path = output / f'{arm}_seed{seed}/selected_VALID.npz'
            expected[seed, arm] = {'path': str(path), 'sha256': sha(path), 'members': members, 'new': True}
            records[seed, arm] = row
    refs = {(r['seed'], r['arm']): r for r in binding['banks']}
    require(len(refs) == len(binding['banks']) == 15 and set(refs) == {(s, a) for s in SEEDS for a in REFS},
            'All15 own-only anchor descriptors')
    for seed in SEEDS:
        for arm in REFS:
            bound, row = refs[seed, arm], old_rows[seed, arm]
            units, members = (4 if 'genuine_I4' in arm else 1), (1 if arm in (O1, F1) else 4)
            routes = 4 if arm == BASE else 1
            require(bound['members'] == members and row['serving'] == 'probability_mean'
                    and [u['member'] for u in row['fits']] == list(range(units))
                    and [u['selected_state'] for u in row['fits']] == bound['checkpoints'], 'All33 selected anchor fit identities')
            for unit in row['fits']:
                starts = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[s, s + cfg['factor_seed_offset'], s + cfg['dropout_seed_offset']] for s in starts],
                        'Original native/factor/dropout seeds')
                checkpoint(unit, arm, seed, old_output, False)
            path = scoped(bound['archive'], old_output)
            require(path == old_output / f'{arm}_seed{seed}/selected_VALID.npz' and sha(path) == bound['archive_sha256'], 'Pinned anchor archive')
            expected[seed, arm] = {'path': str(path), 'sha256': bound['archive_sha256'], 'members': members, 'new': False}
            records[seed, arm] = row
    require(len(expected) == len(records) == 33 and len(checkpoints) == len({r['path'] for r in checkpoints}) == 60,
            'All18fresh/15anchor banks and27fresh/33anchor selected fits before arrays')
    context = {'cfg': cfg, 'freeze': freeze, 'admission': admission, 'reference_admission': reference_admission,
               'qualification': qualification, 'owner_start': start, 'owner': owner, 'old_owner': old_owner,
               'binding': binding, 'complete': complete, 'old_complete': old,
               'complete_sha256': complete_sha, 'freeze_sha256': sha(here / 'FREEZE.json'),
               'owner_end_sha256': sha(here / 'OWNER_END.json')}
    return context, expected, checkpoints, records


def payloads(np, expected):
    arrays, ids, labels = {}, None, None
    for arm in ARMS:
        for seed in SEEDS:
            item = expected[seed, arm]
            require(sha(item['path']) == item['sha256'], 'Admitted archive changed')
            with np.load(item['path'], allow_pickle=False) as archive:
                require(set(archive.files) == BASE_KEYS, 'Exact native six-key selected schema')
                a = {k: archive[k].copy() for k in archive.files}
            require(sha(item['path']) == item['sha256'], 'Archive changed during read')
            require(a['ids'].shape == a['y'].shape == (5274,) and a['ids'].dtype == a['y'].dtype == np.int64, 'Ordered VALID roles')
            if ids is None:
                ids, labels = a['ids'], a['y']
                require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10))
                        and {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in [('ids', ids), ('labels', labels)]} == VALID_HASHES,
                        'Frozen full VALID identity')
            require(np.array_equal(ids, a['ids']) and np.array_equal(labels, a['y']), 'Identical rows/labels for all33 banks')
            raw, pool, member, pooled = [a[k] for k in ('raw_logits', 'probability_mean', 'member_errors', 'pooled_errors')]
            require(raw.dtype == pool.dtype == np.float32 and raw.shape == (item['members'], 5274, 10) and pool.shape == (5274, 10)
                    and np.isfinite(raw).all() and np.isfinite(pool).all() and (pool >= 0).all() and (pool <= 1).all()
                    and member.dtype == pooled.dtype == np.bool_ and member.shape == (item['members'], 5274) and pooled.shape == (5274,)
                    and np.array_equal(pool.argmax(1) != labels, pooled), 'Archived float32 prediction/error authority')
            arrays[seed, arm] = a
    require(len(arrays) == 33, 'All33 schemas before scoring/calibration')
    return arrays, ids, labels


def score(helper, np, archive):
    value = helper.score(archive)
    truth = np.eye(10, dtype=np.float64)[archive['y']]
    raw = archive['raw_logits'].astype(np.float64)
    log_probs = raw - helper.logsumexp(raw, -1)[..., None]
    member_brier = ((np.exp(log_probs) - truth[None]) ** 2).sum(-1)
    pool_brier = ((archive['probability_mean'].astype(np.float64) - truth) ** 2).sum(-1)
    quality = value['quality']
    quality.update(pooled_brier=float(pool_brier.mean()), member_brier=member_brier.mean(1).tolist(),
                   mean_member_brier=float(member_brier.mean()), worst_member_brier=float(member_brier.mean(1).max()),
                   worst_member_nll=max(quality['member_nll']))
    for row in value['classes']:
        mask = archive['y'] == row['class']
        row.update(pooled_brier=float(pool_brier[mask].mean()), member_brier=member_brier[:, mask].mean(1).tolist(),
                   mean_member_brier=float(member_brier[:, mask].mean()),
                   worst_member_brier=float(member_brier[:, mask].mean(1).max()), worst_member_nll=max(row['member_nll']))
    return value


def contrast(helper, values, candidate, reference, labels, scope='full_learning_procedures'):
    value = helper.contrast(values, candidate, reference, labels)
    for metric in ('pooled_brier', 'mean_member_brier', 'worst_member_brier', 'worst_member_nll'):
        value['quality_deltas'][metric] = helper.paired(values[s, candidate]['quality'][metric] - values[s, reference]['quality'][metric] for s in SEEDS)
    for row in value['classes']:
        k = row['class']
        for metric in ('pooled_brier', 'mean_member_brier', 'worst_member_brier', 'worst_member_nll'):
            row[metric] = helper.paired(values[s, candidate]['classes'][k][metric] - values[s, reference]['classes'][k][metric] for s in SEEDS)
    if value['members'] is not None:
        for row in value['members']:
            m = row['member']
            row['brier'] = helper.paired(values[s, candidate]['quality']['member_brier'][m] - values[s, reference]['quality']['member_brier'][m] for s in SEEDS)
    value['comparison_scope'] = scope
    value['error_categories_are_descriptive_not_causal'] = True
    value['scope_explanation'] = ('The same native selected checkpoint/member bank is fixed; only the equal-policy OOF temperature readout changes. No new member alternative is acquired.'
                                  if scope == 'same_selected_state_calibration'
                                  else 'Different arms are separately fitted, stopped and selected. Repairs and coverage changes compare full procedures, not an isolated same-state gradient or serving mechanism.')
    value['coverage_category_note'] = 'Existing-member-alternative-served means both bank coverages are true; cross-procedure counts do not certify the same correct route/vector or exclusive pooling causality.'
    return value


def score_raw(helper, np, arrays, records, context, labels):
    raw, diagnostics = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            a, row = arrays[seed, arm], records[seed, arm]
            value = score(helper, np, a)
            raw[seed, arm] = value
            require(value['counts']['pooled_correct'] == round(row['valid']['accuracy'] * 5274)
                    and value['counts']['coverage'] == row['valid']['correct_alternative_count']
                    and value['counts']['lost_correct_alternatives'] == row['valid']['coverage_lost_in_pooling']
                    and value['counts']['member_correct'] == [round(v * 5274) for v in row['valid']['member_accuracy']],
                    'Archived integer decisions agree with complete selected records')
            raw64 = a['raw_logits'].astype(np.float64)
            probability64 = np.exp(raw64 - helper.logsumexp(raw64, -1)[..., None])
            diagnostics[seed, arm] = {
                'current_bank_common_strict_rival': strict_counts(np, strict_mask(np, a['raw_logits'], labels), value['P'], labels),
                'fixed_original_strict_rival_cohorts': {ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS},
                'FP64_raw_pool_argmax_disagreements_with_archived_errors': int(((probability64.mean(0).argmax(1) != labels) != a['pooled_errors']).sum()),
                'raw_logit_argmax_disagreements_with_archived_member_errors': int(((a['raw_logits'].argmax(-1) != labels) != a['member_errors']).sum()),
                'stable_NLL_minus_original_reported': value['quality']['pooled_nll'] - row['valid']['nll'],
                'scope': 'Selected full-procedure output; original raw float32 error flags remain integer authority.'}
    saved = {(r['seed'], r['arm']): r['versus'] for r in context['complete']['comparisons']}
    require(len(saved) == len(context['complete']['comparisons']) == 18 and set(saved) == {(s, a) for s in SEEDS for a in NEW},
            'Complete fresh saved error roster')
    for (seed, arm), versus in saved.items():
        require(set(versus) == set(NEW), 'All fresh comparisons retained')
        for ref, counts in versus.items():
            a, b = raw[seed, arm]['P'], raw[seed, ref]['P']
            require(counts == {'repairs': int((~b & a).sum()), 'harms': int((b & ~a).sum())}, 'Raw saved repair/harm authority')
    return raw, diagnostics


def calibrate(helper, np, torch, operators, arrays, ids, labels):
    """All33 banks receive the same five fixed terminal fits before analysis."""
    start = time.perf_counter()
    generator = torch.Generator(device='cpu').manual_seed(11709)
    permutation = torch.randperm(len(labels), generator=generator)
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    fold_meta = {'seed': 11709, 'folds': 5, 'counts': [(folds == k).sum().item() for k in range(5)],
                 'assignment_sha256': hashlib.sha256(folds.numpy().tobytes()).hexdigest(),
                 'ordered_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                 'scope': 'Label-free fold assignment after native selection; encountered development, not whole-pipeline confirmation.'}
    y = torch.from_numpy(labels)
    scores, records, binaries, attempted = {}, [], [], 0
    for arm in ARMS:
        for seed in SEEDS:
            a = arrays[seed, arm]
            log_p = torch.from_numpy(a['raw_logits']).to(torch.float64).log_softmax(-1)
            probability = log_p.exp()
            context = torch.empty((len(log_p), len(labels), 0), dtype=torch.float64)
            log_pool = torch.full((5274, 10), float('nan'), dtype=torch.float64)
            log_members = torch.full(log_p.shape, float('nan'), dtype=torch.float64)
            fits = []
            for fold in range(5):
                fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                attempted += 1
                before, returned_record = time.perf_counter(), None
                try:
                    held, record = operators.fit_fold(torch, 'temperature_global', log_p, probability, context, y, fit_ids, held_ids)
                    returned_record = record
                    log_T = torch.tensor(record['state']['log_T'], dtype=torch.float64)
                    temperature = log_T.exp()
                    require(log_T.shape == (1,) and record['updates'] == 500 and record['parameters'] == 1
                            and torch.isfinite(temperature).all().item() and (temperature > 0).all().item(), 'One finite positive fixed final temperature')
                    scaled = (log_p[:, held_ids] / temperature.reshape(-1, 1, 1)).log_softmax(-1)
                    require(torch.isfinite(scaled).all().item() and torch.isfinite(held).all().item(), 'Finite fixed endpoint')
                    log_pool[held_ids], log_members[:, held_ids] = held, scaled
                    record.update(status='finite_fixed_endpoint', fold=fold, temperature=temperature.item(), seconds=time.perf_counter() - before)
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    record = {'status': 'failed_retained', 'fold': fold, 'error': str(error), 'fixed_budget_updates': 500,
                              'actual_updates_before_failure': returned_record['updates'] if returned_record is not None else None,
                              'returned_endpoint_record': returned_record, 'seconds': time.perf_counter() - before}
                fits.append(record)
            finite = torch.isfinite(log_pool).all().item() and torch.isfinite(log_members).all().item()
            record = {'arm': arm, 'seed': seed, 'status': 'finite_complete_five_fold' if finite else 'failed_retained',
                      'fits': fits, 'fit_seconds_sum': sum(f['seconds'] for f in fits), 'attempted_scalar_fits': 5}
            if finite:
                pool_array, member_array = log_pool.numpy(), log_members.numpy()
                compatible = {'raw_logits': member_array, 'y': labels, 'member_errors': a['member_errors'],
                              'pooled_errors': pool_array.argmax(1) != labels, 'probability_mean': np.exp(pool_array)}
                value = score(helper, np, compatible)
                scores[seed, arm] = value
                record.update(quality=value['quality'], counts=value['counts'], classes=value['classes'],
                              FP64_calibrated_member_argmax_disagreements_with_raw_archived_errors=int(((member_array.argmax(-1) != labels) != a['member_errors']).sum()),
                              fixed_original_strict_rival_cohorts={ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS},
                              member_top_class_authority='Raw archived member flags remain fixed. Positive temperature acquires no alternatives; precision/tie changes are separately diagnosed.')
            else:
                scores[seed, arm] = None
            stream = io.BytesIO()
            np.savez_compressed(stream, ids=ids, folds=folds.numpy(), member_log_probability=log_members.numpy(), pool_log_probability=log_pool.numpy())
            name, data = f'CALIBRATED_{arm}_seed{seed}_OOF.npz', stream.getvalue()
            record['OOF_archive'] = {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            binaries.append((name, data))
            records.append(record)
    require(attempted == 165 and len(records) == len(binaries) == 33 and all(len(r['fits']) == 5 for r in records),
            'All165 calibration endpoint attempts terminal before interpretation')
    successful = sum(f['status'] == 'finite_fixed_endpoint' for r in records for f in r['fits'])
    cost = {'attempted_scalar_fits': attempted, 'requested_updates_per_fit': 500, 'requested_fullbatch_scalar_updates': 82500,
            'successful_scalar_fits': successful, 'failed_scalar_fits': 165 - successful,
            'known_successful_fullbatch_scalar_updates': successful * 500,
            'failed_actual_update_counts_known': all(f.get('actual_updates_before_failure') is not None for r in records for f in r['fits'] if f['status'] == 'failed_retained'),
            'fit_seconds_sum': sum(r['fit_seconds_sum'] for r in records), 'wall_seconds': time.perf_counter() - start,
            'native_models_or_forwards_added': 0, 'no_retry_or_final_refit': True,
            'scope': 'Terminal failure does not complete500 updates; unknown failed work remains unknown. Fit sums are nested in wall time.'}
    return scores, records, binaries, fold_meta, cost


def aggregate(helper, values, rows):
    metrics = ('pooled_accuracy_pct', 'pooled_nll', 'pooled_brier', 'mean_member_accuracy_pct',
               'mean_member_nll', 'worst_member_accuracy_pct', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier')
    return {'mean_quality': {k: statistics.mean(v['quality'][k] for v in values) for k in metrics},
            'summed_three_seed_readout_counts': {k: sum(v['counts'][k] for v in values) for k in values[0]['counts'] if k != 'member_correct'},
            'acquisition_costs': helper.cost_summary(rows),
            'count_scope': 'Three dependent readouts of the same5274 development nodes; not independent node samples.'}


def interaction(helper, np, values, labels, readout):
    if any(values[s, a] is None for s in SEEDS for a in (BASE, A, B, AB)):
        return {'failed_retained': True, 'readout': readout, 'reason': 'A required calibrated bank has a terminal failure.'}
    quality = {}
    for metric in ('pooled_accuracy_pct', 'pooled_nll', 'pooled_brier', 'mean_member_accuracy_pct',
                   'mean_member_nll', 'worst_member_accuracy_pct', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier'):
        if metric == 'pooled_accuracy_pct':
            quality[metric] = helper.paired(100 * (values[s, AB]['counts']['pooled_correct'] - values[s, A]['counts']['pooled_correct']
                                                   - values[s, B]['counts']['pooled_correct'] + values[s, BASE]['counts']['pooled_correct']) / 5274 for s in SEEDS)
        else:
            quality[metric] = helper.paired(values[s, AB]['quality'][metric] - values[s, A]['quality'][metric]
                                           - values[s, B]['quality'][metric] + values[s, BASE]['quality'][metric] for s in SEEDS)
    errors = []
    for seed in SEEDS:
        p0, pa, pb, pab = [values[seed, arm]['P'] for arm in (BASE, A, B, AB)]
        ra, rb = ~p0 & pa, ~p0 & pb
        only_a, only_b, both = ra & ~rb, rb & ~ra, ra & rb
        new_ab = ~p0 & pab & ~pa & ~pb
        pattern = p0.astype(np.int64) * 8 + pa.astype(np.int64) * 4 + pb.astype(np.int64) * 2 + pab.astype(np.int64)

        def partition(mask):
            return {'nodes': int(mask.sum()), 'A_only_repairs': int((mask & only_a).sum()),
                    'B_only_repairs': int((mask & only_b).sum()), 'shared_A_B_repairs': int((mask & both).sum()),
                    'A_only_retained_by_AB': int((mask & only_a & pab).sum()),
                    'B_only_retained_by_AB': int((mask & only_b & pab).sum()),
                    'shared_repairs_retained_by_AB': int((mask & both & pab).sum()),
                    'new_AB_repairs_where_A_B_both_wrong': int((mask & new_ab).sum()),
                    'AB_harms_where_base_A_B_correct': int((mask & p0 & pa & pb & ~pab).sum()),
                    'all16_correctness_patterns_base_A_B_AB': np.bincount(pattern[mask], minlength=16).tolist(),
                    'AB_vs_base': helper.comparison(values[seed, AB], values[seed, BASE], labels) if mask.all() else None}
        full = partition(np.ones(len(labels), dtype=np.bool_))
        require(full['AB_vs_base']['repairs'] == full['A_only_retained_by_AB'] + full['B_only_retained_by_AB']
                + full['shared_repairs_retained_by_AB'] + full['new_AB_repairs_where_A_B_both_wrong'], 'Exact AB repair partition')
        errors.append({'seed': seed, **full, 'classes': [{'class': k, **partition(labels == k)} for k in range(10)]})
    classes = []
    for k in range(10):
        row = {'class': k}
        for metric in ('pooled_accuracy_pct', 'pooled_nll', 'pooled_brier'):
            row[metric] = helper.paired(values[s, AB]['classes'][k][metric] - values[s, A]['classes'][k][metric]
                                       - values[s, B]['classes'][k][metric] + values[s, BASE]['classes'][k][metric] for s in SEEDS)
        classes.append(row)
    return {'readout': readout, 'definition': 'AB-A-B+archived_shared_own_base', 'quality_interaction': quality,
            'class_interaction': classes, 'per_seed_error_partitions': errors,
            'scope': 'Full independently fitted/selected procedures; calibrated interaction includes each arm\'s fixed OOF calibration. These are descriptive repair intersections, not fixed-state ingredient interventions or a superiority gate.'}


def analyze(helper, np, raw, calibrated, labels):
    pairs = [(a, b) for a in NEW for b in ARMS if a != b]
    raw_contrasts = {a + '_minus_' + b: contrast(helper, raw, a, b, labels) for a, b in pairs}
    calibrated_contrasts = {a + '_minus_' + b: contrast(helper, calibrated, a, b, labels)
                            if all(calibrated[s, a] is not None and calibrated[s, b] is not None for s in SEEDS)
                            else {'failed_retained': True, 'candidate': a, 'reference': b} for a, b in pairs}
    effects = {}
    for arm in ARMS:
        if all(calibrated[s, arm] is not None for s in SEEDS):
            values = {(s, 'raw'): raw[s, arm] for s in SEEDS}
            values.update({(s, 'calibrated'): calibrated[s, arm] for s in SEEDS})
            effects[arm] = contrast(helper, values, 'calibrated', 'raw', labels, 'same_selected_state_calibration')
        else:
            effects[arm] = {'failed_retained': True}
    required = {A: {BASE: .2, F1: .2, F4: .2, O4: .2},
                B: {BASE: .2, F1SC: .2, F4SC: .2, F4: .2, O4: .2},
                AB: {BASE: .2, F1SC: .2, F4SC: .2, F4: .2, O4: .2, A: .1, B: .1}}
    cells = {}
    for arm, references in required.items():
        accuracy, protection = {}, {}
        for ref, threshold in references.items():
            stat = raw_contrasts[arm + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
            standard = threshold == .2
            flags = {'required_mean_gain_pp': threshold, 'paired_raw_accuracy_pp': stat,
                     'mean_at_least_threshold': stat['mean'] >= threshold,
                     'all_seeds_nonnegative': stat['nonnegative_seed_count'] == 3,
                     'at_least_two_positive_required': standard,
                     'positive_seed_requirement_pass': stat['positive_seed_count'] >= 2 if standard else True}
            flags['pass'] = flags['mean_at_least_threshold'] and flags['all_seeds_nonnegative'] and flags['positive_seed_requirement_pass']
            accuracy[ref] = flags
            value = calibrated_contrasts[arm + '_minus_' + ref]
            if value.get('failed_retained'):
                protection[ref] = {'finite': False, 'pass': False}
            else:
                nll, acc = [value['quality_deltas'][k] for k in ('pooled_nll', 'pooled_accuracy_pct')]
                flags = {'finite': True, 'paired_calibrated_NLL_deterioration': nll, 'paired_calibrated_accuracy_pp': acc,
                         'mean_NLL_deterioration_at_most_0_02': nll['mean'] <= .02,
                         'each_seed_NLL_deterioration_at_most_0_05': nll['max'] <= .05,
                         'calibrated_accuracy_each_seed_nonnegative': acc['nonnegative_seed_count'] == 3}
                flags['pass'] = all(flags[k] for k in ('mean_NLL_deterioration_at_most_0_02',
                                                       'each_seed_NLL_deterioration_at_most_0_05', 'calibrated_accuracy_each_seed_nonnegative'))
                protection[ref] = flags
        raw_pass, protected = all(v['pass'] for v in accuracy.values()), all(v['pass'] for v in protection.values())
        cells[arm] = {'raw_accuracy_criteria': accuracy, 'raw_accuracy_screen_pass': raw_pass,
                      'equal_policy_calibrated_confidence_criteria': protection, 'calibrated_confidence_protection_pass': protected,
                      'joint_accuracy_confidence_development_clue': raw_pass and protected,
                      'accuracy_only_clue_with_confidence_tradeoff': raw_pass and not protected,
                      'member_quality_is_diagnostic_not_veto': True}
    priority = (AB, B, A)
    policy = {'cells': cells, 'predeclared_priority': priority,
              'first_raw_accuracy_qualifier': next((a for a in priority if cells[a]['raw_accuracy_screen_pass']), None),
              'first_joint_accuracy_confidence_qualifier': next((a for a in priority if cells[a]['joint_accuracy_confidence_development_clue']), None),
              'raw_NLL_and_Brier_tradeoffs_preserved': True, 'automatic_confirmation_launch': False,
              'scope': 'One-backbone SAGE encountered development selection among A/B/AB; no GCN/GAT requirement, no novelty/unused-confirmation/acceptance claim. Confidence failure does not erase raw accuracy.'}
    results = {'raw_full_procedure_contrasts': raw_contrasts, 'calibrated_full_procedure_contrasts': calibrated_contrasts,
               'calibration_minus_raw_same_selected_state': effects,
               'raw_2x2_interaction': interaction(helper, np, raw, labels, 'raw'),
               'calibrated_2x2_interaction': interaction(helper, np, calibrated, labels, 'OOF_calibrated')}
    return results, policy


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
    helper = load_source('SAGE_immutable_count_math', root / MATH, MATH_SHA)
    context, expected, checkpoints, records = preflight(root, helper, inputs)
    # Numerical imports and every selected array are downstream of complete custody.
    import numpy as np
    helper.np = np
    arrays, ids, labels = payloads(np, expected)
    raw, diagnostics = score_raw(helper, np, arrays, records, context, labels)
    import torch
    torch.set_num_threads(2)
    operators = load_source('SAGE_fixed_temperature', root / CALIBRATOR, inputs['research_files'][CALIBRATOR])
    calibrated, calibration_records, binaries, folds, calibration_costs = calibrate(helper, np, torch, operators, arrays, ids, labels)
    # No contrast, interaction, criterion or continuation priority before all165 terminal attempts.
    contrasts, policy = analyze(helper, np, raw, calibrated, labels)
    aggregates = {'raw': {a: aggregate(helper, [raw[s, a] for s in SEEDS], [records[s, a] for s in SEEDS]) for a in ARMS},
                  'calibrated': {a: aggregate(helper, [calibrated[s, a] for s in SEEDS], [records[s, a] for s in SEEDS])
                                 if all(calibrated[s, a] is not None for s in SEEDS) else {'failed_retained': True} for a in ARMS}}
    groups = [{'arm': arm, 'seed': seed, 'raw': {k: raw[seed, arm][k] for k in ('quality', 'counts', 'classes')},
               'diagnostics': diagnostics[seed, arm], 'original_group_record': records[seed, arm]} for arm in ARMS for seed in SEEDS]
    costs = {'new27_native_fits': helper.cost_summary(context['complete']['results']),
             'reused33_selected_reference_fits': helper.cost_summary([records[s, a] for s in SEEDS for a in REFS]),
             'original39_parent_context_not_additive': helper.cost_summary(context['old_complete']['results']),
             'calibration_scalar_work': calibration_costs, 'new_owner': context['owner'], 'original_owner': context['old_owner']}
    report = {'complete': True, 'backbone': 'SAGE', 'new_banks': 18, 'new_native_fits': 27,
              'new_native_trajectories_per_complete_roster_forward': 54, 'reused_reference_banks': 15, 'reused_reference_fits': 33,
              'joined_banks': 33, 'joined_selected_fit_records': 60, 'seeds': SEEDS, 'roster': ARMS, 'TEST_access': False,
              'frozen_policy': policy, 'aggregates': aggregates, 'groups_detail': groups, 'contrasts': contrasts,
              'calibration_folds': folds, 'calibration_records': calibration_records, 'costs': costs,
              'source_custody': context, 'selected_archive_custody': list(expected.values()), 'checkpoint_custody': checkpoints,
              'ordered_VALID_hashes': VALID_HASHES, 'analysis_source_sha256': sha(Path(__file__)),
              'immutable_count_math_sha256': MATH_SHA, 'input_bindings': inputs,
              'authority': 'Raw integer decisions use archived float32 error flags. Stable FP64 NLL and Brier remain explicit. Calibrated pooled decisions use fixed OOF FP64 outputs; unchanged raw member flags determine member accuracy/coverage, with precision/tie discrepancies diagnosed.',
              'interpretation_limits': [
                  'All18fresh/15anchor banks and27fresh/33anchor fit identities close before selected arrays. All165 scalar endpoint attempts are terminal before contrasts/criteria/interaction; failed endpoints remain failed.',
                  'Cross-arm raw/calibrated repairs compare full fitted/stopped/selected procedures. They do not isolate one same-state training or serving intervention.',
                  'Calibration-minus-raw comparisons use each exact fixed selected bank. Positive temperature supplies no new member alternatives; numerical argmax/tie changes are separate diagnostics.',
                  'The2x2 interaction is AB-A-B+archived same-custody shared-own base. Repair intersections and interaction do not rescue a failed final-accuracy criterion.',
                  'Member competence remains diagnostic rather than an independent veto. Raw accuracy and proper confidence protection are separate results.',
                  'Five-fold calibration is encountered post-selection development, not untouched validation or whole-pipeline cross-fitting. A future pipeline requires a separately fixed CAL role.',
                  'Three paired optimizer seeds/readouts on one graph are not independent graph/node replications; intervals and sign-flip sensitivities are descriptive.',
                  'Selection among qualifyingAB/B/A is predeclared development selection. One backbone is sufficient; no required GCN/GAT transfer or changed old frozen criteria.',
                  'GNCL/SupCon/BatchEnsemble/native SAGE/calibration are attributed known ingredients; no novelty, guaranteed diversity, heldout, universal-transfer or acceptance claim.',
                  'This reader loads no native model, data loader, checkpoint state or TEST and adds no native forward. Checkpoint byte hashes are current closure snapshots, not earlier historical SHA certificates.']}
    documents = [('SAGE_POLICY_SUMMARY.json', {'complete': True, 'backbone': 'SAGE', 'new_banks': 18, 'new_native_fits': 27,
                  'reused_banks': 15, 'reused_selected_fits': 33, 'frozen_policy': policy,
                  'all_fixed_calibration_fits_attempted': 165, 'calibration_costs': calibration_costs})]
    documents += chunk_documents('SAGE_ARM_AGGREGATES', [{'readout': kind, 'arm': arm, 'aggregate': value}
                                                         for kind, bank in aggregates.items() for arm, value in bank.items()])
    documents += chunk_documents('SAGE_ARM_DETAILS', groups)
    documents += chunk_documents('CALIBRATION_FIT_DETAILS', calibration_records)
    for category, values in contrasts.items():
        if category.endswith('_interaction'):
            documents.append(('SAGE_' + category + '.json', values))
        else:
            documents += chunk_documents('SAGE_' + category, [{'key': name, 'contrast': value} for name, value in values.items()])
    documents += chunk_documents('CHECKPOINT_CUSTODY', checkpoints)
    documents += chunk_documents('ARCHIVE_CUSTODY', [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS])
    prepared = [(report_path, encode(report))]
    prepared += [(report_path.parent / name, encode(value)) for name, value in documents]
    prepared += [(report_path.parent / name, data) for name, data in binaries]
    require(len({path for path, _ in prepared}) == len(prepared), 'Distinct deterministic output names')
    for path, data in prepared:
        require(not path.exists(), 'Every output path must be fresh')
        if path != report_path and path.suffix == '.json':
            require(len(data) < LIMIT, 'Bounded JSON output required')
    for item in list(expected.values()) + checkpoints:
        require(sha(item['path']) == item['sha256'], 'Admitted artifact changed before output')
    receipts = [{'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in prepared]
    summary = {'complete': True, 'backbone': 'SAGE', 'new_banks': 18, 'new_native_fits': 27,
               'original_reference_banks': 15, 'original_selected_fits': 33, 'joined_selected_fit_records': 60,
               'TEST_access': False, 'frozen_policy': policy, 'calibration_folds': folds, 'calibration_costs': calibration_costs,
               'raw_authority_preserved': True, 'all165_scalar_attempts_terminal_before_interpretation': True,
               'post_selection_OOF_scope': 'Encountered development, not unused whole-pipeline confirmation.',
               'repair_scope': 'Cross-arm full procedures; calibration-only effects at each fixed native selected state.',
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
