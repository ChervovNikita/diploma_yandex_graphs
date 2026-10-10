"""Complete sparse feature-kernel SAGE reader; stdlib until full custody.

No native model, loader, checkpoint deserialization, forward, selector or TEST.
Exactly75 admitted original calibration endpoints are reused without fitting;
90 fresh fixed endpoint attempts finish before any score or contrast.
"""
import argparse
import copy
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
PILOT = 'SAGE_sparse_feature_kernel_pilot_root_20261010_v1'
LEARNING = 'native_SAGE_sparse_feature_kernel_source_20261010_v1'
PRIOR = 'SAGE_GNCL_SupCon_2x2_pilot_root_20261010_v1'
PRIOR_READER = 'SAGE_GNCL_SupCon_2x2_complete_reader_source_20261010_v2/analysis.py'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
REFERENCE = 'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
REFERENCE_SHA = 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
CALIBRATOR = 'common_wrapper_graph_reliability_source_20261010_v2/operators.py'
SEEDS = (7301, 7403, 7507)
NEW = ('shared4_private_kernel', 'shared4_common_kernel', 'shared4_fixed_feature_union',
       'ordinary_M1_feature_kernel', 'factorized_M1_feature_kernel', 'genuine_factorized_I4_feature_kernel')
REFS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
OLD = REFS + ('separable_equal_size', 'exchange')
ARMS = NEW + REFS
PRIVATE, COMMON, FIXED, O1K, F1K, F4K = NEW
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


def retained_json(value):
    if isinstance(value, dict):
        return {k: retained_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [retained_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    return value


def check_group_costs(row):
    units, costs = row['fits'], row['costs']
    keys = set(units[0]['costs'])
    require(all(set(u['costs']) == keys for u in units), 'Consistent native fit cost fields')
    for unit in units:
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                    for v in unit['costs'].values()), 'Finite nonnegative complete native costs')
    require(set(costs) == keys | {'selected_pool_seconds', 'selected_serving_readout_seconds'}, 'Complete group cost schema')
    for key in keys:
        expected = max(u['costs'][key] for u in units) if 'peak' in key else sum(u['costs'][key] for u in units)
        require(costs[key] == expected, 'Original sum/peak group accounting: ' + key)
    require(math.isfinite(costs['selected_pool_seconds']) and costs['selected_pool_seconds'] >= 0
            and costs['selected_serving_readout_seconds'] == costs['selected_forward_and_metrics_seconds'] + costs['selected_pool_seconds'],
            'Pool timing and inclusive selected serving accounting')


def admit_reuse(root, here, freeze, inputs, helper, expected, reference_admission):
    """Close all75 original endpoints and prediction bytes before any array."""
    path = here / 'CALIBRATION_REUSE_ADMISSION.json'
    reuse, policy = read(path), read(here / 'CALIBRATION_POLICY.json')
    for name in ('CALIBRATION_REUSE_ADMISSION.json', 'CALIBRATION_POLICY.json'):
        require(helper.freeze_hash(freeze, PILOT + '/' + name) == sha(here / name), 'Reuse and equal law frozen before acquisition')
    require(reference_admission['calibration_reuse_admission'] == reuse
            and reuse['admitted'] is True and reuse['reused_banks'] == 15
            and reuse['reused_scalar_endpoints'] == 75 and reuse['new_scalar_endpoints'] == 90
            and reuse['total_endpoints'] == 165 and reuse['new_scalar_steps'] == 45000
            and reuse['TEST_access'] is False and reuse['no_native_forward'] is True and reuse['no_fit_replay'] is True,
            'Exact root75-endpoint reuse admission;90 new endpoints only')
    expected_policy = {'temperature_parameter': 'one bank-global logT; positive exp(logT)', 'folds': 5,
        'fold_order': 'CPU randperm seed11709; position modulo5', 'dtype': 'float64 CPU', 'optimizer': 'Adam',
        'learning_rate': .01, 'steps_per_endpoint': 500, 'endpoint': 'final step', 'calibrated_banks': 33,
        'scalar_fits': 165, 'scalar_steps': 82500, 'operations_source': CALIBRATOR,
        'equally_applied_to_all_banks': True, 'TEST_access': False, 'all_VALID_refit': False,
        'scope': 'postselection encountered development', 'risk_rule': 'DECISION.md'}
    require(policy == expected_policy and reuse['policy_sha256'] == sha(here / 'CALIBRATION_POLICY.json')
            and reuse['prior_policy_sha256'] == sha(root / PRIOR / 'CALIBRATION_POLICY.json')
            and read(root / PRIOR / 'CALIBRATION_POLICY.json') == policy
            and reuse['calibration_operator_sha256'] == inputs['research_files'][CALIBRATOR]
            and reuse['prior_analysis_source_sha256'] == inputs['research_files'][PRIOR_READER], 'Same exact equal calibration law/source')
    admission = read(here / 'SCIENTIFIC_ADMISSION.json')
    require(admission['calibration_policy_sha256'] == reuse['policy_sha256']
            and admission['calibration_reuse_admission_sha256'] == sha(path)
            and admission['reference_admission_sha256'] == sha(here / 'REFERENCE_ADMISSION.json')
            and admission['source_sha256'] == inputs['research_files'][LEARNING + '/run_family.py']
            and admission['original_reference_banks'] == 15 and admission['original_selected_fits'] == 33
            and admission['all_banks_before_interpretation'] == 33
            and admission['fixed_calibration_fits'] == 165 and admission['reused_scalar_endpoints'] == 75
            and admission['new_scalar_endpoints'] == 90 and admission['no_fit_replay'] is True,
            'Scientific admission closes the whole33-bank/165-endpoint comparative procedure')
    prior = root / PRIOR
    end_path, summary_path = prior / 'READER_JOB_END_V1.json', prior / 'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json'
    require(sha(end_path) == reuse['prior_reader_end_sha256'] and sha(summary_path) == reuse['prior_complete_summary_sha256'],
            'Prior successful reader and complete summary custody')
    end, summary = read(end_path), read(summary_path)
    require(end['success'] is True and end['exit_code'] == 0 and end['TEST_access'] is False
            and summary['complete'] is True and summary['TEST_access'] is False
            and summary['original_reference_banks'] == 15 and summary['original_selected_fits'] == 33
            and summary['all165_scalar_attempts_terminal_before_interpretation'] is True
            and summary['calibration_folds']['assignment_sha256'] == reuse['fold_assignment_sha256'],
            'Complete prior result, finite original endpoints and fixed folds')
    prior_output = summary_path.parent
    prior_documents = {str(scoped(r['path'], prior_output)): r for r in [summary['full_report']] + summary['partitions']}
    require(len(prior_documents) == 1 + len(summary['partitions']), 'Unique prior output descriptors')
    used = []
    def admitted_document(name):
        document = prior_output / name
        receipt = prior_documents[str(document)]
        require(document.stat().st_size == receipt['bytes'] and sha(document) == receipt['sha256'], 'Prior extracted record bytes: ' + name)
        used.append({'path': str(document), 'bytes': receipt['bytes'], 'sha256': receipt['sha256']})
        return read(document)
    fit_document = admitted_document('CALIBRATION_FIT_DETAILS_part1.json')
    archive_document = admitted_document('ARCHIVE_CUSTODY_part1.json')
    require(fit_document['category'] == 'CALIBRATION_FIT_DETAILS' and archive_document['category'] == 'ARCHIVE_CUSTODY'
            and fit_document['offset'] == archive_document['offset'] == 0
            and fit_document['total_rows'] == len(fit_document['rows']) == 33
            and archive_document['total_rows'] == len(archive_document['rows']) == 33, 'Complete prior endpoint/archive record partitions')
    fit_index = {(r['seed'], r['arm']): r for r in fit_document['rows']}
    archive_index = {(r['seed'], r['arm']): r for r in archive_document['rows']}
    require(len(fit_index) == len(archive_index) == 33, 'Unique prior endpoint/archive bank indexes')
    rows = {(r['seed'], r['arm']): r for r in reuse['records']}
    require(len(rows) == len(reuse['records']) == 15 and set(rows) == {(s, a) for s in SEEDS for a in REFS}, 'All15 unchanged original banks')
    descriptors = []
    for arm in REFS:
        for seed in SEEDS:
            row, raw = rows[seed, arm], expected[seed, arm]
            require(row['raw_archive'] == archive_index[seed, arm]
                    and row['raw_archive']['path'] == raw['path'] and row['raw_archive']['sha256'] == raw['sha256']
                    and row['raw_archive']['members'] == raw['members'] and row['raw_archive']['new'] is False,
                    'Exact original source/config/role/selected-prediction binding')
            record = row['endpoint_record']
            require(record == fit_index[seed, arm] and record['arm'] == arm and record['seed'] == seed
                    and record['status'] == 'finite_complete_five_fold' and record['attempted_scalar_fits'] == 5
                    and len(record['fits']) == 5 and [f['fold'] for f in record['fits']] == list(range(5)), 'Admitted complete original five-fold terminal records')
            for fold, fit in enumerate(record['fits']):
                held_nodes = 1055 if fold < 4 else 1054
                require(fit['status'] == 'finite_fixed_endpoint' and fit['updates'] == 500 and fit['parameters'] == 1
                        and fit['fit_nodes'] == 5274 - held_nodes and fit['held_nodes'] == held_nodes
                        and fit['selected_endpoint'] == 'fixed_final_update_no_heldout_selector'
                        and set(fit['state']) == {'log_T'} and len(fit['state']['log_T']) == 1
                        and math.isfinite(fit['state']['log_T'][0]) and math.isfinite(fit['temperature']) and fit['temperature'] > 0
                        and math.isclose(math.exp(fit['state']['log_T'][0]), fit['temperature'], rel_tol=1e-12)
                        and [r['update'] for r in fit['trace']] == [1, 100, 200, 300, 400, 500]
                        and all(math.isfinite(r['fusion_fit_objective_before_update']) for r in fit['trace'])
                        and math.isfinite(fit['seconds']) and fit['seconds'] >= 0, 'Exact75 fixed scalar endpoints without replay')
            require(record['fit_seconds_sum'] == row['prior_fit_seconds'] == sum(f['seconds'] for f in record['fits']), 'Original scalar costs preserved')
            descriptor = record['OOF_archive']
            archive = scoped(row['calibrated_archive'], prior_output)
            require(archive == prior_output / f'CALIBRATED_{arm}_seed{seed}_OOF.npz'
                    and descriptor['name'] == archive.name and archive.stat().st_size == descriptor['bytes']
                    and sha(archive) == row['calibrated_sha256'] == descriptor['sha256'], 'Exact admitted original OOF prediction bytes')
            descriptors.append({'path': str(archive), 'bytes': descriptor['bytes'], 'sha256': descriptor['sha256'], 'seed': seed, 'arm': arm, 'new': False})
    require(len(descriptors) == len({r['path'] for r in descriptors}) == 15
            and sum(r['endpoint_record']['attempted_scalar_fits'] for r in rows.values()) == 75, 'All75 prior endpoints admitted before arrays')
    return reuse, descriptors, {'reader_end': {'path': str(end_path), 'sha256': sha(end_path)},
        'summary': {'path': str(summary_path), 'sha256': sha(summary_path)}, 'record_partitions': used,
        'policy_sha256': reuse['policy_sha256'], 'fold_assignment_sha256': reuse['fold_assignment_sha256'],
        'original_endpoint_scores_are_preserved': True, 'refitted_endpoints': 0}


def support_arrays(np, context):
    """Inspect saved support after all60 fit and75 reused endpoint records close."""
    item, meta = context['support_archive'], context['complete']['feature_support']
    require(sha(item['path']) == item['sha256'], 'Admitted support archive unchanged')
    with np.load(item['path'], allow_pickle=False) as archive:
        require(set(archive.files) == {'original_edge_index', 'union_edge_index'}, 'Exact saved support schema')
        original, union = archive['original_edge_index'].copy(), archive['union_edge_index'].copy()
    require(sha(item['path']) == item['sha256'], 'Support archive unchanged while reading')
    for name, edges, count in [('original', original, meta['original_directed_pairs']), ('union', union, meta['union_directed_pairs'])]:
        require(edges.dtype == np.int64 and edges.shape == (2, count) and edges.min() >= 0 and edges.max() < 11701
                and hashlib.sha256(edges.tobytes(order='C')).hexdigest() == meta[name + '_COO_int64_C_bytes_sha256'], 'Pinned support COO bytes: ' + name)
    original_keys, union_keys = original[1] * 11701 + original[0], union[1] * 11701 + union[0]
    require(len(np.unique(original_keys)) == len(original_keys) and (np.diff(union_keys) > 0).all(), 'Unique original and sorted-unique union support')
    positions = np.searchsorted(union_keys, original_keys)
    require((positions < len(union_keys)).all() and np.array_equal(union_keys[positions], original_keys), 'Union preserves every original directed pair')
    require(int((original[0] == original[1]).sum()) == meta['original_self_loops']
            and int((union[0] == union[1]).sum()) == meta['union_self_loops'], 'Original self-loop preservation')
    degrees = np.bincount(union[1], minlength=11701)
    require(degrees.min() >= 20 and len(union_keys) - len(original_keys) == meta['new_directed_pairs'], 'Candidate coverage and genuine new paths')
    return {'original_directed_pairs': len(original_keys), 'union_directed_pairs': len(union_keys),
        'new_directed_pairs': len(union_keys) - len(original_keys), 'union_incoming_degree_min': int(degrees.min()),
        'union_incoming_degree_max': int(degrees.max()), 'union_incoming_degree_mean': float(degrees.mean()),
        'saved_support_checked': True, 'raw_feature_cosine_construction_replayed': False,
        'scope': 'Saved support custody, inclusion and counts only. The exact frozen acquisition source binds raw-cosine construction.'}


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
    require(header['completion_scope'] == 'fresh six-arm feature-support acquisition only'
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
    admitted_rows = reference_admission['checkpoint_records']
    require(len(admitted_rows) == 33 and all(set(('path', 'sha256', 'bytes', 'arm', 'seed', 'member', 'selected_step')).issubset(r) for r in admitted_rows),
            'All33 frozen reference checkpoint attestations')
    admitted_checkpoints = {str(scoped(r['path'], old_output)): r for r in admitted_rows}
    descriptor_paths = {str(scoped(p, old_output)) for bank in binding['banks'] for p in bank['checkpoints']}
    require(len(admitted_checkpoints) == len(admitted_rows) == len(descriptor_paths) == 33
            and set(admitted_checkpoints) == descriptor_paths, 'Exact unique frozen33-checkpoint path roster')

    def checkpoint(unit, arm, seed, folder, fresh):
        path = scoped(unit['selected_state'], folder)
        require(path == folder / f'{arm}_seed{seed}/member{unit["member"]}/selected.pt'
                and path.stat().st_size == unit['costs']['selected_checkpoint_bytes']
                and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000,
                'Canonical selected fit path/size/horizon')
        digest = sha(path)
        if not fresh:
            attested = admitted_checkpoints[str(path)]
            require(str(scoped(attested['path'], old_output)) == str(path)
                    and attested['sha256'] == digest and attested['bytes'] == path.stat().st_size
                    and attested['arm'] == arm and attested['seed'] == seed
                    and attested['member'] == unit['member'] and attested['selected_step'] == unit['selected_step'],
                    'Frozen checkpoint path/hash/bytes/arm/seed/member/selected-step agreement')
        checkpoints.append({'path': str(path), 'sha256': digest, 'new': fresh, 'arm': arm, 'seed': seed,
                            'fit_member': unit['member'], 'selected_step': unit['selected_step'], 'bytes': path.stat().st_size})

    support = complete['feature_support']
    require(read(output / 'FEATURE_SUPPORT.json') == support, 'Support metadata bound by whole complete closure')
    support_path = output / 'FEATURE_SUPPORT.npz'
    require(support_path.stat().st_size == support['support_archive_bytes']
            and sha(support_path) == support['support_file_sha256'], 'Frozen acquisition support archive custody')
    n, edge_count = 11701, support['union_directed_pairs']
    require(support['nodes'] == n and support['raw_feature_width'] == 300 and support['candidate_k'] == 20
            and support['candidate_directed_pairs'] == n * 20
            and support['new_directed_pairs'] == edge_count - support['original_directed_pairs'] > 0
            and support['union_self_loops'] == support['original_self_loops']
            and support['raw_cosine_dot_MACs'] == n * n * 300
            and support['raw_cosine_scores'] == support['stable_sort_candidate_entries'] == n * n
            and support['sorted_rows'] == n and support['labels_or_predictions_used'] is False
            and support['construction'] == 'incoming top20 raw cosine; self excluded; source-ID ties; sorted union',
            'One frozen label-free top20 support; real path acquisition and exact preparation work')
    require(all(math.isfinite(support[k]) and support[k] >= 0 for k in
                ('inclusive_preparation_seconds', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')),
            'Finite retained preparation cost')
    provider = qualification['native_provider']
    provider_path = scoped(provider['path'], REPO)
    require(provider['class'] == 'torch_geometric.nn.conv.sage_conv.SAGEConv'
            and provider_path.stat().st_size == provider['bytes'] and sha(provider_path) == provider['sha256'],
            'Qualified installed native SAGE provider bytes')
    require(helper.source_hash(complete, LEARNING + '/feature_graph.py')
            == inputs['research_files'][LEARNING + '/feature_graph.py'], 'Exact feature-support and kernel source')
    for seed in SEEDS:
        for arm in NEW:
            row = rows[seed, arm]
            bank = arm.startswith('shared4_')
            units, members = (4 if arm == F4K else 1), (1 if arm in (O1K, F1K) else 4)
            routes = 4 if bank else 1
            mode = 'common' if arm == COMMON else 'fixed' if arm == FIXED else 'private'
            metric_rows = routes if mode == 'private' else 1
            require(row['serving'] == 'probability_mean' and [u['member'] for u in row['fits']] == list(range(units))
                    and read(output / f'{arm}_seed{seed}/RESULT.json') == row, 'Exact fresh bank/body record identity')
            for unit in row['fits']:
                starts = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[x, x + cfg['factor_seed_offset'], x + cfg['dropout_seed_offset']] for x in starts]
                        and unit['learning_objective'] == 'mean_own_CE', 'Unchanged starts and original mean own CE')
                updates = unit['completed_updates']
                require(unit['operation_counts'] == {'updates': updates, 'backwards': updates, 'adam_steps': updates,
                        'train_fullgraph_native_trajectories': routes * updates,
                        'valid_selection_fullgraph_native_trajectories': routes * updates,
                        'selected_valid_fullgraph_native_trajectories': routes,
                        'native_graph_block_calls': 2 * routes * (2 * updates + 1)}, 'Exact original native work accounting')
                kernel, contexts = unit['feature_kernel'], 2 * updates + 1
                learned = mode != 'fixed'
                require(kernel['mode'] == mode and kernel['members'] == routes and kernel['metric_rows'] == metric_rows
                        and kernel['rank'] == 16 and kernel['temperature'] == .2
                        and kernel['normalize_eps'] == 1e-12 and kernel['metric_floor'] == 1e-4
                        and kernel['kernel_seed'] == starts[0] + 5000011
                        and kernel['native_PyG_source'] == provider
                        and kernel['registered_parameters'] == (300 * 16 + metric_rows * 16 if learned else 0)
                        and kernel['projection_shared_within_bank'] is learned
                        and kernel['weights_reused_across_native_layers'] == 2
                        and kernel['score_inputs'] == 'raw X only; no labels, predictions, learned native hidden inputs',
                        'Fixed private/common/uniform kernel parameter and seed identity')
                require(kernel['counters'] == {
                    'contexts': contexts, 'projection_calls': contexts if learned else 0,
                    'projection_MACs': contexts * n * 300 * 16 if learned else 0,
                    'metric_route_rows': contexts * metric_rows if learned else 0,
                    'normalized_coordinates': contexts * metric_rows * n * 16 if learned else 0,
                    'cosine_pairs': contexts * metric_rows * edge_count if learned else 0,
                    'cosine_MACs': contexts * metric_rows * edge_count * 16 if learned else 0,
                    'row_softmax_entries': contexts * metric_rows * edge_count if learned else 0,
                    'fixed_uniform_entries': contexts * edge_count if not learned else 0,
                    'weighted_mean_calls': contexts * routes * 2,
                    'incoming_edge_visits': contexts * routes * 2 * edge_count,
                    'weighted_message_coordinates': contexts * routes * 2 * edge_count * 128},
                    'All factual kernel contexts, projections, scores, softmax and weighted-message work')
                checkpoint(unit, arm, seed, output, True)
                require(read(scoped(unit['selected_state'], output).parent / 'RESULT.json') == unit,
                        'Selected fresh unit result equals complete record')
            check_group_costs(row)
            path = output / f'{arm}_seed{seed}/selected_VALID.npz'
            expected[seed, arm] = {'path': str(path), 'sha256': sha(path), 'members': members, 'new': True}
            records[seed, arm] = row
    kernel_units = [u['feature_kernel']['counters'] for r in complete['results'] for u in r['fits']]
    require({k: sum(r[k] for r in kernel_units) for k in kernel_units[0]}
            == complete['feature_kernel_operation_counts'], 'Complete27-fit kernel work ledger')
    refs = {(r['seed'], r['arm']): r for r in binding['banks']}
    require(len(refs) == len(binding['banks']) == 15 and set(refs) == {(s, a) for s in SEEDS for a in REFS},
            'All15 own-only anchor descriptors')
    for seed in SEEDS:
        for arm in REFS:
            bound, row = refs[seed, arm], old_rows[seed, arm]
            require(read(old_output / f'{arm}_seed{seed}/RESULT.json') == row, 'Original bank result unchanged')
            check_group_costs(row)
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
                require(read(scoped(unit['selected_state'], old_output).parent / 'RESULT.json') == unit, 'Original selected unit result unchanged')
            path = scoped(bound['archive'], old_output)
            require(path == old_output / f'{arm}_seed{seed}/selected_VALID.npz' and sha(path) == bound['archive_sha256'], 'Pinned anchor archive')
            expected[seed, arm] = {'path': str(path), 'sha256': bound['archive_sha256'], 'members': members, 'new': False}
            records[seed, arm] = row
    require(len(expected) == len(records) == 33 and len(checkpoints) == len({r['path'] for r in checkpoints}) == 60,
            'All18fresh/15anchor banks and27fresh/33anchor selected fits before arrays')
    reuse, reused_archives, prior_custody = admit_reuse(root, here, freeze, inputs, helper, expected, reference_admission)
    context = {'cfg': cfg, 'freeze': freeze, 'admission': admission, 'reference_admission': reference_admission,
               'qualification': qualification, 'owner_start': start, 'owner': owner, 'old_owner': old_owner,
               'binding': binding, 'complete': complete, 'old_complete': old,
               'complete_sha256': complete_sha, 'freeze_sha256': sha(here / 'FREEZE.json'),
               'owner_end_sha256': sha(here / 'OWNER_END.json'),
               'support_archive': {'path': str(support_path), 'sha256': support['support_file_sha256'], 'bytes': support['support_archive_bytes']},
               'calibration_reuse_admission': reuse, 'reused_OOF_archives': reused_archives, 'prior_calibration_custody': prior_custody}
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



def calibrate(np, torch, operators, arrays, ids, labels, context):
    """Reuse exact75 admitted endpoints; all90 new attempts terminal before scores."""
    start = time.perf_counter()
    generator = torch.Generator(device='cpu').manual_seed(11709)
    permutation = torch.randperm(len(labels), generator=generator)
    folds = torch.empty(len(labels), dtype=torch.long)
    folds[permutation] = torch.arange(len(labels)) % 5
    fold_array = folds.numpy()
    fold_meta = {'seed': 11709, 'folds': 5, 'counts': [(folds == k).sum().item() for k in range(5)],
        'assignment_sha256': hashlib.sha256(fold_array.tobytes()).hexdigest(),
        'ordered_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
        'scope': 'Label-free fixed folds after native selection; encountered development, not whole-pipeline confirmation.'}
    require(fold_meta['assignment_sha256'] == context['calibration_reuse_admission']['fold_assignment_sha256'], 'Exact original fold prediction binding')
    payloads, record_index, binaries = {}, {}, []
    reuse_index = {(r['seed'], r['arm']): r for r in context['calibration_reuse_admission']['records']}
    for arm in REFS:
        for seed in SEEDS:
            row = reuse_index[seed, arm]
            item = next(r for r in context['reused_OOF_archives'] if r['seed'] == seed and r['arm'] == arm)
            require(sha(item['path']) == item['sha256'], 'Admitted original OOF unchanged')
            with np.load(item['path'], allow_pickle=False) as archive:
                require(set(archive.files) == {'ids', 'folds', 'member_log_probability', 'pool_log_probability'}, 'Exact original OOF schema')
                reused = {k: archive[k].copy() for k in archive.files}
            require(sha(item['path']) == item['sha256'], 'Original OOF unchanged during read')
            members = arrays[seed, arm]['raw_logits'].shape[0]
            require(reused['ids'].dtype == reused['folds'].dtype == np.int64
                    and np.array_equal(reused['ids'], ids) and np.array_equal(reused['folds'], fold_array)
                    and reused['member_log_probability'].dtype == reused['pool_log_probability'].dtype == np.float64
                    and reused['member_log_probability'].shape == (members, 5274, 10)
                    and reused['pool_log_probability'].shape == (5274, 10)
                    and np.isfinite(reused['member_log_probability']).all() and np.isfinite(reused['pool_log_probability']).all(),
                    'Exact admitted original roles/folds/member/pool prediction schema')
            payloads[seed, arm] = reused
            record = copy.deepcopy(row['endpoint_record'])
            record['endpoint_origin'] = 'reused_exact_admitted_original_no_fit_replay'
            record['new_scalar_fit_attempts'] = 0
            record['reused_archive'] = item
            record_index[seed, arm] = record
    y, attempted = torch.from_numpy(labels), 0
    for arm in NEW:
        for seed in SEEDS:
            a = arrays[seed, arm]
            log_p = torch.from_numpy(a['raw_logits']).to(torch.float64).log_softmax(-1)
            probability = log_p.exp()
            empty_context = torch.empty((len(log_p), len(labels), 0), dtype=torch.float64)
            log_pool = torch.full((5274, 10), float('nan'), dtype=torch.float64)
            log_members = torch.full(log_p.shape, float('nan'), dtype=torch.float64)
            fits = []
            for fold in range(5):
                fit_ids, held_ids = torch.where(folds != fold)[0], torch.where(folds == fold)[0]
                attempted += 1
                before, returned_record = time.perf_counter(), None
                try:
                    held, record = operators.fit_fold(torch, 'temperature_global', log_p, probability, empty_context, y, fit_ids, held_ids)
                    returned_record = record
                    log_T = torch.tensor(record['state']['log_T'], dtype=torch.float64)
                    temperature = log_T.exp()
                    require(log_T.shape == (1,) and record['updates'] == 500 and record['parameters'] == 1
                            and record['fit_nodes'] == len(fit_ids) and record['held_nodes'] == len(held_ids)
                            and record['selected_endpoint'] == 'fixed_final_update_no_heldout_selector'
                            and torch.isfinite(temperature).all().item() and (temperature > 0).all().item(), 'One fixed final positive global temperature')
                    scaled = (log_p[:, held_ids] / temperature.reshape(-1, 1, 1)).log_softmax(-1)
                    require(torch.isfinite(scaled).all().item() and torch.isfinite(held).all().item(), 'Finite fixed new endpoint')
                    log_pool[held_ids], log_members[:, held_ids] = held, scaled
                    record.update(status='finite_fixed_endpoint', fold=fold, temperature=temperature.item(), seconds=time.perf_counter() - before)
                except (FloatingPointError, RuntimeError, ValueError) as error:
                    record = {'status': 'failed_retained', 'fold': fold, 'error': str(error), 'fixed_budget_updates': 500,
                        'actual_updates_before_failure': returned_record['updates'] if returned_record is not None else None,
                        'returned_endpoint_record': retained_json(returned_record), 'seconds': time.perf_counter() - before}
                fits.append(record)
            finite = torch.isfinite(log_pool).all().item() and torch.isfinite(log_members).all().item()
            record = {'arm': arm, 'seed': seed, 'status': 'finite_complete_five_fold' if finite else 'failed_retained',
                'fits': fits, 'fit_seconds_sum': sum(f['seconds'] for f in fits), 'attempted_scalar_fits': 5,
                'endpoint_origin': 'new_fixed_endpoint', 'new_scalar_fit_attempts': 5}
            value = {'ids': ids, 'folds': fold_array, 'member_log_probability': log_members.numpy(), 'pool_log_probability': log_pool.numpy()}
            payloads[seed, arm] = value
            stream = io.BytesIO()
            np.savez_compressed(stream, **value)
            name, data = f'CALIBRATED_{arm}_seed{seed}_OOF.npz', stream.getvalue()
            record['OOF_archive'] = {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            binaries.append((name, data))
            record_index[seed, arm] = record
    records = [record_index[seed, arm] for arm in ARMS for seed in SEEDS]
    require(attempted == 90 and len(records) == len(payloads) == 33 and len(binaries) == 18
            and all(len(r['fits']) == 5 for r in records)
            and sum(r['attempted_scalar_fits'] for r in records) == 165
            and sum(r['new_scalar_fit_attempts'] for r in records) == 90, 'All165 terminal endpoints, exactly90 newly attempted')
    new_records = [r for r in records if r['arm'] in NEW]
    successful = sum(f['status'] == 'finite_fixed_endpoint' for r in new_records for f in r['fits'])
    reused_seconds = sum(r['prior_fit_seconds'] for r in reuse_index.values())
    cost = {'reused_scalar_endpoints': 75, 'reused_successful_scalar_updates_already_paid': 37500,
        'reused_prior_fit_seconds_sum_already_paid': reused_seconds, 'reused_scalar_fit_replays': 0,
        'new_attempted_scalar_fits': attempted, 'new_requested_updates_per_fit': 500,
        'new_requested_fullbatch_scalar_updates': 45000, 'new_successful_scalar_fits': successful,
        'new_failed_scalar_fits': 90 - successful, 'new_known_successful_fullbatch_scalar_updates': successful * 500,
        'new_known_completed_fullbatch_scalar_updates_lower_bound': successful * 500 + sum(f['actual_updates_before_failure'] or 0 for r in new_records for f in r['fits'] if f['status'] == 'failed_retained'),
        'new_failed_actual_update_counts_known': all(f.get('actual_updates_before_failure') is not None for r in new_records for f in r['fits'] if f['status'] == 'failed_retained'),
        'new_fit_seconds_sum': sum(r['fit_seconds_sum'] for r in new_records),
        'current_readout_wall_seconds_including_reuse_IO': time.perf_counter() - start,
        'total_comparative_endpoint_records': 165, 'total_comparative_requested_scalar_updates': 82500,
        'native_models_or_forwards_added': 0, 'no_retry_or_final_refit': True,
        'scope': '75 original endpoint costs are retained separately and not charged as new work.90 new attempts only; failure does not imply500 updates. Unknown failed work stays unknown. Fit seconds are nested in current wall time.'}
    return payloads, records, binaries, fold_meta, cost


def calibrated_scores(helper, np, payloads, records, arrays, labels):
    """Scoring begins only after every new and reused endpoint is terminal."""
    scores, index = {}, {(r['seed'], r['arm']): r for r in records}
    for arm in ARMS:
        for seed in SEEDS:
            record, a, p = index[seed, arm], arrays[seed, arm], payloads[seed, arm]
            if record['status'] != 'finite_complete_five_fold':
                scores[seed, arm] = None
                continue
            pool, members = p['pool_log_probability'], p['member_log_probability']
            require(np.allclose(helper.logsumexp(members, -1), 0, rtol=0, atol=1e-12)
                    and np.allclose(helper.logsumexp(pool, -1), 0, rtol=0, atol=1e-12)
                    and np.allclose(pool, helper.logsumexp(members, 0) - math.log(len(members)), rtol=1e-12, atol=1e-12), 'Complete normalized equal-policy OOF predictions')
            compatible = {'raw_logits': members, 'y': labels, 'member_errors': a['member_errors'],
                'pooled_errors': pool.argmax(1) != labels, 'probability_mean': np.exp(pool)}
            value = score(helper, np, compatible)
            disagreements = int(((members.argmax(-1) != labels) != a['member_errors']).sum())
            if arm in REFS:
                require(value['counts'] == record['counts'], 'Original calibrated integer scores unchanged')
                for key, stored in record['quality'].items():
                    require(np.allclose(value['quality'][key], stored, rtol=1e-12, atol=1e-12), 'Original admitted calibrated score/prediction binding: ' + key)
                require(disagreements == record['FP64_calibrated_member_argmax_disagreements_with_raw_archived_errors'], 'Original member precision diagnostics unchanged')
                # Preserve original per-seed scores exactly, including their float values.
                value.update(quality=copy.deepcopy(record['quality']), counts=copy.deepcopy(record['counts']), classes=copy.deepcopy(record['classes']))
            else:
                record.update(quality=value['quality'], counts=value['counts'], classes=value['classes'],
                    FP64_calibrated_member_argmax_disagreements_with_raw_archived_errors=disagreements,
                    fixed_original_strict_rival_cohorts={ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in REFS},
                    member_top_class_authority='Raw archived member flags remain fixed. Positive temperature acquires no alternatives; precision/tie changes are separately diagnosed.')
            record['fixed_complete_roster_strict_rival_cohorts'] = {ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), value['P'], labels) for ref in ARMS}
            scores[seed, arm] = value
    return scores


def aggregate(helper, values, rows):
    metrics = ('pooled_accuracy_pct', 'pooled_nll', 'pooled_brier', 'mean_member_accuracy_pct',
               'mean_member_nll', 'worst_member_accuracy_pct', 'worst_member_nll', 'mean_member_brier', 'worst_member_brier')
    return {'mean_quality': {k: statistics.mean(v['quality'][k] for v in values) for k in metrics},
            'summed_three_seed_readout_counts': {k: sum(v['counts'][k] for v in values) for k in values[0]['counts'] if k != 'member_correct'},
            'acquisition_costs': helper.cost_summary(rows),
            'count_scope': 'Three dependent readouts of the same5274 development nodes; not independent node samples.'}



def native_cost_summary(helper, rows):
    value = helper.cost_summary(rows)
    restoration_recorded = all('restore_seconds' in r['costs'] for r in rows)
    value.update(restore_seconds_sum=sum(r['costs']['restore_seconds'] for r in rows) if restoration_recorded else None,
        restore_seconds_recorded_for_all_groups=restoration_recorded,
        original_unrecorded_restore_cost_is_not_zero=True,
        selected_forward_and_metrics_seconds_sum=sum(r['costs']['selected_forward_and_metrics_seconds'] for r in rows),
        selected_pool_seconds_sum=sum(r['costs']['selected_pool_seconds'] for r in rows),
        parameters_sum=sum(r['costs']['parameters'] for r in rows),
        inference_parameter_bytes_sum=sum(r['costs']['inference_parameter_bytes'] for r in rows),
        selected_checkpoint_bytes_sum=sum(r['costs']['selected_checkpoint_bytes'] for r in rows),
        process_lifetime_peak_RSS_bytes_max=max(r['costs']['process_peak_rss_bytes'] for r in rows),
        cuda_peak_allocated_bytes_max=max(r['costs']['cuda_peak_allocated_bytes'] for r in rows),
        cuda_peak_reserved_bytes_max=max(r['costs']['cuda_peak_reserved_bytes'] for r in rows),
        scope='Observed costs; acquisition, restoration, selected serving and one-time support preparation have separate scopes. Peak memory is a maximum, not a sum. Process RSS is lifetime peak.')
    return value


def transport_error_partitions(helper, np, values, labels, readout):
    arms = (BASE, PRIVATE, COMMON, FIXED)
    if any(values[s, a] is None for s in SEEDS for a in arms):
        return {'failed_retained': True, 'readout': readout, 'reason': 'A required calibrated bank failed.'}
    rows = []
    for seed in SEEDS:
        base, private, common, fixed = [values[seed, a]['P'] for a in arms]
        repair = ~base & private
        pattern = base.astype(np.int64) * 8 + private.astype(np.int64) * 4 + common.astype(np.int64) * 2 + fixed.astype(np.int64)
        def partition(mask):
            common_only = repair & common & ~fixed
            fixed_only = repair & fixed & ~common
            both = repair & common & fixed
            neither = repair & ~common & ~fixed
            result = {'nodes': int(mask.sum()), 'private_repairs_vs_BASE': int((mask & repair).sum()),
                'private_repairs_also_correct_in_common_only': int((mask & common_only).sum()),
                'private_repairs_also_correct_in_fixed_only': int((mask & fixed_only).sum()),
                'private_repairs_also_correct_in_both_controls': int((mask & both).sum()),
                'private_repairs_not_correct_in_either_control': int((mask & neither).sum()),
                'private_harms_where_BASE_common_fixed_all_correct': int((mask & base & common & fixed & ~private).sum()),
                'all16_correctness_patterns_BASE_private_common_fixed': np.bincount(pattern[mask], minlength=16).tolist()}
            require(result['private_repairs_vs_BASE'] == sum(result[k] for k in
                ('private_repairs_also_correct_in_common_only', 'private_repairs_also_correct_in_fixed_only',
                 'private_repairs_also_correct_in_both_controls', 'private_repairs_not_correct_in_either_control')), 'Exact private repair partition')
            return result
        rows.append({'seed': seed, **partition(np.ones(len(labels), dtype=np.bool_)),
            'classes': [{'class': k, **partition(labels == k)} for k in range(10)]})
    return {'readout': readout, 'pattern_order': list(arms), 'per_seed': rows,
        'scope': 'Full independently fitted/stopped/selected procedures. These control repair intersections describe specificity; they do not isolate a same-state cause or replace frozen accuracy/NLL gates.'}


def analyze(helper, np, raw, calibrated, labels, records):
    pairs = [(a, b) for a in NEW for b in ARMS if a != b]
    raw_contrasts = {a + '_minus_' + b: contrast(helper, raw, a, b, labels) for a, b in pairs}
    calibrated_contrasts = {a + '_minus_' + b: contrast(helper, calibrated, a, b, labels)
        if all(calibrated[s, a] is not None and calibrated[s, b] is not None for s in SEEDS)
        else {'failed_retained': True, 'candidate': a, 'reference': b} for a, b in pairs}
    for a, b in pairs:
        fields = set(records[SEEDS[0], a]['costs']) & set(records[SEEDS[0], b]['costs'])
        cost_deltas = {key: helper.paired(records[s, a]['costs'][key] - records[s, b]['costs'][key] for s in SEEDS) for key in sorted(fields)}
        for contrasts in (raw_contrasts, calibrated_contrasts):
            contrasts[a + '_minus_' + b]['native_cost_deltas'] = cost_deltas
            contrasts[a + '_minus_' + b]['native_cost_fields_unavailable_for_delta'] = sorted(set(records[SEEDS[0], a]['costs']) ^ set(records[SEEDS[0], b]['costs']))
            contrasts[a + '_minus_' + b]['native_cost_scope'] = 'Observed full-procedure costs; original work already paid. One-time shared support preparation is reported once outside arm-specific fit/restore/serving deltas.'
    effects = {}
    for arm in ARMS:
        if all(calibrated[s, arm] is not None for s in SEEDS):
            values = {(s, 'raw'): raw[s, arm] for s in SEEDS}
            values.update({(s, 'calibrated'): calibrated[s, arm] for s in SEEDS})
            effects[arm] = contrast(helper, values, 'calibrated', 'raw', labels, 'same_selected_state_calibration')
        else:
            effects[arm] = {'failed_retained': True}
    required = {BASE: .2, F4K: .2, COMMON: .1, FIXED: .1}
    accuracy, protection = {}, {}
    for ref, threshold in required.items():
        stat = raw_contrasts[PRIVATE + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        flags = {'required_mean_gain_pp': threshold, 'paired_raw_accuracy_pp': stat,
            'mean_at_least_threshold': stat['mean'] >= threshold,
            'all_seeds_nonnegative': stat['nonnegative_seed_count'] == 3,
            'at_least_two_positive': stat['positive_seed_count'] >= 2}
        flags['pass'] = all(flags[k] for k in ('mean_at_least_threshold', 'all_seeds_nonnegative', 'at_least_two_positive'))
        accuracy[ref] = flags
        value = calibrated_contrasts[PRIVATE + '_minus_' + ref]
        if value.get('failed_retained'):
            protection[ref] = {'finite': False, 'pass': False, 'failure_retained': True}
        else:
            nll, acc = [value['quality_deltas'][k] for k in ('pooled_nll', 'pooled_accuracy_pct')]
            flags = {'finite': True, 'paired_calibrated_NLL_deterioration': nll, 'paired_calibrated_accuracy_pp': acc,
                'mean_NLL_deterioration_at_most_0_02': nll['mean'] <= .02,
                'each_seed_NLL_deterioration_at_most_0_05': nll['max'] <= .05,
                'calibrated_accuracy_reported_without_extra_veto': True}
            flags['pass'] = flags['mean_NLL_deterioration_at_most_0_02'] and flags['each_seed_NLL_deterioration_at_most_0_05']
            protection[ref] = flags
    raw_pass, protected = all(v['pass'] for v in accuracy.values()), all(v['pass'] for v in protection.values())
    capable = {}
    for ref in (O1K, F1K, O1, F1, O4, F4):
        stat = raw_contrasts[PRIVATE + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        value = calibrated_contrasts[PRIVATE + '_minus_' + ref]
        capable[ref] = {'paired_raw_pooled_accuracy_pp': stat,
            'adverse_stronger_reference_in_mean_raw_accuracy': stat['mean'] < 0,
            'any_negative_raw_seed': stat['nonnegative_seed_count'] < 3,
            'calibrated_quality_deltas': value.get('quality_deltas'),
            'calibration_failure_retained': bool(value.get('failed_retained')),
            'role': 'capable reference; adverse stronger reference limits a broad superiority statement; no extra private attribution gate'}
    generic = {arm: {'raw_quality_deltas_vs_BASE': raw_contrasts[arm + '_minus_' + BASE]['quality_deltas'],
        'calibrated_quality_deltas_vs_BASE': calibrated_contrasts[arm + '_minus_' + BASE].get('quality_deltas'),
        'calibration_failure_retained': bool(calibrated_contrasts[arm + '_minus_' + BASE].get('failed_retained')),
        'scope': 'Generic feature-path/learned-kernel contrast; descriptive. This control cannot replace the declared private candidate gate.'} for arm in (COMMON, FIXED)}
    policy = {'candidate': PRIVATE, 'raw_accuracy_criteria': accuracy, 'raw_private_attribution_screen_pass': raw_pass,
        'equal_policy_calibrated_confidence_criteria': protection, 'calibrated_confidence_protection_pass': protected,
        'joint_accuracy_confidence_development_clue': raw_pass and protected,
        'accuracy_only_clue_with_confidence_tradeoff': raw_pass and not protected,
        'capable_reference_diagnostics': capable,
        'broad_superiority_statement_blocked_by_stronger_equipped_single': any(capable[a]['adverse_stronger_reference_in_mean_raw_accuracy'] for a in (O1K, F1K)),
        'broad_superiority_statement_blocked_by_stronger_capable_reference': any(v['adverse_stronger_reference_in_mean_raw_accuracy'] for v in capable.values()),
        'generic_transport_control_findings': generic, 'member_quality_is_diagnostic_not_veto': True,
        'calibrated_accuracy_reported_without_extra_veto': True, 'raw_NLL_and_Brier_tradeoffs_preserved': True,
        'historical_A_B_AB_not_candidate_or_K': True, 'automatic_confirmation_launch': False,
        'scope': 'One native SAGE backbone on encountered WikiCS development. Private attribution has four fixed references. Capable singles and original I4s remain visible; control gains do not replace the private gate. No generic novelty, unused confirmation, universal transfer or manuscript verdict.'}
    results = {'raw_full_procedure_contrasts': raw_contrasts, 'calibrated_full_procedure_contrasts': calibrated_contrasts,
        'calibration_minus_raw_same_selected_state': effects,
        'raw_transport_error_partitions': transport_error_partitions(helper, np, raw, labels, 'raw'),
        'calibrated_transport_error_partitions': transport_error_partitions(helper, np, calibrated, labels, 'OOF_calibrated')}
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
    # NumPy and all33 selected/support/OOF arrays are downstream of complete metadata custody.
    import numpy as np
    helper.np = np
    support_diagnostics = support_arrays(np, context)
    arrays, ids, labels = payloads(np, expected)
    import torch
    torch.set_num_threads(2)
    operators = load_source('SAGE_fixed_temperature', root / CALIBRATOR, inputs['research_files'][CALIBRATOR])
    calibrated_payloads, calibration_records, binaries, folds, calibration_costs = calibrate(np, torch, operators, arrays, ids, labels, context)
    # No score, contrast, criterion, transport repair partition or selection before all165 terminal endpoints.
    raw, diagnostics = score_raw(helper, np, arrays, records, context, labels)
    for arm in ARMS:
        for seed in SEEDS:
            diagnostics[seed, arm]['fixed_complete_roster_strict_rival_cohorts'] = {
                ref: strict_counts(np, strict_mask(np, arrays[seed, ref]['raw_logits'], labels), raw[seed, arm]['P'], labels) for ref in ARMS}
    calibrated = calibrated_scores(helper, np, calibrated_payloads, calibration_records, arrays, labels)
    contrasts, policy = analyze(helper, np, raw, calibrated, labels, records)
    aggregates = {'raw': {a: aggregate(helper, [raw[s, a] for s in SEEDS], [records[s, a] for s in SEEDS]) for a in ARMS},
        'calibrated': {a: aggregate(helper, [calibrated[s, a] for s in SEEDS], [records[s, a] for s in SEEDS])
            if all(calibrated[s, a] is not None for s in SEEDS) else {'failed_retained': True} for a in ARMS}}
    for values in aggregates.values():
        for arm, value in values.items():
            if not value.get('failed_retained'):
                value['acquisition_costs'] = native_cost_summary(helper, [records[s, arm] for s in SEEDS])
    groups = [{'arm': arm, 'seed': seed, 'raw': {k: raw[seed, arm][k] for k in ('quality', 'counts', 'classes')},
        'diagnostics': diagnostics[seed, arm], 'original_group_record': records[seed, arm],
        'acquisition_origin': 'new_feature_kernel' if arm in NEW else 'unchanged_original_reference'} for arm in ARMS for seed in SEEDS]
    kernels_by_arm = {arm: {key: sum(u['feature_kernel']['counters'][key] for s in SEEDS for u in records[s, arm]['fits'])
        for key in records[SEEDS[0], arm]['fits'][0]['feature_kernel']['counters']} for arm in NEW}
    costs = {'new27_native_fits': native_cost_summary(helper, context['complete']['results']),
        'reused33_selected_reference_fits_already_paid': native_cost_summary(helper, [records[s, a] for s in SEEDS for a in REFS]),
        'original39_parent_context_not_additive': native_cost_summary(helper, context['old_complete']['results']),
        'one_shared_feature_support_preparation': context['complete']['feature_support'],
        'feature_kernel_operation_counts': context['complete']['feature_kernel_operation_counts'],
        'feature_kernel_operation_counts_by_arm': kernels_by_arm,
        'calibration_scalar_work': calibration_costs, 'new_owner': context['owner'], 'original_owner': context['old_owner'],
        'scope': 'Support preparation is paid once in the fresh family and nested in owner wall time. Native acquisition, restore, selected forward/metrics, pool and scalar work remain separate. Original native/scalar costs are retained as already-paid references. Original restoration time was not recorded and remains unavailable, never zero. Timings are observed, not isolated benchmarks.'}
    limits = [
        'All18 fresh/15 original banks and27 fresh/33 original selected fits close before arrays. Exact75 old calibration endpoints are admitted and not replayed;90 new attempts finish before any scoring/contrasts/gates. Failures remain terminal failures.',
        'Original selected records, per-seed raw/calibrated scores and costs remain visible. New work charges27 native fits, one support preparation and90 scalar attempts only.',
        'Cross-arm raw/calibrated repairs, controls and rival-cohort corrections compare full fitted/stopped/selected procedures, not a causal same-state intervention.',
        'Calibration-minus-raw holds each native selected bank fixed. Positive temperature acquires no new member alternatives; raw member flags and coverage remain authoritative, with precision/tie changes diagnosed.',
        'The private raw gate requires0.2 pp over BASE/equipped genuineI4 and0.1 pp over common/fixed; all three contrasts nonnegative and at least two positive for every reference. NLL protection is at most0.02 mean/0.05 per seed. Calibrated accuracy is reported without a new veto.',
        'Both equipped singles and original ordinary/factorized references remain capable comparators. Adverse stronger references limit broad superiority language; member weakness alone is not a veto.',
        'Common/fixed control findings can support generic path-acquisition observations; they cannot replace the declared private attribution gate. No failed historical A/B/AB is selected asK.',
        'Fixed five-fold calibration is encountered post-selection development, not untouched validation or whole-pipeline cross-fitting. A future pipeline needs a separately fixed CAL role.',
        'Three paired optimizer readouts on the same graph/5274 nodes are not independent node or graph replications. Intervals and sign flips are descriptive.',
        'Attention/AM-GCN/DGM/native SAGE/factorization and positive-temperature calibration are known ingredients. One backbone suffices; no novelty, universal transfer, unused confirmation or acceptance claim.',
        'No model/checkpoint state, native loader, new native forward, cosine support replay or TEST is used by this reader. Checkpoint hashes match frozen root admission; original historical hashes are not invented.']
    report = {'complete': True, 'backbone': 'SAGE', 'new_banks': 18, 'new_native_fits': 27,
        'new_native_trajectories_per_complete_roster_forward': 54, 'reused_reference_banks': 15, 'reused_reference_fits': 33,
        'joined_banks': 33, 'joined_selected_fit_records': 60, 'seeds': SEEDS, 'roster': ARMS, 'TEST_access': False,
        'calibration_reused_endpoints': 75, 'calibration_new_attempts': 90, 'calibration_complete_terminal_endpoints': 165,
        'frozen_policy': policy, 'aggregates': aggregates, 'groups_detail': groups, 'contrasts': contrasts,
        'calibration_folds': folds, 'calibration_records': calibration_records, 'costs': costs,
        'feature_support_diagnostics': support_diagnostics, 'source_custody': context,
        'selected_archive_custody': [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS],
        'checkpoint_custody': checkpoints, 'reused_calibrated_archive_custody': context['reused_OOF_archives'],
        'ordered_VALID_hashes': VALID_HASHES, 'analysis_source_sha256': sha(Path(__file__)),
        'immutable_count_math_sha256': MATH_SHA, 'reviewed_reader_parent_sha256': inputs['research_files'][PRIOR_READER],
        'input_bindings': inputs,
        'authority': 'Raw counts use archived native float32 flags. Stable float64 NLL/Brier remain explicit. Calibrated pooled decisions use fixed OOF float64 outputs; raw member flags/coverage remain authority. Original admitted per-seed calibrated scores are retained exactly.',
        'interpretation_limits': limits}
    documents = [('SAGE_POLICY_SUMMARY.json', {'complete': True, 'backbone': 'SAGE', 'new_banks': 18, 'new_native_fits': 27,
        'reused_banks': 15, 'reused_selected_fits': 33, 'frozen_policy': policy, 'all_terminal_endpoints': 165,
        'reused_endpoints': 75, 'new_endpoint_attempts': 90, 'calibration_costs': calibration_costs})]
    documents += [('FEATURE_SUPPORT_AND_KERNEL_COSTS.json', {'support': context['complete']['feature_support'],
        'support_diagnostics': support_diagnostics, 'kernel_operations': context['complete']['feature_kernel_operation_counts'],
        'kernel_operations_by_arm': kernels_by_arm, 'costs': {k: v for k, v in costs.items() if k not in ('new_owner', 'original_owner')}})]
    documents += chunk_documents('SAGE_ARM_AGGREGATES', [{'readout': kind, 'arm': arm, 'aggregate': value}
        for kind, bank in aggregates.items() for arm, value in bank.items()])
    documents += chunk_documents('SAGE_ARM_DETAILS', groups)
    documents += chunk_documents('CALIBRATION_FIT_DETAILS', calibration_records)
    for category, values in contrasts.items():
        if category.endswith('_error_partitions'):
            documents.append(('SAGE_' + category + '.json', values))
        else:
            documents += chunk_documents('SAGE_' + category, [{'key': name, 'contrast': value} for name, value in values.items()])
    documents += chunk_documents('CHECKPOINT_CUSTODY', checkpoints)
    documents += chunk_documents('ARCHIVE_CUSTODY', [{'seed': s, 'arm': a, **expected[s, a]} for a in ARMS for s in SEEDS])
    documents += chunk_documents('REUSED_OOF_ARCHIVE_CUSTODY', context['reused_OOF_archives'])
    prepared = [(report_path, encode(report))]
    prepared += [(report_path.parent / name, encode(value)) for name, value in documents]
    prepared += [(report_path.parent / name, data) for name, data in binaries]
    require(len({path for path, _ in prepared}) == len(prepared), 'Distinct deterministic output names')
    for path, data in prepared:
        require(not path.exists(), 'Every output path must be fresh')
        if path != report_path and path.suffix == '.json':
            require(len(data) < LIMIT, 'Bounded JSON export required')
    # Rehash every admitted selected/support/OOF byte and source/receipt before first output write.
    for item in list(expected.values()) + checkpoints + [context['support_archive']] + context['reused_OOF_archives'] + context['prior_calibration_custody']['record_partitions']:
        require(sha(item['path']) == item['sha256'], 'Admitted artifact changed before output')
    for name, digest in inputs['research_files'].items():
        require(sha(root / name) == digest, 'Pinned source/receipt changed during readout')
    for name, digest in source['files'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Reader seal changed during readout')
    require(sha(root / PILOT / 'actual_family_v1/COMPLETE_FAMILY.json') == context['complete_sha256']
            and sha(root / PILOT / 'OWNER_END.json') == context['owner_end_sha256']
            and sha(Path(context['binding']['family_root']) / 'COMPLETE_FAMILY.json') == context['binding']['complete_sha256']
            and sha(Path(context['binding']['family_root']).parent / 'OWNER_END.json') == context['binding']['owner_end_sha256']
            and sha(root / PRIOR / 'READER_JOB_END_V1.json') == context['calibration_reuse_admission']['prior_reader_end_sha256']
            and sha(root / PRIOR / 'complete_analysis_v1/COMPLETE_ANALYSIS_SUMMARY.json') == context['calibration_reuse_admission']['prior_complete_summary_sha256'], 'Whole current/original/reused readout closure unchanged')
    receipts = [{'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in prepared]
    summary = {'complete': True, 'backbone': 'SAGE', 'new_banks': 18, 'new_native_fits': 27,
        'original_reference_banks': 15, 'original_selected_fits': 33, 'joined_banks': 33, 'joined_selected_fit_records': 60,
        'TEST_access': False, 'frozen_policy': policy, 'calibration_folds': folds, 'calibration_costs': calibration_costs,
        'raw_authority_preserved': True, 'original_raw_and_calibrated_scores_preserved': True,
        'all165_endpoint_records_terminal_before_interpretation': True, 'reused_endpoints': 75, 'new_scalar_fit_attempts': 90,
        'post_selection_OOF_scope': 'Encountered development, not unused whole-pipeline confirmation.',
        'repair_scope': 'Cross-arm full procedures; calibration-only effects at each fixed native selected state.',
        'full_report': receipts[0], 'partitions': receipts[1:], 'reused_OOF_archives': context['reused_OOF_archives'],
        'input_binding_sha256': sha(Path(__file__).with_name('INPUT_BINDINGS.json')),
        'new_complete_sha256': context['complete_sha256'], 'new_freeze_sha256': context['freeze_sha256'],
        'new_owner_end_sha256': context['owner_end_sha256'], 'original_complete_sha256': context['binding']['complete_sha256'],
        'analysis_source_sha256': sha(Path(__file__))}
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
        'reused_scalar_endpoints': 75, 'new_scalar_fit_attempts': 90, 'partitions': len(receipts) - 1}))


if __name__ == '__main__':
    main()

