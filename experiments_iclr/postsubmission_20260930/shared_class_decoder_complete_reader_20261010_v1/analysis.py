"""Complete fixed SAGE decoder readout; stdlib until full artifact custody.

No model construction, forward, fitting, owner, data loader, or re-decoding.
All 36 archives and all 30 selected checkpoint paths are admitted together.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import statistics
import subprocess

REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE = REPO / 'experiments_iclr/postsubmission_20260930'
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PILOT = 'shared_class_decoder_pilot_root_20261010_v1'
SOURCE = 'shared_recurrent_class_decoder_native_source_20261010_v1'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
REFERENCE_BINDING = 'nonlocal_label_retrieval_pilot_root_20261010_v1/SAGE/REFERENCE_BINDINGS.json'
REFERENCE_SHA = 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'
SEEDS = (7301, 7403, 7507)
NEW = ('ordinary_M1_graphP_commonB', 'factorized_allmap_M1_graphP_commonB',
       'factorized_allmap_genuine_I4_graphP_ownB', 'joint_untied4_graphP_commonB',
       'shared4_graphP_commonB', 'shared4_graphP_privateB4', 'shared4_Pidentity_commonB')
REFERENCES = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
              'factorized_allmap_genuine_I4', 'shared4_unchanged')
OLD_ARMS = REFERENCES + ('separable_equal_size', 'exchange')
ARMS = REFERENCES + NEW
PRIMARY, I4, PRIVATE, IDENTITY = NEW[4], NEW[2], NEW[5], NEW[6]
VALID_HASHES = {'ids': '48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801',
                'labels': '2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'}
POLICY = {'DECISION.md': '4ab7531dcaa8584ca7b9216c46d36057125031158ba25ee8e753c74a86b331ca',
          'CONFIG.json': 'ab20efc8e98aac6819844948737ce97b0a112fc479c4aebb29c78b699ef63ed9',
          'FREEZE.json': 'b0167e29fc898d3e5ea6aae00e100b9d8e6da6e37fa687f30e92371162de1435',
          'run.py': 'aef5a6057a6c7c5f19a4a9e01035d24a39adbff7f54d7bd34cd0003532df94a3',
          'owner.py': 'cbd5c17e3d34dc4d31b39c02c4f5327e4020877d3d4e80d90919123bcb6bd606',
          'ACTUAL_QUALIFICATION_V1.json': 'afa29d13ad30c04fabb4f7ef319f2e02f0c60d9405d16d69ccadc3e48cec7c00',
          'QUALIFICATION_FREEZE.json': 'cf412974aa81e5e71ece610ff1258d34cb2b96a4c01493d3476dfccf41fdefd8'}
LEARNING = {SOURCE + '/native_extension.py': 'ce7d9c5276595583ce579ea8a3f4be68be4fed48b6c185b68605661a36be4a98',
            SOURCE + '/decoder.py': 'e2f4da1953b10f81ba20817083e11b070ef4acad419af0fe0e52badfb5b7d1f8',
            SOURCE + '/SOURCE.json': 'a2403110f263d882a8d473a1a1ea1365272d5b2b819bdaae1228517302e64dac',
            SOURCE + '/ROSTER.json': 'a3ba31c9f67d36464926dd60f9e2dd69158fe44bccf3338d435411d8bdf3cef3',
            'common_wrapper_native_backbone_full_family_source_20261010_v1/run_family.py':
            'def9e67b42e618f3b14fb98e616c145706e367d4347d3ce9d6251e390c9402c4'}
NATIVE = {'models.py': '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303',
          'run_base.py': 'ca78469a7c1dcc9e98fba054ae1df714f6b19c3e91690ee7b48538005b4e52b1',
          'run_common.py': 'e5256b39aef9fdc688832320a9e7d8aa24f244f48f6939d0b0daa342639a1ebc',
          'shared_fast_graph_model_interface_20261010_v1/common_routes.py':
          '84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773',
          'portable_internal_be_public_interface_20261007_v2/core/factors.py':
          '9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'}
TAIL_METADATA = ('native_body_fits', 'declared_acquisition_arms',
                 'decoder_operation_counts', 'decoder_graph_preparation_once')
STATES = ('no_alternative_wrong', 'aggregation_only_correct', 'alternative_lost', 'alternative_served')
LIMIT = 2000000


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


def bound_path(path, parent):
    path = Path(path).resolve()
    require(path.is_relative_to(parent.resolve()), 'Artifact path outside its declared scope: ' + str(path))
    return path


def load_math(root):
    path = root / MATH
    require(sha(path) == MATH_SHA, 'Immutable count/math source required')
    spec = importlib.util.spec_from_file_location('decoder_immutable_count_math', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # The pinned helper imports stdlib only.
    return helper


def arm_semantics(arm):
    return {'training': 'joint untied four bodies/common B' if arm == NEW[3] else
            'four separately initialized/fitted/selected native+own-B models' if arm == I4 else
            'joint shared-native-weight bank' if arm in NEW[4:] else 'single complete native model',
            'genuine_independent': arm == I4, 'factorized_native_maps': arm != NEW[0],
            'decoder_B': 'four private matrices' if arm == PRIVATE else
            'own matrix per independent fit' if arm == I4 else 'one common matrix per fit',
            'decoder_P': 'I' if arm == IDENTITY else 'incoming D^-1(A_off+I)',
            'selection': 'each member own decoded VALID accuracy' if arm == I4 else 'whole-role decoded VALID pool accuracy',
            'raw_logits_archive_authority': 'five-step decoded scores; native_logits is separate at same selected state'}


def preflight(root, helper):
    here = root / PILOT
    for name, digest in POLICY.items():
        require(sha(here / name) == digest, 'Frozen pilot policy changed: ' + name)
    for name, digest in LEARNING.items():
        require(sha(root / name) == digest, 'Frozen decoder learning source changed: ' + name)
    require(sha(root / REFERENCE_BINDING) == REFERENCE_SHA, 'Exact original SAGE reference bindings required')
    cfg, freeze, binding = read(here / 'CONFIG.json'), read(here / 'FREEZE.json'), read(root / REFERENCE_BINDING)
    require(cfg == dict(binding['configuration'], backbone='SAGE') and cfg['seeds'] == list(SEEDS)
            and cfg['hidden'] == 128 and cfg['depth'] == 2 and cfg['dropout'] == .2
            and cfg['learning_rate'] == .001 and cfg['max_updates'] == 1000 and cfg['patience_updates'] == 300,
            'Matched frozen original SAGE operating point')
    require(freeze['qualification_passed'] is True and freeze['new_banks'] == 21
            and freeze['new_optimizer_fits'] == 30 and freeze['native_bodies_fitted'] == 39
            and freeze['route_trajectories_per_complete_forward'] == 66 and freeze['seeds'] == list(SEEDS)
            and freeze['backbone'] == 'SAGE' and freeze['TEST_access'] is False
            and freeze['all30_fits_before_interpretation'] is True, 'Frozen full 21/30/39/66 study required')
    for row in freeze['bound_files']:
        require(sha(bound_path(REPO / row['path'], REPO)) == row['sha256'], 'Frozen source/config/data bytes changed')
    qualification = read(here / 'ACTUAL_QUALIFICATION_V1.json')
    qfreeze = read(here / 'QUALIFICATION_FREEZE.json')
    require(qualification['qualified'] is True and qualification['VALID_quality_scored'] is False
            and qualification['TEST_access'] is False and qualification['scientific_quality_evidence'] is False
            and [r['arm'] for r in qualification['cases']] == list(NEW)
            and qfreeze['TRAIN_only_derivatives'] is True and qfreeze['VALID_quality_scored'] is False
            and qfreeze['TEST_access'] is False and qfreeze['new_fits'] == 0 and qfreeze['cases'] == 7,
            'Seven-arm actual TRAIN qualification, separate from predictive evidence')
    for row in qfreeze['bound_files']:
        require(sha(bound_path(root / row['path'], root)) == row['sha256'], 'Qualification source changed')
    for row in qfreeze['repo_files']:
        require(sha(bound_path(REPO / row['path'], REPO)) == row['sha256'], 'Qualification repo binding changed')
    # No complete outcomes or payloads before BOTH successful closures and headers.
    output = here / 'actual_family_v1'
    old_output = bound_path(binding['family_root'], root)
    require(old_output == root / 'common_wrapper_SAGE_root_20261010_v1/actual_family_v1', 'Exact original family root')
    require(sha(old_output.parent / 'OWNER_END.json') == binding['owner_end_sha256'], 'Pinned original owner receipt')
    owner, old_owner = helper.closed_owner(here / 'OWNER_END.json'), helper.closed_owner(old_output.parent / 'OWNER_END.json')
    header = helper.completion_header(output / 'COMPLETE_FAMILY.json')
    old_header = helper.completion_header(old_output / 'COMPLETE_FAMILY.json')
    for head, fits in ((header, 30), (old_header, 39)):
        require(head['complete'] is True and head['groups'] == head['expected_groups'] == 21
                and head['fit_units'] == head['expected_fit_units'] == fits and head['TEST_access'] is False,
                'Both complete 21-bank families before full outcome records')
    complete_sha = sha(output / 'COMPLETE_FAMILY.json')
    require(complete_sha == owner['complete_sha256']
            and sha(old_output / 'COMPLETE_FAMILY.json') == binding['complete_sha256'] == old_owner['complete_sha256'],
            'Both owners bind complete records')
    complete, old = read(output / 'COMPLETE_FAMILY.json'), read(old_output / 'COMPLETE_FAMILY.json')
    # The extension appends body/decoder metadata after results. Preserve it,
    # while comparing the actual pre-results header without those appended keys.
    excluded = {'results', 'comparisons', 'cost_scope', *TAIL_METADATA}
    require({k: v for k, v in complete.items() if k not in excluded} == header
            and {k: v for k, v in old.items() if k not in ('results', 'comparisons', 'cost_scope')} == old_header,
            'Complete headers unchanged across custody reads')
    require(complete['native_body_fits'] == 39 and complete['declared_acquisition_arms'] == list(NEW)
            and complete['config_sha256'] == POLICY['CONFIG.json'] and read(output / 'CONFIG.json') == cfg,
            'Full appended metadata and exact configuration')
    old_cfg = bound_path(binding['config'], root)
    require(sha(old_cfg) == binding['config_sha256'] == old['config_sha256']
            and read(old_cfg) == binding['configuration'] and read(old_output / 'CONFIG.json') == binding['configuration'],
            'Original immutable config authority')
    old_freeze = read(old_output.parent / 'FREEZE.json')
    for row in binding['frozen_inputs']:
        require(sha(bound_path(REPO / row['path'], REPO)) == row['sha256'], 'Original frozen source/data changed')
    for suffix, digest in NATIVE.items():
        require(helper.source_hash(complete, suffix) == helper.source_hash(old, suffix) == digest
                and helper.freeze_hash(freeze, suffix) == helper.freeze_hash(old_freeze, suffix) == digest,
                'Shared native source identity: ' + suffix)
    for name in (SOURCE + '/native_extension.py', SOURCE + '/decoder.py',
                 'common_wrapper_native_backbone_full_family_source_20261010_v1/run_family.py'):
        require(helper.source_hash(complete, name) == LEARNING[name] == helper.freeze_hash(freeze, name), 'Exact decoder acquisition source')
    old_source = 'common_wrapper_SAGE_full_family_source_20261010_v1/run_family.py'
    require(helper.source_hash(old, old_source) == helper.freeze_hash(old_freeze, old_source) ==
            '12859e58b9f8a490a38c69c04e161c0b9b0c741cf2438cc402466a0c8261a8f3', 'Original SAGE acquisition source')
    rows = {(r['seed'], r['arm']): r for r in complete['results']}
    old_rows = {(r['seed'], r['arm']): r for r in old['results']}
    require(len(rows) == len(complete['results']) == 21 and set(rows) == {(s, a) for s in SEEDS for a in NEW}
            and sum(len(r['fits']) for r in rows.values()) == 30
            and helper.sum_operations(complete['results']) == complete['operation_counts'], 'Full unique new fit/operation roster')
    require(len(old_rows) == len(old['results']) == 21 and set(old_rows) == {(s, a) for s in SEEDS for a in OLD_ARMS}
            and sum(len(r['fits']) for r in old_rows.values()) == 39
            and helper.sum_operations(old['results']) == old['operation_counts'], 'Full original 21/39 family authority')
    graph = complete['decoder_graph_preparation_once']
    require(graph['normalization'] == 'incoming D^-1(A_off+I)' and graph['factual_offdiagonal_multiplicity'] == 'retained'
            and graph['identity_records'] == 11701 and graph['replaced_factual_self_records'] == 11701
            and graph['decoder_records'] == 442907 and graph['native_graph_unchanged'] is True
            and graph['support_tensor_bytes'] == 7133316, 'Frozen self-inclusive multiplicity-preserving decoder support')
    archives, checkpoints, records = {}, [], {}
    for (seed, arm), row in rows.items():
        fits, members = (4 if arm == I4 else 1), (1 if arm in NEW[:2] else 4)
        require(row['serving'] == 'probability_mean' and row['semantics'] == arm_semantics(arm)
                and [u['member'] for u in row['fits']] == list(range(fits)), 'Exact arm semantics and independent fit identities')
        for unit in row['fits']:
            routes = 4 if arm in NEW[3:] else 1
            native_seeds = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
            require(unit['native_factor_dropout_seeds'] == [[s, s + cfg['factor_seed_offset'], s + cfg['dropout_seed_offset']] for s in native_seeds]
                    and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000
                    and unit['semantics'] == row['semantics'] and unit['decoder_graph'] == graph
                    and unit['native_bodies_fitted'] == (4 if arm == NEW[3] else 1)
                    and unit['decoder_parameters'] == (400 if arm == PRIVATE else 100)
                    and unit['optimizer_policy'] == 'one AdamW group, lr=.001, weight_decay=0, native+factors+B; joint own-route loss mean',
                    'Native route seeds, selected states, body count and optimizer policy')
            updates, op = unit['completed_updates'], unit['operation_counts']
            require(op == {'updates': updates, 'backwards': updates, 'adam_steps': updates,
                    'train_fullgraph_native_trajectories': routes * updates,
                    'valid_selection_fullgraph_native_trajectories': routes * updates,
                    'selected_valid_fullgraph_native_trajectories': routes,
                    'native_graph_block_calls': 2 * routes * (2 * updates + 1)}, 'Exact native operation counters')
            passes = 5 * routes * (2 * updates + 1)
            require(unit['decoder_operation_counts'] == {'route_updates': passes,
                    'spatial_row_mean_passes': passes if arm != IDENTITY else 0,
                    'edge_class_values': passes * 442907 * 10 if arm != IDENTITY else 0,
                    'dense_class_MACs': passes * 11701 * 100}, 'Exact five-step decoder operation counters')
            checkpoint = bound_path(unit['selected_state'], output)
            require(checkpoint == output / f'{arm}_seed{seed}/member{unit["member"]}/selected.pt'
                    and checkpoint.stat().st_size == unit['costs']['selected_checkpoint_bytes'], 'Owner-bound canonical selected checkpoint')
            checkpoints.append({'path': checkpoint, 'sha256': sha(checkpoint), 'arm': arm, 'seed': seed,
                                'fit_member': unit['member'], 'selected_step': unit['selected_step'],
                                'selection': unit['selection'], 'shape': (4, 10, 10) if arm == PRIVATE else (10, 10)})
        path = output / f'{arm}_seed{seed}/selected_VALID.npz'
        archives[seed, arm] = {'path': path, 'sha256': sha(path), 'members': members, 'new': True}
        records[seed, arm] = row
    require(sum(u['native_bodies_fitted'] for r in rows.values() for u in r['fits']) == 39
            and len(checkpoints) == len({d['path'] for d in checkpoints}) == 30, 'All 30 checkpoints and 39 native bodies before numerical import')
    decoder_keys = set(next(iter(rows.values()))['fits'][0]['decoder_operation_counts'])
    require(all(set(u['decoder_operation_counts']) == decoder_keys for r in rows.values() for u in r['fits'])
            and {k: sum(u['decoder_operation_counts'][k] for r in rows.values() for u in r['fits']) for k in decoder_keys}
            == complete['decoder_operation_counts'], 'Complete decoder counter sum')
    refs = {(r['seed'], r['arm']): r for r in binding['banks']}
    require(binding['backbone'] == 'SAGE' and len(refs) == len(binding['banks']) == 15
            and set(refs) == {(s, a) for s in SEEDS for a in REFERENCES}, 'All 15 original reference archives required')
    for (seed, arm), bound in refs.items():
        row = old_rows[seed, arm]
        members, fits = (1 if arm in (REFERENCES[0], REFERENCES[2]) else 4), (4 if 'genuine_I4' in arm else 1)
        require(bound['members'] == members and row['serving'] == 'probability_mean'
                and [u['member'] for u in row['fits']] == list(range(fits))
                and [u['selected_state'] for u in row['fits']] == bound['checkpoints'], 'Original 33 selected fit records')
        for unit in row['fits']:
            routes = 4 if arm == REFERENCES[4] else 1
            seeds = [seed + cfg['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
            require(unit['native_factor_dropout_seeds'] == [[s, s + cfg['factor_seed_offset'], s + cfg['dropout_seed_offset']] for s in seeds]
                    and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000, 'Matched original route/selection seed laws')
        path = bound_path(bound['archive'], old_output)
        require(path == old_output / f'{arm}_seed{seed}/selected_VALID.npz'
                and sha(path) == bound['archive_sha256'], 'Immutable reference archive authority')
        archives[seed, arm] = {'path': path, 'sha256': bound['archive_sha256'], 'members': members, 'new': False}
        records[seed, arm] = row
    require(len(archives) == len(records) == 36 and sum(len(records[s, a]['fits']) for s in SEEDS for a in REFERENCES) == 33,
            'All 21 new and 15 reference banks before numerical import')
    context = {'cfg': cfg, 'freeze': freeze, 'qualification': qualification, 'qualification_freeze': qfreeze,
               'binding': binding, 'owner': owner, 'old_owner': old_owner, 'complete': complete, 'old_complete': old,
               'old_freeze': old_freeze, 'complete_sha256': complete_sha,
               'owner_end_sha256': sha(here / 'OWNER_END.json'), 'old_freeze_sha256': sha(old_output.parent / 'FREEZE.json')}
    return context, archives, checkpoints, records


def payloads(np, expected):
    arrays, ids, labels = {}, None, None
    original_keys = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}
    native_keys = {'native_logits', 'native_probability_mean', 'native_member_errors', 'native_pooled_errors'}
    for key, descriptor in expected.items():
        path = descriptor['path']
        require(sha(path) == descriptor['sha256'], 'Archive changed after complete custody')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == original_keys | (native_keys if descriptor['new'] else set()), 'Exact declared VALID schema')
            a = {k: archive[k].copy() for k in archive.files}
        require(sha(path) == descriptor['sha256'], 'Archive changed during payload read')
        require(a['ids'].dtype == a['y'].dtype == np.int64 and a['ids'].shape == a['y'].shape == (5274,), 'Exact VALID role dtype/shape')
        if ids is None:
            ids, labels = a['ids'], a['y']
            require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701
                    and set(labels.tolist()) == set(range(10))
                    and {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in [('ids', ids), ('labels', labels)]} == VALID_HASHES,
                    'Frozen ordered ten-class VALID identity')
        require(np.array_equal(ids, a['ids']) and np.array_equal(labels, a['y']), 'Identical ordered roles across all 36 archives')
        for native in (False, True) if descriptor['new'] else (False,):
            prefix, raw_name = ('native_', 'native_logits') if native else ('', 'raw_logits')
            raw, pool = a[raw_name], a[prefix + 'probability_mean']
            errors, pooled_errors = a[prefix + 'member_errors'], a[prefix + 'pooled_errors']
            require(raw.dtype == pool.dtype == np.float32 and raw.shape == (descriptor['members'], 5274, 10)
                    and pool.shape == (5274, 10) and np.isfinite(raw).all() and np.isfinite(pool).all()
                    and errors.dtype == pooled_errors.dtype == np.bool_
                    and errors.shape == (descriptor['members'], 5274) and pooled_errors.shape == (5274,)
                    and (pool >= 0).all() and (pool <= 1).all()
                    and np.array_equal(pool.argmax(1) != labels, pooled_errors), 'Exact finite archived native/decoded serving authority')
        arrays[key] = a
    require(len(arrays) == 36, 'Every archive schema/identity before scoring')
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


def representation_diagnostics(helper, np, a, value, reported, labels):
    raw = a['raw_logits'].astype(np.float64)
    probabilities = np.exp(raw - helper.logsumexp(raw, -1)[..., None])
    require(value['counts']['pooled_correct'] == round(reported['accuracy'] * 5274)
            and value['counts']['coverage'] == reported['correct_alternative_count']
            and value['counts']['lost_correct_alternatives'] == reported['coverage_lost_in_pooling']
            and value['counts']['member_correct'] == [round(v * 5274) for v in reported['member_accuracy']],
            'Exact archived counts agree with original selected/restored record')
    return {'common_strict_wrong_rival': strict_counts(np, strict_mask(np, raw, labels), value['P'], labels),
            'stable_FP64_NLL_minus_original_reported_NLL': value['quality']['pooled_nll'] - reported['nll'],
            'FP64_member_argmax_disagreements_with_archived_errors': int(((probabilities.argmax(-1) != labels) != a['member_errors']).sum()),
            'raw_argmax_disagreements_with_archived_member_errors': int(((raw.argmax(-1) != labels) != a['member_errors']).sum()),
            'FP64_uniform_pool_decision_disagreements_with_archived_errors': int(((probabilities.mean(0).argmax(1) != labels) != a['pooled_errors']).sum()),
            'FP64_uniform_pool_correct_minus_archived_correct': int((probabilities.mean(0).argmax(1) == labels).sum()) - value['counts']['pooled_correct']}


def score_all(helper, np, arrays, records, labels, context):
    scores, native, diagnostics = {}, {}, {}
    for key, a in arrays.items():
        row = records[key]
        value = helper.score(a)
        scores[key] = value
        diagnostics[key] = {'served': representation_diagnostics(helper, np, a, value, row['valid'], labels)}
        if key[1] in NEW:
            n = native_archive(a)
            component = helper.score(n)  # Exact archived native errors, never reconstructed errors.
            native[key] = component
            diagnostics[key].update(native_at_decoded_selected=representation_diagnostics(
                helper, np, n, component, row['native_at_decoded_selected'], labels),
                decoded_minus_exact_native=helper.comparison(value, component, labels),
                native_common_strict_wrong_rival_corrected_by_decoded=strict_counts(
                    np, strict_mask(np, n['raw_logits'], labels), value['P'], labels))
    comparisons = {(r['seed'], r['arm']): r['versus'] for r in context['complete']['comparisons']}
    require(len(comparisons) == len(context['complete']['comparisons']) == 21 and set(comparisons) == {(s, a) for s in SEEDS for a in NEW}, 'All archived comparison records')
    for (seed, arm), versus in comparisons.items():
        require(set(versus) == set(NEW), 'All seven original within-new comparison authorities')
        for ref, saved in versus.items():
            a, b = scores[seed, arm]['P'], scores[seed, ref]['P']
            require(saved == {'repairs': int((~b & a).sum()), 'harms': int((b & ~a).sum())}, 'Stored archive errors agree with complete comparison authority')
    return scores, native, diagnostics


def matrices(np, checkpoints):
    # No raw checkpoint payload is opened until all archives have passed their
    # schemas and all 30 checkpoint hashes/paths have been admitted together.
    import torch
    rows = []
    j = np.ones((10, 10), dtype=np.float64) / 10
    h = np.eye(10) - j
    for descriptor in checkpoints:
        path = descriptor['path']
        require(sha(path) == descriptor['sha256'], 'Selected checkpoint changed after full custody')
        state = torch.load(path, map_location='cpu', weights_only=True)
        require(sha(path) == descriptor['sha256'], 'Checkpoint changed during read-only matrix extraction')
        require(state['step'] == descriptor['selected_step'] and state['selection'] == descriptor['selection'], 'Checkpoint step/selector matches complete fit record')
        tensor = state['model']['class_decoder_B']
        require(tuple(tensor.shape) == descriptor['shape'] and tensor.dtype == torch.float32
                and tensor.device.type == 'cpu' and torch.isfinite(tensor).all().item(), 'Exact finite learned B layout')
        values = tensor.detach().numpy().astype(np.float64)
        values = values[None] if values.ndim == 2 else values
        for matrix_member, b in enumerate(values):
            graph, bias, null = h @ b @ h, j @ b @ h, b @ j
            residual = b - graph - bias - null
            rows.append({'arm': descriptor['arm'], 'seed': descriptor['seed'], 'fit_member': descriptor['fit_member'],
                         'matrix_member': matrix_member if len(values) == 4 else None,
                         'selected_step': descriptor['selected_step'], 'checkpoint_path': str(path),
                         'checkpoint_sha256': descriptor['sha256'], 'raw_B': b.tolist(),
                         'graph_class_interaction_HBH': graph.tolist(), 'constant_class_bias_JBH': bias.tolist(),
                         'softmax_null_BJ': null.tolist(), 'input_class_row_means': b.mean(1).tolist(),
                         'output_class_column_means': b.mean(0).tolist(), 'effective_constant_class_bias': bias[0].tolist(),
                         'frobenius_norms': {name: float(np.linalg.norm(value)) for name, value in
                                             [('raw_B', b), ('HBH', graph), ('JBH', bias), ('BJ', null)]},
                         'decomposition_max_abs_residual': float(np.abs(residual).max())})
        del state, tensor
    require(len(rows) == 39 and {(r['seed'], r['arm']) for r in rows} == {(s, a) for s in SEEDS for a in NEW}, 'All 39 learned matrices, no outcome selection')
    return {'selected_checkpoints': 30, 'learned_class_matrices': 39, 'matrices': rows,
            'identity': 'B = HBH + JBH + BJ, J = 11^T/10, H = I-J',
            'scope': 'Known gauge arithmetic: row-stochastic P and normalized q make JBH a constant class-bias vector; BJ shifts all output classes equally and cancels in softmax. This does not establish graph utility or necessity; the retrained P=I control measures neighbor-graph value.',
            'raw_checkpoints_exported': False, 'model_construction_or_forward': False}


def aggregate(helper, values, records):
    return {'mean_quality': {k: statistics.mean(v['quality'][k] for v in values) for k in
                            ('pooled_accuracy_pct', 'pooled_nll', 'mean_member_accuracy_pct', 'mean_member_nll', 'worst_member_accuracy_pct')},
            'summed_three_seed_readout_counts': {k: sum(v['counts'][k] for v in values) for k in values[0]['counts'] if k != 'member_correct'},
            'costs': helper.cost_summary(records)}


def analyze(helper, scores, native, diagnostics, records, context, labels):
    pairs = [(PRIMARY, a) for a in ARMS if a != PRIMARY]
    pairs += [(a, ref) for a in NEW if a != PRIMARY for ref in REFERENCES]
    contrasts = {a + '_minus_' + b: helper.contrast(scores, a, b, labels) for a, b in pairs}
    native_scores = dict(native)
    native_scores.update({(s, a): scores[s, a] for s in SEEDS for a in REFERENCES})
    native_contrasts = {a + '_minus_' + b: helper.contrast(native_scores, a, b, labels) for a, b in pairs}
    same_state = {}
    for arm in NEW:
        values = {(s, 'decoded'): scores[s, arm] for s in SEEDS}
        values.update({(s, 'exact_archived_native'): native[s, arm] for s in SEEDS})
        same_state[arm] = helper.contrast(values, 'decoded', 'exact_archived_native', labels)
    accuracy, nll = {}, {}
    required_accuracy = {a: (.2 if a == I4 else .1) for a in NEW if a != PRIMARY}
    required_accuracy.update({a: .2 for a in (REFERENCES[1], REFERENCES[3], REFERENCES[4])})
    for ref, threshold in required_accuracy.items():
        stat = contrasts[PRIMARY + '_minus_' + ref]['quality_deltas']['pooled_accuracy_pct']
        accuracy[ref] = {'mean_required_accuracy_pp': threshold, 'observed_paired_accuracy_pp': stat,
                         'mean_at_least_threshold': stat['mean'] >= threshold,
                         'all_three_seed_nonnegative': stat['nonnegative_seed_count'] == 3,
                         'at_least_two_positive': stat['positive_seed_count'] >= 2}
    for ref in tuple(a for a in NEW if a != PRIMARY) + (REFERENCES[1], REFERENCES[3]):
        stat = contrasts[PRIMARY + '_minus_' + ref]['quality_deltas']['pooled_nll']
        nll[ref] = {'observed_paired_NLL_deterioration': stat,
                    'mean_deterioration_at_most_0_02': stat['mean'] <= .02,
                    'each_seed_deterioration_at_most_0_05': stat['max'] <= .05}
    accuracy_pass = all(all(v[k] for k in ('mean_at_least_threshold', 'all_three_seed_nonnegative', 'at_least_two_positive')) for v in accuracy.values())
    nll_pass = all(v['mean_deterioration_at_most_0_02'] and v['each_seed_deterioration_at_most_0_05'] for v in nll.values())
    groups, aggregates, native_aggregates = [], {}, {}
    for arm in ARMS:
        aggregates[arm] = aggregate(helper, [scores[s, arm] for s in SEEDS], [records[s, arm] for s in SEEDS])
        if arm in NEW:
            native_aggregates[arm] = aggregate(helper, [native[s, arm] for s in SEEDS], [records[s, arm] for s in SEEDS])
        for seed in SEEDS:
            value = scores[seed, arm]
            row = {'arm': arm, 'seed': seed, 'acquisition_origin': 'decoder_pilot' if arm in NEW else 'immutable_original_reference',
                   **{k: value[k] for k in ('quality', 'counts', 'classes')}, 'diagnostics': diagnostics[seed, arm],
                   'original_group_record': records[seed, arm]}
            if arm in NEW:
                row['exact_archived_native_at_decoded_selected'] = {k: native[seed, arm][k] for k in ('quality', 'counts', 'classes')}
            groups.append(row)
    return {'backbone': 'SAGE', 'primary_candidate': PRIMARY, 'new_banks': 21, 'new_optimizer_fits': 30,
            'new_native_bodies_fitted': 39, 'reused_reference_banks': 15, 'reused_reference_fits': 33,
            'groups_detail': groups, 'aggregates': aggregates, 'exact_native_aggregates_at_decoded_selected': native_aggregates,
            'all_fixed_context_contrasts': contrasts, 'exact_native_contrasts_at_decoded_selected': native_contrasts,
            'decoded_minus_exact_archived_native_each_new_arm': same_state,
            'frozen_decision': {'accuracy_criteria': accuracy, 'accuracy_screen_pass': accuracy_pass,
                                'pooled_NLL_protection_criteria': nll, 'pooled_NLL_protection_pass': nll_pass,
                                'protected_pooled_quality_development_clue': accuracy_pass and nll_pass},
            'member_quality_scope': 'Member quality is reported; no member accuracy gate vetoes pooled utility.',
            'native_component_causal_limit': 'Native states share decoded-selected checkpoints, stopping and selection; comparisons do not isolate pure training causality.',
            'costs': {'new30_optimizer_fits': helper.cost_summary(context['complete']['results']),
                      'reused33_selected_reference_fits': helper.cost_summary([records[s, a] for s in SEEDS for a in REFERENCES]),
                      'entire_original39_fit_parent_context_not_additive': helper.cost_summary(context['old_complete']['results']),
                      'native_operation_counts': context['complete']['operation_counts'],
                      'decoder_operation_counts': context['complete']['decoder_operation_counts'],
                      'decoder_graph_preparation_once': context['complete']['decoder_graph_preparation_once'],
                      'new_owner': context['owner'], 'original_owner': context['old_owner']}}


def encode(value, compact=False):
    return json.dumps(value, allow_nan=False, indent=None if compact else 2,
                      separators=(',', ':') if compact else None).encode() + b'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True, help='Fresh full JSON report path inside the research workspace')
    args = parser.parse_args()
    # Literal scientific route check precedes all project metadata reads.
    require(socket.gethostname() == 'anogena-2-0', 'Authorized scientific hostname required')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == [UUID], 'Sole authorized GPU UUID required')
    root, report_path = args.research_root.resolve(), args.report.resolve()
    require(root == PHASE and Path(__file__).resolve().is_relative_to(root), 'Literal authorized research workspace required')
    require(report_path.is_relative_to(root) and not report_path.exists(), 'Fresh in-scope report path required')
    manifest_path = Path(__file__).with_name('SOURCE.json')
    manifest = read(manifest_path)
    for name, digest in manifest['files'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Sealed reader source changed: ' + name)
    helper = load_math(root)
    context, expected, checkpoints, records = preflight(root, helper)
    import numpy as np
    helper.np = np
    arrays, ids, labels = payloads(np, expected)
    learned = matrices(np, checkpoints)
    scores, native, diagnostics = score_all(helper, np, arrays, records, labels, context)
    family = analyze(helper, scores, native, diagnostics, records, context, labels)
    report = {'complete': True, 'backbone': 'SAGE', 'seeds': SEEDS, 'roster': ARMS, 'banks': 36,
              'new_banks': 21, 'new_optimizer_fits': 30, 'new_native_bodies_fitted': 39,
              'route_trajectories_per_complete_roster_forward': 66, 'reused_reference_banks': 15,
              'reused_reference_fits': 33, 'total_optimizer_fits_represented': 63, 'TEST_access': False,
              'analysis_source_sha256': sha(Path(__file__)), 'reader_manifest_sha256': sha(manifest_path),
              'immutable_math_sha256': MATH_SHA, 'decision_sha256': POLICY['DECISION.md'],
              'reference_bindings_sha256': REFERENCE_SHA, 'ordered_VALID_hashes': VALID_HASHES,
              'selected_archive_hashes': {str(d['path']): d['sha256'] for d in expected.values()},
              'selected_checkpoint_custody': [{**d, 'path': str(d['path'])} for d in checkpoints],
              'checkpoint_digest_scope': 'No fit record contained a prior checkpoint SHA. These all-30 closure-time digests bind canonical paths/bytes to owner-pinned complete fit records; every digest is rechecked before and after read-only extraction.',
              'family': family, 'learned_decoder_matrices': learned,
              'source_custody': context,
              'native_authority': 'Exact archived float32 decoded and native pooled/member errors drive all integer counts. Stable FP64 NLL uses immutable log-softmax arithmetic. FP64 decision discrepancies remain separate diagnostics.',
              'strict_rival_definition': 'A wrong class strictly outranks truth in every member in archived raw scores; no epsilon. Same-state native strict-rival corrections use decoded archived served errors. Floating/tie caveats remain.',
              'transition_state_order': STATES,
              'coverage_identity': 'pooled correct = any-member coverage - lost correct alternatives + aggregation-only correct',
              'interpretation_limits': ['All new fits, original archives and new checkpoints close before numerical scoring; no partial outcome selects continuation.',
                  'Pooled accuracy and NLL are decisive and separately gated. Individual member weakness is diagnostic.',
                  'Decoded VALID selects checkpoints and stopping. Native comparisons at those states include selection and cannot isolate pure learning effects.',
                  'This is an encountered WikiCS development screen. Three optimizer seeds on one graph are not graph/split replication or independent node samples.',
                  'Paired df2 intervals and sign-flip summaries are descriptive optimization uncertainty; screening thresholds are not population significance.',
                  'B gauge arithmetic has established softmax ancestry. The retrained P=I contrast addresses neighbor-message utility; matrix norms do not establish it.',
                  'CPGNN, GBPN, CRF-RNN and collective ensemble classification are close ancestry. This reader makes no primitive novelty, exact posterior or manuscript acceptance claim.',
                  'All original records, failures/qualification, costs and overlap are preserved. The original 39-fit parent cost is context and is not added to the reused 33-fit subtotal.',
                  'Raw checkpoints remain host-side. No model/optimizer is constructed, no forward or re-decoding runs, and no TEST label is loaded.']}
    # Prepare every partition and size-check it before writing any output.
    documents = [(report_path, report, False, None)]
    contrast_partition = {k: v for k, v in family.items() if k != 'groups_detail'}
    documents.append((report_path.parent / 'SAGE_COMPLETE_CONTRASTS.json', contrast_partition, True, LIMIT))
    for arm in ARMS:
        documents.append((report_path.parent / ('SAGE_' + arm + '_DETAIL.json'),
                          {'backbone': 'SAGE', 'arm': arm, 'groups_detail': [r for r in family['groups_detail'] if r['arm'] == arm]}, True, LIMIT))
    documents.append((report_path.parent / 'ALL_LEARNED_DECODER_MATRICES.json', learned, True, LIMIT))
    summary = {k: v for k, v in report.items() if k not in ('family', 'source_custody', 'learned_decoder_matrices', 'selected_checkpoint_custody', 'selected_archive_hashes')}
    summary.update(frozen_decision=family['frozen_decision'], all_arm_aggregates=family['aggregates'],
                   exact_native_aggregates_at_decoded_selected=family['exact_native_aggregates_at_decoded_selected'],
                   costs=family['costs'], qualification=context['qualification'],
                   source_hashes={'new': context['complete']['source_sha256'], 'original': context['old_complete']['source_sha256']},
                   new_owner_end_sha256=context['owner_end_sha256'], new_complete_sha256=context['complete_sha256'],
                   original_owner_end_sha256=context['binding']['owner_end_sha256'], original_complete_sha256=context['binding']['complete_sha256'],
                   policy_sha256=POLICY, matrix_checkpoint_count=30, learned_matrix_count=39)
    prepared = []
    for path, value, compact, limit in documents:
        require(not path.exists(), 'Every output path must be new: ' + str(path))
        data = encode(value, compact)
        require(limit is None or len(data) < limit, 'Complete partition exceeds fixed 2 MB limit: ' + str(path))
        prepared.append((path, data))
    receipts = [{'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for path, data in prepared]
    summary.update(full_report=receipts[0], partitions=receipts[1:])
    summary_path = report_path.parent / 'COMPLETE_ANALYSIS_SUMMARY.json'
    require(not summary_path.exists() and len({p for p, _ in prepared}) == len(prepared)
            and summary_path not in {p for p, _ in prepared}, 'Every report/partition/summary path must be distinct and new')
    summary_data = encode(summary, True)
    require(len(summary_data) < LIMIT, 'Summary exceeds fixed 2 MB limit')
    prepared.append((summary_path, summary_data))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for path, data in prepared:
        with path.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'complete': True, 'summary': str(summary_path), 'full_report': receipts[0],
                      'partitions': receipts[1:], 'frozen_decision': family['frozen_decision']}))


if __name__ == '__main__':
    main()
