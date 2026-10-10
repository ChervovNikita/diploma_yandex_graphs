"""Complete stored-output retrieval reader; stdlib until all six closures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

PILOT = 'nonlocal_label_retrieval_pilot_root_20261010_v1'
MATH = 'private_feature_rotation_joined_complete_reader_20261010_v1/analysis.py'
MATH_SHA = '82482b235f9b3930f48fc0e3e8104302ab8878c1affc0e3357fbe4d246e7c84d'
DECISION_SHA = '850edaaee096deb093dbe6741dee7e9e070c19ea5fca3e7ddd85a463a07a1eb7'
LEARNING_SOURCE = 'nonlocal_train_label_retrieval_accuracy_hypothesis_20261010_v1/run_retrieval_family.py'
LEARNING_SHA = 'a6144527a6f5e551deeecb89cf312cb54473df69a48e4c0cd6c27a08ae01e0fc'
MEMORY_SOURCE = 'nonlocal_train_label_retrieval_accuracy_hypothesis_20261010_v1/label_memory.py'
MEMORY_SHA = 'c568c41f58e741f96b13a552bbac4df6ef137e280db8629a6165d732c760145f'
BACKBONES, SEEDS = ('SAGE', 'GCN', 'GAT'), (7301, 7403, 7507)
NEW = ('live_shared4', 'detached_shared4', 'ordinary_M1_live', 'genuine_ordinary_I4_live')
REFERENCES = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1', 'factorized_allmap_genuine_I4', 'shared4_unchanged')
ARMS = REFERENCES + NEW
LIVE, DETACHED, LIVE_M1, LIVE_I4 = NEW
VALID_HASHES = {'ids': '48d17843cf300ef7ec3d09e5aaaff26bd81f55a0af73f7ae03d6df8a7a700801',
                'labels': '2ab8078de1ca949e111b8cfec4a41c8c04ca8ca4be86478daf58d2e683f4cb12'}
POLICY = {
 'SAGE': {'FREEZE.json': 'c0d3d876633cfa1e0efb2055e68b00cf941dc3e8b4e3b9c07bc709014503337a', 'CONFIG.json': '5a52f32492b8c6a18ae256859eca821bfe5cd7320b7e9de8b983a1372bca7144', 'REFERENCE_BINDINGS.json': 'bb11feed32943dc7d0e3d83c831605254a432b497ff7546a60b87e5d29fb52b4'},
 'GCN': {'FREEZE.json': '6591c53aed1fb86daef71e9588a23a6200777b55592cbf73fd5b3fa30d220bee', 'CONFIG.json': 'd605f3e15f50542f3786b73fd68e4ffee7bffcc6a708ce15a1cc9308194ec604', 'REFERENCE_BINDINGS.json': '6f01622ac03d965fb37748f286734d2d6eeb212486285252fb08617b3795b6ef'},
 'GAT': {'FREEZE.json': '495d0ee30c573ce1c9675bcb3ecca7a993f2556468de74b037dd3887adc530bd', 'CONFIG.json': 'd4a83e94b7ca11ee23eadea4686b31892a52f0de8155531743a0c5c2dccb9f59', 'REFERENCE_BINDINGS.json': 'ab33b85c99ec4484c8fe68780c791447bfd71447d116faacd8bd337395942034'}}
QUALIFICATION_HASHES = {'ACTUAL_QUALIFICATION_V2.json': '9ac8f392859979649b2ba424dd908ac6e78a83ca9075edae4232ab6702de2342',
                        'QUALIFICATION_TRANSPORT.json': 'c6c442f3234a3b9a6b381c4d4b3091dddd8a473dfbaef698c4eba9b6aacc4e15'}


def sha(path):
    with Path(path).open('rb') as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
        return value.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def load_math(root):
    path = root / MATH
    require(sha(path) == MATH_SHA, 'Immutable stored-output math required')
    spec = importlib.util.spec_from_file_location('retrieval_stored_count_math', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(root, helper):
    here, repo = root / PILOT, root.parent.parent
    require(sha(here / 'DECISION.md') == DECISION_SHA and sha(root / LEARNING_SOURCE) == LEARNING_SHA
            and sha(root / MEMORY_SOURCE) == MEMORY_SHA, 'Sealed pilot rule/learning source required')
    metadata = {}
    # Gate every new and reused owner and header before any complete outcome
    # object. No partial family can choose a contrast or continuation.
    for backbone in BACKBONES:
        folder = here / backbone
        for name, digest in POLICY[backbone].items():
            require(sha(folder / name) == digest, 'Frozen pilot policy changed')
        cfg, freeze, binding = read(folder / 'CONFIG.json'), read(folder / 'FREEZE.json'), read(folder / 'REFERENCE_BINDINGS.json')
        for bound in freeze['bound_files']:
            require(sha(repo / bound['path']) == bound['sha256'], 'Frozen source/config/data binding changed')
        require(cfg['backbone'] == backbone and cfg['seeds'] == list(SEEDS) and cfg['query_seed_offset'] == 5000011
                and cfg['hidden'] == 128 and cfg['max_updates'] == 1000 and cfg['patience_updates'] == 300
                and freeze['new_fit_units'] == 21 and freeze['expected_groups'] == 12
                and freeze['all_family_fits_across_backbones'] == 63 and freeze['expected_complete_pilot_groups'] == 36
                and freeze['TEST_access'] is False, 'Fixed new12/21 family and complete36/63 pilot')
        owner = helper.closed_owner(folder / 'OWNER_END.json')
        output = folder / 'actual_family_v1'
        header = helper.completion_header(output / 'COMPLETE_FAMILY.json')
        require(header['complete'] and header['groups'] == header['expected_groups'] == 12
                and header['fit_units'] == header['expected_fit_units'] == 21 and header['TEST_access'] is False,
                'Every new family complete12/21 before metrics')
        old_output = Path(binding['family_root'])
        require(old_output.is_relative_to(root) and sha(old_output / 'COMPLETE_FAMILY.json') == binding['complete_sha256']
                and sha(old_output.parent / 'OWNER_END.json') == binding['owner_end_sha256'], 'Exact original reference closure')
        old_owner = helper.closed_owner(old_output.parent / 'OWNER_END.json')
        old_header = helper.completion_header(old_output / 'COMPLETE_FAMILY.json')
        require(old_header['complete'] and old_header['groups'] == old_header['expected_groups'] == 21
                and old_header['fit_units'] == old_header['expected_fit_units'] == 39 and old_header['TEST_access'] is False,
                'Every original native family complete21/39')
        metadata[backbone] = {'folder': folder, 'output': output, 'cfg': cfg, 'freeze': freeze, 'owner': owner,
                              'header': header, 'binding': binding, 'old_output': old_output,
                              'old_owner': old_owner, 'old_header': old_header}
    require(all({k: v for k, v in metadata[b]['cfg'].items() if k != 'backbone'}
                == {k: v for k, v in metadata[BACKBONES[0]]['cfg'].items() if k != 'backbone'} for b in BACKBONES),
            'Companion pilot configs differ only by backbone')
    expected, original_records = {}, {}
    for backbone, context in metadata.items():
        complete_path = context['output'] / 'COMPLETE_FAMILY.json'
        require(sha(complete_path) == context['owner']['complete_sha256'], 'Owner binds complete new results')
        complete = read(complete_path)
        old = read(context['old_output'] / 'COMPLETE_FAMILY.json')
        require({k: v for k, v in complete.items() if k not in ('results', 'comparisons', 'cost_scope')} == context['header']
                and complete['config_sha256'] == POLICY[backbone]['CONFIG.json']
                and read(context['output'] / 'CONFIG.json') == context['cfg']
                and helper.source_hash(complete, LEARNING_SOURCE) == LEARNING_SHA
                and helper.source_hash(complete, MEMORY_SOURCE) == MEMORY_SHA, 'Complete source/config identity unchanged')
        binding = context['binding']
        require(sha(binding['config']) == binding['config_sha256'] == old['config_sha256']
                and read(binding['config']) == binding['configuration'], 'Immutable native reference config')
        comparison_cfg = {k: v for k, v in context['cfg'].items() if k not in ('label_memory', 'query_seed_offset')}
        require(comparison_cfg == dict(binding['configuration'], backbone=backbone), 'Common native operating point')
        for bound in binding['frozen_inputs']:
            require(sha(repo / bound['path']) == bound['sha256'], 'Original native reference source/data changed')
        rows = {(r['seed'], r['arm']): r for r in complete['results']}
        require(len(rows) == len(complete['results']) == 12 and set(rows) == {(s, a) for s in SEEDS for a in NEW}
                and sum(len(r['fits']) for r in rows.values()) == 21
                and helper.sum_operations(complete['results']) == complete['operation_counts'], 'Full native acquisition/operation roster')
        old_rows = {(r['seed'], r['arm']): r for r in old['results']}
        require(len(old_rows) == 21 and len(old['results']) == 21, 'Original family complete unique groups')
        for (seed, arm), row in rows.items():
            members, fits = (1 if arm == LIVE_M1 else 4), (4 if arm == LIVE_I4 else 1)
            require(row['serving'] == 'probability_mean' and [u['member'] for u in row['fits']] == list(range(fits)), 'Fixed independent/native bank units')
            for unit in row['fits']:
                routes = 4 if arm in (LIVE, DETACHED) else 1
                native_seeds = [seed + context['cfg']['member_seed_stride'] * (unit['member'] + m) for m in range(routes)]
                require(unit['native_factor_dropout_seeds'] == [[s, s + context['cfg']['factor_seed_offset'], s + context['cfg']['dropout_seed_offset']] for s in native_seeds]
                        and unit['query_rng_seed'] == seed + 5000011 and unit['query_rng_restored_selected_step'] == unit['selected_step']
                        and 1 <= unit['selected_step'] <= unit['completed_updates'] <= 1000, 'Original route/query/selection identities')
            path = context['output'] / f'{arm}_seed{seed}/selected_VALID.npz'
            key = (backbone, seed, arm)
            expected[key] = {'path': path, 'sha256': sha(path), 'members': members, 'new': True}
            original_records[key] = row
        reference_index = {(b['seed'], b['arm']): b for b in binding['banks']}
        require(len(reference_index) == len(binding['banks']) == 15
                and set(reference_index) == {(s, a) for s in SEEDS for a in REFERENCES}, 'Exact15 reference banks per backbone')
        for (seed, arm), bound in reference_index.items():
            path = Path(bound['archive'])
            require(path.is_relative_to(root) and sha(path) == bound['archive_sha256'], 'Original reference archive authority')
            key = (backbone, seed, arm)
            expected[key] = {'path': path, 'sha256': bound['archive_sha256'], 'members': bound['members'], 'new': False}
            original_records[key] = old_rows[seed, arm]
        context.update(complete=complete, old_complete=old, complete_sha256=sha(complete_path), old_complete_sha256=binding['complete_sha256'])
    require(len(expected) == len(original_records) == 81, 'All36 new plus45 reference archive identities before numerical import')
    qualification = {}
    for name, digest in QUALIFICATION_HASHES.items():
        require(sha(here / name) == digest, 'Qualification history changed')
        qualification[name] = read(here / name)
    q = qualification['ACTUAL_QUALIFICATION_V2.json']
    require(q['qualified'] is True and q['TRAIN_only'] is True and q['validation_accuracy_or_NLL_scored'] is False
            and q['TEST_access'] is False and q['scientific_quality_evidence'] is False
            and qualification['QUALIFICATION_TRANSPORT.json']['exit_code'] != 0, 'Successful TRAIN-only and preserved failed qualification')
    return metadata, expected, original_records, qualification


def payloads(np, expected):
    arrays, ids, labels = {}, None, None
    old_keys = {'ids', 'y', 'raw_logits', 'probability_mean', 'member_errors', 'pooled_errors'}
    new_keys = old_keys - {'raw_logits'} | {'mixed_log_probs', 'native_logits'}
    for key, descriptor in expected.items():
        require(sha(descriptor['path']) == descriptor['sha256'], 'Archive changed after full identity admission')
        with np.load(descriptor['path'], allow_pickle=False) as archive:
            require(set(archive.files) == (new_keys if descriptor['new'] else old_keys), 'Exact original/new stored VALID schema')
            a = {k: archive[k].copy() for k in archive.files}
        raw_name = 'mixed_log_probs' if descriptor['new'] else 'raw_logits'
        m = descriptor['members']
        require(a['ids'].dtype == a['y'].dtype == np.int64 and a['ids'].shape == a['y'].shape == (5274,)
                and a[raw_name].dtype == a['probability_mean'].dtype == np.float32
                and a[raw_name].shape == (m, 5274, 10) and a['probability_mean'].shape == (5274, 10)
                and a['member_errors'].shape == (m, 5274) and a['pooled_errors'].shape == (5274,)
                and a['member_errors'].dtype == a['pooled_errors'].dtype == np.bool_
                and np.isfinite(a[raw_name]).all() and np.isfinite(a['probability_mean']).all(), 'Finite exact native role/layout')
        if descriptor['new']:
            require(a['native_logits'].shape == (m, 5274, 10) and a['native_logits'].dtype == np.float32
                    and np.isfinite(a['native_logits']).all(), 'Separate finite native component logits')
        if ids is None:
            ids, labels = a['ids'], a['y']
            require(len(np.unique(ids)) == 5274 and ids.min() >= 0 and ids.max() < 11701 and set(labels.tolist()) == set(range(10))
                    and {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in [('ids', ids), ('labels', labels)]} == VALID_HASHES,
                    'Exact frozen ordered VALID roles')
        require(np.array_equal(ids, a['ids']) and np.array_equal(labels, a['y'])
                and np.array_equal(a['probability_mean'].argmax(1) != labels, a['pooled_errors']), 'All81 common roles and authoritative served decisions')
        arrays[key] = a
    return arrays, ids, labels


def probabilities(helper, np, logits):
    raw = logits.astype(np.float64)
    return np.exp(raw - helper.logsumexp(raw, -1)[..., None])


def strict_mask(np, probability, labels):
    truth = np.take_along_axis(probability, labels[None, :, None], axis=2)
    return (probability > truth).all(0).any(1)


def strict_counts(np, mask, correct, labels):
    return {'nodes': int(mask.sum()), 'corrected': int((mask & correct).sum()),
            'classes': [{'class': k, 'nodes': int((mask & (labels == k)).sum()),
                         'corrected': int((mask & correct & (labels == k)).sum())} for k in range(10)]}


def score_all(helper, np, arrays, labels, records):
    scores, native, diagnostics = {}, {}, {}
    for key, a in arrays.items():
        if key[2] not in NEW:
            scores[key] = helper.score(a)
            p = probabilities(helper, np, a['raw_logits'])
            diagnostics[key] = {'common_strict_wrong_rival': strict_counts(np, strict_mask(np, p, labels), scores[key]['P'], labels)}
            continue
        mixed = {**a, 'raw_logits': a['mixed_log_probs']}
        value = helper.score(mixed)
        scores[key] = value
        p_native = probabilities(helper, np, a['native_logits'])
        n = {'ids': a['ids'], 'y': a['y'], 'raw_logits': a['native_logits'],
             'probability_mean': p_native.mean(0), 'member_errors': p_native.argmax(-1) != labels,
             'pooled_errors': p_native.mean(0).argmax(1) != labels}
        component = helper.score(n)
        native[key] = component
        p_mixed = probabilities(helper, np, a['mixed_log_probs'])
        mixed_strict, native_strict = strict_mask(np, p_mixed, labels), strict_mask(np, p_native, labels)
        reported = records[key]['valid']
        require(value['counts']['coverage'] == reported['correct_alternative_count']
                and value['counts']['lost_correct_alternatives'] == reported['coverage_lost_in_pooling']
                and value['counts']['pooled_correct'] == round(reported['accuracy'] * 5274), 'Original mixed selected score/count authority')
        diagnostics[key] = {'mixed_common_strict_wrong_rival': strict_counts(np, mixed_strict, value['P'], labels),
             'native_component_FP64_common_strict_wrong_rival_corrected_by_mixed': strict_counts(np, native_strict, value['P'], labels),
             'mixed_vs_native_component_FP64': helper.comparison(value, component, labels),
             'stable_mixed_NLL_minus_original_reported_NLL': value['quality']['pooled_nll'] - reported['nll'],
             'stable_native_component_NLL_minus_original_reported_NLL': component['quality']['pooled_nll'] - reported['native_probability_pool_nll'],
             'reported_native_component_at_mixed_selected_state': {k: v for k, v in reported.items() if k.startswith('native_')},
             'native_component_FP64_correct_count_minus_reported_native_float32': component['counts']['pooled_correct'] - round(reported['native_probability_pool_accuracy'] * 5274),
             'native_component_FP64_member_count_minus_reported_native_float32': [int(component['C'][m].sum()) - round(reported['native_member_accuracy'][m] * 5274) for m in range(len(component['C']))],
             'mixed_raw_argmax_disagreements_with_archived_member_errors': int(((a['mixed_log_probs'].argmax(-1) != labels) != a['member_errors']).sum()),
             'mixed_uniform_FP64_pool_correct_minus_archived_float32': int((p_mixed.mean(0).argmax(1) == labels).sum()) - value['counts']['pooled_correct'],
             'native_component_scope': 'FP64 nodewise reconstruction at the mixed-selected checkpoint; reported float32 native component scores remain separate authority; selection/stopping differs from original native controls.'}
    return scores, native, diagnostics


def family_report(helper, backbone, scores, native, diagnostics, records, context, labels):
    local = {(s, a): scores[backbone, s, a] for s in SEEDS for a in ARMS}
    pairs = [(LIVE, a) for a in ARMS if a != LIVE]
    pairs += [(a, ref) for a in NEW if a != LIVE for ref in REFERENCES]
    pairs += [(LIVE_I4, LIVE_M1), (DETACHED, REFERENCES[-1])]
    contrasts = {a + '_minus_' + b: helper.contrast(local, a, b, labels) for a, b in pairs}
    primary, accuracy, nll = {}, {}, {}
    for control in (LIVE_I4, LIVE_M1, DETACHED):
        value = contrasts[LIVE + '_minus_' + control]
        acc, risk = value['quality_deltas']['pooled_accuracy_pct'], value['quality_deltas']['pooled_nll']
        threshold = .2 if control == LIVE_I4 else .1
        primary[control] = value
        accuracy[control] = {'mean_required_accuracy_pp': threshold, 'mean_at_least_threshold': acc['mean'] >= threshold,
                             'all_three_seed_nonnegative': acc['nonnegative_seed_count'] == 3, 'at_least_two_positive': acc['positive_seed_count'] >= 2}
        nll[control] = {'mean_deterioration_at_most_0_02': risk['mean'] <= .02, 'each_seed_deterioration_at_most_0_05': risk['max'] <= .05}
    native_local = {(s, a): native[backbone, s, a] for s in SEEDS for a in NEW}
    component_contrasts = {LIVE + '_minus_' + a: helper.contrast(native_local, LIVE, a, labels) for a in NEW if a != LIVE}
    mixed_component = {}
    for arm in NEW:
        values = {(s, 'mixed'): local[s, arm] for s in SEEDS}
        values.update({(s, 'native_FP64'): native_local[s, arm] for s in SEEDS})
        mixed_component[arm] = helper.contrast(values, 'mixed', 'native_FP64', labels)
    groups, aggregates = [], {}
    for arm in ARMS:
        rows = [local[s, arm] for s in SEEDS]
        aggregates[arm] = {'mean_quality': {k: statistics.mean(v['quality'][k] for v in rows) for k in ('pooled_accuracy_pct', 'pooled_nll', 'mean_member_accuracy_pct', 'mean_member_nll', 'worst_member_accuracy_pct')},
                           'summed_three_seed_readout_counts': {k: sum(v['counts'][k] for v in rows) for k in rows[0]['counts'] if k != 'member_correct'},
                           'costs': helper.cost_summary([records[backbone, s, arm] for s in SEEDS])}
        for seed in SEEDS:
            key, value = (backbone, seed, arm), local[seed, arm]
            row = {'arm': arm, 'seed': seed, 'acquisition_origin': 'retrieval_pilot' if arm in NEW else 'immutable_native_reference',
                   **{k: value[k] for k in ('quality', 'counts', 'classes')}, 'diagnostics': diagnostics[key], 'original_group_record': records[key]}
            if arm in NEW:
                row['native_component_FP64_at_mixed_selected_state'] = {k: native[key][k] for k in ('quality', 'counts', 'classes')}
            groups.append(row)
    return {'backbone': backbone, 'new_groups': 12, 'reused_reference_groups': 15, 'new_fit_units': 21,
            'reused_reference_fit_units': 33, 'groups_detail': groups, 'aggregates': aggregates,
            'primary_same_information_contrasts': primary, 'all_fixed_context_contrasts': contrasts,
            'accuracy_transfer_criteria': accuracy, 'accuracy_transfer_within_backbone': all(all(v[k] for k in ('mean_at_least_threshold', 'all_three_seed_nonnegative', 'at_least_two_positive')) for v in accuracy.values()),
            'proper_risk_protection_criteria': nll, 'proper_risk_protection_within_backbone': all(all(v.values()) for v in nll.values()),
            'native_component_FP64_contrasts_at_mixed_selected_states': component_contrasts,
            'mixed_minus_native_component_FP64_each_new_arm': mixed_component,
            'native_component_causal_limit': 'New checkpoints/stopping selected by mixed VALID accuracy. Native component contrasts combine acquisition and selection; they are not a pure causal acquisition effect.',
            'costs': {'new21_acquisitions': helper.cost_summary(context['complete']['results']),
                      'reused33_reference_acquisitions': helper.cost_summary([records[backbone, s, a] for s in SEEDS for a in REFERENCES]),
                      'entire_original39_acquisition_parent': helper.cost_summary(context['old_complete']['results']),
                      'actual_new_process_operation_counts': context['complete']['actual_process_operation_counts'],
                      'new_owner': context['owner'], 'original_owner': context['old_owner']}}


def write(path, value, compact=False, limit=None):
    encoded = json.dumps(value, allow_nan=False, indent=None if compact else 2,
                         separators=(',', ':') if compact else None).encode() + b'\n'
    require(limit is None or len(encoded) < limit, 'Partition exceeds fixed retrieval size limit')
    with Path(path).open('xb') as stream:
        stream.write(encoded)
    return {'path': str(path), 'bytes': len(encoded), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root, report_path = args.research_root.resolve(), args.report.resolve()
    require(report_path.is_relative_to(root) and not report_path.exists(), 'Fresh in-scope report path')
    manifest = read(Path(__file__).with_name('SOURCE.json'))
    require(manifest['files']['analysis.py'] == sha(Path(__file__)), 'Sealed reader source required')
    helper = load_math(root)
    contexts, expected, records, qualification = preflight(root, helper)
    import numpy as np
    helper.np = np
    arrays, ids, labels = payloads(np, expected)
    require(len(arrays) == 81, 'All81 array identities before scoring')
    scores, native, diagnostics = score_all(helper, np, arrays, labels, records)
    families = {b: family_report(helper, b, scores, native, diagnostics, records, contexts[b], labels) for b in BACKBONES}
    accuracy = all(families[b]['accuracy_transfer_within_backbone'] for b in BACKBONES)
    protection = all(families[b]['proper_risk_protection_within_backbone'] for b in BACKBONES)
    report = {'complete': True, 'banks': 81, 'new_banks': 36, 'new_fit_units': 63, 'reused_reference_banks': 45,
              'reused_reference_fit_units': 99, 'backbones': BACKBONES, 'seeds': SEEDS, 'roster': ARMS, 'TEST_access': False,
              'analysis_source_sha256': sha(Path(__file__)), 'reader_manifest_sha256': sha(Path(__file__).with_name('SOURCE.json')),
              'immutable_math_reader_sha256': MATH_SHA, 'decision_sha256': DECISION_SHA,
              'ordered_VALID_hashes': VALID_HASHES, 'selected_archive_hashes': {str(d['path']): d['sha256'] for d in expected.values()},
              'families': families, 'frozen_decision': {'accuracy_rule_all_three_backbones': accuracy,
                                                       'pooled_NLL_protection_all_three_backbones': protection,
                                                       'protected_pooled_quality_development_clue': accuracy and protection},
              'source_custody': {b: {'new_owner_end_sha256': sha(c['folder'] / 'OWNER_END.json'),
                  'new_complete_sha256': c['complete_sha256'], 'freeze_sha256': POLICY[b]['FREEZE.json'],
                  'config_sha256': POLICY[b]['CONFIG.json'], 'reference_binding_sha256': POLICY[b]['REFERENCE_BINDINGS.json'],
                  'original_complete_sha256': c['old_complete_sha256'], 'original_owner_end_sha256': c['binding']['owner_end_sha256'],
                  'original_source_sha256': c['old_complete']['source_sha256'], 'new_source_sha256': c['complete']['source_sha256']}
                  for b, c in contexts.items()},
              'qualification_and_failed_prefix': qualification,
              'native_authority': 'Archived mixed and original-reference float32 pooled/member decisions unchanged; stable FP64 NLL uses the original raw/mixed log-score convention. Native nodewise FP64 component is separate from reported float32 native metrics.',
              'strict_rival_definition': 'A wrong class strictly outranks truth in every member in the named probability representation. No epsilon. Retrieval can change native probabilities; mixed common strict rivals still obstruct scalar convex pooling in exact arithmetic. Floating/tie caveats remain.',
              'coverage_identity': 'pooled correct = any-member coverage - lost correct alternatives + aggregation-only correct',
              'interpretation_limits': ['All three new families and all reused original families close before metrics; no partial outcome chooses a continuation.',
                  'Mixed VALID accuracy selects new checkpoints and stopping. Native components at those states cannot isolate a causal acquisition effect.',
                  'Queries are excluded from every retrieval support, but native loss still supervises queries; no uniform-mask unbiasedness claim.',
                  'Half-support training and all-anchor serving differ. This is encountered development, not unused whole-pipeline confirmation.',
                  'Three seeds are optimization repeats on one connected graph; summed readouts and node counts are not independent graph replications.',
                  'Qualification derivatives/tolerances are engineering evidence, not predictive quality. Actual concurrent costs establish no isolated speed claim.',
                  'Final mixed accuracy and NLL are separate from member/native quality. Weak members do not forbid useful pooled predictions.',
                  'Matching Networks/NCA/TPN/UniMP ancestry is retained. A passing screen requires capable published references and unused confirmation before broader claims.']}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    full = write(report_path, report)
    partitions = []
    for backbone in BACKBONES:
        family = families[backbone]
        # Keep each complete contrast partition small; detailed records split
        # by declared arm, without selecting by observed quality.
        result = {k: v for k, v in family.items() if k != 'groups_detail'}
        partitions.append(write(report_path.parent / (backbone + '_COMPLETE_RESULTS.json'), result, True, 2000000))
        for arm in ARMS:
            value = {'backbone': backbone, 'arm': arm, 'groups_detail': [r for r in family['groups_detail'] if r['arm'] == arm]}
            partitions.append(write(report_path.parent / (backbone + '_' + arm + '_DETAIL.json'), value, True, 2000000))
    summary = {k: v for k, v in report.items() if k not in ('families', 'selected_archive_hashes')}
    summary.update(full_report=full, partitions=partitions,
                   per_backbone_flags={b: {k: families[b][k] for k in ('accuracy_transfer_criteria', 'accuracy_transfer_within_backbone', 'proper_risk_protection_criteria', 'proper_risk_protection_within_backbone')} for b in BACKBONES},
                   all_arm_aggregates={b: families[b]['aggregates'] for b in BACKBONES})
    saved = write(report_path.parent / 'COMPLETE_ANALYSIS_SUMMARY.json', summary, True, 2000000)
    print(json.dumps({'complete': True, 'summary': saved, 'full_report': full,
                      'partitions': partitions, 'frozen_decision': report['frozen_decision']}))


if __name__ == '__main__':
    main()
