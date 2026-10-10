"""Local text/JSON/hash bookkeeping only; never opens scientific payloads."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
P = D.parent
REMOTE_REPO = '/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
REMOTE_PHASE = REMOTE_REPO + '/experiments_iclr/postsubmission_20260930'
BACKBONES = ('SAGE', 'GCN', 'GAT')
SEEDS = (7301, 7403, 7507)
ARMS = ('ordinary_M1', 'ordinary_genuine_I4', 'factorized_allmap_M1',
        'factorized_allmap_genuine_I4', 'shared4_unchanged')
if (D / 'SEAL.json').exists():
    raise SystemExit('Preserve sealed source; use a successor.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (D / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read(path):
    value = json.loads((P / path).read_text())
    input_bindings.append({'path': path, 'sha256': digest(P / path), 'bytes': (P / path).stat().st_size,
                           'scope': 'Selected closed descriptor/hash/config metadata; not numerical payload.'})
    return value


UTC, input_bindings = datetime.datetime.now(datetime.timezone.utc).isoformat(), []
sage = read('common_wrapper_SAGE_root_20261010_v1/ACTUAL_ANALYSIS.json')
native = read('common_wrapper_GCN_GAT_complete_family_root_20261010_v1/ACTUAL_ANALYSIS.json')
analyses = {'SAGE': sage, **native['families']}
families = []
for backbone in BACKBONES:
    local_root = 'common_wrapper_' + backbone + '_root_20261010_v1'
    cfg = read(local_root + '/CONFIG.json')
    frozen = read(local_root + '/FREEZE.json')
    a = analyses[backbone]
    assert a['complete'] and a['TEST_access'] is False and a['groups'] == 21 and a['fit_units'] == 39
    bindings = a['input_bindings']
    assert digest(P / (local_root + '/CONFIG.json')) == bindings['config_sha256']
    source_dir = ('common_wrapper_SAGE_full_family_source_20261010_v1' if backbone == 'SAGE'
                  else 'common_wrapper_native_backbone_full_family_source_20261010_v1')
    remote_source = REMOTE_PHASE + '/' + source_dir + '/run_family.py'
    assert digest(P / source_dir / 'run_family.py') == bindings['training_source_sha256'][remote_source]
    root = REMOTE_PHASE + '/' + local_root + '/actual_family_v1'
    banks = []
    for seed in SEEDS:
        for arm in ARMS:
            folder = root + '/' + arm + '_seed' + str(seed)
            archive = folder + '/selected_VALID.npz'
            assert archive in bindings['selected_archives']
            independent = 'genuine_I4' in arm
            members = 4 if independent or arm == 'shared4_unchanged' else 1
            banks.append({'key': f'{backbone}_{arm}_seed{seed}', 'arm': arm, 'seed': seed,
                          'members': members, 'archive': archive,
                          'archive_sha256': bindings['selected_archives'][archive],
                          'checkpoints': [folder + '/member' + str(m) + '/selected.pt'
                                          for m in range(4 if independent else 1)],
                          'historical_checkpoint_digest_in_local_descriptor': False})
    families.append({'backbone': backbone, 'family_root': root,
                     'family_source': REMOTE_PHASE + '/' + source_dir + '/run_family.py',
                     'config': REMOTE_PHASE + '/' + local_root + '/CONFIG.json',
                     'configuration': cfg, 'config_sha256': bindings['config_sha256'],
                     'complete_sha256': bindings['complete_family_sha256'],
                     'owner_end_sha256': bindings['owner_end_sha256'],
                     'frozen_inputs': frozen['bound_files'], 'banks': banks})
role_hashes = {'ids': sage['input_bindings']['ordered_VALID_ids_sha256'],
               'labels': sage['input_bindings']['ordered_VALID_labels_sha256']}
for a in analyses.values():
    assert a['input_bindings']['ordered_VALID_ids_sha256'] == role_hashes['ids']
    assert a['input_bindings']['ordered_VALID_labels_sha256'] == role_hashes['labels']
source_files = [{'path': str(p.relative_to(P)), 'sha256': digest(p)}
                for p in sorted(D.glob('*.py')) if p.name not in ('BUILD_SUPPORT.py', 'check_static.py')]
support = {'schema': 'common-wrapper-reliability-support-v1', 'UTC': UTC,
           'status': 'SOURCE_FROZEN_NO_RUNTIME_OR_FIT_ADMISSION', 'backbones': BACKBONES, 'seeds': SEEDS,
           'arms': ARMS, 'banks': 45, 'selected_checkpoint_units': 99, 'fullgraph_member_trajectories_for_export': 126,
           'source_files': source_files, 'families': families, 'ordered_VALID_hashes': role_hashes,
           'TEST_access': False, 'new_base_fits': 0,
           'graph_information': 'Existing TRAIN container contains permitted full11701-node x/edge_index and only580 TRAIN labels. VALID container contains5274 IDs/labels. No TEST labels exist in these containers.',
           'minimal_new_cache': 'VALID IDs and Mx5274x10 float64 neighbor means only. Full-node logits/probabilities are ephemeral unlabeled prediction support, never persisted. Existing selected_VALID archive owns every own-node field and fusion label.',
           'checkpoint_custody': 'Exact path/layout/selected step from the digest-bound closed complete descriptor. Original checkpoint digests were not retained locally; runtime captures/stability-checks current selected.pt digests and performs one finite tolerance-qualified VALID comparison. No invented historical digest.',
           'comparison_tolerance': {'atol': 1e-6, 'rtol': 1e-5, 'bitwise_required': False, 'one_export_no_replay_loop': True},
           'fusion': {'folds': 5, 'fold_seed': 11709, 'updates': 500, 'optimizer': 'full_batch_Adam_lr0.01_betas0.9/0.999_eps1e-8_weight_decay0',
                      'rule_order': ['temperature_global', 'temperature_member', 'reliability_self', 'linear_stacking', 'reliability_graph'],
                      'graph_and_self_parameters': 56, 'gate_shrinkage': 0.01, 'linear_mean_square_penalty': 0.01,
                      'duplicate_M1_rules_collapsed_before_outcomes': ['both_reliability_rules_equal_M1', 'temperature_member_equal_global'],
                      'nonduplicate_fit_calls': 855, 'maximum_small_head_updates': 427500,
                      'endpoint': 'fixed_last_update_no_held_fold_epoch_or_setting_selection', 'no_final_refit': True,
                      'supervision': 'Same5274 encountered VALID labels; each aggregator excludes its held fold. Base checkpoints already used all VALID for selection. No independent/whole-pipeline crossfit claim.'},
           'prior_dispositions': 'Saved5October graph stacker is direct ancestry; Amazon99 fixedNO_GO and direct12 exact-refit failure preserved. Current24/full9on77 and separable/exchange banks excluded.',
           'falsifier': 'No net served accuracy gain with acceptable NLL/Brier beyond strongest temperature/self/linear control; P=I matching graph rejects neighbor ingredient. Unused confirmation essential after a fully frozen pipeline.',
           'no_novelty_generalization_or_predictive_gain_claim': True}
write('FROZEN_SUPPORT.json', support)
for path in ['common_wrapper117_stored_error_diagnosis_root_20261010_v1/REPORT.md',
             'graph_aware_saved_prediction_aggregation_prior_scout_20261005_v1/REPORT.md',
             'graph_aware_aggregation_development_reuse_pairwise_amendment_20261005_v1/REPORT.md',
             'amazon_polynormer_logits_graph_moment_result_independent_review_20261005_v1/REPORT.md',
             'direct12_result_synthesis_root_20261008_v1/REPORT.md',
             'graph_context_postprediction_reliability_triage_20261010_v1/REPORT.md',
             'graph_context_postprediction_reliability_triage_20261010_v1/METHOD.json',
             'graph_context_postprediction_reliability_triage_20261010_v1/SEAL.json',
             'shared_fast_graph_model_interface_20261010_v1/common_routes.py',
             'common_wrapper_native_source_20261010_v1/models.py',
             'common_wrapper_native_source_20261010_v1/run_base.py',
             'common_wrapper_native_source_20261010_v1/run_common.py',
             'common_wrapper_SAGE_full_family_source_20261010_v1/run_family.py',
             'common_wrapper_native_backbone_full_family_source_20261010_v1/run_family.py',
             'common_wrapper_WikiCS_allocation_data_source_20261010_v1/acquire.py']:
    input_bindings.append({'path': path, 'sha256': digest(P / path), 'bytes': (P / path).stat().st_size,
                           'scope': 'Saved reports/prior method or native source inspection; no import or execution.'})
write('SOURCE_BINDINGS.json', {'UTC': UTC, 'inputs': input_bindings,
                              'model_payload_or_dataset_reads': False, 'remote_operations': False,
                              'imports_or_calls_of_inspected_native_source': False,
                              'primary_retrievals_or_new_read_credit': 0})
print(json.dumps({'support': str(D / 'FROZEN_SUPPORT.json'), 'banks': 45,
                  'selected_checkpoint_units': 99, 'fit_calls_if_root_admits': 855,
                  'no_model_or_fit_execution': True}))
