"""Stdlib custody gates for one separately released frozen-all25 TEST invocation."""
from pathlib import Path
import importlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
PHASE = Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git/experiments_iclr/postsubmission_20260930')
EXECUTION = PHASE / 'ncnc_frozen_all25_heldout_release_preparation_20261004_v2'
OUTPUT = EXECUTION / 'heldout/run01'
AUDIT_SOURCE = PHASE / 'ncnc_selected_checkpoint_metric_audit_v4_source_20261004'
AUDIT_SOURCE_SHA = 'fd59e048a53695c945c6cfb613a77ce0e4e21c2631ce56085348c3941f5ec6d6'
AUDIT_RESULT_SHA = '7065c9c10c1afb8afee60aebe8fbfd3ed7e34a737f1feedb2ddc084f2e4da9e6'
AUDIT_PHYSICAL_SHA = '2a3ef829fba96f77e0330efbc5509ecee778601c2a47c03318b1ab9e21711528'
TEST_META_SHA = '442d5f4a6d861da6b97431f49197412df3d4ac957cb0769df61327dbd7e14a9a'
TEST_FILE_SHA = '75315d4e3f3d73fe5cc1079758adecf1a87d17bd4dc4466701b5cbdfcbcac006'
LOCK_SHA = 'ab1d6a4b3a0fd02bcbb2db0a50764362cff7b20206eda6a805e89f8880393f50'
PROFILE = 'True2'
POLICY = {'graph': 'native_TRAIN_plus_VALID', 'graph_records': 'original_TRAIN_then_original_VALID_positive_order', 'TEST_rows_added_to_graph': False, 'query_target_removal': False, 'edge_weights_used': False, 'query_order': 'original_official_TEST_positive_and_shared_negative_order', 'eval_batch_size': 131072, 'serving_pool': 'mean_raw_logits', 'metric': 'official_ogbl_collab_Hits@50', 'profile': PROFILE, 'scorer_calls': 40, 'served_cells': 25, 'training_updates': 0, 'retry': False, 'refit_reselection_calibration': False}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha(path):
    from hashlib import sha256
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def pin_value(pin, *, decode=False):
    path = Path(pin['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'Noncanonical input pin')
    require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Input pin changed: ' + str(path))
    return json.loads(path.read_text()) if decode else path

def manifest(root, expected):
    require(root.is_absolute() and root.resolve() == root and sha(root / 'MANIFEST.json') == expected, 'Source manifest changed')
    rows = json.loads((root / 'MANIFEST.json').read_text())['files']
    require(rows and len({r['path'] for r in rows}) == len(rows), 'Duplicate/empty source manifest')
    for row in rows:
        path = root / row['path']
        require(path.resolve() == path and path.is_relative_to(root), 'Source payload escaped')
        require(path.stat().st_size == row.get('bytes', row.get('size')) and sha(path) == row['sha256'], 'Source payload changed')

def audit_gate(release, api):
    audit = pin_value(release['v4_audit_result'], decode=True)
    physical = pin_value(release['v4_audit_physical_terminal'], decode=True)
    require(release['v4_audit_result']['sha256'] == AUDIT_RESULT_SHA, 'Different original all25 audit')
    require(release['v4_audit_physical_terminal']['sha256'] == AUDIT_PHYSICAL_SHA, 'Different original audit physical receipt')
    require(audit['status'] == 'ALL25_SELECTED_CHECKPOINT_METRIC_CONSISTENCY_PASS' and audit['failures'] == [], 'All25 v4 audit must pass')
    require(physical['status'] == 'PHYSICALLY_COMPLETE' and physical['exit_code'] == 0 and physical['root_release_sha256'] == release['v4_audit_release']['sha256'], 'All25 audit physical terminal differs')
    require(audit['root_release_sha256'] == release['v4_audit_release']['sha256'] and audit['family_lock'] == release['family_lock'], 'Audit release/lock identity differs')
    require(audit['sidecar_manifest_sha256'] == physical['sidecar_manifest_sha256'] == AUDIT_SOURCE_SHA, 'Audit source identity differs')
    require(len(audit['cells']) == 25 and {(c['arm'], c['base_seed']) for c in audit['cells']} == api.EXPECTED_CELLS and all(c['status'] == 'PASS' for c in audit['cells']), 'Whole audited cohort required')
    require(all(audit['work'][k] == 120 for k in ('planned', 'attempted', 'entered_original_scorer', 'returned', 'completed_validated')), 'All120 audited slots required')
    require(audit['training_updates'] == 0 and audit['TEST_opened'] is False and audit['old_exact_replay_qualified'] is False, 'Audit scope changed')
    custody = audit['final_input_custody']
    require(all(custody[k] == 'PASS' for k in ('source_release_authority_qualification_actual_data_runtime_custody', 'family_journal_selected_bytes', 'loaded_typed_canonical_rows')), 'Audit final custody incomplete')
    return audit

def metadata_admission(release_path, output, *, expected_sha, require_fresh=True):
    release_path, output = Path(release_path), Path(output)
    require(release_path.is_absolute() and release_path.resolve() == release_path and release_path.parent == EXECUTION, 'Foreign heldout release path')
    require(output == OUTPUT and output.resolve() == output and sha(release_path) == expected_sha, 'Heldout release/output binding differs')
    release = json.loads(release_path.read_text())
    require(release['schema'] == 'ncnc-frozen-all25-one-time-heldout-root-release-v1' and release['execution_enabled'] is True and release['root_authorization_reference'], 'Separate heldout root release required')
    require(release['TEST_access_authorized'] is True and release['evaluation_once'] is True and release['policy'] == POLICY, 'Frozen one-time TEST policy differs')
    invocation = dict(stage='frozen_all25_heldout_confirmation', output_directory=str(output), cuda_visible_devices=release['cuda_visible_devices'])
    require(release['authorized_stages'] == [invocation['stage']] and release['authorized_invocations'] == [invocation], 'Exact one heldout invocation required')
    require(release['cuda_visible_devices'] == 'GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998' and os.environ.get('CUDA_VISIBLE_DEVICES') == release['cuda_visible_devices'], 'Exact admitted GPU required')
    manifest(HERE, release['heldout_source_manifest_sha256'])
    manifest(AUDIT_SOURCE, AUDIT_SOURCE_SHA)
    sys.path.insert(0, str(AUDIT_SOURCE))
    api = importlib.import_module('replay_gate')
    require(Path(api.__file__).resolve() == AUDIT_SOURCE / 'replay_gate.py', 'Original audit gate shadowed')
    audit_release_path = pin_value(release['v4_audit_release'])
    require(release['family_lock']['sha256'] == LOCK_SHA, 'Original frozen lock changed')
    audit = audit_gate(release, api)
    context = api.admission_context(audit_release_path, release['v4_audit_output_directory'], expected_release_sha256=release['v4_audit_release']['sha256'])
    require(context['identity'] == release['identity'] == audit['identity'] and context['release']['family_lock'] == release['family_lock'], 'Heldout original family identity differs')
    require(context['release']['data_authority'] == release['TRAIN_VALID_data_authority'] and context['release']['runtime_authority'] == release['runtime_authority'], 'Inherited data/runtime pins differ')
    require(context['release']['unit_custody'] == release['unit_custody'], 'Original selected-state/unit pins changed')
    context['output'] = output
    context.update(heldout_release=release, heldout_release_path=release_path, heldout_release_sha256=expected_sha, heldout_source_sha256=release['heldout_source_manifest_sha256'])
    plan = context['plan']
    require(plan['heldout_barrier']['root_separate_release_required'] and plan['heldout_barrier']['TEST_open_now'] is False, 'Frozen heldout barrier differs')
    require(plan['data_graph']['eventual_test_graph_policy'] == 'native-collab-VAL-at-TEST; complete TRAIN+VALID positives for encoder and all query structure', 'Frozen TEST topology differs')
    require(plan['primary_contrast']['candidate'] == api.ARMS[2] and plan['primary_contrast']['control'] == api.ARMS[3], 'Frozen primary contrast differs')
    require(plan['analysis']['quality_margin_or_continue_threshold'] is None, 'New threshold forbidden')
    test = pin_value(release['TEST_data_authority'], decode=True)
    require(test['schema'] == 'ncnc-official-TEST-data-authority-proposal-v1' and test['graph_authority'] == 'frozen_NCNC_PILOT_PLAN_not_BUDDY_cache', 'Separate TEST query authority required')
    require(test['existing_query_metadata']['sha256'] == TEST_META_SHA and test['official_TEST_file']['sha256'] == TEST_FILE_SHA and test['query_topology_policy'] == POLICY, 'Original TEST metadata/file/policy changed')
    pin_value(test['typed_digest_source'])
    test_metadata = pin_value(test['existing_query_metadata'], decode=True)
    require(test_metadata['official_test_file_sha256'] == test['official_TEST_file']['sha256'] and test['typed_query_digests'] == {k:test_metadata[k] for k in ('positive_sha256', 'negative_sha256', 'pair_order_sha256')}, 'TEST query metadata provenance differs')
    require(test['official_TEST_file']['path'] == str(Path(context['authority']['dataset_root']) / 'split/time/test.pt') and test['official_TEST_file']['bytes'] == 3404768, 'Different official TEST file')
    # No TEST binary is opened or hashed in stdlib metadata admission.
    context['TEST_authority'] = test
    for key, schema in (('independent_heldout_source_review', 'ncnc-frozen-all25-heldout-source-review-v1'), ('runtime_resource_admission', 'ncnc-frozen-all25-heldout-resource-admission-v1')):
        value = pin_value(release[key], decode=True)
        require(value['schema'] == schema and value['status'] == 'PASS' and value['identity'] == context['identity'] and value['heldout_source_manifest_sha256'] == release['heldout_source_manifest_sha256'] and value['family_lock_sha256'] == LOCK_SHA and value['v4_audit_result_sha256'] == AUDIT_RESULT_SHA and value['invocation'] == invocation and value['policy'] == POLICY, 'Separate heldout review/resource admission missing')
        if key == 'runtime_resource_admission':
            require(value['dispatch_recheck_required'] is True and value['minimum_GPU_free_MiB'] == 24576 and value['minimum_host_MemAvailable_bytes'] == 16 * 1024**3, 'Heldout resource bounds differ')
    qualification = pin_value(release['heldout_wrapper_fabricated_qualification'], decode=True)
    qualification_plan = json.loads((HERE / 'FABRICATED_QUALIFICATION_PLAN.json').read_text())
    expected_qualification_invocation = dict(stage='fabricated_heldout_wrapper_qualification', output_directory=str(EXECUTION / 'qualification/run01'), cuda_visible_devices=release['cuda_visible_devices'])
    require(qualification['schema'] == 'ncnc-heldout-wrapper-fabricated-qualification-v2' and qualification['status'] == 'PASS' and qualification['heldout_source_manifest_sha256'] == release['heldout_source_manifest_sha256'] and qualification['source_entry'] == dict(path=str(HERE / 'heldout_qualification.py'), bytes=(HERE / 'heldout_qualification.py').stat().st_size, sha256=sha(HERE / 'heldout_qualification.py')), 'Actual successor wrapper qualification required')
    require(qualification['runtime_authority'] == release['runtime_authority'] and qualification['original_family_identity'] == context['identity'] and qualification['invocation'] == expected_qualification_invocation and qualification['policy'] == POLICY, 'Qualification runtime/identity/invocation/policy binding differs')
    require(qualification['fabricated_inputs_only'] is True and qualification['TEST_opened'] is False and qualification['study_data_or_selected_checkpoint_accessed'] is False and qualification['training_updates'] == 0 and qualification['automatic_retry'] is False, 'Qualification scope differs')
    require(qualification['case_count'] == len(qualification['cases']) == len(qualification_plan['cases']) and {r['case'] for r in qualification['cases']} == set(qualification_plan['cases']) and all(r['status'] == 'PASS' for r in qualification['cases']), 'Every minimal fabricated case must actually pass')
    require(qualification['actual_original_scorer_calls'] == sum(r.get('original_scorer_calls',0) for r in qualification['cases']) == 40 and qualification['actual_stub_scorer_entries'] == sum(r.get('stub_scorer_entries',0) for r in qualification['cases']) == 52, 'Qualification literal work differs')
    family = api.family_gate(context)
    require(family['full_success'] and family['unique_fits_completed'] == 35 and family['complete_served_cells'] == 25, 'Complete original35/25 closure required')
    if require_fresh:
        require(not output.exists() and not (EXECUTION / 'ONE_TIME_TEST_CLAIM.json').exists(), 'Heldout invocation already claimed; no retry')
        require(not output.is_relative_to(HERE) and not any((p / 'MANIFEST.json').exists() for p in (output, *output.parents)), 'Heldout output overlaps sealed source')
    return context, family

def final_custody(context, family):
    current, after = metadata_admission(context['heldout_release_path'], context['output'], expected_sha=context['heldout_release_sha256'], require_fresh=False)
    require(after['lock'] == family['lock'] and current['heldout_release'] == context['heldout_release'], 'Frozen inputs changed at final custody')
    import replay_gate
    replay_gate.runtime_and_data_custody(context)
    pin_value(context['TEST_authority']['official_TEST_file'])
    context['loaded_tensor_guard']()
    import torch
    from replay_numeric import profile_receipt
    profile_receipt(torch, PROFILE)
    return dict(original_family_source_runtime_TRAIN_VALID_TEST_file_custody='PASS', loaded_canonical_graph_and_TEST_rows='PASS', fixed_strict_profile='PASS', timing='immediately_before_public_all25_terminal_serialization')
