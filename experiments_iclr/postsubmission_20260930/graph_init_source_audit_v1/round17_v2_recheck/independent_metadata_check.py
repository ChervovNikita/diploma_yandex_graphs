"""Independent stdlib-only metadata audit; never import supplied source."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODERN = BASE / 'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
ISSUED = BASE / 'modern_teacher_execution_root_v1/certification_v3/issued_v1'
CHECKS, FILES, SKIPPED, CACHE = [], {}, {}, {}
SEEDS = [17, 29, 43]
AUDITS = {
    'optimizer_exact_live_membership', 'native_parameter_grouping',
    'all_graph_logits_finite', 'all_active_parameter_gradients_finite',
    'every_member_connected', 'boundary_bias_copies_and_dormant_owner_removal',
    'exact_local_model_and_adam_restore_including_step_counters',
    'frozen_tolerance_identity_logits_shared_weight_and_private_bias_gradients',
    'selected_checkpoint_restore_and_replay', 'primitive_metadata_weights_only_safe_load',
}
COVERAGE = {
    'identical_graph_provider_and_rows', 'exact_role_nodes_and_source_pack_hashes',
    'independent_train_validation_label_extraction',
    'target_label_shapes_ranges_and_node_alignment', 'numerical_runtime_tested_seed17_only',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canon(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def check(name, value):
    CHECKS.append({'check': name, 'ok': bool(value)})


def local(path):
    if '/postsubmission_20260930/' in path:
        return BASE / path.split('/postsubmission_20260930/', 1)[1]
    return Path(path)


def verify(record):
    """Preserve the original descriptor and translate only an allowed read path."""
    key = json.dumps(record, sort_keys=True)
    if key in FILES or key in SKIPPED:
        return local(record['path'])
    valid = (isinstance(record, dict) and {'path', 'sha256'} <= set(record)
             <= {'path', 'sha256', 'bytes'} and isinstance(record.get('path'), str)
             and isinstance(record.get('sha256'), str) and len(record['sha256']) == 64
             and all(c in '0123456789abcdef' for c in record['sha256']))
    if 'bytes' in record:
        valid = valid and type(record['bytes']) is int and record['bytes'] >= 0
    check('descriptor_schema:' + str(record.get('path')), valid)
    if not valid:
        raise ValueError('Invalid descriptor: ' + str(record))
    path = local(record['path'])
    if path.suffix not in {'.json', '.py'}:
        SKIPPED[key] = {'descriptor': record, 'status': 'metadata schema only; scientific bytes/hash not inspected'}
        return path
    actual_sha, actual_bytes = digest(path), path.stat().st_size
    ok = actual_sha == record['sha256'] and ('bytes' not in record or actual_bytes == record['bytes'])
    check('allowed_mirror_hash_and_optional_bytes:' + record['path'], ok)
    FILES[key] = dict(descriptor=record, local_mirror=str(path), actual_sha256=actual_sha,
                      actual_bytes=actual_bytes, ok=ok)
    return path


def load(record):
    path = verify(record)
    if path.suffix != '.json':
        raise ValueError('Non-JSON metadata read requested: ' + str(path))
    if str(path) not in CACHE:
        CACHE[str(path)] = json.loads(path.read_text())
    return CACHE[str(path)]


def observed(path):
    """Starting JSON roots have no upstream descriptor in this audit."""
    return load({'path': str(path), 'sha256': digest(path)})


PROTOCOL = observed(MODERN / 'TEACHER_PROTOCOL.json')
MODEL = observed(MODERN / 'MODEL_IMPLEMENTATION_BINDINGS.json')
IMPLEMENTATION = {p.name: digest(p) for p in sorted((MODERN / 'prototype').glob('*.py'))}
check('modern_model_hashes_match_source', all(IMPLEMENTATION[n] == h for n, h in MODEL.items()))
for p in sorted((MODERN / 'prototype').glob('*.py')):
    ast.parse(p.read_text(), filename=str(p))


def preprocessing(admission, identity):
    name = admission['backbone']
    binding = {
        'backbone': name,
        'input_binding': {'graph_input': identity['graph_input'],
                          'implementation_sha256': admission['implementation_sha256']},
        'recipe': ('raw features; to_undirected; native gcn_norm+SciPy COO FP32; X..Ahat^12X'
                   if name == 'polyformer_mono' else
                   'verified provider raw features; PyG NormalizeFeatures once; undirected; remove/add loops once'),
        'environment': admission['environment'], 'order': 12 if name == 'polyformer_mono' else None,
        'dtype': 'FP32', 'cache_policy': 'no persistent reuse; full recomputation',
        'source_commit': PROTOCOL['native'][name]['author_commit'],
    }
    return dict(identity=canon(binding), **binding)


def specification(admission, seed):
    native = dict(PROTOCOL['native'][admission['backbone']])
    cfg = PROTOCOL['configurations'][admission['config']]
    native['dropout'] = max(0.0, native['dropout'] + cfg['dropout_delta'])
    if admission['backbone'] == 'polynormer_r':
        native['global_dropout'] = max(0.0, native['global_dropout'] + cfg['dropout_delta'])
    return dict(backbone=admission['backbone'], family=admission['family'], config=admission['config'],
                seed=seed, members=4, native=native, lr_multiplier=cfg['lr_multiplier'],
                initialization='paired native core seed; independent member m seed+100003*m; GNNM stem R Rademacher',
                primary_pooling=PROTOCOL['primary_pooling'], protocol_version=PROTOCOL['version'])


def source_identity(admission, tag):
    bundle = load(admission['source_label_binding'])
    role = load(admission['role_freeze'])
    graph = load(role['graph_input'])
    pairs = [p for p in bundle['pairs'] if (p['backbone'], p['seed']) == (admission['backbone'], role['seed'])]
    check(tag + ':source_bundle_v2_six_unique_pairs', bundle['schema'] == 'modern-source-label-bindings-v2'
          and len(bundle['pairs']) == 6 and len({(p['backbone'], p['seed']) for p in bundle['pairs']}) == 6)
    check(tag + ':one_exact_bundle_pair', len(pairs) == 1)
    pair = pairs[0]
    check(tag + ':exact_original_descriptors_preserved', pair['role_freeze'] == admission['role_freeze']
          and pair['graph_input'] == role['graph_input'] and pair['source_labels'] == admission['source_labels']
          and pair['independently_extracted_and_row_verified'] is True)
    check(tag + ':label_blind_roles', role['graph'] == admission['graph'] and role['labels_read'] is False
          and role['role_derivation_version'] == 'derived_roles_v2' and role['source_split_index'] == SEEDS.index(role['seed'])
          and role['preparation_driver_sha256'] == '376e8b779d8d71cf21785a62b8543ecadc9c09c9db8174dc230b55b8d0dbd253')
    check(tag + ':exact_source_roles', set(admission['source_labels']) == {'train', 'validation'})
    for record in admission['source_labels'].values():
        verify(record)
    for k in ['features', 'edges']:
        verify(graph[k])
    provider = load(pair['provider_row_identity'])
    extraction = load(pair['extraction_provenance'])
    check(tag + ':provider_rows_metadata', provider['full_feature_and_canonical_edge_rows_exact'] is True
          and provider['labels_used_to_establish_rows'] is False
          and provider['graph_input']['path'] == pair['graph_input']['path']
          and provider['graph_input']['sha256'] == pair['graph_input']['sha256'])
    verify(provider['graph_input'])
    check(tag + ':source_only_extraction_metadata', bundle['independent_audit'] is True
          and bundle['final_pool_labels_opened'] is False and bundle['fits_or_scores_performed'] is False
          and extraction['source_roles'] == ['train', 'validation']
          and extraction['public_raw_label_vector_decoded_for_source_reextraction'] is True
          and extraction['non_source_label_values_used'] is False
          and extraction['final_label_pack_files_opened'] is False and extraction['fits_or_scores_performed'] is False)
    extraction_row = next(p for p in extraction['checks'] if p['seed'] == role['seed'])
    check(tag + ':exact_extraction_descriptors', extraction_row['role_freeze'] == pair['role_freeze']
          and all(extraction_row['roles'][k]['pack'] == pair['source_labels'][k]
                  and extraction_row['roles'][k]['row_alignment_verified'] is True
                  and extraction_row['roles'][k]['independent_raw_source_reextraction_exact'] is True
                  for k in ['train', 'validation']))
    for record in [extraction['auditor_source'], extraction['label_pack_manifest']]:
        verify(record)
    dims = {k: graph[k] for k in ['num_nodes', 'num_features', 'num_classes', 'num_edges']}
    check(tag + ':provider_dimensions', provider['dimensions'] == dims)
    return dict(graph=admission['graph'], backbone=admission['backbone'], seed=role['seed'],
                graph_input=role['graph_input'], role_freeze=admission['role_freeze'],
                source_labels=admission['source_labels'], source_label_binding=admission['source_label_binding'],
                provider_row_identity=pair['provider_row_identity'], extraction_provenance=pair['extraction_provenance'],
                dimensions=dims)


CASES = []
for name, graph_name in [('polyformer_mono', 'Squirrel'), ('polynormer_r', 'Photo')]:
    for seed in SEEDS:
        tag = name + ':target_seed' + str(seed)
        cp = ISSUED / 'certificates' / (name + '__gnnm_boundary_4__config0__target_seed' + str(seed) + '.json')
        cert = observed(cp)
        req, partial, external, coverage, tested = [load(cert[k]) for k in
            ['request', 'partial_receipt', 'external_audit', 'coverage_authorization', 'tested_admission']]
        target = req['target']
        identity = source_identity(target, tag)
        tested_identity = source_identity(dict(tested, source_label_binding=target['source_label_binding']), tag + ':tested')
        check(tag + ':exact_target_identity', cert['input_identity'] == cert['authorized_target_input'] == identity)
        check(tag + ':cfg0_GNNM_source_scope', cert['schema'] == 'modern-teacher-qualification-certificate-v2'
              and cert['admission_eligible'] is True and cert['report_eligible'] is False
              and cert['qualified_config'] == target['config'] == tested['config'] == 0
              and cert['backbone'] == target['backbone'] == tested['backbone'] == name
              and cert['family'] == target['family'] == tested['family'] == 'gnnm_boundary_4'
              and identity['graph'] == graph_name and identity['seed'] == seed)
        check(tag + ':numerical_vs_coverage_seed_semantics', cert['numerical_tested_seeds'] == [17]
              and cert['tested_seed17_input'] == tested_identity and tested_identity['seed'] == 17
              and cert['numerical_tests_on_target_seed'] is (seed == 17)
              and cert['numerical_tested_configurations'] == [0] and cert['registered_configuration_scope'] == [0, 1, 2, 3]
              and cert['configuration_scope_is_authorization_not_four_config_numerical_testing'] is True
              and cert['full_schedule_feasibility_separately_required'] is True)
        check(tag + ':source_environment_protocol', cert['environment'] == target['environment'] == tested['environment']
              and cert['model_sha256'] == MODEL and cert['implementation_sha256'] == target['implementation_sha256'] == IMPLEMENTATION
              and target['teacher_protocol'] == tested['teacher_protocol'] == PROTOCOL)
        check(tag + ':current_v3_source_seal', cert['source_seal']['sha256'] == digest(MODERN / 'SEAL.json'))
        check(tag + ':target_preprocessing', cert['preprocessing'] == preprocessing(target, identity))
        check(tag + ':partial_request_exact', req['partial_receipt'] == cert['partial_receipt'] == external['partial_receipt']
              and req['external_audit'] == cert['external_audit'] and partial['report_eligible'] is False
              and partial['final_labels_read'] is False and partial['labels_read'] == ['train', 'validation'])
        check(tag + ':coverage_checks_and_scope', coverage['schema'] == 'modern-exact-role-label-coverage-v2'
              and coverage['tested_seed17_input'] == tested_identity and coverage['backbone'] == name
              and coverage['family'] == 'gnnm_boundary_4' and coverage['environment'] == target['environment']
              and coverage['model_sha256'] == MODEL and set(coverage['checks']) == COVERAGE
              and all(v is True for v in coverage['checks'].values())
              and coverage['numerical_tests_on_untested_seeds'] is False
              and len(coverage['authorized_target_inputs']) == 3
              and {r['seed'] for r in coverage['authorized_target_inputs']} == set(SEEDS)
              and identity in coverage['authorized_target_inputs'])
        for case in coverage['authorized_target_inputs']:
            recovered = source_identity(dict(target, role_freeze=case['role_freeze'], source_labels=case['source_labels'],
                                            source_label_binding=case['source_label_binding']), tag + ':coverage' + str(case['seed']))
            check(tag + ':coverage_exact_identity' + str(case['seed']), recovered == case
                  and all(case[k] == tested_identity[k] for k in
                          ['graph', 'backbone', 'graph_input', 'source_label_binding', 'provider_row_identity', 'dimensions']))
        check(tag + ':external_scope_checks', external['schema'] == 'modern-external-runtime-audit-v2'
              and external['tested_seed17_input'] == tested_identity and external['coverage_authorization'] == cert['coverage_authorization']
              and set(external['checks']) == AUDITS and all(v is True for v in external['checks'].values())
              and external['frozen_tolerances'] == {'logit_rtol': 1e-5, 'logit_atol': 1e-6, 'gradient_rtol': 1e-4, 'gradient_atol': 1e-6}
              and external['failed_attempts_and_fresh_version_retries_retained'] is True)
        for k in ['attempt_history', 'instrumentation_sources', 'evidence']:
            check(tag + ':nonempty_' + k, bool(external[k]) and isinstance(external[k], list))
            for record in external[k]:
                verify(record)
        check(tag + ':nonempty_coverage_evidence', bool(coverage['evidence']))
        for record in coverage['evidence']:
            verify(record)
        for k in ['source_seal', 'tested_source_seal', 'preprocessing_evidence']:
            verify(cert[k])
        selection = load(external['preprocessing_evidence'])
        tested_pre = preprocessing(tested, tested_identity)
        check(tag + ':tested_preprocessing_exact', selection['preprocessing'] == external['tested_preprocessing']
              == cert['tested_preprocessing'] == tested_pre)
        binding = external['partial_run_binding']
        root, terminal, cell, optimizer = [load(binding[k]) for k in
            ['original_root_request', 'root_terminal', 'teacher_cell_freeze', 'original_optimizer_audit']]
        run = root['output'].rstrip('/')
        check(tag + ':typed_legacy_original_run', partial['schema'] == 'modern-teacher-qualification-v1'
              and binding['schema'] == 'modern-legacy-qualification-run-binding-v3'
              and root['root_admitted'] is True and root['action'] == 'qualification'
              and root['full_fit_admitted'] is False and root['heldout_scoring_admitted'] is False
              and root['teacher_admission'] == external['tested_admission'] == cert['tested_admission']
              and root['deterministic'] is True and root['aps_score_backend'] == 'cpu_fixed_algorithm')
        check(tag + ':original_run_path_and_terminal_links', binding['root_terminal']['path'] == run + '/TERMINAL.json'
              and binding['teacher_cell_freeze']['path'] == run + '/cell/TEACHER_CELL_FREEZE.json'
              and cert['partial_receipt']['path'] == run + '/cell/QUALIFICATION_RECEIPT.json'
              and binding['original_optimizer_audit']['path'] == run + '/EXTERNAL_OPTIMIZER_AUDIT.json'
              and terminal['complete'] is True and terminal['report_eligible'] is False and terminal['final_labels_read'] is False
              and terminal['request_sha256'] == binding['original_root_request']['sha256']
              and terminal['qualification_receipt_sha256'] == cert['partial_receipt']['sha256']
              and terminal['optimizer_audit_sha256'] == binding['original_optimizer_audit']['sha256'])
        spec = specification(tested, 17)
        check(tag + ':original_cell_exact_scope', cell['schema'] == 'modern-teacher-cell-freeze-v1'
              and cell['admission'] == cert['tested_admission'] and cell['role_freeze'] == tested['role_freeze']
              and cell['graph'] == graph_name and cell['environment'] == tested['environment']
              and cell['implementation_sha256'] == partial['implementation_sha256'] == tested['implementation_sha256']
              and cell['teacher_protocol'] == tested['teacher_protocol'] == PROTOCOL and cell['specification'] == spec
              and cell['label_scope'] == ['train', 'validation'] and cell['report_eligible'] is False and cell['final_labels_read'] is False)
        payload = {p['path']: p for p in cell['payload']}
        check(tag + ':partial_and_preprocessing_cell_payload_links', len(payload) == len(cell['payload'])
              and external['preprocessing_evidence']['path'] == run + '/cell/teacher_selection.json'
              and payload['QUALIFICATION_RECEIPT.json']['sha256'] == cert['partial_receipt']['sha256']
              and payload['teacher_selection.json']['sha256'] == external['preprocessing_evidence']['sha256']
              and cell['selection'] == selection and selection['specification'] == spec
              and selection['qualification_only'] is True and selection['report_eligible'] is False
              and partial['updates_completed'] == selection['updates_completed'] == terminal['training_steps']
              and partial['global_stage'] == selection['global_stage'])
        manifest, seal = load(root['source_manifest']), load(external['tested_source_seal'])
        check(tag + ':original_seal_manifest_instrumentation_links',
              root['source_manifest']['path'].rsplit('/', 1)[0] == external['tested_source_seal']['path'].rsplit('/', 1)[0] == root['source_packet']
              and seal['manifest_sha256'] == root['source_manifest']['sha256']
              and external['tested_source_seal'] == cert['tested_source_seal']
              and any(p['path'] == external['tested_source_seal']['path'] and p['sha256'] == external['tested_source_seal']['sha256'] for p in root['protected_files'])
              and any(p['path'] == root['entry_script'] and any(q['path'] == p['path'] and q['sha256'] == p['sha256']
                     for q in root['protected_files']) for p in external['instrumentation_sources']))
        custody = optimizer['exact_input_custody']
        check(tag + ':original_optimizer_exact_input_custody',
              all(custody[k] == tested[k] for k in ['role_freeze', 'source_labels', 'environment', 'backbone', 'family', 'implementation_sha256'])
              and custody['graph_input'] == tested_identity['graph_input'] and custody['source_manifest'] == root['source_manifest'])
        expected_linkage = dict(schema='modern-partial-run-linkage-v3', partial_schema=partial['schema'],
                                partial_receipt=cert['partial_receipt'], tested_admission=external['tested_admission'],
                                tested_input=tested_identity, tested_preprocessing=tested_pre,
                                tested_source_seal=external['tested_source_seal'], tested_config=tested['config'], binding=binding)
        check(tag + ':entire_reconstructed_v3_linkage', cert['partial_run_linkage'] == expected_linkage)
        CASES.append(dict(certificate=str(cp), backbone=name, graph=graph_name, target_seed=seed,
                          tested_seed=17, tested_config=0, numerical_tests_on_target_seed=cert['numerical_tests_on_target_seed'],
                          role_descriptor_has_bytes='bytes' in identity['role_freeze'],
                          source_label_descriptors_have_bytes=all('bytes' in r for r in identity['source_labels'].values())))

RESULT = dict(schema='independent-round17-v2-actual-issued-metadata-audit-v1',
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              checks=CHECKS, cases=CASES, allowed_mirror_receipts=list(FILES.values()),
              scientific_descriptors_not_opened=list(SKIPPED.values()),
              summary=dict(certificates=len(CASES), predicates=len(CHECKS), failed=sum(not c['ok'] for c in CHECKS),
                           allowed_mirror_descriptors=len(FILES), allowed_unique_files=len(CACHE),
                           scientific_descriptors_schema_only=len(SKIPPED)),
              scope='Independently written stdlib JSON/source/hash checks. No supplied source imported or executed; no data/label arrays or checkpoints read or hashed; no remote access or scientific runtime.')
(OUT / 'ACTUAL_ISSUED_METADATA_RECEIPTS.json').write_text(json.dumps(RESULT, indent=2, allow_nan=False) + '\n')
print(json.dumps(RESULT['summary']))
for item in CHECKS:
    if not item['ok']:
        print(json.dumps(item))
