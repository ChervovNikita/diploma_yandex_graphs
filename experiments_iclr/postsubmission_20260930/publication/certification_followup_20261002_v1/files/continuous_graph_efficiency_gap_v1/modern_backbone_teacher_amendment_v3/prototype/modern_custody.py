"""Interface-only custody v3. Stdlib metadata/hashes; no scientific imports.

Prospective registries and exclusive claim/terminal records forbid a second
attempt at a registered cell, including a retry in a fresh output directory.
Failures remain in closure and block selection/release for this protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent.parent
AUDITS = ('optimizer_exact_live_membership', 'native_parameter_grouping',
    'all_graph_logits_finite', 'all_active_parameter_gradients_finite',
    'every_member_connected', 'boundary_bias_copies_and_dormant_owner_removal',
    'exact_local_model_and_adam_restore_including_step_counters',
    'frozen_tolerance_identity_logits_shared_weight_and_private_bias_gradients',
    'selected_checkpoint_restore_and_replay', 'primitive_metadata_weights_only_safe_load')
COVERAGE_CHECKS = ('identical_graph_provider_and_rows','exact_role_nodes_and_source_pack_hashes',
    'independent_train_validation_label_extraction','target_label_shapes_ranges_and_node_alignment',
    'numerical_runtime_tested_seed17_only')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verified(record):
    require(isinstance(record, dict) and set(record) >= {'path','sha256'} and
        isinstance(record['path'],str) and isinstance(record['sha256'],str) and len(record['sha256']) == 64,
        'Complete descriptor required')
    path = Path(record['path']).resolve()
    require(sha(path) == record['sha256'], 'Custody hash mismatch: '+str(path))
    return path


def write(path, value):
    with open(path,'x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write('\n')


def implementations():
    return {p.name:sha(p) for p in sorted((ROOT/'prototype').glob('*.py'))}


def models():
    values = read(ROOT/'MODEL_IMPLEMENTATION_BINDINGS.json')
    require(all(sha(ROOT/'prototype'/name) == value for name,value in values.items()), 'Model code identity changed')
    return values


def protocol():
    return read(ROOT/'TEACHER_PROTOCOL.json')


def source_identity(admission, roles, manifest):
    """Verify the one independent extraction bundle before opening label arrays."""
    bundle_record = admission['source_label_binding']
    bundle = read(verified(bundle_record))
    require(bundle['schema'] == 'modern-source-label-bindings-v2' and len(bundle['pairs']) == 6,
        'One frozen independently extracted pair per graph/seed required')
    pair = [r for r in bundle['pairs'] if (r['backbone'],r['seed']) == (admission['backbone'],roles['seed'])]
    require(len(pair) == 1, 'Missing/repeated source label pair')
    pair = pair[0]
    packs = {k:admission['source_labels'][k] for k in ('train','validation')}
    require(pair['role_freeze'] == admission['role_freeze'] and pair['graph_input'] == roles['graph_input'] and
        pair['source_labels'] == packs and pair['independently_extracted_and_row_verified'] is True,
        'Train/validation pack descriptors or extraction provenance differ')
    verified(pair['provider_row_identity']); verified(pair['extraction_provenance'])
    for record in packs.values(): verified(record)
    for key in ('features','edges'): verified(manifest[key])
    return dict(graph=admission['graph'], backbone=admission['backbone'], seed=roles['seed'],
        graph_input=roles['graph_input'], role_freeze=admission['role_freeze'],
        source_labels=packs, source_label_binding=bundle_record,
        provider_row_identity=pair['provider_row_identity'], extraction_provenance=pair['extraction_provenance'],
        dimensions={k:manifest[k] for k in ('num_nodes','num_features','num_classes','num_edges')})


def preprocessing(admission, identity):
    backbone = admission['backbone']
    recipe = ('raw features; to_undirected; native gcn_norm+SciPy COO FP32; X..Ahat^12X' if backbone == 'polyformer_mono'
        else 'verified provider raw features; PyG NormalizeFeatures once; undirected; remove/add loops once')
    binding = dict(backbone=backbone, input_binding=dict(graph_input=identity['graph_input'],
        implementation_sha256=admission['implementation_sha256']), recipe=recipe,
        environment=admission['environment'],order=12 if backbone == 'polyformer_mono' else None,
        dtype='FP32',cache_policy='no persistent reuse; full recomputation',
        source_commit=protocol()['native'][backbone]['author_commit'])
    identity_hash = hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(identity=identity_hash,**binding)


def registered_specification(admission, seed):
    """Reconstruct frozen specification metadata without importing a model."""
    recipe = protocol()
    config = admission['config']
    require(config in range(4) and seed in recipe['seeds'] and admission['family'] in recipe['families'],
        'Unregistered tested specification')
    native = dict(recipe['native'][admission['backbone']])
    configuration = recipe['configurations'][config]
    native['dropout'] = max(0.0,native['dropout']+configuration['dropout_delta'])
    if admission['backbone'] == 'polynormer_r':
        native['global_dropout'] = max(0.0,native['global_dropout']+configuration['dropout_delta'])
    return dict(backbone=admission['backbone'],family=admission['family'],config=config,seed=seed,
        members=1 if admission['family']=='single_author' else 4,native=native,
        lr_multiplier=configuration['lr_multiplier'],
        initialization='paired native core seed; independent member m seed+100003*m; GNNM stem R Rademacher',
        primary_pooling=recipe['primary_pooling'],protocol_version=recipe['version'])


def qualification_run_linkage(request, partial, audit, tested_admission, tested_identity, tested_preprocessing):
    """Prove the partial and supplement describe one exact tested attempt."""
    require(partial['implementation_sha256'] == tested_admission['implementation_sha256'] and
        tested_preprocessing == preprocessing(tested_admission,tested_identity),
        'Partial implementation/tested preprocessing differs from the original admission')
    schema = partial['schema']
    if schema == 'modern-teacher-partial-qualification-v2':
        require(partial['admission_eligible'] is False and
            partial['admission'] == audit['tested_admission'] and
            partial['input_identity'] == tested_identity and
            partial['preprocessing'] == tested_preprocessing and
            partial['source_seal'] == audit['tested_source_seal'] and partial['model_sha256'] == models(),
            'Native partial is not linked to the exact audited admission/input/preprocessing/source seal')
        emitted_admission = read(verified(partial['admission']))
        require(emitted_admission == tested_admission and
            emitted_admission['config'] == tested_admission['config'] and
            partial.get('config',emitted_admission['config']) == tested_admission['config'],
            'Native partial tested configuration differs')
        binding = dict(kind='native-v2',emitted_admission=partial['admission'])
    else:
        require(schema == 'modern-teacher-qualification-v1', 'Unsupported partial qualification schema')
        binding = audit['partial_run_binding']
        require(binding['schema'] == 'modern-legacy-qualification-run-binding-v3',
            'Typed original legacy run binding required')
        request_path = verified(binding['original_root_request'])
        terminal_path = verified(binding['root_terminal'])
        cell_path = verified(binding['teacher_cell_freeze'])
        optimizer_path = verified(binding['original_optimizer_audit'])
        root_request, terminal, cell, optimizer = map(read,(request_path,terminal_path,cell_path,optimizer_path))
        run = Path(root_request['output']).resolve()
        cell_root = run/'cell'
        require(root_request['root_admitted'] is True and root_request['action'] == 'qualification' and
            root_request['full_fit_admitted'] is False and root_request['heldout_scoring_admitted'] is False and
            root_request['teacher_admission'] == audit['tested_admission'] and
            root_request['deterministic'] is True and root_request['aps_score_backend'] == 'cpu_fixed_algorithm',
            'Original root request does not authorize this exact tested legacy admission/backend')
        require(terminal_path == run/'TERMINAL.json' and cell_path == cell_root/'TEACHER_CELL_FREEZE.json' and
            verified(request['partial_receipt']) == cell_root/'QUALIFICATION_RECEIPT.json' and
            optimizer_path == run/'EXTERNAL_OPTIMIZER_AUDIT.json' and
            terminal['complete'] is True and terminal['report_eligible'] is False and terminal['final_labels_read'] is False and
            terminal['request_sha256'] == binding['original_root_request']['sha256'] and
            terminal['qualification_receipt_sha256'] == request['partial_receipt']['sha256'] and
            terminal['optimizer_audit_sha256'] == binding['original_optimizer_audit']['sha256'],
            'Legacy terminal/request/partial/optimizer artifacts are not one completed tested run')
        require(cell['schema'] == 'modern-teacher-cell-freeze-v1' and
            cell['admission'] == audit['tested_admission'] and cell['role_freeze'] == tested_admission['role_freeze'] and
            cell['graph'] == tested_identity['graph'] and cell['environment'] == tested_admission['environment'] and
            cell['implementation_sha256'] == tested_admission['implementation_sha256'] and
            cell['teacher_protocol'] == tested_admission['teacher_protocol'] == protocol() and
            cell['specification'] == registered_specification(tested_admission,tested_identity['seed']) and
            cell['label_scope'] == ['train','validation'] and cell['report_eligible'] is False and cell['final_labels_read'] is False,
            'Legacy cell freeze admission/role/config/specification/source identity differs')
        selection_path = verified(audit['preprocessing_evidence'])
        selection = read(selection_path)
        payload = {row['path']:row for row in cell['payload']}
        require(len(payload) == len(cell['payload']) and selection_path == cell_root/'teacher_selection.json' and
            payload['QUALIFICATION_RECEIPT.json']['sha256'] == request['partial_receipt']['sha256'] and
            payload['teacher_selection.json']['sha256'] == audit['preprocessing_evidence']['sha256'] and
            cell['selection'] == selection and selection['specification'] == cell['specification'] and
            selection['preprocessing'] == tested_preprocessing and selection['qualification_only'] is True and
            selection['report_eligible'] is False and partial['updates_completed'] == selection['updates_completed'] == terminal['training_steps'] and
            partial['global_stage'] == selection['global_stage'],
            'Legacy partial/preprocessing metadata is not linked through the actual frozen cell payload')
        source_manifest_path = verified(root_request['source_manifest'])
        tested_seal_path = verified(audit['tested_source_seal'])
        require(source_manifest_path.parent == tested_seal_path.parent == Path(root_request['source_packet']).resolve() and
            read(tested_seal_path)['manifest_sha256'] == root_request['source_manifest']['sha256'] and
            any(row['path'] == audit['tested_source_seal']['path'] and row['sha256'] == audit['tested_source_seal']['sha256']
                for row in root_request['protected_files']) and
            any(Path(row['path']).resolve() == Path(root_request['entry_script']).resolve() and
                any(Path(original['path']).resolve() == Path(row['path']).resolve() and original['sha256'] == row['sha256']
                    for original in root_request['protected_files']) for row in audit['instrumentation_sources']),
            'Legacy root request lacks the tested source seal/manifest/instrumentation linkage')
        original_custody = optimizer['exact_input_custody']
        require(all(original_custody[k] == tested_admission[k] for k in
            ('role_freeze','source_labels','environment','backbone','family','implementation_sha256')) and
            original_custody['graph_input'] == tested_identity['graph_input'] and
            original_custody['source_manifest'] == root_request['source_manifest'],
            'Original optimizer audit exact role/label/source custody differs from the tested admission')
    return dict(schema='modern-partial-run-linkage-v3',partial_schema=schema,
        partial_receipt=request['partial_receipt'],tested_admission=audit['tested_admission'],
        tested_input=tested_identity,tested_preprocessing=tested_preprocessing,
        tested_source_seal=audit['tested_source_seal'],tested_config=tested_admission['config'],binding=binding)


def qualification_gate(admission, identity):
    certificate = read(verified(admission['qualification_receipt']))
    require(certificate['schema'] == 'modern-teacher-qualification-certificate-v2' and certificate['admission_eligible'] is True and
        certificate['input_identity'] == identity and certificate['preprocessing'] == preprocessing(admission,identity) and
        certificate['backbone'] == admission['backbone'] and certificate['family'] == admission['family'] and
        certificate['environment'] == admission['environment'] and certificate['model_sha256'] == models() and
        certificate['score_backend'] == 'randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2' and
        certificate['implementation_sha256'] == implementations() == admission['implementation_sha256'],
        'Exact graph/role/label/source-bound supplemented qualification required')
    verified(certificate['partial_receipt']); verified(certificate['external_audit']); verified(certificate['source_seal'])
    linkage = certificate['partial_run_linkage']
    require(linkage['schema'] == 'modern-partial-run-linkage-v3' and
        linkage['partial_receipt'] == certificate['partial_receipt'] and
        linkage['tested_admission'] == certificate['tested_admission'] and
        linkage['tested_input'] == certificate['tested_seed17_input'] and
        linkage['tested_preprocessing'] == certificate['tested_preprocessing'] and
        linkage['tested_source_seal'] == certificate['tested_source_seal'] and
        linkage['tested_config'] == certificate['qualified_config'] and
        certificate['numerical_tested_configurations'] == [linkage['tested_config']] and
        certificate['registered_configuration_scope'] == list(range(4)) and
        admission['config'] in certificate['registered_configuration_scope'],
        'Supplemented certificate lacks exact partial-to-tested-run linkage')
    coverage = read(verified(certificate['coverage_authorization']))
    require(certificate['authorized_target_input'] == identity and certificate['numerical_tested_seeds'] == [17] and
        certificate['tested_seed17_input']['seed'] == 17 and
        coverage['schema'] == 'modern-exact-role-label-coverage-v2' and
        identity in coverage['authorized_target_inputs'] and coverage['tested_seed17_input'] == certificate['tested_seed17_input'] and
        coverage['numerical_tests_on_untested_seeds'] is False,
        'Exact authorized target coverage must remain distinct from tested seed17')
    feasibility = read(verified(admission['full_schedule_feasibility']))
    require(feasibility['schema'] == 'modern-full-schedule-feasibility-v2' and feasibility['approved'] is True and
        feasibility['input_identity'] == identity and feasibility['backbone'] == admission['backbone'] and
        feasibility['family'] == admission['family'] and feasibility['environment'] == admission['environment'] and
        feasibility['model_sha256'] == models() and feasibility['teacher_protocol'] == protocol() and
        feasibility['short_qualification_is_not_full_fit_evidence'] is True,
        'Separate full-schedule feasibility evidence required')
    for record in feasibility['resource_evidence']: verified(record)
    require(bool(feasibility['resource_evidence']), 'Resource evidence is missing')
    return certificate


def certify(request_path, output):
    request = read(request_path)
    admission = request['target']
    require(admission['implementation_sha256'] == implementations() and admission['teacher_protocol'] == protocol(), 'Wrong target interfaces')
    roles = read(verified(admission['role_freeze']))
    manifest = read(verified(roles['graph_input']))
    identity = source_identity(admission,roles,manifest)
    partial = read(verified(request['partial_receipt']))
    audit = read(verified(request['external_audit']))
    tested_admission = read(verified(audit['tested_admission']))
    tested_roles = read(verified(tested_admission['role_freeze']))
    tested_target = dict(tested_admission,source_label_binding=admission['source_label_binding'])
    tested_identity = source_identity(tested_target,tested_roles,read(verified(tested_roles['graph_input'])))
    coverage = read(verified(audit['coverage_authorization']))
    require(coverage['schema'] == 'modern-exact-role-label-coverage-v2' and
        coverage['tested_seed17_input'] == tested_identity and tested_identity['seed'] == 17 and
        coverage['backbone'] == admission['backbone'] and coverage['family'] == admission['family'] and
        coverage['environment'] == admission['environment'] and coverage['model_sha256'] == models() and
        set(coverage['checks']) == set(COVERAGE_CHECKS) and all(v is True for v in coverage['checks'].values()) and
        coverage['numerical_tests_on_untested_seeds'] is False and
        len(coverage['authorized_target_inputs']) == 3,
        'Explicit same-graph role/label coverage authorization required')
    seen_targets = set()
    for case in coverage['authorized_target_inputs']:
        case_admission = dict(admission,role_freeze=case['role_freeze'],source_labels=case['source_labels'],
            source_label_binding=case['source_label_binding'])
        case_roles = read(verified(case['role_freeze']))
        require(source_identity(case_admission,case_roles,read(verified(case_roles['graph_input']))) == case and
            all(case[k] == tested_identity[k] for k in ('graph','backbone','graph_input','source_label_binding','provider_row_identity','dimensions')),
            'Authorized target differs from exact graph/provider/source extraction')
        seen_targets.add(case['seed'])
    require(seen_targets == set(protocol()['seeds']) and identity in coverage['authorized_target_inputs'],
        'Coverage must enumerate the three exact authorized target cases')
    for record in coverage['evidence']: verified(record)
    require(bool(coverage['evidence']), 'Independent coverage audit evidence is missing')
    require(partial['report_eligible'] is False and partial['final_labels_read'] is False and
        partial['labels_read'] == ['train','validation'] and
        partial.get('partial_checks_passed',partial.get('passed')) is True and
        partial['backbone'] == admission['backbone'] and partial['family'] == admission['family'] and
        partial['environment'] == admission['environment'] and
        all(partial['implementation_sha256'][n] == value for n,value in models().items()), 'Partial evidence/model identity differs')
    require(audit['schema'] == 'modern-external-runtime-audit-v2' and audit['partial_receipt'] == request['partial_receipt'] and
        audit['score_backend'] == 'randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2' and
        audit['tested_seed17_input'] == tested_identity and audit['model_sha256'] == models() and
        audit['environment'] == admission['environment'] and
        tested_admission['backbone'] == admission['backbone'] and tested_admission['family'] == admission['family'] and
        tested_admission['environment'] == admission['environment'] and tested_admission['config'] == admission['config'],
        'External evidence must bind the tested graph/role/label packs/config')
    require(tested_roles['graph_input'] == identity['graph_input'], 'Qualified graph input differs')
    require(set(audit['checks']) == set(AUDITS) and all(value is True for value in audit['checks'].values()), 'External optimizer/member/Adam checks incomplete')
    require(audit['frozen_tolerances'] == dict(logit_rtol=1e-5,logit_atol=1e-6,gradient_rtol=1e-4,gradient_atol=1e-6),
        'Qualification tolerance amendment is outside this interface repair')
    verified(audit['tested_source_seal'])
    compatible_seals = {read(ROOT/'V1_MODEL_IDENTITY.json')['v1_seal_sha256'],
        read(ROOT/'V2_SOURCE_IDENTITY.json')['v2_seal_sha256'],sha(ROOT/'SEAL.json')}
    require(audit['tested_source_seal']['sha256'] in compatible_seals, 'Tested immutable source seal is incompatible')
    tested_preprocessing = read(verified(audit['preprocessing_evidence']))['preprocessing']
    expected_preprocessing = preprocessing(admission,identity)
    require(tested_preprocessing == audit['tested_preprocessing'] and
        tested_preprocessing['input_binding']['graph_input'] == identity['graph_input'] and
        all(tested_preprocessing[k] == expected_preprocessing[k] for k in
            ('backbone','recipe','environment','order','dtype','cache_policy','source_commit')),
        'Tested preprocessing/graph contract differs')
    tested_binding = {k:v for k,v in tested_preprocessing.items() if k != 'identity'}
    require(tested_preprocessing['identity'] == hashlib.sha256(json.dumps(tested_binding,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'Tested preprocessing identity is not canonical')
    run_linkage = qualification_run_linkage(request,partial,audit,tested_admission,tested_identity,tested_preprocessing)
    require(audit['failed_attempts_and_fresh_version_retries_retained'] is True and bool(audit['attempt_history']) and
        bool(audit['instrumentation_sources']), 'External qualification attempt/instrumentation custody is missing')
    for record in audit['attempt_history'] + audit['instrumentation_sources']: verified(record)
    for record in audit['evidence']: verified(record)
    require(bool(audit['evidence']), 'External audit evidence is missing')
    write(output,dict(schema='modern-teacher-qualification-certificate-v2',request=descriptor(request_path),
        partial_receipt=request['partial_receipt'],external_audit=request['external_audit'],
        input_identity=identity,preprocessing=preprocessing(admission,identity),
        authorized_target_input=identity,tested_seed17_input=tested_identity,numerical_tested_seeds=[17],
        coverage_authorization=audit['coverage_authorization'],numerical_tests_on_target_seed=identity['seed']==17,
        backbone=admission['backbone'],family=admission['family'],environment=admission['environment'],
        model_sha256=models(),implementation_sha256=implementations(),source_seal=descriptor(ROOT/'SEAL.json'),
        tested_source_seal=audit['tested_source_seal'],admission_eligible=True,report_eligible=False,
        tested_admission=audit['tested_admission'],qualified_config=admission['config'],
        partial_run_linkage=run_linkage,
        registered_configuration_scope=list(range(4)),
        numerical_tested_configurations=[tested_admission['config']],
        configuration_scope_is_authorization_not_four_config_numerical_testing=True,
        score_backend='randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2',
        preprocessing_evidence=audit['preprocessing_evidence'],tested_preprocessing=tested_preprocessing,
        neural_operation_changed=False,full_schedule_feasibility_separately_required=True))


def registry(record):
    value = read(verified(record))
    require(value['schema'] == 'modern-attempt-registry-v2' and value['implementation_sha256'] == implementations(), 'Wrong attempt registry')
    identity = read(verified(value['study_identity']))
    require(identity['schema'] == 'modern-study-identity-v2' and identity['study_id'] == value['study_id'] and
        identity['teacher_protocol'] == protocol() and identity['custody_version'] == 2,
        'Wrong prospective study identity')
    name = 'TEACHER_ATTEMPT_REGISTRY.json' if value['kind'] == 'teacher' else 'CORRECTION_ATTEMPT_REGISTRY.json'
    require(Path(record['path']).resolve() == Path(identity['anchor_directory']).resolve()/name, 'Registry moved outside fixed study anchor')
    expected = {(b,f,c,s) for b in protocol()['native'] for f in protocol()['families'] for c in range(4) for s in protocol()['seeds']} if value['kind']=='teacher' else {
        (b,'gnnm_boundary_4',None,s) for b in protocol()['native'] for s in protocol()['seeds']}
    keys = {(e['cell']['backbone'],e['cell']['family'],e['cell'].get('config'),e['cell']['seed']) for e in value['entries']}
    require(keys==expected and len(value['entries'])==len(expected) and
        len({e['attempt_id'] for e in value['entries']})==len(expected) and value['model_sha256']==models(),
        'Registry is not the complete unique fixed model grid')
    return value


def freeze_registry(request_path, output):
    request = read(request_path)
    identity = read(verified(request['study_identity']))
    require(identity['schema'] == 'modern-study-identity-v2' and identity['custody_version'] == 2 and
        identity['teacher_protocol'] == protocol(), 'Freeze a separate v2 study identity first')
    verified(identity['source_label_binding'])
    require(identity['separate_declared_protocol_for_any_retry'] is True and bool(identity['prior_exposure_disclosure']),
        'Study must disclose predecessor exposure and forbid undeclared retry')
    for record in identity['prior_attempt_history']: verified(record)
    kind = request['kind']; require(kind in ('teacher','correction'), 'Unknown registry kind')
    expected = {(b,f,c,s) for b in protocol()['native'] for f in protocol()['families'] for c in range(4) for s in protocol()['seeds']} if kind == 'teacher' else {
        (b,'gnnm_boundary_4',None,s) for b in protocol()['native'] for s in protocol()['seeds']}
    seen,ids,outputs = set(),set(),set()
    for entry in request['entries']:
        admission = read(verified(entry['admission']))
        cell = entry['cell'];key=(cell['backbone'],cell['family'],cell.get('config'),cell['seed'])
        require(key in expected and key not in seen and entry['attempt_id'] not in ids and
            re.fullmatch('[a-z][a-z0-9_-]{1,80}',entry['attempt_id']) and
            admission['attempt_id'] == entry['attempt_id'] and admission['study_id'] == identity['study_id'] and
            admission['execution_authorized'] is True and admission['implementation_sha256'] == implementations() and
            admission['source_label_binding'] == identity['source_label_binding'], 'Wrong/duplicate registered attempt or source extraction bundle')
        role = read(verified(admission['role_freeze']))
        require(role['seed'] == cell['seed'] and admission['backbone'] == cell['backbone'], 'Registry role/cell mismatch')
        if kind == 'teacher': require(admission['family'] == cell['family'] and admission['config'] == cell['config'], 'Wrong teacher cell')
        source_identity(admission,role,read(verified(role['graph_input'])))
        path = str(Path(entry['output']).resolve())
        require(path == entry['output'] and path not in outputs and not Path(path).exists(), 'Output must be a unique fresh registered absolute directory')
        seen.add(key);ids.add(entry['attempt_id']);outputs.add(path)
    require(seen == expected, 'Register the complete prospective grid; no partial registry')
    anchor = Path(identity['anchor_directory']).resolve()
    name = 'TEACHER_ATTEMPT_REGISTRY.json' if kind == 'teacher' else 'CORRECTION_ATTEMPT_REGISTRY.json'
    require(Path(output).resolve() == anchor/name and anchor.is_dir(), 'Fixed study anchor output required')
    ledger = anchor/(kind+'_attempt_ledger');ledger.mkdir(exist_ok=False)
    write(output,dict(schema='modern-attempt-registry-v2',kind=kind,study_id=identity['study_id'],study_identity=request['study_identity'],
        request=descriptor(request_path),implementation_sha256=implementations(),model_sha256=models(),
        entries=request['entries'],ledger_directory=str(ledger),retry_policy='none; source repair/retry requires a new declared protocol preserving predecessor registry/closure',
        prospective=True))


def ledger_paths(value, attempt_id):
    key = hashlib.sha256(attempt_id.encode()).hexdigest()
    return Path(value['ledger_directory'])/(key+'.claim.json'),Path(value['ledger_directory'])/(key+'.terminal.json')


def begin(record, admission_record, output, kind):
    value = registry(record); admission = read(verified(admission_record))
    entries = [e for e in value['entries'] if e['attempt_id'] == admission['attempt_id']]
    require(value['kind'] == kind and len(entries) == 1 and entries[0]['admission'] == admission_record and
        entries[0]['output'] == str(Path(output).resolve()) and admission['study_id'] == value['study_id'], 'Attempt/output differs from prospective registry')
    claim_path,terminal_path = ledger_paths(value,admission['attempt_id'])
    require(not terminal_path.exists(), 'Registered attempt already terminal; fresh-directory retry forbidden')
    write(claim_path,dict(schema='modern-attempt-claim-v2',registry=record,entry=entries[0],timestamp=time.time()))
    return dict(registry=record,attempt_id=admission['attempt_id'],claim=descriptor(claim_path),entry=entries[0])


def finish(binding, artifact=None, failure=None, costs=None):
    value = registry(binding['registry'])
    claim_path,path = ledger_paths(value,binding['attempt_id'])
    require(descriptor(claim_path) == binding['claim'], 'Attempt claim differs')
    write(path,dict(schema='modern-attempt-terminal-v2',binding=binding,state='completed' if artifact else 'failed',
        artifact=artifact,failure=failure,costs=costs,timestamp=time.time(),automatic_retry=False))


def close_registry(record, output):
    value = registry(record); rows=[];allowed=set()
    for entry in value['entries']:
        claim_path,terminal_path = ledger_paths(value,entry['attempt_id']);allowed.update((claim_path.name,terminal_path.name))
        claim_record = descriptor(claim_path) if claim_path.exists() else None
        terminal_record = descriptor(terminal_path) if terminal_path.exists() else None
        terminal = read(terminal_path) if terminal_record else None
        if terminal:
            require(terminal['binding']['registry'] == record and terminal['binding']['entry'] == entry and terminal['binding']['claim'] == claim_record,
                'Terminal attempt differs from registry')
            if terminal['artifact']:verified(terminal['artifact'])
            if terminal['failure'] and isinstance(terminal['failure'],dict) and set(terminal['failure']) >= {'path','sha256'}:verified(terminal['failure'])
        rows.append(dict(entry=entry,claim=claim_record,terminal=terminal_record,
            state=terminal['state'] if terminal else ('incomplete' if claim_record else 'not_started'),
            costs=terminal['costs'] if terminal else None))
    require(all(p.name in allowed for p in Path(value['ledger_directory']).iterdir()), 'Unregistered attempt ledger content')
    write(output,dict(schema='modern-attempt-closure-v2',registry=record,kind=value['kind'],rows=rows,
        all_registered_attempts_enumerated=True,all_completed=all(r['state']=='completed' for r in rows),
        failures_and_incomplete_count=sum(r['state'] != 'completed' for r in rows),retry_policy=value['retry_policy']))


def closure(record, kind):
    value=read(verified(record));reg=registry(value['registry'])
    require(value['schema']=='modern-attempt-closure-v2' and value['kind']==kind and reg['kind']==kind and
        value['all_registered_attempts_enumerated'] is True and value['all_completed'] is True and
        len(value['rows'])==len(reg['entries']) and value['failures_and_incomplete_count']==0,'Complete registered terminal closure required')
    require([r['entry'] for r in value['rows']]==reg['entries'],'Attempt closure omits/reorders entries')
    for row in value['rows']:
        terminal=read(verified(row['terminal']));verified(row['claim'])
        require(terminal['state']=='completed' and terminal['binding']['entry']==row['entry'] and
            terminal['binding']['registry']==value['registry'] and terminal['binding']['claim']==row['claim'],'Wrong terminal closure')
        verified(terminal['artifact'])
    return value


def closed_artifact(closure_record, artifact_record, kind):
    value=closure(closure_record,kind)
    rows=[r for r in value['rows'] if read(verified(r['terminal']))['artifact']==artifact_record]
    require(len(rows)==1,'Artifact is not the one registered completed terminal attempt')
    terminal=read(verified(rows[0]['terminal']));artifact=read(verified(artifact_record))
    require(artifact['admission']==rows[0]['entry']['admission'] and artifact['attempt_binding']==terminal['binding'],
        'Frozen artifact differs from its one registered admission/claim/terminal')
    return rows[0]


def release(request_path,output):
    import correction_screen_driver as core
    request=read(request_path);study=read(verified(request['teacher_study_selection_freeze']))
    require(study['schema']=='modern-teacher-study-selection-v2' and study['source_selection_closed'] is True,'Teacher source selection is not closed')
    closure(study['attempt_closure'],'teacher');closure(request['correction_attempt_closure'],'correction')
    require(len(request['score_freezes'])==6,'All six primary correction freezes required')
    expected={(b,s) for b in protocol()['native'] for s in protocol()['seeds']};seen=set();cells=[]
    for record in request['score_freezes']:
        closed_artifact(request['correction_attempt_closure'],record,'correction')
        path=verified(record);score=read(path);core.verify_tree(path.parent,score['payload'])
        admission=read(verified(score['admission']));role=read(verified(score['role_freeze']))
        spec=score['modern_teacher']['specification'];key=(spec['backbone'],spec['seed'])
        require(key in expected and key not in seen and spec['family']=='gnnm_boundary_4' and
            score['final_labels_read'] is False and score['implementation_sha256']==implementations() and
            score['score_backend']=='randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2' and
            admission['teacher_study_selection_freeze']==request['teacher_study_selection_freeze'] and
            admission['role_freeze']==score['role_freeze'] and role['seed']==spec['seed'], 'Wrong/repeated primary correction freeze')
        identity=source_identity(admission,role,read(verified(role['graph_input'])))
        require(identity==score['source_identity'],'Correction input/label custody differs')
        cells.append(dict(score_freeze=record,admission=score['admission'],role_freeze=score['role_freeze'],
            backbone=spec['backbone'],seed=spec['seed'],source_identity=identity));seen.add(key)
    require(seen==expected,'Six exact graph/seed source freezes required')
    write(output,dict(schema='modern-final-label-release-v2',request=descriptor(request_path),
        teacher_study_selection_freeze=request['teacher_study_selection_freeze'],teacher_attempt_closure=study['attempt_closure'],
        correction_attempt_closure=request['correction_attempt_closure'],cells=cells,
        implementation_sha256=implementations(),final_labels_not_opened=True,release_eligible=True))


def verify_release(record,target_score=None):
    import correction_screen_driver as core
    value=read(verified(record))
    require(value['schema']=='modern-final-label-release-v2' and value['release_eligible'] is True and
        value['implementation_sha256']==implementations() and len(value['cells'])==6,'Shared six-cell final release certificate required')
    closure(value['teacher_attempt_closure'],'teacher');closure(value['correction_attempt_closure'],'correction')
    study=read(verified(value['teacher_study_selection_freeze']))
    require(study['schema']=='modern-teacher-study-selection-v2' and study['implementation_sha256']==implementations() and
        study['attempt_closure']==value['teacher_attempt_closure'] and study['source_selection_closed'] is True,'Study closure differs')
    seen=set()
    for cell in value['cells']:
        path=verified(cell['score_freeze']);score=read(path);core.verify_tree(path.parent,score['payload'])
        closed_artifact(value['correction_attempt_closure'],cell['score_freeze'],'correction')
        admission=read(verified(cell['admission']));role=read(verified(cell['role_freeze']))
        spec=score['modern_teacher']['specification']
        require(score['admission']==cell['admission'] and score['role_freeze']==cell['role_freeze'] and
            spec['backbone']==cell['backbone'] and spec['seed']==cell['seed'] and spec['family']=='gnnm_boundary_4' and
            score['implementation_sha256']==implementations() and
            score['score_backend']=='randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2' and
            admission['teacher_study_selection_freeze']==value['teacher_study_selection_freeze'] and
            source_identity(admission,role,read(verified(role['graph_input'])))==cell['source_identity']==score['source_identity'] and
            score['final_labels_read'] is False,'Release cell/admission/role/label binding differs')
        seen.add((cell['backbone'],cell['seed']))
    require(seen=={(b,s) for b in protocol()['native'] for s in protocol()['seeds']},'Release is not the six paired cells')
    if target_score is not None:require(any(c['score_freeze']==target_score for c in value['cells']),'Target score freeze is outside shared release')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('certify-qualification','freeze-registry','release-final-labels'):
        command=sub.add_parser(name);command.add_argument('--request',required=True);command.add_argument('--output',required=True)
    command=sub.add_parser('close-registry');command.add_argument('--registry',required=True);command.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='certify-qualification':certify(args.request,args.output)
    elif args.command=='freeze-registry':freeze_registry(args.request,args.output)
    elif args.command=='close-registry':close_registry(descriptor(args.registry),args.output)
    else:release(args.request,args.output)


if __name__=='__main__':main()
