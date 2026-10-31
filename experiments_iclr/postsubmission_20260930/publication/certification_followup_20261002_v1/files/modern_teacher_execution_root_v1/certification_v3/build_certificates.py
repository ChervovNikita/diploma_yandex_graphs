"""Root metadata preparation/issuance; no scientific imports or array decoding.

prepare only reads JSON/source text and writes ineligible drafts. issue requires
root approval on the original host and calls the sealed stdlib custody certifier;
that existing gate hashes frozen binary descriptors, never decodes arrays.
Neither command launches training, forwards, GPUs, SSH or reporting.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
RUNS = PHASE/'modern_teacher_execution_root_v1'
REMOTE = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
PACKET = PHASE/'continuous_graph_efficiency_gap_v1/modern_backbone_teacher_amendment_v3'
PACKET_SHA = 'e2fd767c4d08e93ca7ec9c9c526440c4cc5b8f4e7d8bb345de90308e9c57302b'
SCORE_BACKEND = 'randomized_APS_CPU_preserve_FP32_fixed_uniforms_v2'
SEEDS = (17,29,43)
FAMILIES = ('single_author','gnnm_boundary_4','independent_author_4_same_width')
TEXT_SUFFIXES = {'.json','.py','.md','.diff','.sh','.txt','.log','.jsonl'}
AUDIT_SOURCE_PINS = {
    'qualification_entry_v5.py':'a6d8c72c22d56f129b8008c97cdd5aae48eddc7710adcc88308c94a9a38894b2',
    'full_parity_checks_v1.py':'f9a7736a3608b8db42380fc1372b0f74c87517ec8112d36d022cb48e0c1b7e20',
    'source_pack_audit_entry_v2.py':'412765d43166be718e1a85ac7666824ef95ac04bbb459192ae826f3d0da2c296'}


def require(condition,message):
    if not condition: raise ValueError(message)


def confined(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PHASE) and path != PHASE,'Path must stay in this phase')
    return path


def sha_text(path):
    path = confined(path)
    require(path.suffix in TEXT_SUFFIXES,'Metadata/source text only: '+str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path = confined(path)
    require(path.suffix == '.json','Only JSON metadata may be decoded')
    return json.loads(path.read_text())


def desc(path):
    path = confined(path)
    return dict(path=str(REMOTE/path.relative_to(PHASE)),sha256=sha_text(path))


def local(record):
    remote = Path(record['path'])
    require(remote.is_absolute() and remote.is_relative_to(REMOTE),'Original remote phase descriptor required')
    path = confined(PHASE/remote.relative_to(REMOTE))
    require(sha_text(path) == record['sha256'],'Original metadata fingerprint differs: '+str(path))
    return path


def same_descriptor(a,b):
    return all(a[k] == b[k] for k in ('path','sha256'))


def write(path,value):
    path = confined(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def packet_identity():
    require(sha_text(PACKET/'MANIFEST.json') == PACKET_SHA and
        read(PACKET/'SEAL.json')['manifest_sha256'] == PACKET_SHA,'Sealed v3 source identity differs')
    for row in read(PACKET/'MANIFEST.json')['payload']:
        require(sha_text(PACKET/row['path']) == row['sha256'],'Sealed v3 source payload differs')
    for name,value in AUDIT_SOURCE_PINS.items():
        require(sha_text(RUNS/name) == value,'Original audit instrumentation source differs')
    return read(PACKET/'TEACHER_PROTOCOL.json'),read(PACKET/'MODEL_IMPLEMENTATION_BINDINGS.json'),read(PACKET/'IMPLEMENTATION_BINDINGS.json')


def source_cases():
    terminal_path = RUNS/'source_pack_audit_run03/TERMINAL.json'
    terminal = read(terminal_path)
    bundle_record = terminal['source_binding'];bundle = read(local(bundle_record))
    require(terminal['complete'] is True and terminal['training_steps'] == 0 and
        terminal['final_label_pack_files_opened'] is False and terminal['scientific_metrics_calculated'] is False and
        bundle['schema'] == 'modern-source-label-bindings-v2' and bundle['independent_audit'] is True and
        bundle['final_pool_labels_opened'] is False and bundle['fits_or_scores_performed'] is False,
        'Successful independent source-pack audit required')
    source_request = RUNS/'SOURCE_PACK_AUDIT_REQUEST_v3.json'
    require(sha_text(source_request) == terminal['request_sha256'],'Source-pack terminal/request linkage differs')
    identities = {}
    for pair in bundle['pairs']:
        roles = read(local(pair['role_freeze']));graph = read(local(pair['graph_input']))
        provider = read(local(pair['provider_row_identity']));extraction = read(local(pair['extraction_provenance']))
        require(pair['independently_extracted_and_row_verified'] is True and
            roles['seed'] == pair['seed'] and roles['graph_input'] == pair['graph_input'] and roles['labels_read'] is False and
            roles['role_derivation_version'] == 'derived_roles_v2' and
            provider['schema'] == 'modern-provider-row-identity-audit-v1' and
            same_descriptor(provider['graph_input'],pair['graph_input']) and
            provider['full_feature_and_canonical_edge_rows_exact'] is True and provider['labels_used_to_establish_rows'] is False and
            extraction['schema'] == 'modern-independent-source-extraction-audit-v1' and
            extraction['source_roles'] == ['train','validation'] and extraction['non_source_label_values_used'] is False and
            extraction['final_label_pack_files_opened'] is False and extraction['fits_or_scores_performed'] is False,
            'Exact graph/provider/extraction provenance differs')
        require(local(extraction['auditor_source']) == RUNS/'source_pack_audit_entry_v2.py','Unexpected independent extraction source')
        cases = [row for row in extraction['checks'] if row['seed'] == pair['seed']]
        require(len(cases) == 1 and cases[0]['role_freeze'] == pair['role_freeze'],'Extraction role linkage differs')
        payload = {row['path']:row for row in roles['payload']}
        for name in ('train','validation'):
            check = cases[0]['roles'][name]
            require(check['pack'] == pair['source_labels'][name] and check['row_alignment_verified'] is True and
                check['independent_raw_source_reextraction_exact'] is True and
                check['nodes'] == roles['source_counts'][name] and
                payload[name+'_nodes.npy']['sha256'] == check['nodes_sha256'],'Exact source-role/node/label metadata linkage differs')
        dimensions = {key:graph[key] for key in ('num_nodes','num_features','num_classes','num_edges')}
        require(provider['dimensions'] == dimensions,'Provider dimensions differ')
        identity = dict(graph=graph['graph'],backbone=pair['backbone'],seed=pair['seed'],graph_input=pair['graph_input'],
            role_freeze=pair['role_freeze'],source_labels=pair['source_labels'],source_label_binding=bundle_record,
            provider_row_identity=pair['provider_row_identity'],extraction_provenance=pair['extraction_provenance'],dimensions=dimensions)
        key = (pair['backbone'],pair['seed']);require(key not in identities,'Repeated source case');identities[key] = identity
    require(set(identities) == {(b,s) for b in ('polyformer_mono','polynormer_r') for s in SEEDS},'Exact six source cases required')
    for backbone in ('polyformer_mono','polynormer_r'):
        reference = identities[backbone,17]
        require(all(all(identities[backbone,s][key] == reference[key] for key in
            ('graph','backbone','graph_input','source_label_binding','provider_row_identity','dimensions')) for s in SEEDS),
            'Seed coverage graph/provider/rows differ')
    return identities,[desc(terminal_path),desc(source_request),bundle_record]


def evidence_checks(row,identities,models):
    request = read(local(row['request']));admission = read(local(row['admission']))
    partial = read(local(row['qualification']));terminal = read(local(row['root_terminal']))
    cell = read(local(row['cell_freeze']));optimizer = read(local(row['external_optimizer_audit']))
    out = Path(request['output']);cell_local = local(row['cell_freeze']).parent
    selection_record = desc(cell_local/'teacher_selection.json');selection = read(cell_local/'teacher_selection.json')
    costs_record = desc(cell_local/'COSTS.json');costs = read(cell_local/'COSTS.json')
    b,f = row['backbone'],row['family'];identity = identities[b,17]
    require(request['teacher_admission'] == row['admission'] and request['root_admitted'] is True and
        request['action'] == 'qualification' and request['full_fit_admitted'] is False and request['heldout_scoring_admitted'] is False and
        request['deterministic'] is True and request['aps_score_backend'] == 'cpu_fixed_algorithm' and
        terminal['complete'] is True and terminal['request_sha256'] == row['request']['sha256'] and
        terminal['qualification_receipt_sha256'] == row['qualification']['sha256'] and
        terminal['optimizer_audit_sha256'] == row['external_optimizer_audit']['sha256'] and
        terminal['report_eligible'] is False and terminal['final_labels_read'] is False,'Exact successful original root attempt required')
    require(row['root_terminal']['path'] == str(out/'TERMINAL.json') and
        row['cell_freeze']['path'] == str(out/'cell/TEACHER_CELL_FREEZE.json') and
        row['qualification']['path'] == str(out/'cell/QUALIFICATION_RECEIPT.json') and
        row['external_optimizer_audit']['path'] == str(out/'EXTERNAL_OPTIMIZER_AUDIT.json'),'Original run artifact paths differ')
    require(admission['backbone'] == b and admission['family'] == f and admission['config'] == 0 and
        admission['role_freeze'] == identity['role_freeze'] and admission['source_labels'] == identity['source_labels'] and
        admission['environment'] == partial['environment'] == cell['environment'] and
        admission['environment']['deterministic_algorithms'] is True and
        admission['environment']['cublas_workspace_config'] == ':4096:8' and
        partial['schema'] == 'modern-teacher-qualification-v1' and partial['passed'] is True and
        partial['report_eligible'] is False and partial['final_labels_read'] is False and
        partial['implementation_sha256'] == admission['implementation_sha256'] and
        all(partial['implementation_sha256'][key] == value for key,value in models.items()),'Original tested config0/seed17/model/input differs')
    payload = {r['path']:r for r in cell['payload']}
    require(cell['admission'] == row['admission'] and cell['role_freeze'] == identity['role_freeze'] and
        cell['selection'] == selection and cell['specification'] == selection['specification'] and
        (selection['specification']['backbone'],selection['specification']['family'],selection['specification']['config'],selection['specification']['seed']) == (b,f,0,17) and
        selection['qualification_only'] is True and selection['report_eligible'] is False and
        payload['QUALIFICATION_RECEIPT.json']['sha256'] == row['qualification']['sha256'] and
        payload['teacher_selection.json']['sha256'] == selection_record['sha256'] and cell['costs'] == costs == row['costs'],
        'Original cell/selection/partial/preprocessing/cost payload links differ')
    parity = partial['parity'];stages = [False,True] if b == 'polynormer_r' else [False]
    require((parity['rtol_logits'],parity['atol_logits'],parity['rtol_gradients'],parity['atol_gradients']) == (1e-5,1e-6,1e-4,1e-6) and
        parity['native_vs_identity_boundary_passed'] is True and parity['shortened_fit_report_eligible'] is False and
        parity['selected_checkpoint_replay_max_absolute_logit_difference'] == 0 and
        [r['global_stage'] for r in parity['checks']] == stages and
        all(r['full_graph'] is True and r['all_graph_logits_and_active_gradients_finite'] is True and
            r['dormant_owner_removal_verified'] is True and r['max_absolute_logit_difference'] == 0 and
            r['common_body_gradient_tensors_compared'] > 0 and r['private_bias_gradient_rows_compared'] == 8 for r in parity['checks']),
        'Original fixed-tolerance full-stage parity evidence incomplete')
    steps = 4 if b == 'polynormer_r' else 3;members = 1 if f == 'single_author' else 4
    require(len(optimizer['steps']) == steps == partial['updates_completed'] == selection['updates_completed'] == terminal['training_steps'] and
        optimizer['selected_full_graph_logits_finite'] is True and
        all(r['finite_gradient_tensors'] > 0 and len(r['member_gradient_energy']) == members and
            all(v > 0 for v in r['member_gradient_energy']) for r in optimizer['steps']) and
        len(optimizer['optimizer_groups']) == 1 and optimizer['optimizer_groups'][0]['live_membership_exact'] is True and
        len(optimizer['restores']) == (1 if b == 'polynormer_r' else 0) and
        all(r['exact_optimizer_state_restored'] is True and r['step_counters_checked'] is True for r in optimizer['restores']) and
        len(optimizer['model_restores']) == (2 if b == 'polynormer_r' else 1) and
        all(r['exact_model_state_restored'] is True for r in optimizer['model_restores']),'Original optimizer/member/restore checks incomplete')
    exact = optimizer['exact_input_custody']
    require(all(exact[k] == admission[k] for k in ('role_freeze','source_labels','environment','backbone','family','implementation_sha256')) and
        exact['graph_input'] == identity['graph_input'] and exact['source_manifest'] == request['source_manifest'],
        'Original external audit source/role/label custody differs')
    seal_path = local(request['source_manifest']).parent/'SEAL.json';tested_seal = desc(seal_path)
    require(read(seal_path)['manifest_sha256'] == request['source_manifest']['sha256'] and
        any(same_descriptor(r,tested_seal) for r in request['protected_files']),'Original tested source seal linkage differs')
    entry = RUNS/'qualification_entry_v5.py'
    require(Path(request['entry_script']).name == entry.name and
        any(same_descriptor(r,desc(entry)) for r in request['protected_files']),'Original guarded primitive/CPU-APS instrumentation differs')
    bound = read(RUNS/(out.name+'_bound')/'TERMINAL.json')
    require(bound['complete'] is True and bound['child_exit_code'] == 0 and bound['timed_out'] is False and
        bound['root_request_unchanged'] is True,'Original whole-attempt supervisor did not close')
    linkage = dict(schema='modern-legacy-qualification-run-binding-v3',original_root_request=row['request'],
        root_terminal=row['root_terminal'],teacher_cell_freeze=row['cell_freeze'],original_optimizer_audit=row['external_optimizer_audit'])
    return dict(admission=admission,partial=partial,selection=selection,preprocessing_evidence=selection_record,
        costs_record=costs_record,tested_source_seal=tested_seal,partial_run_binding=linkage,
        evidence=[row[key] for key in ('request','admission','qualification','external_optimizer_audit','root_terminal','cell_freeze')]+
            [selection_record,costs_record,desc(RUNS/(out.name+'_bound')/'TERMINAL.json')])


def history_records():
    """Retain all original root JSON history, including every failed/guarded attempt."""
    return [desc(path) for path in sorted(RUNS.rglob('*.json')) if not path.is_relative_to(HERE)]


def output_directory(value):
    path = confined(value)
    require(path.is_relative_to(HERE) and path != HERE and not path.exists(),'Use a new output directory inside certification_v3')
    path.mkdir(parents=True,exist_ok=False)
    return path


def prepare(args):
    protocol,models,implementations = packet_identity();identities,source_evidence = source_cases()
    qualification_path = RUNS/'SIX_QUALIFICATION_AUDIT_v1.json';qualification = read(qualification_path)
    require(qualification['schema'] == 'gnnm-modern-six-full-graph-qualification-audit-v1' and
        qualification['all_six_metadata_gates_verified'] is True and qualification['numerically_tested_seed'] == 17 and
        len(qualification['rows']) == 6,'Original six-qualification metadata audit required')
    expected = {(b,f) for b in protocol['native'] for f in FAMILIES};seen = set()
    checked = []
    for row in qualification['rows']:
        key = (row['backbone'],row['family']);require(key in expected and key not in seen,'Wrong/duplicate qualification family')
        seen.add(key);checked.append((row,evidence_checks(row,identities,models)))
    require(seen == expected,'All six family qualifications required')
    history = history_records();out = output_directory(args.output)
    write(out/'HISTORY_BINDINGS.json',dict(schema='modern-original-qualification-history-v3',records=history,
        all_original_json_history_retained=True,automatic_retry=False,scope='Original root metadata and costs, including failed and guard-blocked attempts; no new execution'))
    history_record = desc(out/'HISTORY_BINDINGS.json');index = []
    for row,evidence in checked:
        b,f = row['backbone'],row['family'];family_dir = out/'families'/(b+'__'+f)
        coverage = dict(schema='modern-exact-role-label-coverage-v2',backbone=b,family=f,
            environment=evidence['admission']['environment'],model_sha256=models,tested_seed17_input=identities[b,17],
            authorized_target_inputs=[identities[b,s] for s in SEEDS],
            checks={name:True for name in ('identical_graph_provider_and_rows','exact_role_nodes_and_source_pack_hashes',
                'independent_train_validation_label_extraction','target_label_shapes_ranges_and_node_alignment','numerical_runtime_tested_seed17_only')},
            numerical_tests_on_untested_seeds=False,numerical_tested_configurations=[0],
            evidence=source_evidence+[identities[b,17]['provider_row_identity'],identities[b,17]['extraction_provenance'],desc(qualification_path)],
            check_provenance='Exact successful independent source audit, audited source assertions at their pinned hash, role/pack/node metadata links and original seed17 qualification; no new label or numerical inspection',
            root_coverage_authorized=False,root_signature=None)
        coverage_path = family_dir/'COVERAGE.json';write(coverage_path,coverage)
        audit = copy.deepcopy(read(PACKET/'templates/EXTERNAL_RUNTIME_AUDIT_TEMPLATE.json'))
        audit.update(partial_receipt=row['qualification'],tested_admission=row['admission'],tested_source_seal=evidence['tested_source_seal'],
            model_sha256=models,environment=evidence['admission']['environment'],checks={name:True for name in audit['checks']},
            preprocessing_evidence=evidence['preprocessing_evidence'],tested_preprocessing=evidence['selection']['preprocessing'],
            evidence=evidence['evidence']+source_evidence+[desc(qualification_path)],
            instrumentation_sources=[desc(RUNS/name) for name in AUDIT_SOURCE_PINS],attempt_history=[history_record],
            failed_attempts_and_fresh_version_retries_retained=True,score_backend=SCORE_BACKEND,
            tested_seed17_input=identities[b,17],coverage_authorization=desc(coverage_path),
            partial_run_binding=evidence['partial_run_binding'],numerical_tested_seeds=[17],numerical_tested_configurations=[0],
            root_evidence_accepted=False,root_signature=None,
            check_provenance=dict(optimizer_membership_and_native_grouping='Original pinned guarded factory; successful terminal and optimizer_groups live membership receipt',
                logits_gradients_member_connectivity='Original parity checks, selected_full_graph_logits_finite and every guarded training step/member energy receipt',
                boundary_bias_ownership_and_gradients='Original pinned full_parity source, every stage dormant removal/common-body/private-B receipt',
                model_and_Adam_restore='Original exact model_restores and optimizer restores/step counter receipts; Squirrel has no local/global transition',
                selected_replay_and_primitive_safe_loader='Original successful pinned JSON-primitive runtime + weights_only replay source, exact replay difference and terminal hashes',
                scope='Projection of existing root evidence only; no new numerical tests, no four-config numerical testing'))
        audit_path = family_dir/'EXTERNAL_AUDIT.json';write(audit_path,audit)
        for seed in SEEDS:
            target = dict(evidence['admission'],execution_authorized=False,implementation_sha256=implementations,teacher_protocol=protocol,
                source_label_binding=identities[b,seed]['source_label_binding'],role_freeze=identities[b,seed]['role_freeze'],
                source_labels=identities[b,seed]['source_labels'],config=0,root_signature=None)
            name = b+'__'+f+'__config0__target_seed'+str(seed)
            request_path = out/'requests'/(name+'.json')
            write(request_path,dict(target=target,partial_receipt=row['qualification'],external_audit=desc(audit_path),root_approval=None))
            feasibility = copy.deepcopy(read(PACKET/'templates/FULL_SCHEDULE_FEASIBILITY_TEMPLATE.json'))
            feasibility.update(approved=False,input_identity=identities[b,seed],backbone=b,family=f,
                environment=target['environment'],model_sha256=models,teacher_protocol=protocol,
                short_qualification_is_not_full_fit_evidence=True,resource_evidence=[evidence['costs_record'],row['cell_freeze'],row['root_terminal'],desc(qualification_path)],
                root_signature=None,root_resource_accounting=None,
                observed_short_complete_graph_costs=row['costs'],
                prospective_native_schedule=dict(local_updates=200,global_updates=1000,total_updates=1200) if b == 'polynormer_r' else dict(max_updates=2000,patience=250),
                limitations='Short measured fit aggregate is not per-stage timing or full-schedule completion; root must budget full stages, peak, optimizer/checkpoint/transfers and history before approval. R17 uses its own resource_forecast instead.')
            feasibility_path = out/'feasibility_templates'/(name+'.json');write(feasibility_path,feasibility)
            index.append(dict(backbone=b,family=f,seed=seed,config=0,request=desc(request_path),
                coverage=desc(coverage_path),external_audit=desc(audit_path),feasibility_template=desc(feasibility_path),
                certificate_relative_path='certificates/'+name+'.json',input_identity=identities[b,seed],
                numerical_tested_seed=17,numerical_tested_config=0,scientific_execution_authorized=False))
    write(out/'INDEX.json',dict(schema='modern-certificate-preparation-index-v3',source_manifest=desc(PACKET/'MANIFEST.json'),
        source_seal=desc(PACKET/'SEAL.json'),qualification_audit=desc(qualification_path),source_evidence=source_evidence,
        history=history_record,rows=index,certificate_count=0,target_requests=18,root_approval_required=True,
        scientific_fit_admitted=False,numerical_tests_performed=False))
    print(json.dumps(dict(prepared_directory=str(out),target_requests=18,eligible_certificates=0,root_approval_required=True)))


def issue(args):
    require(PHASE == REMOTE,'Eligible issuance runs only at the original remote phase; local mirrors may prepare/check only')
    packet_identity()
    prepared = confined(args.prepared);index = read(prepared/'INDEX.json')
    require(index['schema'] == 'modern-certificate-preparation-index-v3' and
        index['source_manifest'] == desc(PACKET/'MANIFEST.json') and len(index['rows']) == 18 and
        {(r['backbone'],r['family'],r['seed'],r['config']) for r in index['rows']} ==
            {(b,f,s,0) for b in ('polyformer_mono','polynormer_r') for f in FAMILIES for s in SEEDS},
        'Exact eighteen prepared config0 targets and sealed v3 source required')
    approval_path = confined(args.root_approval);approval = read(approval_path)
    require(approval['schema'] == 'modern-root-certification-approval-v3' and approval['approved'] is True and
        isinstance(approval['root_signature'],str) and bool(approval['root_signature'].strip()) and
        isinstance(approval['signed_utc'],str) and bool(approval['signed_utc'].strip()) and
        approval['prepared_index'] == desc(prepared/'INDEX.json') and approval['source_manifest'] == desc(PACKET/'MANIFEST.json') and
        all(approval[name] is True for name in ('source_engineering_audit_reviewed_and_accepted','original_qualification_evidence_accepted',
            'exact_seed_role_label_coverage_authorized','unchanged_four_config_scope_authorized','failed_attempt_history_retained')),
        'Root must sign exact reviewed source/evidence/coverage/config scope before eligible issuance')
    local(approval['independent_source_engineering_audit'])
    for record in read(local(index['history']))['records']:local(record)
    # Run source/evidence metadata guards again; no draft booleans supply evidence.
    identities,_ = source_cases();models = read(PACKET/'MODEL_IMPLEMENTATION_BINDINGS.json')
    qualification = read(local(index['qualification_audit']))
    for row in qualification['rows']:evidence_checks(row,identities,models)
    for row in index['rows']:
        for key in ('request','coverage','external_audit','feasibility_template'):local(row[key])
    out = output_directory(args.output);approval_record = desc(approval_path)
    write(out/'ISSUE_STARTED.json',dict(root_approval=approval_record,prepared_index=desc(prepared/'INDEX.json'),
        source_manifest=desc(PACKET/'MANIFEST.json'),scientific_fit_admitted=False))
    spec = importlib.util.spec_from_file_location('modern_custody_certification_v3',PACKET/'prototype/modern_custody.py')
    custody = importlib.util.module_from_spec(spec);spec.loader.exec_module(custody)
    emitted = [];family_records = {}
    try:
        for row in index['rows']:
            key = (row['backbone'],row['family'])
            if key not in family_records:
                folder = out/'families'/(key[0]+'__'+key[1])
                coverage = read(local(row['coverage']));coverage.update(root_coverage_authorized=True,root_signature=approval['root_signature'],root_approval=approval_record)
                coverage_path = folder/'COVERAGE.json';write(coverage_path,coverage)
                audit = read(local(row['external_audit']));audit.update(coverage_authorization=desc(coverage_path),
                    root_evidence_accepted=True,root_signature=approval['root_signature'],root_approval=approval_record)
                audit_path = folder/'EXTERNAL_AUDIT.json';write(audit_path,audit)
                family_records[key] = desc(audit_path)
            request = read(local(row['request']));request.update(external_audit=family_records[key],root_approval=approval_record)
            certificate_path = out/row['certificate_relative_path'];certificate_path.parent.mkdir(parents=True,exist_ok=True)
            request_path = out/'requests'/certificate_path.name;write(request_path,request)
            custody.certify(str(request_path),str(certificate_path))
            emitted.append(dict(backbone=row['backbone'],family=row['family'],seed=row['seed'],config=0,
                modern_qualification_certificate=desc(certificate_path),request=desc(request_path),
                feasibility_template=row['feasibility_template'],input_identity=row['input_identity']))
        write(out/'INDEX.json',dict(schema='modern-issued-certificate-index-v3',source_manifest=desc(PACKET/'MANIFEST.json'),
            root_approval=approval_record,rows=emitted,certificates=18,numerical_tested_seeds=[17],numerical_tested_configurations=[0],
            registered_configuration_scope=[0,1,2,3],scientific_fit_admitted=False,final_labels_read=False))
        write(out/'ISSUE_TERMINAL.json',dict(complete=True,certificates=18,index=desc(out/'INDEX.json'),root_approval=approval_record,
            scientific_fit_admitted=False,final_labels_read=False))
    except Exception as error:
        write(out/'ISSUE_FAILED.json',dict(error_type=type(error).__name__,message=str(error),emitted=emitted,
            automatic_retry=False,scientific_fit_admitted=False))
        raise
    print(json.dumps(dict(issued_directory=str(out),certificates=18,scientific_fit_admitted=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__);commands = parser.add_subparsers(dest='command',required=True)
    prepare_parser = commands.add_parser('prepare');prepare_parser.add_argument('--output',required=True)
    issue_parser = commands.add_parser('issue');issue_parser.add_argument('--prepared',required=True)
    issue_parser.add_argument('--root-approval',required=True);issue_parser.add_argument('--output',required=True)
    args = parser.parse_args();{'prepare':prepare,'issue':issue}[args.command](args)


if __name__ == '__main__':main()
