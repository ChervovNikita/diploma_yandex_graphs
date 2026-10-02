"""AST/hash/JSON-only admission checks; no runner, SSH, model or array access."""
import ast
import difflib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
from admission_support import HERE, PHASE, R17_REL, CELLS, SEEDS, ARMS, require, read, sha, bound, descriptor, write, object_hash, expected_attempts, verify_sources, source_descriptors, bindings
from lineage_support import verify_lineage, prior_registries
from continuation_support import finite_plan, PHASES, decision_guard
from build_continuation import template

def main():
    allowed={'argparse','ast','copy','datetime','difflib','hashlib','io','json','math','os','pathlib','shlex',
             'subprocess','sys','tarfile','time','traceback','admission_support','build_admissions',
             'continuation_support','build_continuation','lineage_support','finite_coordinator'}
    sources=[]
    for p in sorted(HERE.glob('*.py')):
        tree=ast.parse(p.read_text(),filename=str(p))
        for n in ast.walk(tree):
            names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) else []
            require(all(x in allowed for x in names),'Only stdlib/source metadata imports allowed: '+p.name)
        sources.append(descriptor(p))
    verify_sources();lineage=verify_lineage()
    contexts=read(HERE/'prepared_v1/CONTEXTS_DRAFT.json')
    require(len(contexts)==6 and [(c['graph'],c['backbone'],c['seed'],c['source_split_index']) for c in contexts]==
            [(g,b,s,i) for g,b in CELLS for i,s in enumerate(SEEDS)],'Exact unchanged six contexts required')
    expected_keys=set(read(PHASE/'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2/templates/CELL_CONTEXT_TEMPLATE.json'))
    for c in contexts:
        require(set(c)==expected_keys and c['config']==0 and c['resource_forecast']['admitted'] is False and
            c['source_bindings']==bindings()[0] and c['protocol']==bindings()[1] and
            c['roles_frozen_before_label_extraction'] is True and c['independent_of_stage1_outcomes'] is True and
            set(c['source_labels'])=={'train','validation'} and 'known Photo17 v2' in c['prior_exposure_disclosure'] and
            'All six cold and every actual-warm' in c['prior_exposure_disclosure'],'Exact disclosed context/control boundary required')
        cert=read(bound(c['modern_qualification_certificate']))
        for k in ('role_freeze','graph_input','source_labels','source_label_binding'):
            require(c[k]==cert['authorized_target_input'][k],'Preserve exact source metadata including optional bytes')
        require(c['environment']==cert['environment'],'Certified runtime unchanged')
    draft=read(HERE/'prepared_v1/PROSPECTIVE_72_ATTEMPTS_DRAFT.json')
    require(draft['attempts']==expected_attempts(contexts,draft['anchor_directory']),'Canonical draft72 inventory required')
    registry={'contexts':contexts,'attempts':draft['attempts'],'anchor_directory':draft['anchor_directory']}
    plan=finite_plan(registry)
    counts={p:sum(r['phase']==p for r in plan) for p in PHASES}
    require(counts=={'qualify':6,'warm':6,'initialize':30,'fit':30} and len(plan)==72 and
            len({r['key'] for r in plan})==len({r['output'] for r in plan})==72,'All72fresh phases and six cold gates required')
    require([PHASES.index(r['phase']) for r in plan]==sorted(PHASES.index(r['phase']) for r in plan),
            'All six cold qualifiers before warm; all warm before actual-warm initialization; then fits')
    require([(r['phase'],r['arm']) for r in plan[:6]]==[('qualify',None)]*6,'First cold qualifier must not be inherited')
    for r in plan:
        c=next(c for c in contexts if object_hash(c)==r['context_sha256'])
        require(r['whole_cap_seconds']==c['resource_forecast']['forecast']['wall_seconds_by_phase'][r['phase']], 'Registered caps unchanged')
        require('/graph_init_precision_execution_root_v2/study_v2/' in r['output'],'Every phase gets fresh precision output identity')
    root=read(HERE/'prepared_v1/ROOT_REGISTRATION_DECISION_TEMPLATE.json')
    method_audit=descriptor(PHASE/'graph_init_source_audit_v1/round17_v3_precision_recheck/REPORT.json')
    require(method_audit['sha256']=='e5adf2f5ddafae9b38ad9ff7a03f069152297618d88a494ebe481c221be5a96f' and
            root['independent_source_audit']==method_audit,'Exact accepted v3 scoped report in false template required')
    req=read(HERE/'prepared_v1/REGISTRY_REQUEST_DRAFT.json')
    lin=read(HERE/'prepared_v1/LINEAGE_DRAFT.json')
    require(root['approved'] is False and root['registration_authorized'] is False and root['cold_qualification_authorized'] is False and
        req['registration_authorized'] is False and lin['approved'] is False and lin['predecessor_history_complete'] is False and
        req['prior_attempt_registries']==lin['prior_attempt_registries']==prior_registries(), 'Unsigned registration and full failure lineage required')
    for record in read(HERE/'prepared_v1/ROOT_REGISTER_REQUEST_DRAFT.json')['protected_files']:
        bound(record)
    decision=template()
    decision.update({'study_id':'graph_init_cfg0_outcome_aware_precision_v2',
        'contexts_sha256':object_hash(contexts),'finite_plan':plan,'finite_plan_sha256':object_hash(plan),
        'identities_are_draft_until_realized_registry':True})
    audit=PHASE/'graph_init_source_audit_v1/round17_v3_precision_recheck/REPORT.json'
    if audit.is_file(): decision['independent_R17_source_audit']=descriptor(audit)
    require(decision['approved'] is False and decision['scientific_phases_authorized'] is False and
        decision['compare_authorized'] is False and decision['report_authorized'] is False and decision['final_labels_authorized'] is False and
        decision['automatic_retry_authorized'] is False,'Source/template approval never admits execution')
    write(HERE/'ROOT_FINITE_SCIENCE_DECISION_TEMPLATE.json',decision)
    try: decision_guard(HERE/'ROOT_FINITE_SCIENCE_DECISION_TEMPLATE.json')
    except ValueError: pass
    else: raise ValueError('Unsigned finite allowance must be rejected')
    write(HERE/'FINITE_72_PLAN_DRAFT.json',{'schema':'graph-init-precision-finite-plan-draft-v1',
        'scientific_execution_authorized':False,'anchor_directory':draft['anchor_directory'],
        'identities_are_draft_until_forecasts_finalized_and_registry_created':True,
        'counts':counts,'plan_sha256':object_hash(plan),'attempts':plan,
        'whole_cap_budget_seconds':sum(r['whole_cap_seconds'] for r in plan),
        'caps_are_bounds_not_runtime_forecasts':True,'final_labels_compare_report_authorized':False})
    origins=read(HERE/'ASSEMBLY_ORIGINS.json');diffs=[]
    for row in origins['files']:
        old=PHASE/row['original_path'];new=HERE/row['target']
        require(sha(old)==row['original_sha256'],'Original wrapper changed')
        row['revised_sha256']=sha(new)
        diffs.extend(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),
                    fromfile=row['original_path'],tofile=str(new.relative_to(PHASE))))
    (HERE/'ASSEMBLY_ORIGINS.json').write_text(json.dumps(origins,indent=2)+'\n')
    (HERE/'REUSE_DIFF.patch').write_text(''.join(diffs))
    protocol=read(PHASE/R17_REL/'PROTOCOL.json')
    old_protocol=read(PHASE/'continuous_method_gap_search_v1/round17_graph_init_driver_integration_v2/PROTOCOL.json')
    require({k:v for k,v in protocol.items() if k!='qualification_measurement_amendment'}==old_protocol,
            'Only declared qualification measurement amendment may change protocol')
    require(protocol['arms']==list(ARMS) and protocol['seeds']==list(SEEDS) and protocol['registered_phase_attempts']==72 and
            protocol['qualification']['actual_warm_Adam_equivalence_required_every_arm'] is True and
            protocol['qualification']['actual_warm_AD_required_every_arm'] is True,'All original actual-warm gates remain required')
    write(HERE/'DEPLOYMENT_DEPENDENCIES.json',{'schema':'graph-init-precision-explicit-deployment-dependencies-v1',
        'sealed_method_and_modern_sources':source_descriptors(),'protected_original_failure_and_diagnostic_lineage_count':len(lineage['all_bound_text_files']),
        'prior_attempt_registries':prior_registries(),'bootstrap_uploads_explicit_manifest_payloads_and_protected_text_metadata':True,
        'whole_results_folder_upload_allowed':False,'array_model_checkpoint_transfer_or_inheritance_allowed':False,
        'local_and_remote_launch_parent_creation_required':True})
    result={'schema':'graph-init-precision-wrapper-static-check-v1','passed':True,'AST_sources':sources,
        'contexts':6,'fresh_attempt_counts':counts,'fresh_plan_sha256':object_hash(plan),'all72caps_equal_original_forecasts':True,
        'protected_original_failure_and_diagnostic_text_files':len(lineage['all_bound_text_files']),
        'prior_failed_registries':len(prior_registries()),'old_status':lineage['old_v2_status'],
        'unsigned_finite_decision_rejected':True,'all_registration_science_report_approvals_false':True,
        'protocol_identical_except_explicit_precision_measurement_amendment':True,
        'all_cold_and_every_actual_warm_requalification_required':True,'no_old_qualification_or_phase_output_inherited':True,
        'scope':'Source/AST/JSON/text-hash only; no scientific imports, arrays, models, checkpoints, runner, SSH or GPU executed.',
        'independent_wrapper_audit_required_before_execution':True}
    write(HERE/'STATIC_CHECKS.json',result)
    print(json.dumps({'passed':True,'AST_sources':len(sources),'fresh_attempt_counts':counts,'protected_lineage_files':len(lineage['all_bound_text_files'])}))

if __name__=='__main__': main()
