"""Freeze exact matched H16 scopes after independent fit-supervisor review."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json

D=Path(__file__).resolve().parent; P=D.parent
CALLER='matched_first_order_private_gradient_utility_common400_H16_preparation_20261006_v4/run_utility_H16.py'
SUPERVISOR='matched_first_order_private_gradient_utility_common400_H16_owned_supervisor_preparation_20261006_v1/supervise_H16.py'
CALLER_REVIEW='matched_first_order_private_gradient_utility_common400_H16_v4_monolithic_independent_source_review_20261006_v1/FINDINGS.json'
TEMPLATES='matched_first_order_private_gradient_utility_common400_H16_v4_template_correction_root_20261006_v1'
NATIVE='allocation_monolithic_utility_native_root_admission_20261006_v1'
NATIVE_EXEC='allocation_monolithic_utility_native_execution_root_20261006_v1/owned_run01'
NATIVE_HANDOFF='allocation_monolithic_utility_native_terminal_handoff_root_20261006_v1'
EXEC='allocation_common400_monolithic_utility_H16_execution_root_20261006_v1'

def desc(rel,local_alias=None):
    b=(P/(local_alias or rel)).read_bytes()
    return {'path':rel,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def write(name,data):
    f=D/name;assert not f.exists()
    f.write_text(json.dumps(data,indent=2)+'\n');f.chmod(0o444)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--supervisor-review',required=True);args=parser.parse_args()
    rel=Path(args.supervisor_review);assert not rel.is_absolute() and '..' not in rel.parts
    review=json.loads((P/rel).read_text());caller_review=json.loads((P/CALLER_REVIEW).read_text())
    assert review['status'].startswith('PASS_SOURCE') and review['execution_or_fit_authorized'] is False
    assert review['supervisor_sha256']==desc(SUPERVISOR)['sha256']
    assert caller_review['status'].startswith('PASS_SOURCE') and caller_review['caller_sha256']==desc(CALLER)['sha256']
    native_worker=json.loads((P/NATIVE/'WORKER_SCOPE.json').read_text())
    native_sup=json.loads((P/NATIVE/'SUPERVISION_SCOPE.json').read_text())
    qualified=json.loads((P/NATIVE_HANDOFF/'WORKER_RESULT.json').read_text())
    terminal=json.loads((P/NATIVE_HANDOFF/'TERMINAL.json').read_text())
    assert terminal['wrapper_resource_closure_PASS'] and terminal['owned_child']['exit_code']==0 and qualified['restoration_errors']==[]
    resource=json.loads((D/'RESOURCE_REVIEW.json').read_text())
    runtime=json.loads((D/'RUNTIME_COMPARISON.json').read_text());assert runtime['disposition']=='SAME_LIVE_RUNTIME_VERIFIED'
    now=datetime.now(timezone.utc).isoformat();invocation='allocation_exact_common400_original_monolithic_first_order_H16_20261006_v1'
    supervision=json.loads((P/TEMPLATES/'FIT_SUPERVISION_ADMISSION_TEMPLATE.json').read_text())
    supervision.update({'UTC':now,'status':'ROOT_ADMITTED_ONCE_ONLY_MONOLITHIC_UTILITY_H16_FIT_SUPERVISION','root_supervision_fit_authorized':True,'fixed_before_launch':True,'closed_A_VALID_TEST':True,'fit_invocation_id':invocation,'resource_limits':resource['resource_limits'],'external_watchdog_seconds':resource['external_watchdog_seconds'],'reviewed_supervisor_sha256':desc(SUPERVISOR)['sha256'],'supervisor_source':desc(SUPERVISOR),'source_review':desc(str(rel)),'supervisor_output_relative_path':EXEC+'/supervisor','child_cwd':native_worker['repository'],'terminate_grace_seconds':10,'kill_reap_seconds':30,'minimum_initial_cuda_free_bytes':resource['minimum_fresh_free_bytes']})
    write('FIT_SUPERVISION_ADMISSION.json',supervision)
    fit=json.loads((P/TEMPLATES/'ROOT_FIT_SCOPE_TEMPLATE.json').read_text())
    for key in ('hostname','repository','GPU_UUID','python_executable','site_packages','public_b_dir'):
        fit[key]=native_worker[key]
    fit.update({'UTC':now,'status':'ROOT_ADMITTED_ONE_EXACT_COMMON400_MONOLITHIC_UTILITY_H16_NO_HELD_SCORING','root_fit_authorized':True,'fixed_before_fit':True,'caller_sha256':desc(CALLER)['sha256'],'caller_review':desc(CALLER_REVIEW),'fit_invocation_id':invocation,'native_utility_terminal':desc(NATIVE_EXEC+'/TERMINAL.json',NATIVE_HANDOFF+'/TERMINAL.json'),'native_utility_worker_result':desc(NATIVE_EXEC+'/child/RESULT.json',NATIVE_HANDOFF+'/WORKER_RESULT.json'),'native_utility_worker_scope':desc(NATIVE+'/WORKER_SCOPE.json'),'native_utility_supervision_scope':desc(NATIVE+'/SUPERVISION_SCOPE.json'),'runtime_identity_expected':qualified['runtime_identity'],'strict_backend_expected':qualified['backend_during_native'],'python_executable_sha256':native_sup['child_python_sha256'],'comparison_runtime_disposition':runtime['disposition'],'fit_resource_limits':resource['resource_limits'],'external_fit_supervision_admission':desc(str((D/'FIT_SUPERVISION_ADMISSION.json').relative_to(P))),'output_relative_path':EXEC+'/fit','resource_review':desc(str((D/'RESOURCE_REVIEW.json').relative_to(P))),'runtime_comparison':desc(str((D/'RUNTIME_COMPARISON.json').relative_to(P)))})
    write('FIT_SCOPE.json',fit)
    write('ROOT_ADOPTION.json',{'UTC':now,'source_reviews':[desc(CALLER_REVIEW),desc(str(rel))],'caller':desc(CALLER),'supervisor':desc(SUPERVISOR),'fit_scope':desc(str((D/'FIT_SCOPE.json').relative_to(P))),'no_new_W_acquisition':True,'original_six_conditions_unchanged':True,'H':16,'expected_work':{'native_forwards':648,'native_reverse_constructions':832,'ordinary_grad_APIs':1008},'retry':False,'held_scoring':False,'fresh_native_reference_cohort_starts_after_owned_H16_closure':True})
    print(json.dumps({'fit_scope':desc(str((D/'FIT_SCOPE.json').relative_to(P))),'fit_launches':0,'held_scoring':False}))

if __name__=='__main__':main()
