"""Bind the independently reviewed original monolithic check; no numerical run."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

D=Path(__file__).resolve().parent
P=D.parent
REVIEW='matched_first_order_private_gradient_utility_native_v6_monolithic_supervisor_v4_independent_source_review_20261006_v1'
OLD='allocation_streamed_utility_native_joint_oracle_root_admission_20261006_v1'
QUAL='matched_first_order_private_gradient_utility_native_qualification_preparation_20261006_v6'
SUP='matched_first_order_private_gradient_utility_native_supervisor_preparation_20261006_v4'

def desc(relative):
    b=(P/relative).read_bytes()
    return {'path':relative,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def save(name,value):
    f=D/name
    assert not f.exists()
    f.write_text(json.dumps(value,indent=2)+'\n');f.chmod(0o444)

def main():
    review=json.loads((P/REVIEW/'FINDINGS.json').read_text())
    assert review['status'].startswith('PASS_SOURCE') and review['new_source_blockers']==[]
    assert review['qualifier']==desc(QUAL+'/qualify.py')
    assert review['supervisor_sha256']==desc(SUP+'/supervise.py')['sha256']
    assert desc(REVIEW+'/FINDINGS.json')['sha256']=='79a1d084d89e66a1ae180c8d071adfcdc714936322b90612acb8f7760206b726'
    ready=json.loads((D/'READINESS.json').read_text())
    assert ready['hostname']=='anogena-2-0' and ready['GPU']['free_MiB']*1048576>=70*1073741824
    now=datetime.now(timezone.utc).isoformat()
    caps={'max_elapsed_seconds':1800,'max_process_rss_bytes':8*1073741824,'max_cuda_allocated_bytes':64*1073741824,'max_cuda_reserved_bytes':68*1073741824}
    resource={'UTC':now,'status':'ROOT_PROSPECTIVE_ORIGINAL_MONOLITHIC_NATIVE_RESOURCE_ADMISSION','new_caps':caps,'external_watchdog_seconds':1920,'minimum_fresh_free_bytes':70*1073741824,'readiness':desc(str((D/'READINESS.json').relative_to(P))),'basis':'Original monolithic77 check failed for memory on an occupied40GB card. Allocation has78,708MiB free; preceding qualification owners have terminal receipts. Keep full graph, native FP32 and unchanged coordinate tolerances; allow64/68GiB CUDA and whole-child1800s/8GiB RSS. No predictive outcome, graph reduction, retry or other-process interference.','previous_streamed_failure_preserved':True,'no_fit_or_scoring':True}
    save('RESOURCE_REVIEW.json',resource)
    old=json.loads((P/OLD/'WORKER_SCOPE.json').read_text())
    worker=json.loads((P/QUAL/'ROOT_SCOPE_TEMPLATE.json').read_text())
    for key in ('CUDA_VISIBLE_DEVICES','GPU_UUID','device','hostname','repository','python_executable','site_packages','runtime_identity_expected','strict_backend_expected','current_six_arm_terminal_evidence'):
        worker[key]=old[key]
    worker.update({'UTC':now,'fixed_before_execution':True,'root_engineering_invocation_authorized':True,'status':'ROOT_ADMITTED_ONE_ORIGINAL_MONOLITHIC_NATIVE_CHECK_NO_FIT','resource_limits':caps,'external_watchdog_seconds':1920,'qualifier_review':desc(REVIEW+'/FINDINGS.json'),'explicit_resource_review':desc(str((D/'RESOURCE_REVIEW.json').relative_to(P))),'fresh_available_allocation_evidence':desc(str((D/'READINESS.json').relative_to(P)))})
    save('WORKER_SCOPE.json',worker)
    scope=json.loads((P/SUP/'ROOT_SCOPE_TEMPLATE.json').read_text())
    output='allocation_monolithic_utility_native_execution_root_20261006_v1/owned_run01'
    scope.update({'UTC':now,'fixed_before_execution':True,'root_engineering_supervision_authorized':True,'status':'ROOT_ADMITTED_ONE_ORIGINAL_MONOLITHIC_NATIVE_SUPERVISION_NO_FIT','resource_limits':caps,'external_watchdog_seconds':1920,'terminate_grace_seconds':10,'kill_reap_seconds':30,'worker_scope':desc(str((D/'WORKER_SCOPE.json').relative_to(P))),'child_python_sha256':ready['python_binary_sha256'],'qualifier_review':desc(REVIEW+'/FINDINGS.json'),'supervisor_review':desc(REVIEW+'/FINDINGS.json'),'output_relative_path':output,'explicit_resource_review':worker['explicit_resource_review'],'fresh_available_allocation_evidence':worker['fresh_available_allocation_evidence'],'current_six_arm_terminal_evidence':worker['current_six_arm_terminal_evidence'],'runtime_caps_watchdog_and_cleanup_durations_unselected':False})
    phase=Path(worker['repository'])/'experiments_iclr/postsubmission_20260930'
    scope['child_argv']=[worker['python_executable'],'-B',str(phase/QUAL/'qualify.py'),'--execute-authorized','--source-root',str(phase),'--execution-scope',str(phase/scope['worker_scope']['path']),'--execution-scope-sha256',scope['worker_scope']['sha256'],'--output',str(phase/output/'child')]
    save('SUPERVISION_SCOPE.json',scope)
    note='''# Root admission of one original-monolithic numerical check

Independent V6/V4 source review passed. Original utility83967 and actual CPU96ce are exact prerequisites. The scalar reference, joint self-check, fixed cotangents, negative controls, native full context and FP32 thresholds remain unchanged. All streamed candidate failures stay FAIL. This invocation tests the original monolithic implementation on the authorized80GB allocation with prospectively fixed64/68GiB CUDA,1800s whole-child wall and1920s watchdog. No fit or held scoring is authorized. Expected54F/77native/88API; actual terminal counts and restoration must close before any H16 admission. Fresh free-memory floor70GiB is enforced again at dispatch. No77cofit or automatic retry.
'''
    f=D/'ROOT_REVIEW.md';assert not f.exists();f.write_text(note);f.chmod(0o444)
    source=(P/OLD/'run_once.py').read_text()
    for before,after in (
        (OLD,D.name),
        ('allocation_streamed_utility_native_joint_oracle_launch_root_20261006_v1','allocation_monolithic_utility_native_launch_root_20261006_v1'),
        ('matched_first_order_private_gradient_utility_native_qualification_preparation_20261006_v5',QUAL),
        ('matched_first_order_private_gradient_utility_native_supervisor_preparation_20261006_v3',SUP),
        ('matched_first_order_private_gradient_utility_native_v5_joint_oracle_supervisor_v3_independent_source_review_20261006_v1',REVIEW),
        ('allocation_streamed_utility_native_readiness_root_20261006_v1',D.name),
        ('matched_first_order_private_gradient_utility_control_source_preparation_20261006_v3','matched_first_order_private_gradient_utility_control_source_preparation_20261006_v1'),
        ('matched_first_order_streamed_utility_and_synthetic_root_independent_source_review_20261006_v1/FINDINGS.json','matched_first_order_private_gradient_utility_control_independent_source_review_20261006_v1/SEAL.json')):
        assert before in source
        source=source.replace(before,after)
    source=source.replace("rows=[]\n for rel in stage:","stage += ['matched_first_order_private_gradient_utility_synthetic_execution_root_20261006_v1/RESULT.json','matched_first_order_private_gradient_utility_control_independent_source_review_20261006_v1/SEAL.json']\n rows=[]\n for rel in stage:")
    f=D/'run_once.py';assert not f.exists();f.write_text(source)
    source=(P/OLD/'monitor.py').read_text()
    source=source.replace('allocation_streamed_utility_native_joint_oracle_execution_root_20261006_v1','allocation_monolithic_utility_native_execution_root_20261006_v1').replace('allocation_streamed_utility_native_joint_oracle_launch_root_20261006_v1','allocation_monolithic_utility_native_launch_root_20261006_v1').replace('allocation_streamed_utility_native_joint_oracle_terminal_handoff_root_20261006_v1','allocation_monolithic_utility_native_terminal_handoff_root_20261006_v1')
    f=D/'monitor.py';assert not f.exists();f.write_text(source)
    print(json.dumps({'review':desc(REVIEW+'/FINDINGS.json'),'worker_scope':scope['worker_scope'],'supervision_scope':desc(str((D/'SUPERVISION_SCOPE.json').relative_to(P))),'launches':0,'fits':0,'held_scoring':False}))

if __name__=='__main__':main()
