"""Guarded one-use metadata registration launcher; never accepts a science action."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback
from admission_support import HERE, PHASE, REMOTE_PHASE, require, mirror, read, bound, descriptor, write
from build_admissions import approved_decision, ANCHOR
from continuation_support import own_sources, verify_own_sources
from finite_coordinator import ensure_upload, prepare_launch_parents, sync_metadata, utc

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    require(PHASE!=REMOTE_PHASE, 'Root registration launcher is local')
    manifests=own_sources();verify_own_sources(*manifests)
    request=read(args.request)
    require(request['schema']=='graph-init-root-launch-request-v1' and request['root_admitted'] is True and
            request['action']=='register' and request['full_fit_admitted'] is False and request['heldout_scoring_admitted'] is False and
            request['whole_cap_seconds']==600 and request['output']==str(Path(ANCHOR)/'GRAPH_INIT_ATTEMPT_REGISTRY.json'),
            'Only separately admitted new metadata registration allowed')
    decision=approved_decision(bound(request['root_decision']),'graph-init-root-registration-decision-v1')
    require(decision['registration_authorized'] is True and decision['cold_qualification_authorized'] is False,
            'Source approval cannot authorize science')
    receipt=mirror(args.receipt)
    require(receipt==HERE/'registration_launch_v1.json' and not receipt.exists(), 'Fresh exact registration launcher identity required')
    records=request['protected_files']+[descriptor(args.request),request['root_decision'],request['phase_payload'],*manifests]
    payload=read(bound(request['phase_payload']))
    records += [payload['lineage_authorization'],decision['independent_source_audit'],decision['predecessor_history']]
    unique={r['path']:r for r in records}
    # Claim before deployment/parent preparation, so even pre-launch failures cannot be retried.
    write(receipt,{'UTC':utc(),'request':descriptor(args.request),
                   'automatic_retry':False,'source_metadata_registration_only':True})
    try:
        transfer=ensure_upload([unique[k] for k in sorted(unique)])
        parents=prepare_launch_parents(request)
        write(HERE/'registration_launch_preparation_v1.json',{'deployment':transfer,'launch_parents':parents})
        argv=[sys.executable,str(PHASE/'protocols/launch_modern_root_v1.py'),
              '--request',str(mirror(args.request).relative_to(PHASE)),
              '--outer',str(Path(request['outer_supervisor_directory']).relative_to(REMOTE_PHASE)),
              '--inner',str(Path(request['supervisor_directory']).relative_to(REMOTE_PHASE)),
              '--receipt',str(Path(request['local_launch_receipt']).relative_to(REMOTE_PHASE))]
        result=subprocess.run(argv,capture_output=True,text=True,check=False,timeout=780)
        write(HERE/'registration_launch_terminal_v1.json',{'UTC':utc(),'exit_code':result.returncode,'stdout':result.stdout,
            'stderr':result.stderr,'automatic_retry':False,'source_metadata_registration_only':True})
        if result.returncode==0:
            # Registration root/whole terminals are required for the one finite72 decision.
            sync_metadata(request['output'], REMOTE_PHASE/HERE.name)
        require(result.returncode==0,'Registration launch failed; preserved and never retried')
    except BaseException as error:
        write(HERE/'registration_launch_failed_v1.json',{'UTC':utc(),'error_type':type(error).__name__,
            'message':str(error),'traceback':traceback.format_exc(),'automatic_retry':False,
            'this_registration_launch_blocked':True})
        raise

if __name__=='__main__':
    main()
