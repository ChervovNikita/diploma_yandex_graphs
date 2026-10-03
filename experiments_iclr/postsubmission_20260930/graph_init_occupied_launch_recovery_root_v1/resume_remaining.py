"""Root-admitted administrative recovery after a verified pre-child refusal.

All scientific entrypoints, registered keys, states, caps and selectors remain
unchanged. No successful or scientifically attempted key is restarted.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
PACKET=Path(__file__).resolve().parent
PHASE=PACKET.parent
sys.path.insert(0,str(PHASE/'graph_init_precision_continuation_v3_cap_binding'))
import finite_coordinator as fc
from continuation_support import decision_guard,scan_state,mirror,remote,write,descriptor,read,completed_phase,require
from build_continuation import make_admission,next_row


def main():
    decision_path=PACKET/'ROOT_RECOVERY_DECISION.json'
    decision,registry,plan=decision_guard(decision_path)
    recovery=decision['administrative_recovery']
    require(recovery['approved'] and recovery['pre_child_refusal_only'] and recovery['scientific_retry_authorized'] is False,'Root recovery scope')
    proof=read(PACKET/'PRELAUNCH_PROOF.json')['proof']
    require(proof['failed_key']==recovery['failed_key'] and not proof['GPU_compute_processes'],'Verified idle pre-child refusal required')
    require(not any(proof[k] for k in ('claim_exists','phase_terminal_exists','science_output_exists','inner_supervisor_exists','root_terminal_exists')),'No previous scientific attempt may be recovered')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==recovery['coordinator_sha256'],'Recovery coordinator binding')
    completed=scan_state(decision['attempt_registry'],registry)
    prefix=[r['key'] for r in plan[:len(completed)]]
    require(set(prefix)==set(completed) and len(prefix)==recovery['completed_prefix_count']==64,'Exact complete prefix required')
    remaining=plan[len(prefix):]
    require(len(remaining)==8 and remaining[0]['key']==proof['failed_key'],'Exact untouched eight fits required')
    require(all(r['phase']=='fit' for r in remaining),'Fit-only recovery')
    out=mirror(decision['coordinator_run_root']);out.mkdir(exist_ok=False)
    (out/'requests').mkdir();(out/'supervision').mkdir()
    started=time.monotonic()
    write(out/'START.json',dict(UTC=datetime.now(timezone.utc).isoformat(),decision=descriptor(decision_path),completed_prefix=prefix,remaining_keys=[r['key'] for r in remaining],scientific_source_changed=False,scientific_retry=False,final_labels_authorized=False))
    try:
        fc.ensure_upload([descriptor(decision_path),descriptor(Path(__file__)),descriptor(PACKET/'PRELAUNCH_PROOF.json')])
        fc.sync_metadata(decision['attempt_registry']['path'],decision['coordinator_run_root'])
        require(set(scan_state(decision['attempt_registry'],registry))==set(prefix),'Fresh remote canonical prefix differs')
        for planned in remaining:
            decision,registry,current=decision_guard(decision_path)
            require(next_row(decision,registry,current)==planned,'Unchanged next registered attempt required')
            key=planned['key'];request_dir=out/'requests'/key
            request=make_admission(decision_path,key,request_dir)
            write(out/(key+'_UPLOAD.json'),fc.ensure_upload([descriptor(request_dir/'ADMISSION.json'),descriptor(request_dir/'REQUEST.json')]))
            write(out/(key+'_LAUNCH_CLAIM.json'),dict(UTC=datetime.now(timezone.utc).isoformat(),attempt=planned,request=descriptor(request_dir/'REQUEST.json'),automatic_retry=False))
            write(out/(key+'_LAUNCH_PARENTS.json'),fc.prepare_launch_parents(request))
            argv=[sys.executable,str(PHASE/'protocols/launch_modern_root_v1.py'),'--request',str(request_dir.relative_to(PHASE)/'REQUEST.json'),'--outer',str(Path(request['outer_supervisor_directory']).relative_to(fc.REMOTE_PHASE)),'--inner',str(Path(request['supervisor_directory']).relative_to(fc.REMOTE_PHASE)),'--receipt',str(Path(request['local_launch_receipt']).relative_to(fc.REMOTE_PHASE))]
            run=subprocess.run(argv,capture_output=True,text=True,timeout=planned['whole_cap_seconds']+180)
            write(out/(key+'_LAUNCH_RESULT.json'),dict(UTC=datetime.now(timezone.utc).isoformat(),attempt=planned,exit_code=run.returncode,stdout=run.stdout,stderr=run.stderr,automatic_retry=False))
            imported=fc.sync_metadata(decision['attempt_registry']['path'],decision['coordinator_run_root'])
            write(out/(key+'_METADATA_FETCH.json'),dict(UTC=datetime.now(timezone.utc).isoformat(),files=imported))
            require(run.returncode==0,'Stop on any new failed launch or scientific phase')
            outer=read(mirror(Path(request['outer_supervisor_directory'])/'TERMINAL.json'))
            root=read(mirror(Path(request['receipt_directory'])/'TERMINAL.json'))
            require(outer['complete'] and outer['within_whole_cap'] and outer['root_request_unchanged'] and root['completed'],'Full supervision completion required')
            canonical=next(r for r in registry['attempts'] if r['key']==key)
            _,terminal,_=completed_phase(decision['attempt_registry'],registry,canonical)
            write(out/(key+'_COMPLETED.json'),dict(UTC=datetime.now(timezone.utc).isoformat(),phase_terminal=terminal,whole_terminal=descriptor(mirror(Path(request['outer_supervisor_directory'])/'TERMINAL.json')),whole_supervised_seconds=outer['whole_supervised_seconds'],whole_cap_seconds=planned['whole_cap_seconds']))
        require(len(scan_state(decision['attempt_registry'],registry))==72 and next_row(decision,registry,plan) is None,'All72 source phases must close')
        write(out/'COMPLETED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=True,registered_successful_terminals=72,seconds=time.monotonic()-started,compare_or_final_labels_executed=False))
    except BaseException as error:
        write(out/'FAILED.json',dict(UTC=datetime.now(timezone.utc).isoformat(),completed=False,error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    main()
