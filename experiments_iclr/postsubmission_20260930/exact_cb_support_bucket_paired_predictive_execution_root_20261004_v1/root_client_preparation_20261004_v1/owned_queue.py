"""One detached fixed-order queue of nine original supervised fresh fits."""
from pathlib import Path
import argparse
import json
import os
import subprocess
from time import monotonic
from remote_control import (HERE,ROOT,REPO,PYTHON,SOURCE,SOURCE_SHA,GPU,bound_queue,
                            require,sha,utc,write_new,atomic,descriptor,physical)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-release-sha256',required=True)
    args=parser.parse_args()
    q,plan=bound_queue(args.queue_release_sha256)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
            and os.environ.get('GNNM_SSH_DESTINATION')=='shmelev@192.168.18.77','Fixed GPU1 native route required')
    output=ROOT/'queue/run01'
    output.mkdir(parents=True,exist_ok=False)
    write_new(ROOT/'QUEUE_ATTEMPT_SPENT.json',{'UTC':utc(),'queue_release_sha256':args.queue_release_sha256,
              'queue_order_sha256':q['queue_order_sha256'],'automatic_retry_or_resume':False})
    self_identity=physical(os.getpid())
    started=monotonic(); records=[]; attempted=[]; failure=None; active=None
    write_new(output/'STARTED.json',{'UTC':utc(),'physical_identity':self_identity,
              'source_manifest_sha256':SOURCE_SHA,'client_manifest_sha256':q['client_manifest_sha256'],
              'queue_release_sha256':args.queue_release_sha256,'queue_order_sha256':q['queue_order_sha256'],
              'queue_order':q['queue_order'],'GPU_UUID':GPU,'automatic_retry_or_resume':False,
              'private_selection_or_logits_read':False})
    try:
        for number,(cell,release_pin) in enumerate(zip(q['queue_order'],q['cell_releases']),1):
            bound_queue(args.queue_release_sha256)
            inv=cell['invocation']; name=cell['cell']
            require(not Path(inv['output_directory']).exists()
                    and not Path(cell['supervision_output_directory']).exists(),'Fresh cell outputs required')
            write_new(ROOT/'spent'/(name+'.json'),{'UTC':utc(),'queue_release_sha256':args.queue_release_sha256,
                      'queue_order':number,'cell':name,'release':release_pin,'automatic_retry_or_resume':False})
            attempted.append(name)
            command=[str(PYTHON),'-B',str(SOURCE/'supervise_fit.py'),'--root-release',release_pin['path'],
                     '--arm',inv['unit'],'--seed',str(inv['base_seed']),
                     '--output',cell['supervision_output_directory']]
            began=monotonic()
            atomic(output/'STATUS.json',{'UTC':utc(),'status':'RUNNING','active_cell':name,
                   'queue_order':number,'completed_cells':len(records),'elapsed_seconds':monotonic()-started})
            with (output/(name+'.stdout.txt')).open('xb') as out, (output/(name+'.stderr.txt')).open('xb') as err:
                active=subprocess.Popen(command,cwd=REPO,env=os.environ.copy(),stdin=subprocess.DEVNULL,
                                        stdout=out,stderr=err,start_new_session=True)
                identity=physical(active.pid)
                require(identity is not None and identity['argv']==command and identity['cwd']==str(REPO)
                        and identity['exe']==str(PYTHON.resolve()) and identity['session']==active.pid
                        and identity['process_group']==active.pid,'Owned fit supervisor handle differs')
                write_new(output/(name+'_STARTED.json'),{'UTC':utc(),'physical_identity':identity,
                          'command':command,'cell':name,'queue_order':number,'root_release':release_pin})
                code=active.wait(); active=None
            terminal_path=Path(cell['supervision_output_directory'])/'SUPERVISOR_TERMINAL.json'
            terminal=json.loads(terminal_path.read_text()) if terminal_path.exists() else None
            ok=bool(code==0 and terminal and terminal.get('status')=='COMPLETE'
                    and terminal.get('child_exit_code')==0 and terminal.get('cap_violation') is None
                    and terminal.get('root_release_sha256')==release_pin['sha256']
                    and terminal.get('arm')==inv['unit'] and terminal.get('seed')==inv['base_seed'])
            if ok:
                result_path=Path(inv['output_directory'])/'COMPLETE.json'
                ok=bool(result_path.is_file() and not result_path.is_symlink()
                        and sha(result_path)==terminal['fit_receipt']['sha256'])
            row={'UTC':utc(),'queue_order':number,'cell':name,'status':'COMPLETE' if ok else 'FAILED',
                 'supervisor_physical_identity':identity,'supervisor_exit_code':code,'wall_seconds':monotonic()-began,
                 'supervisor_terminal':descriptor(terminal_path) if terminal_path.is_file() else None,
                 'fit_receipt':terminal.get('fit_receipt') if terminal else None,
                 'private_selection_or_logits_read':False}
            records.append(row); write_new(output/(name+'_TERMINAL.json'),row)
            if not ok:
                failure={'condition':'owned_fit_did_not_complete','cell':name,'supervisor_exit_code':code}
                break
    except BaseException as error:
        failure={'type':type(error).__name__,'condition':str(error)}
        if active is not None:
            # The original supervisor continues to own its bounded scientific
            # child. Reap that supervisor before closing this failed queue.
            failure['active_supervisor_PID']=active.pid
            failure['active_supervisor_exit_code']=active.wait()
            active=None
    finally:
        complete=not failure and len(records)==9 and all(x['status']=='COMPLETE' for x in records)
        terminal={'UTC':utc(),'schema':'ncnc-exact-CB-owned-nine-fit-queue-terminal-v1',
                  'status':'COMPLETE' if complete else 'FAILED','physical_identity':self_identity,
                  'source_manifest_sha256':SOURCE_SHA,'client_manifest_sha256':q['client_manifest_sha256'],
                  'queue_release_sha256':args.queue_release_sha256,'queue_order_sha256':q['queue_order_sha256'],
                  'records':records,'attempted_cells':attempted,'failure':failure,'elapsed_seconds':monotonic()-started,
                  'unattempted_cells':q['queue_order'][len(attempted):],
                  'automatic_retry_or_resume':False,'other_jobs_mutated':False,'signals_sent_by_queue':False,
                  'private_selection_or_logits_read':False,'TEST_access':False,
                  'family_analysis_automatic':False}
        write_new(output/'TERMINAL.json',terminal)
        atomic(output/'STATUS.json',{'UTC':utc(),'status':terminal['status'],'completed_cells':len(records),
                                   'elapsed_seconds':terminal['elapsed_seconds']})
    return 0 if complete else 1


if __name__=='__main__':
    raise SystemExit(main())
