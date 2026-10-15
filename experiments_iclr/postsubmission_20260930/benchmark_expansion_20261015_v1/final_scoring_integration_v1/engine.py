"""Minimal TEST terminal successor of frozen portablev3 Engine.

Launch, run, lease, STOP, reservation, drain and failure-loop control are
retained. Changes are enumerated in SOURCE_DELTA.md. No model imports.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def require(ok,message):
    if not ok:raise ValueError(message)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def pin(path):
    path=Path(path);raw=path.read_bytes();return {'path':str(path.resolve()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def verify(value):
    actual=pin(value['path']);require(all(actual[k]==value[k]for k in('bytes','sha256')if k in value),'Receipt changed: '+value['path'])
def read(path):return json.loads(Path(path).read_text())
def exclusive(path,value):
    with Path(path).open('x')as out:json.dump(value,out,indent=2,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
def atomic(path,value):
    temporary=Path(str(path)+'.tmp-'+str(os.getpid())+'-'+str(time.time_ns()));exclusive(temporary,value);temporary.replace(path)


class Engine:
    def __init__(self,tasks,output_root,manifest_sha256,worker,backend,lease_paths,*,resumes=None,continuation=None,reconciliations=None,allowed_root=None,poll_seconds=.2):
        self.tasks=tasks;self.root=Path(output_root).resolve();self.allowed_root=Path(allowed_root or output_root).resolve();require(self.root.is_relative_to(self.allowed_root),'Output root escapes approved repository');self.manifest=manifest_sha256;self.worker=worker;self.backend=backend
        self.leases=lease_paths;self.resumes=resumes or {};self.continuation=continuation;self.reconciliations=reconciliations or {};self.poll_seconds=poll_seconds
        self.index={t['id']:t for t in tasks};require(len(self.index)==len(tasks),'Duplicate physical task IDs')
        self.control=self.root/'control';self.task_control=self.control/'tasks';self.stops=self.backend.ledger.root/'stops'
        self.active={};self.states={};self.records=[];self.max_active=0;self.started_children=0;self.stopped=False
        self.extra_authorities=[];self.lease_handles=[];self.owns_leases=False
        for t in tasks:
            require(t['worker']==worker and len(t['dependencies'])in(0,4),'Foreign worker or invalid dependency count')
            require(set(t['dependencies']).issubset(self.index),'Dependencies must remain on physical worker')
            p=self.output(t);require(p.is_relative_to(self.root)and p!=self.root,'Output escapes reviewed release root')
        require(len({str(self.output(t))for t in tasks})==len(tasks),'Duplicate task output paths')

    def output(self,task):return (self.root/task['output_relative']).resolve()
    def folder(self,task):
        folder=self.task_control/digest(task['id']);require(folder.resolve().is_relative_to(self.root),'Task control folder escapes output root');return folder
    def verify(self,value):
        require(Path(value['path']).resolve().is_relative_to(self.allowed_root),'Receipt escapes approved repository')
        verify(value)
    def save_state(self,task,value):
        self.states[task['id']]=value;atomic(self.folder(task)/'STATE.json',value)
    def stop(self,reason):
        self.stopped=True
        self.backend.ledger.stop({'manifest_sha256':self.manifest,'worker':self.worker,'reason':reason})
    def latest_stops(self):return {str(p)for p in self.stops.glob('*.json')}

    def establish(self):
        require(self.task_control.resolve().is_relative_to(self.root)and self.stops.resolve().is_relative_to(self.allowed_root),'Control paths escape output root')
        self.task_control.mkdir(parents=True,exist_ok=True);self.stops.mkdir(parents=True,exist_ok=True)
        for path in self.leases:
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            self.lease_handles.append(handle)
        self.owns_leases=True
        previous=self.latest_stops()
        if previous:
            require(self.continuation,'Existing STOP requires a reviewed continuation; it is never cleared')
            self.verify(self.continuation);value=read(self.continuation['path'])
            require(value['schema']=='expanded-reviewed-continuation-v1'and value['reviewed_continue']is True and value['manifest_sha256']==self.manifest,'Continuation must bind unchanged release')
            require(set(x['path']for x in value['STOP_receipts'])==previous,'Continuation must account for every retained STOP')
            for value_pin in value['STOP_receipts']:self.verify(value_pin)
            self.verify(value['human_review_evidence']);self.extra_authorities+=[self.continuation,value['human_review_evidence']]+value['STOP_receipts']
        self.baseline_stops=previous
        self.backend.baseline_stops=previous
        self.backend.ledger.establish()  # after actual GPU/worker leases; same host lock phase handoff

    def immutable_claim(self,task):
        return {'schema':'expanded-permanent-task-claim-v1','id':task['id'],'runtime_job_hash':task.get('runtime_job_hash'),
                'science_contract_hash':task['science_contract_hash'],'manifest_sha256':self.manifest,'worker':self.worker,'output':str(self.output(task))}

    def reconcile(self,task):
        folder=self.folder(task);claim=folder/'CLAIM.json';output=self.output(task)
        if not claim.exists():
            require(not output.exists(),'Unclaimed existing output; never repeat or overwrite: '+task['id'])
            self.states[task['id']]={'status':'pending'};return
        require(read(claim)==self.immutable_claim(task),'Permanent claim belongs to changed job/release/worker')
        state_path=folder/'STATE.json'
        require(state_path.exists(),'Claim without state requires reviewed reconciliation')
        state=read(state_path)
        if state['status']in('complete','no_selected_state','blocked_dependency_no_selected_state','blocked_dependency_failed'):
            self.verify(state['attempt_terminal'])
            if not state['status'].startswith('blocked_dependency_'):
                self.verify(state['native_result']);self.verify(state['live_native_result_receipt']);raw=read(state['native_result']['path'])
                self.validate_terminal(task,raw)
                if state['status']=='complete':self.verify(state['TEST_export'])
            self.states[task['id']]=state;self.records.append({'id':task['id'],'action':'skip_already_'+state['status'],'extra_fits':0});return
        if state['status'] in ('failed','launch_failed') and task['id'] not in self.resumes:
            self.verify(state['attempt_terminal'])
            self.states[task['id']]=state
            self.records.append({'id':task['id'],'action':'retain_failure_without_retry','extra_fits':0})
            return
        if state['status']=='running'and self.backend.alive(state['child_identity']):
            raise ValueError('Live/orphan child identity remains; no duplicate admission: '+task['id'])
        require(state['status']in('running','failed','interrupted','launch_failed','claimed'),'Unknown durable task state')
        if state['status']=='running' and self.recover_terminal(task,state):return
        if state['status']=='running':
            state=dict(state,status='interrupted',partial_work_after_last_checkpoint='unknown;prior attempt retained',exit_code='unknown_after_supervisor_loss')
            self.save_state(task,state)
        self.resume_for(task,state)  # Every interrupted/failed attempt needs explicit review before restarting queue.
        self.states[task['id']]=state

    def recover_terminal(self,task,state):
        directory=self.folder(task)/'attempts'/f"{state['attempt']:04d}"
        terminal=directory/'TERMINAL.json'
        if not terminal.exists():return False
        record=read(terminal)
        require(record['id']==task['id'] and record['attempt']==state['attempt'],'Wrong reaped terminal')
        require(record['status']in('complete','no_selected_state','failed'),'Unknown scoring terminal')
        if record.get('native_result'):
            self.verify(record['native_result']);self.verify(record['live_native_result_receipt'])
            raw=read(record['native_result']['path']);self.validate_terminal(task,raw)
        if record.get('TEST_export'):self.verify(record['TEST_export'])
        recovered={k:record.get(k)for k in('status','attempt','native_result','TEST_export','live_native_result_receipt','TEST_accessed')}
        recovered['attempt_terminal']=pin(terminal)
        self.save_state(task,recovered)
        self.records.append({'id':task['id'],'action':'recover_exact_own_durably_reaped_terminal','extra_scoring':0})
        return True

    def resume_for(self,task,state):
        raise ValueError('Final TEST attempts are never automatically repeated; preserve interrupted/claimed/failed costs and review disposition: '+task['id'])

    def validate_terminal(self,task,raw):
        expected='final-selected-fit-scoring-result-v2'if task['kind']=='selected_fit_TEST'else'final-I4-TEST-scoring-result-v2'
        require(raw.get('schema')==expected and raw.get('task_id')==task['id'] and raw.get('role')=='TEST','Final terminal schema/task/role mismatch')
        require(raw.get('physical_training_fits')==0 and raw.get('optimizer_updates')==0 and raw.get('automatic_retries')==0,'Final scoring cannot train/retry')
        allowed=('complete','failed','missing_selected_state')if task['kind']=='selected_fit_TEST'else('complete','failed','missing_component','missing_training_I4')
        require(raw.get('status')in allowed and type(raw.get('TEST_accessed'))is bool,'Unknown final status/access state')
        if task['kind']=='selected_fit_TEST':
            require(raw.get('training_fit_id')==task['training_fit_id']and raw.get('job_hash')==task['job_hash'],'Foreign selected fit')
        if raw['status']=='complete':require(raw['TEST_accessed']is True,'Complete TEST score must declare role access')
        self.backend.validate_result(task,raw)

    def collect(self,task,code,attempt,elapsed):
        output=self.output(task);native_path=output/'RESULT.json';raw=None
        error=None;status='failed';selected=None;control_error=None
        try:
            raw=read(native_path)if native_path.is_file()else None
            if raw is not None:self.validate_terminal(task,raw)
            elif code==0:raise ValueError('Successful scoring child has no final terminal authority')
        except Exception as exc:
            control_error=type(exc).__name__+': '+str(exc);raw=None
        try:
            require(raw is not None,'Missing final terminal');require(code==0 and raw['status']!='failed','Child or final scoring failed')
            status='complete'if raw['status']=='complete'else'no_selected_state'
            if status=='complete':
                selected=raw['TEST_export'];self.verify(selected)
                expected='selected_TEST.pt'if task['kind']=='selected_fit_TEST'else'I4_TEST.pt'
                require(Path(selected['path']).resolve()==output/expected,'TEST export escapes its own scoring task')
        except Exception as exc:
            status='failed';selected=None;error=type(exc).__name__+': '+str(exc)
            if code==0 and raw is not None and raw.get('status')=='complete':control_error=error
        archive=self.folder(task)/'attempts'/f'{attempt:04d}'/'NATIVE_RESULT.json'
        if native_path.is_file():
            with archive.open('xb')as out:out.write(native_path.read_bytes())
        record={'schema':'portable-final-attempt-terminal-v1','id':task['id'],'attempt':attempt,'status':status,'exit_code':code,
            'observed_launch_to_reap_seconds':elapsed,'native_claimed_status':None if raw is None else raw.get('status'),
            'native_result':pin(archive)if archive.is_file()else None,'live_native_result_receipt':pin(native_path)if native_path.is_file()else None,
            'native_invocation_wall_seconds':None if raw is None else raw.get('worker_wall_seconds'),
            'native_optimizer_calls_this_invocation':None if raw is None else raw.get('optimizer_updates'),
            'native_resources':None if raw is None else raw.get('resources'),'failure':error,'control_error':control_error,
            'unknown_native_costs':raw is None,'TEST_export':selected,'automatic_retries':0,
            'TEST_accessed':None if raw is None else raw.get('TEST_accessed')}
        terminal=self.folder(task)/'attempts'/f'{attempt:04d}'/'TERMINAL.json';exclusive(terminal,record)
        state={k:record[k]for k in('status','attempt','native_result','TEST_export','live_native_result_receipt','TEST_accessed')}
        state['attempt_terminal']=pin(terminal)
        self.save_state(task,state);self.records.append(record)
        if control_error:self.stop({'reason':'final_terminal_identity_or_schema_corruption','id':task['id'],'error':control_error})
        self.backend.finished(task,pin(terminal))
        return status

    def launch(self,task):
        previous=self.states[task['id']];checkpoint=None
        if previous['status']!='pending':checkpoint=self.resume_for(task,previous)
        folder=self.folder(task);folder.mkdir(exist_ok=True);claim=folder/'CLAIM.json'
        if not claim.exists():exclusive(claim,self.immutable_claim(task))
        attempt=previous.get('attempt',0)+1;directory=folder/'attempts'/f'{attempt:04d}';directory.mkdir(parents=True,exist_ok=False)
        start={'schema':'expanded-attempt-start-v1','id':task['id'],'attempt':attempt,'manifest_sha256':self.manifest,
               'resume':checkpoint,'native_state_reused_only_from_same_fit':checkpoint is not None,'previous_attempts_preserved':True,'UTC_unix':time.time()}
        exclusive(directory/'START.json',start);self.save_state(task,{'status':'claimed','attempt':attempt})
        log=(directory/'STDOUT.log').open('x');started=time.monotonic()
        try:
            child=self.backend.start(task,self.output(task),checkpoint,log,self.states)
        except BaseException as exc:
            log.close();record={'id':task['id'],'attempt':attempt,'status':'launch_failed','children_started':0,'optimizer_transitions':0,'failure':str(exc)}
            exclusive(directory/'TERMINAL.json',record);self.save_state(task,{'status':'launch_failed','attempt':attempt,'attempt_terminal':pin(directory/'TERMINAL.json')})
            raise
        self.started_children+=1
        self.active[child.pid]={'child':child,'task':task,'attempt':attempt,'log':log,'started':started}
        self.max_active=max(self.max_active,len(self.active));require(len(self.active)<=2,'Two-slot invariant violated')
        try:identity=self.backend.identity(child)
        except Exception as error:identity={'pid':child.pid,'identity_unavailable':str(error)}
        exclusive(directory/'PID.json',identity)
        self.save_state(task,{'status':'running','attempt':attempt,'child_identity':identity,'started_UTC_unix':time.time()})

    def dependency(self,task):
        if not task['dependencies']:return 'ready'
        states=[self.states[x]['status']for x in task['dependencies']]
        if any(x in ('failed','launch_failed','blocked_dependency_failed') for x in states):return 'failed'
        if 'no_selected_state'in states or 'blocked_dependency_no_selected_state'in states:return 'no_selected_state'
        if not all(x=='complete'for x in states):return 'wait'
        for dep in task['dependencies']:self.verify(self.states[dep]['TEST_export'])
        return 'ready'

    def block_export(self,task,reason='no_selected_state'):
        require(task['kind']=='I4_TEST_aggregate','Only final I4 can be dependency-blocked')
        folder=self.folder(task);folder.mkdir(exist_ok=True);exclusive(folder/'CLAIM.json',self.immutable_claim(task))
        output=self.output(task);require(not output.exists(),'Never overwrite a prior final role outcome');output.mkdir(parents=True)
        raw={'schema':'final-I4-TEST-scoring-result-v2','task_id':task['id'],'role':'TEST','status':'missing_component',
             'physical_training_fits':0,'optimizer_updates':0,'automatic_retries':0,'native_forwards':0,'TEST_accessed':False,
             'dependencies':task['dependencies'],'dependency_statuses':{key:self.states[key]['status']for key in task['dependencies']},
             'reason':'Retained upstream '+reason+'; no fallback/retry/TEST tensor load'}
        exclusive(output/'RESULT.json',raw)
        record={'id':task['id'],'status':'blocked_dependency_'+reason,'dependencies':task['dependencies'],
                'live_native_result_receipt':pin(output/'RESULT.json'),'extra_training_fits':0,'model_forwards':0,'TEST_accessed':False}
        terminal=folder/'DEPENDENCY_TERMINAL.json';exclusive(terminal,record)
        self.save_state(task,{'status':record['status'],'attempt_terminal':pin(terminal),'TEST_accessed':False});self.records.append(record)

    def drain(self):
        errors=[]
        for pid,entry in list(self.active.items()):
            try:
                code=self.backend.wait(entry['child']);entry['log'].close()
                self.collect(entry['task'],code,entry['attempt'],time.monotonic()-entry['started'])
            except BaseException as exc:
                errors.append(str(exc))
            finally:self.active.pop(pid)
        if errors:
            self.records.append({'status':'drain_collection_errors','errors':errors,'reservations_retained':True})
            try:self.stop({'reason':'drain_collection_errors','errors':errors})
            except OSError:pass

    def run(self):
        outcome='complete';blocked=[];started=time.monotonic()
        try:
            self.establish()
            for task in self.tasks:self.reconcile(task)
            pending=[t for t in self.tasks if self.states[t['id']]['status']not in('complete','no_selected_state','blocked_dependency_no_selected_state','blocked_dependency_failed') and (self.states[t['id']]['status']not in('failed','launch_failed') or t['id']in self.resumes)]
            while pending or self.active:
                for pid,entry in list(self.active.items()):
                    code=self.backend.poll(entry['child'])
                    if code is None:continue
                    entry['log'].close();self.active.pop(pid)
                    status=self.collect(entry['task'],code,entry['attempt'],time.monotonic()-entry['started'])
                    # Ordinary fit failure retains costs and never retries; reviewed policy allows peers to continue.
                if self.latest_stops()-self.baseline_stops:self.stopped=True
                admitted=False;last_reason=None
                if not self.stopped:
                    for task in list(pending):
                        if len(self.active)>=2:break
                        dependency=self.dependency(task)
                        if dependency=='wait':continue
                        if dependency in('no_selected_state','failed'):self.block_export(task,dependency);pending.remove(task);continue
                        self.backend.recheck(task)
                        for value in self.extra_authorities:self.verify(value)
                        active=[{'pid':pid,'worker':self.worker,'resources':entry['task']['binding']['resources']}for pid,entry in self.active.items()]
                        ok,reason=self.backend.admit(task,active)
                        if not ok:
                            last_reason=reason
                            if reason=='untracked_owned_GPU_process':self.stop({'reason':reason});break
                            continue
                        self.launch(task);pending.remove(task);admitted=True
                        if getattr(self.backend,'control_error',None):
                            self.stop({'reason':'backend_control_error','error':self.backend.control_error});break
                atomic(self.control/('PROGRESS-'+self.worker.replace('/','_')+'.json'),{'manifest_sha256':self.manifest,
                    'active':[e['task']['id']for e in self.active.values()],'pending':[t['id']for t in pending],
                    'states':self.states,'maximum_owned_children_observed':self.max_active,'automatic_retries':0})
                if self.stopped:
                    self.drain();outcome='stopped_drained';blocked=[t['id']for t in pending];break
                if pending and not self.active and not admitted and not self.backend.peer_activity():
                    outcome='blocked_resources_or_dependencies';blocked=[t['id']for t in pending];self.records.append({'status':outcome,'reason':last_reason or 'dependency_not_terminal'});break
                if pending or self.active:time.sleep(self.poll_seconds)
        except BaseException as exc:
            outcome='blocked_or_failed_drained'
            try:
                if self.owns_leases and self.stops.is_dir():self.stop({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(limit=6)})
            except OSError:pass
            try:self.backend.cancel_pending(str(exc))
            except Exception as cancel_error:self.records.append({'status':'startup_custody_uncertain','error':str(cancel_error)})
            self.drain();blocked=[t['id']for t in self.tasks if self.states.get(t['id'],{}).get('status')not in('complete','no_selected_state','blocked_dependency_no_selected_state','blocked_dependency_failed')];self.records.append({'status':'dispatcher_error','type':type(exc).__name__,'message':str(exc)})
        finally:
            try:self.backend.cancel_pending('supervisor_exit_before_spawn')
            except Exception as cancel_error:self.records.append({'status':'startup_custody_uncertain','error':str(cancel_error)})
            for handle in self.lease_handles:fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        if outcome=='complete'and any(s['status']in('failed','launch_failed','blocked_dependency_failed')for s in self.states.values()):outcome='complete_with_failed_outcomes'
        result={'schema':'expanded-dispatch-worker-result-v1','status':outcome,'manifest_sha256':self.manifest,'worker':self.worker,
            'children_started_this_invocation':self.started_children,'maximum_owned_children_observed':self.max_active,
            'elapsed_seconds':time.monotonic()-started,'records':self.records,'not_admitted':blocked,'states':self.states,
            'automatic_retries':0,'ordinary_failure_policy':'retain_failure_continue_other_admissible_tasks','scored_role':'TEST','TEST_accessed':any(s.get('TEST_accessed')is True for s in self.states.values()),'TEST_access_unknown':any(s.get('TEST_accessed')is None and s['status']in('failed','interrupted','running','claimed')for s in self.states.values()),'native_model_qualification_claim':False}
        if self.control.is_dir():exclusive(self.control/('WORKER_RESULT-'+str(time.time_ns())+'.json'),result)
        return result
