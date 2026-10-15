"""Durable two-slot dispatcher engine. Backend injection is for bounded tests.

Production CLI supplies only its fixed allocation backend. No fake/backend
selection flag exists in production. This module imports no model libraries.
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
        self.control=self.root/'control';self.task_control=self.control/'tasks';self.stops=self.control/'stops'
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
        path=self.stops/(str(time.time_ns())+'-'+str(os.getpid())+'.json')
        exclusive(path,{'schema':'expanded-dispatch-STOP-v1','manifest_sha256':self.manifest,'worker':self.worker,'reason':reason,'UTC_unix':time.time(),'live_children_drained_not_killed':True})
    def latest_stops(self):return {str(p)for p in self.stops.glob('*.json')}

    def establish(self):
        require(self.task_control.resolve().is_relative_to(self.root)and self.stops.resolve().is_relative_to(self.root),'Control paths escape output root')
        self.task_control.mkdir(parents=True,exist_ok=True);self.stops.mkdir(exist_ok=True)
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
        if state['status']in('complete','no_selected_state','blocked_dependency_no_selected_state'):
            self.verify(state['attempt_terminal'])
            if state['status']!='blocked_dependency_no_selected_state':
                self.verify(state['native_result']);self.verify(state['live_native_result_receipt']);raw=read(state['native_result']['path'])
                self.validate_terminal(task,raw)
                if state['status']=='complete'and task['kind']=='fit':self.verify(state['selected_VALID_export'])
            self.states[task['id']]=state;self.records.append({'id':task['id'],'action':'skip_already_'+state['status'],'extra_fits':0});return
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
        existing=directory/'TERMINAL.json'
        if existing.exists():
            record=read(existing)
            if record.get('status')in('complete','no_selected_state')and record.get('exit_code')==0:
                self.verify(record['native_result']);raw=read(record['native_result']['path']);self.validate_terminal(task,raw)
                if record['status']=='complete'and task['kind']=='fit':self.verify(record['selected_VALID_export'])
                recovered={k:record[k]for k in('status','attempt','native_result','selected_VALID_export','live_native_result_receipt')}
                self.verify(recovered['live_native_result_receipt']);recovered['attempt_terminal']=pin(existing)
                self.save_state(task,recovered);self.records.append({'id':task['id'],'action':'recovered_own_reaped_terminal_transaction','extra_scoring':0});return True
        approval=self.reconciliations.get(task['id'])
        if approval:require(not existing.exists(),'Known failed reap cannot be adopted as an unknown-exit completion')
        if not approval:return False
        self.verify(approval);value=read(approval['path'])
        require(value['schema']=='expanded-native-terminal-reconciliation-v1'and value['reviewed_accept_terminal']is True,'Reviewed lost-supervisor terminal reconciliation required')
        require(value['manifest_sha256']==self.manifest and value['task_id']==task['id']and value['runtime_job_hash']==task['runtime_job_hash']and value['science_contract_hash']==task['science_contract_hash'],'Reconciliation cannot change job/science/release')
        require(value['previous_attempt']==pin(directory/'START.json'),'Reconcile exact orphaned attempt')
        terminal=value['native_terminal'];self.verify(terminal)
        require(Path(terminal['path']).resolve()==self.output(task)/'RESULT.json','Reconcile only own native terminal')
        raw=read(terminal['path']);self.validate_terminal(task,raw);require(raw['status']in('complete','no_selected_state'),'Only complete/no-selected terminals can be adopted without scoring')
        selected=raw.get('selected_VALID_export')
        if raw['status']=='complete'and task['kind']=='fit':
            self.verify(selected);require(Path(selected['path']).resolve()==self.output(task)/'selected_VALID.pt','Own selected export required')
        self.verify(value['human_review_evidence']);self.extra_authorities.extend([approval,value['human_review_evidence']])
        archive=directory/'RECONCILED_NATIVE_RESULT.json'
        with archive.open('xb')as out:out.write(Path(terminal['path']).read_bytes())
        record={'schema':'expanded-reconciled-attempt-terminal-v1','id':task['id'],'attempt':state['attempt'],'status':raw['status'],
                'exit_code':'unknown_after_supervisor_loss','native_result':pin(archive),'live_native_result_receipt':terminal,'selected_VALID_export':selected,
                'native_invocation_wall_seconds':raw.get('invocation_wall_seconds'),'native_optimizer_calls_this_invocation':raw.get('optimizer_calls_this_invocation'),
                'partial_work_after_terminal':'not_inferred','reconciliation':approval,'new_children':0,'extra_scoring':0}
        target=directory/'TERMINAL_RECONCILED.json';exclusive(target,record)
        recovered={k:record[k]for k in('status','attempt','native_result','live_native_result_receipt','selected_VALID_export')};recovered['attempt_terminal']=pin(target)
        self.save_state(task,recovered);self.records.append(record);return True

    def resume_for(self,task,state):
        value_pin=self.resumes.get(task['id']);require(value_pin,'Interrupted/failed/claimed task requires reviewed same-fit disposition: '+task['id'])
        require(task['kind']=='fit','Export failure needs separately reviewed reconciliation; no automatic repeat scoring')
        self.verify(value_pin);value=read(value_pin['path'])
        require(value['schema']=='expanded-same-fit-resume-v1'and value['explicit_reviewed_resume']is True,'Explicit same-fit resume required')
        require(value['task_id']==task['id']and value['runtime_job_hash']==task['runtime_job_hash']and value['science_contract_hash']==task['science_contract_hash']and value['manifest_sha256']==self.manifest,'Resume cannot change physical job/science/release')
        previous=self.folder(task)/'attempts'/f"{state['attempt']:04d}"/'START.json'
        require(value['previous_attempt']==pin(previous),'Resume must bind exact previous attempt')
        checkpoint=value['complete_epoch_checkpoint'];self.verify(checkpoint)
        require(Path(checkpoint['path']).resolve()==self.output(task)/'recovery.pt','Only own unchanged complete-epoch recovery file may resume')
        require(value['prior_partial_work_is_preserved']is True,'Preserve unknown partial work and all failed attempt costs')
        self.verify(value['human_review_evidence']);self.extra_authorities.extend([value_pin,value['human_review_evidence']])
        return checkpoint

    def validate_terminal(self,task,raw):
        require(raw.get('TEST_accessed')is False,'Terminal TEST role violation')
        if task['kind']=='fit':
            require(raw.get('schema')=='native-full-epoch-fit-result-v1'and raw.get('job_id')==task['id']and raw.get('job_hash')==task['runtime_job_hash'],'Native terminal belongs to another fit')
        else:
            require(raw.get('schema')=='independent-own-selected-VALID-bank-result-v1','I4 export terminal schema mismatch')
            require(sorted(m['job_id']for m in raw.get('members',[]))==sorted(task['dependencies']),'I4 terminal members changed')
            expected={dep:self.index[dep]['runtime_job_hash']for dep in task['dependencies']}
            require(all(m.get('job_hash')==expected[m['job_id']]for m in raw['members']),'I4 member job hashes changed')
        require(raw.get('status')in('complete','no_selected_state','failed'),'Unknown native terminal status')

    def collect(self,task,code,attempt,elapsed):
        output=self.output(task);native_path=output/'RESULT.json';raw=read(native_path)if native_path.is_file()else None
        error=None;status='failed';selected=None
        try:
            require(raw is not None,'Missing native terminal');self.validate_terminal(task,raw)
            require(code==0 and raw['status']in('complete','no_selected_state'),'Child failed or native terminal is failed')
            status=raw['status']
            if status=='complete'and task['kind']=='fit':
                selected=raw.get('selected_VALID_export');require(selected,'Complete fit must retain own selected VALID authority');self.verify(selected)
                require(Path(selected['path']).resolve()==output/'selected_VALID.pt','Selected export escapes own physical fit')
        except Exception as exc:
            status='failed';selected=None;error=type(exc).__name__+': '+str(exc)
        archive=self.folder(task)/'attempts'/f'{attempt:04d}'/'NATIVE_RESULT.json'
        if native_path.is_file():
            with archive.open('xb')as out:out.write(native_path.read_bytes())
        record={'schema':'expanded-attempt-terminal-v1','id':task['id'],'attempt':attempt,'status':status,'exit_code':code,
            'observed_launch_to_reap_seconds':elapsed,'native_claimed_status':None if raw is None else raw.get('status'),'native_result':pin(archive)if archive.is_file()else None,'live_native_result_receipt':pin(native_path)if native_path.is_file()else None,
            'native_invocation_wall_seconds':None if raw is None else raw.get('invocation_wall_seconds'),
            'native_optimizer_calls_this_invocation':None if raw is None else raw.get('optimizer_calls_this_invocation'),
            'native_resources':None if raw is None else raw.get('resources'),'failure':error,
            'unknown_native_costs':raw is None or raw.get('optimizer_calls_this_invocation')is None,
            'selected_VALID_export':selected,'automatic_retries':0,'TEST_accessed':False}
        terminal=self.folder(task)/'attempts'/f'{attempt:04d}'/'TERMINAL.json';exclusive(terminal,record)
        state={k:record[k]for k in('status','attempt','native_result','selected_VALID_export','live_native_result_receipt')};state['attempt_terminal']=pin(terminal)
        self.save_state(task,state);self.records.append(record)
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
        if 'no_selected_state'in states or 'blocked_dependency_no_selected_state'in states:return 'no_selected_state'
        if not all(x=='complete'for x in states):return 'wait'
        for dep in task['dependencies']:self.verify(self.states[dep]['selected_VALID_export'])
        return 'ready'

    def block_export(self,task):
        folder=self.folder(task);folder.mkdir(exist_ok=True);exclusive(folder/'CLAIM.json',self.immutable_claim(task))
        record={'id':task['id'],'status':'blocked_dependency_no_selected_state','dependencies':task['dependencies'],'extra_training_fits':0,'model_forwards':0}
        terminal=folder/'DEPENDENCY_TERMINAL.json';exclusive(terminal,record)
        self.save_state(task,{'status':record['status'],'attempt_terminal':pin(terminal)});self.records.append(record)

    def drain(self):
        for pid,entry in list(self.active.items()):
            code=self.backend.wait(entry['child']);entry['log'].close()
            self.collect(entry['task'],code,entry['attempt'],time.monotonic()-entry['started']);self.active.pop(pid)

    def run(self):
        outcome='complete';blocked=[];started=time.monotonic()
        try:
            self.establish()
            for task in self.tasks:self.reconcile(task)
            pending=[t for t in self.tasks if self.states[t['id']]['status']not in('complete','no_selected_state','blocked_dependency_no_selected_state')]
            while pending or self.active:
                for pid,entry in list(self.active.items()):
                    code=self.backend.poll(entry['child'])
                    if code is None:continue
                    entry['log'].close();self.active.pop(pid)
                    status=self.collect(entry['task'],code,entry['attempt'],time.monotonic()-entry['started'])
                    if status=='failed':self.stop({'reason':'child_failed','id':entry['task']['id']})
                if self.latest_stops()-self.baseline_stops:self.stopped=True
                admitted=False;last_reason=None
                if not self.stopped:
                    for task in list(pending):
                        if len(self.active)>=2:break
                        dependency=self.dependency(task)
                        if dependency=='wait':continue
                        if dependency=='no_selected_state':self.block_export(task);pending.remove(task);continue
                        self.backend.recheck(task)
                        for value in self.extra_authorities:self.verify(value)
                        active=[{'pid':pid,'worker':self.worker,'resources':entry['task']['binding']['resources']}for pid,entry in self.active.items()]
                        ok,reason=self.backend.admit(task,active)
                        if not ok:
                            last_reason=reason
                            if reason=='untracked_owned_GPU_process':self.stop({'reason':reason});break
                            continue
                        self.launch(task);pending.remove(task);admitted=True
                atomic(self.control/('PROGRESS-'+self.worker.replace('/','_')+'.json'),{'manifest_sha256':self.manifest,
                    'active':[e['task']['id']for e in self.active.values()],'pending':[t['id']for t in pending],
                    'states':self.states,'maximum_owned_children_observed':self.max_active,'automatic_retries':0})
                if self.stopped:
                    self.drain();outcome='stopped_drained';blocked=[t['id']for t in pending];break
                if pending and not self.active and not admitted:
                    outcome='blocked_resources_or_dependencies';blocked=[t['id']for t in pending];self.records.append({'status':outcome,'reason':last_reason or 'dependency_not_terminal'});break
                if pending or self.active:time.sleep(self.poll_seconds)
        except BaseException as exc:
            outcome='blocked_or_failed_drained'
            if self.owns_leases and self.stops.is_dir():self.stop({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(limit=6)})
            self.drain();blocked=[t['id']for t in self.tasks if self.states.get(t['id'],{}).get('status')not in('complete','no_selected_state','blocked_dependency_no_selected_state')];self.records.append({'status':'dispatcher_error','type':type(exc).__name__,'message':str(exc)})
        finally:
            for handle in self.lease_handles:fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        result={'schema':'expanded-dispatch-worker-result-v1','status':outcome,'manifest_sha256':self.manifest,'worker':self.worker,
            'children_started_this_invocation':self.started_children,'maximum_owned_children_observed':self.max_active,
            'elapsed_seconds':time.monotonic()-started,'records':self.records,'not_admitted':blocked,'states':self.states,
            'automatic_retries':0,'TEST_accessed':False,'native_model_qualification_claim':False}
        if self.control.is_dir():exclusive(self.control/('WORKER_RESULT-'+str(time.time_ns())+'.json'),result)
        return result
