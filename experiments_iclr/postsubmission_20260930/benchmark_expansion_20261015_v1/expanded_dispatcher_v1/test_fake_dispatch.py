"""Finite fake-subprocess contract checks. No native/scientific process is run."""
from __future__ import annotations
import fcntl
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from engine import Engine,pin,read,exclusive,digest
import contracts as production

HERE=Path(__file__).resolve().parent
PIDS={}


class FakeBackend:
    def __init__(self,modes=None,serial=False,incompatible=False,fail_recheck_after=None,resource_block=False,fail_start=False):
        self.modes=modes or {};self.serial=serial;self.incompatible=incompatible;self.fail_recheck_after=fail_recheck_after
        self.resource_block=resource_block;self.fail_start=fail_start;self.rechecks=0;self.launches=[];self.processes=[]
    def recheck(self,task):
        self.rechecks+=1
        if self.fail_recheck_after and self.rechecks>=self.fail_recheck_after:raise ValueError('FAKE source authority changed')
    def admit(self,task,active):
        if self.resource_block:return False,'VRAM_reservation'
        if len(active)>=2:return False,'two_slot_limit'
        if active and self.serial:return False,'serial_only_resource_profile'
        if active and self.incompatible:return False,'pair_profile_not_qualified'
        return True,'FAKE admitted'
    def start(self,task,output,checkpoint,log,states):
        if self.fail_start:raise OSError('FAKE Popen refused before child construction')
        case={'id':task['id'],'kind':task['kind'],'job_hash':task.get('runtime_job_hash'),'output':str(output),'mode':self.modes.get(task['id'],'complete'),
              'sleep_seconds':.10,'resume':None if checkpoint is None else checkpoint['path']}
        if task['kind']!='fit':case['members']=[{'job_id':x,'job_hash':task_index[x]['runtime_job_hash']}for x in task['dependencies']]
        child=subprocess.Popen([sys.executable,'-B',str(HERE/'fake_worker.py'),json.dumps(case)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        token={'pid':child.pid,'fake_token':str(time.time_ns())};PIDS[child.pid]=(child,token);self.processes.append(child);self.launches.append(task['id']);return child
    def identity(self,child):return PIDS[child.pid][1]
    def alive(self,identity):return identity['pid']in PIDS and PIDS[identity['pid']][1]==identity and PIDS[identity['pid']][0].poll()is None
    @staticmethod
    def poll(child):return child.poll()
    @staticmethod
    def wait(child):return child.wait()


def task(name,kind='fit',deps=()):
    return {'id':name,'kind':kind,'dependencies':list(deps),'worker':'allocation/gpu0','runtime_job_hash':digest(name)if kind=='fit'else None,
        'science_contract_hash':digest({'FAKE':name}),'output_relative':'outputs/'+digest(name),'binding':{'resources':{'FAKE_METADATA_ONLY':True}}}

def execute(folder,tasks,backend,**kwargs):
    global task_index;task_index={t['id']:t for t in tasks}
    value=Engine(tasks,folder,'FAKE_manifest_sha','allocation/gpu0',backend,[folder/'FAKE_GPU.lock'],poll_seconds=.01,**kwargs).run()
    assert all(p.poll()is not None for p in backend.processes),'Fake child leaked'
    return value

def review(folder):
    p=folder/'FAKE_REVIEW.json'
    if not p.exists():exclusive(p,{'fake_metadata_only':True,'not_actual_human_GO':True})
    return pin(p)

def continuation(folder):
    p=folder/('FAKE_CONTINUE-'+str(time.time_ns())+'.json')
    exclusive(p,{'schema':'expanded-reviewed-continuation-v1','reviewed_continue':True,'manifest_sha256':'FAKE_manifest_sha',
        'STOP_receipts':[pin(x)for x in sorted((folder/'control/stops').glob('*.json'))],'human_review_evidence':review(folder)})
    return pin(p)


def checks(root):
    results=[]
    def passed(name,value=None):results.append({'name':name,'status':'pass','detail':value})
    tasks=[task('A'),task('B'),task('C')];b=FakeBackend();r=execute(root/'two_slots',tasks,b)
    assert r['status']=='complete'and r['maximum_owned_children_observed']==2 and r['children_started_this_invocation']==3
    passed('at most two children including startup; queue drains all three')
    b=FakeBackend();r2=execute(root/'two_slots',tasks,b)
    assert r2['children_started_this_invocation']==0 and len(b.launches)==0
    passed('restart skips all own completed fits without scoring or retraining')
    for name,kwargs in [('serial',{'serial':True}),('incompatible',{'incompatible':True})]:
        r=execute(root/name,[task(name+'A'),task(name+'B')],FakeBackend(**kwargs));assert r['status']=='complete'and r['maximum_owned_children_observed']==1
        passed(name+' profiles prohibit overlap')
    folder=root/'failure';b=FakeBackend(modes={'F':'fail'});r=execute(folder,[task('F'),task('P'),task('Q')],b)
    assert r['status']=='stopped_drained'and r['children_started_this_invocation']==2 and 'Q'not in b.launches
    assert r['states']['P']['status']=='complete'and list((folder/'control/stops').glob('*.json'))
    passed('failure stops new admissions and drains live peer without killing it')
    before=len(list((folder/'control/stops').glob('*.json')));r=execute(folder,[task('F'),task('P'),task('Q')],FakeBackend())
    assert r['children_started_this_invocation']==0 and r['status']=='blocked_or_failed_drained'
    passed('retained STOP prevents automatic retries')
    folder=root/'resume';t=task('R');r=execute(folder,[t],FakeBackend(modes={'R':'interrupt'}))
    assert r['states']['R']['status']=='failed'and r['children_started_this_invocation']==1
    control=folder/'control/tasks'/digest('R');state=read(control/'STATE.json');resume=folder/'FAKE_RESUME.json'
    exclusive(resume,{'schema':'expanded-same-fit-resume-v1','explicit_reviewed_resume':True,'task_id':'R','runtime_job_hash':t['runtime_job_hash'],
        'science_contract_hash':t['science_contract_hash'],'manifest_sha256':'FAKE_manifest_sha','previous_attempt':pin(control/'attempts/0001/START.json'),
        'complete_epoch_checkpoint':pin(folder/t['output_relative']/'recovery.pt'),'prior_partial_work_is_preserved':True,'human_review_evidence':review(folder)})
    r=execute(folder,[t],FakeBackend(),resumes={'R':pin(resume)},continuation=continuation(folder))
    assert r['status']=='complete'and r['children_started_this_invocation']==1 and r['states']['R']['attempt']==2
    assert (control/'attempts/0001/TERMINAL.json').exists()and (control/'attempts/0002/TERMINAL.json').exists()
    raw=read(folder/t['output_relative']/'RESULT.json');assert raw['resume_argument_recorded'].endswith('recovery.pt')
    passed('reviewed same-fit epoch resume retains both attempts and unchanged job ID/hash')
    tasks=[task('D'+str(i))for i in range(4)];tasks+=[task('I4','aggregate_I4_exports',[t['id']for t in tasks])]
    b=FakeBackend();r=execute(root/'I4',tasks,b);assert r['status']=='complete'and b.launches[-1]=='I4'and len(b.launches)==5
    passed('I4 export task waits for four own selected VALID authorities; no member scoring')
    b=FakeBackend(modes={'D0':'tampered_export'});r=execute(root/'tampered_selected_export',tasks,b)
    assert r['status']=='stopped_drained'and r['states']['D0']['status']=='failed'and r['states']['D0']['selected_VALID_export']is None
    assert r['states']['D1']['status']=='complete'and 'I4'not in b.launches and 'D2'not in b.launches and 'D3'not in b.launches
    bad=next(x for x in r['records']if x.get('id')=='D0')
    assert bad['native_claimed_status']=='complete'and bad['failure']and bad['status']=='failed'
    passed('complete native claim with tampered selected export fails,drains peer,and never admits I4')
    b=FakeBackend(modes={'D0':'no_selected'});r=execute(root/'no_selected',tasks,b)
    assert 'I4'not in b.launches and r['states']['I4']['status']=='blocked_dependency_no_selected_state'
    passed('no selected state blocks dependent I4 without repeating a fit')
    r=execute(root/'resource_block',[task('Z')],FakeBackend(resource_block=True));assert r['children_started_this_invocation']==0 and r['status']=='blocked_resources_or_dependencies'
    passed('insufficient reservations retain pending work with no child and no invented capacity')
    r=execute(root/'wrong_identity',[task('W')],FakeBackend(modes={'W':'wrong_identity'}));assert r['states']['W']['status']=='failed'
    passed('wrong native terminal identity is not accepted as complete')
    b=FakeBackend(fail_recheck_after=2);r=execute(root/'source_change',[task('S1'),task('S2')],b)
    assert r['children_started_this_invocation']==1 and r['states']['S1']['status']=='complete'and r['status']=='blocked_or_failed_drained'
    passed('source authority is rechecked before every admission and live work drains')
    r=execute(root/'launch_error',[task('L')],FakeBackend(fail_start=True));assert r['states']['L']['status']=='launch_failed'and r['children_started_this_invocation']==0
    passed('Popen failure records zero started children and blocks retries')
    folder=root/'lease';folder.mkdir();lease=(folder/'FAKE_GPU.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:r=execute(folder,[task('LEASE')],FakeBackend());assert r['children_started_this_invocation']==0 and not list((folder/'control/stops').glob('*.json'))
    finally:fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    passed('second physical GPU supervisor refuses lease without stopping legitimate owner')
    try:production.guard('server77/gpu0');assert False
    except ValueError as exc:assert 'disabled'in str(exc)
    passed('server77 refused before project/host/GPU access')
    # Interrupted supervisor after writing its own reaped terminal: transaction recovery skips scoring.
    folder=root/'transaction';t=task('T');execute(folder,[t],FakeBackend())
    control=folder/'control/tasks'/digest('T');state=read(control/'STATE.json')
    (control/'STATE.json').write_text(json.dumps({'status':'running','attempt':1,'child_identity':{'pid':999999,'fake_token':'dead'}}))
    b=FakeBackend();r=execute(folder,[t],b);assert r['children_started_this_invocation']==0 and r['states']['T']['status']=='complete'
    passed('own reaped terminal transaction recovers without repeated scoring')
    # Full native terminal without a supervisor reap requires an explicit reviewed adoption.
    folder=root/'reconcile';t=task('REC');execute(folder,[t],FakeBackend())
    control=folder/'control/tasks'/digest('REC');directory=control/'attempts/0001'
    # Keep immutable original evidence; simulate only the transaction visibility of a separate fresh fixture.
    original_terminal=directory/'TERMINAL.json';saved=directory/'SIMULATED_PRIOR_TERMINAL.json';original_terminal.rename(saved)
    (control/'STATE.json').write_text(json.dumps({'status':'running','attempt':1,'child_identity':{'pid':999998,'fake_token':'dead'}}))
    approval=folder/'FAKE_RECONCILE.json'
    exclusive(approval,{'schema':'expanded-native-terminal-reconciliation-v1','reviewed_accept_terminal':True,'manifest_sha256':'FAKE_manifest_sha',
        'task_id':'REC','runtime_job_hash':t['runtime_job_hash'],'science_contract_hash':t['science_contract_hash'],
        'previous_attempt':pin(directory/'START.json'),'native_terminal':pin(folder/t['output_relative']/'RESULT.json'),'human_review_evidence':review(folder)})
    b=FakeBackend();r=execute(folder,[t],b,reconciliations={'REC':pin(approval)})
    assert r['status']=='complete'and r['children_started_this_invocation']==0 and (directory/'TERMINAL_RECONCILED.json').exists()
    passed('reviewed lost-supervisor native terminal adoption avoids all repeated scoring')
    folder=root/'unclaimed';t=task('U');(folder/t['output_relative']).mkdir(parents=True)
    r=execute(folder,[t],FakeBackend());assert r['children_started_this_invocation']==0
    passed('existing unclaimed native output is never overwritten or repeated')
    try:Engine([task('dup'),task('dup')],root/'duplicate','FAKE','allocation/gpu0',FakeBackend(),[]);assert False
    except ValueError:pass
    passed('duplicate physical task IDs rejected before claims or children')
    folder=root/'orphan_live';t=task('ORPHAN');b=FakeBackend();e=Engine([t],folder,'FAKE_manifest_sha','allocation/gpu0',b,[folder/'FAKE_GPU.lock'],poll_seconds=.01)
    control=e.folder(t);(control/'attempts/0001').mkdir(parents=True)
    exclusive(control/'CLAIM.json',e.immutable_claim(t));exclusive(control/'attempts/0001/START.json',{'FAKE':True})
    log=(control/'FAKE_ORPHAN.log').open('x');child=b.start(t,e.output(t),None,log,{})
    exclusive(control/'STATE.json',{'status':'running','attempt':1,'child_identity':b.identity(child)})
    r=e.run();assert r['children_started_this_invocation']==0 and len(b.launches)==1
    code=child.wait(timeout=5);log.close();assert code==0
    passed('live orphan claim is neither duplicated nor killed;bounded fake child finishes itself')
    # Structural GO/readiness tests use explicit FAKE receipt files, not human authorization.
    folder=root/'FAKE_AUTHORITY';folder.mkdir();evidence=folder/'FAKE_NOT_HUMAN_GO.json';exclusive(evidence,{'FAKE_METADATA_ONLY':True,'is_human_GO':False})
    old_verify=production.verify
    def fixture_verify(value):
        assert Path(value['path']).resolve().is_relative_to(folder.resolve())
        from engine import verify as check
        check(value)
    production.verify=fixture_verify
    try:
        release={'source_bundle_sha256':'FAKE_source'};release_pin={'sha256':'FAKE_manifest'}
        ready={'schema':'expanded-production-readiness-v1','manifest_sha256':'FAKE_manifest','source_bundle_sha256':'FAKE_source',
            'workers':{'allocation/gpu0':{'budget':{'gpu_headroom_bytes':0,'host_ram_headroom_bytes':0,'disk_headroom_bytes':0},
            'gates':{g:{'status':'PASS','evidence':pin(evidence)}for g in('route_identity_environment','complete_per_task_qualification','shared_claim_and_STOP_custody','failure_cost_and_resume_custody')}}}}
        ready_pin={'path':str(folder/'FAKE_READY.json'),'bytes':1,'sha256':'FAKE_ready'}
        go={'schema':'expanded-production-human-GO-v1','user_explicit_GO':True,'manifest_sha256':'FAKE_manifest','source_bundle_sha256':'FAKE_source',
            'readiness':ready_pin,'allowed_workers':['allocation/gpu0'],'human_instruction_evidence':pin(evidence),'review_evidence':'FAKE structural fixture, not human GO'}
        production.validate_reviewed_authority(release,release_pin,go,{},ready,ready_pin,'allocation/gpu0')
        passed('structural authority helper accepts a complete bounded FAKE fixture only;no actual GO created')
        variants=[('GO false',lambda g,r:g.update(user_explicit_GO=False)),('GO manifest mismatch',lambda g,r:g.update(manifest_sha256='wrong')),
          ('GO source mismatch',lambda g,r:g.update(source_bundle_sha256='wrong')),('wrong readiness receipt',lambda g,r:g.update(readiness={})),
          ('worker not allowed',lambda g,r:g.update(allowed_workers=[])),('readiness source mismatch',lambda g,r:r.update(source_bundle_sha256='wrong')),
          ('non-PASS worker gate',lambda g,r:r['workers']['allocation/gpu0']['gates']['shared_claim_and_STOP_custody'].update(status='FAIL')),
          ('negative headroom',lambda g,r:r['workers']['allocation/gpu0']['budget'].update(gpu_headroom_bytes=-1))]
        for label,change in variants:
            wrong_go,wrong_ready=copy.deepcopy(go),copy.deepcopy(ready);change(wrong_go,wrong_ready)
            try:production.validate_reviewed_authority(release,release_pin,wrong_go,{},wrong_ready,ready_pin,'allocation/gpu0');assert False,label
            except ValueError:passed('authority rejects '+label)
    finally:production.verify=old_verify
    # Exercise the actual predecessor reservation code with labelled synthetic inputs.
    parent=HERE.parent/'campaign_launcher_v1';old_verify=production.verify
    def local_source_verify(value):
        assert Path(value['path']).resolve().is_relative_to(parent.resolve())
        from engine import verify as check
        check(value)
    production.verify=local_source_verify
    try:old,old_contract=production.load_legacy({name:pin(parent/name)for name in('contracts.py','supervisor.py')})
    finally:production.verify=old_verify
    r={'vram_bytes':100,'ram_bytes':200,'disk_bytes':300,'max_concurrent_owned_children':2,'profile':'FAKE_A','qualified_peer_profiles':['FAKE_A']}
    probe={'untracked_own_gpu_pids':[],'allocated_vram_by_pid':{},'rss_by_pid':{},'free_vram_bytes':1000,'available_ram_bytes':2000,'free_disk_bytes':3000}
    budget={'gpu_headroom_bytes':0,'host_ram_headroom_bytes':0,'disk_headroom_bytes':0}
    active=[{'pid':123,'worker':'allocation/gpu0','resources':r}]
    assert old.admissible({'resources':r},'allocation/gpu0',active,probe,budget)==(True,'admitted')
    passed('actual reused admission permits only reciprocal measured-profile fixture pair')
    wrong=copy.deepcopy(r);wrong['qualified_peer_profiles']=[]
    assert old.admissible({'resources':wrong},'allocation/gpu0',active,probe,budget)[1]=='pair_profile_not_qualified'
    passed('actual reused admission rejects missing reciprocal peer profile')
    scarce=copy.deepcopy(probe);scarce['free_vram_bytes']=150
    assert old.admissible({'resources':r},'allocation/gpu0',active,scarce,budget)[1]=='VRAM_reservation'
    passed('actual reused admission reserves startup memory before a child appears in CUDA')
    crowded=copy.deepcopy(probe);crowded['untracked_own_gpu_pids']=[999999]
    assert old.admissible({'resources':r},'allocation/gpu0',[],crowded,budget)[1]=='untracked_owned_GPU_process'
    passed('actual reused admission blocks unknown owned GPU processes')
    assert old.admissible({'resources':r},'allocation/gpu0',active+active,probe,budget)[1]=='two_slot_limit'
    passed('actual reused admission enforces two-slot ceiling')
    limited=copy.deepcopy(probe);limited['available_ram_bytes']=300
    assert old.admissible({'resources':r},'allocation/gpu0',active,limited,budget)[1]=='host_shared_RAM_reservation'
    passed('actual reused admission keeps remaining host RAM reservations')
    assert production.digest({'method':'variance λ'})==hashlib.sha256(json.dumps({'method':'variance λ'},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    passed('UTF-8 scientific metadata canonical hashes match descriptor compiler')
    assert not any(k in sys.modules for k in('torch','numpy','scipy','ogb','torch_geometric'))
    passed('no model/data/numerical libraries imported')
    return results


if __name__=='__main__':
    destination=HERE/('FAKE_DISPATCH_FIXTURE_'+str(time.time_ns()));destination.mkdir()
    started=time.monotonic();results=checks(destination)
    value={'schema':'expanded-dispatcher-fake-subprocess-checks-v1','status':'pass','checks':results,'check_count':len(results),
        'fixture_directory':str(destination),'elapsed_seconds':time.monotonic()-started,'scope':'bounded real OS fake subprocesses;plain-text fake .pt files;no native runtime/model/data/TEST/GPU work',
        'native_models_constructed':0,'native_optimizer_updates':0,'TEST_accessed':False,'native_or_mature_resource_qualification':False,
        'fake_children_bounded_and_reaped':True,'synthetic_reviews_are_not_human_GO':True}
    target=HERE/('FAKE_CHECKS_'+str(time.time_ns())+'.json');exclusive(target,value);print(json.dumps({'evidence':pin(target),'check_count':len(results),'status':'pass'},sort_keys=True))
