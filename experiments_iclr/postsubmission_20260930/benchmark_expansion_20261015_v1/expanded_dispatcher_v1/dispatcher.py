"""Production interface: one guarded supervisor per authorized physical GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
# CLI route guard runs before importing any project module or reading a manifest.
if __name__=='__main__':
    import socket
    requested=sys.argv[sys.argv.index('--worker')+1]if '--worker'in sys.argv and sys.argv.index('--worker')+1<len(sys.argv)else'allocation/gpu0'
    if requested!='allocation/gpu0':
        print('{"status":"BLOCKED","reason":"Server77 disabled before project/host access","children_started":0}');raise SystemExit(2)
    if socket.gethostname()!='anogena-2-0' or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()!=['GPU-44039938-fd82-41d2-fefd-de71514e2fac']:
        print('{"status":"BLOCKED","reason":"Wrong allocation identity;no project access","children_started":0}');raise SystemExit(2)

from contracts import (ROOT,PHASE,HERE,UUID,WORKERS,GATES,guard,inside,read,pin,verify,require,
                       compile_release,authority)
from engine import Engine


class AllocationBackend:
    def __init__(self,legacy,budget,recheck):self.legacy=legacy;self.budget=budget;self.recheck=recheck
    def admit(self,task,active):
        probe=self.legacy.probe_system(ROOT,active)
        return self.legacy.admissible({'resources':task['binding']['resources']},'allocation/gpu0',active,probe,self.budget)
    def start(self,task,output,checkpoint,log,states):
        execution=task['binding']['execution'];entry=execution['runtime_entrypoint'];verify(entry)
        argv=[execution['python_executable'],'-B','-u',entry['path'],'--project-root',str(ROOT)]
        if task['kind']=='fit':
            descriptor=execution['job_descriptor'];verify(descriptor)
            argv+=['--job',descriptor['path'],'--output',str(output)]
            if checkpoint:verify(checkpoint);argv+=['--resume',checkpoint['path']]
        else:
            members=[states[x]['selected_VALID_export']for x in task['dependencies']]
            require(len(members)==4,'I4 requires four own selected exports')
            for value in members:verify(value)
            argv+=['--members',*[v['path']for v in members],'--output',str(output)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        return subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    def identity(self,child):
        folder=Path('/proc')/str(child.pid);raw=(folder/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return {'pid':child.pid,'UID':folder.stat().st_uid,'linux_start_ticks':fields[19],
                'cmdline_sha256':hashlib.sha256((folder/'cmdline').read_bytes()).hexdigest()}
    def alive(self,identity):
        pid=identity['pid'];folder=Path('/proc')/str(pid)
        if not folder.exists():return False
        if identity.get('identity_unavailable'):return True  # Fail closed; never kill/adopt a possibly reused PID.
        try:return self.identity(type('Recorded',(),{'pid':pid})())==identity
        except FileNotFoundError:return False
    @staticmethod
    def poll(child):return child.poll()
    @staticmethod
    def wait(child):return child.wait()


def run(release_path,go_path,readiness_path,worker,resumes_path=None,continuation_path=None,reconciliations_path=None):
    release,release_pin,old,budget,recheck=authority(release_path,go_path,readiness_path,worker)
    resumes={};extra=[]
    if resumes_path:
        resume_map=pin(inside(resumes_path));verify(resume_map);value=read(resumes_path)
        require(value['schema']=='expanded-reviewed-resume-map-v1'and value['manifest_sha256']==release_pin['sha256'],'Resume map must bind unchanged release')
        resumes=value['tasks'];require(set(resumes).issubset({t['id']for t in release['tasks']if t['worker']==worker}),'Resume map has foreign IDs');extra.append(resume_map)
    reconciliations={}
    if reconciliations_path:
        reconciliation_map=pin(inside(reconciliations_path));value=read(reconciliations_path)
        require(value['schema']=='expanded-reviewed-reconciliation-map-v1'and value['manifest_sha256']==release_pin['sha256'],'Reconciliation map must bind unchanged release')
        reconciliations=value['tasks'];require(set(reconciliations).issubset({t['id']for t in release['tasks']if t['worker']==worker}),'Reconciliation map has foreign IDs');extra.append(reconciliation_map)
    continuation=pin(inside(continuation_path))if continuation_path else None
    if continuation:extra.append(continuation)
    original_recheck=recheck
    def full_recheck(task):
        original_recheck(task)
        for value in extra:verify(value)
    backend=AllocationBackend(old,budget,full_recheck)
    # Coordinate with both this successor and the predecessor's allocation lease.
    leases=[inside(PHASE/'.expanded_dispatcher_leases'/f'{UUID}.lock'),inside(PHASE/'.campaign_worker_leases/allocation_gpu0.lock')]
    tasks=[t for t in release['tasks']if t['worker']==worker]
    return Engine(tasks,release['output_root'],release_pin['sha256'],worker,backend,leases,resumes=resumes,continuation=continuation,reconciliations=reconciliations,allowed_root=ROOT).run()


def inspect_release(path,worker):
    guard(worker);release=read(inside(path))
    require(release['schema']=='expanded-production-dispatch-release-v1','Production release required')
    own=[t for t in release['tasks']if t['worker']==worker]
    return {'schema':'expanded-production-dispatch-preflight-v1','status':'BLOCKED_QUALIFICATION_AND_GO',
        'release':pin(path),'physical_fit_intentions':sum(t['kind']=='fit'for t in own),
        'I4_export_only_tasks':sum(t['kind']!='fit'for t in own),'qualified_tasks':sum(t['binding']is not None for t in own),
        'blocked_tasks':sum(t['binding']is None for t in own),'actual_production_dispatch_path_implemented':True,
        'native_runtime_qualification_by_fake_tests':False,'user_GO_received':False,'resource_profiles_invented':False,
        'source_reference_and_matched_protocols_preserved':True,'maximum_owned_children':2,'server77_enabled':False,
        'unresolved_gates':list(GATES)+['human_full_campaign_GO','worker_route_environment_claim_STOP_resume_evidence'],
        'models_constructed':0,'children_started':0,'TEST_accessed':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    c=s.add_parser('compile');c.add_argument('--skeleton',required=True);c.add_argument('--bindings',required=True);c.add_argument('--output',required=True);c.add_argument('--output-root',required=True)
    for action in ('run','dry-run'):
        q=s.add_parser(action);q.add_argument('--release',required=True);q.add_argument('--worker',choices=WORKERS,required=True)
        if action=='run':q.add_argument('--go',required=True);q.add_argument('--readiness',required=True);q.add_argument('--resumes');q.add_argument('--continuation');q.add_argument('--reconciliations')
    a=p.parse_args()
    try:
        if a.action=='compile':
            guard('allocation/gpu0');value=compile_release(pin(a.skeleton),a.bindings,a.output,a.output_root)
        elif a.action=='dry-run':value=inspect_release(a.release,a.worker)
        else:value=run(a.release,a.go,a.readiness,a.worker,a.resumes,a.continuation,a.reconciliations)
        print(json.dumps(value,sort_keys=True));raise SystemExit(0 if value.get('status')in(None,'complete')else 2)
    except (ValueError,OSError)as error:
        print(json.dumps({'status':'BLOCKED','reason':str(error),'children_started':0}));raise SystemExit(2)
