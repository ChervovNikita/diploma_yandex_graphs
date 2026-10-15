"""Run a single frozen pilot, with at most two total own training workers.

Existing declared older workers reserve a slot until exit. Every fit runs once;
failed fits remain failed. A child failure stops new admission and allows live
peers to finish. This queue cannot launch a giant campaign or score TEST.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import atomic_json, digest, inside, require_allocation, utc
from run_pilot import cost_manifest


def alive(pid):
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--existing-training-pid',action='append',type=int,default=[])
    args=parser.parse_args();require_allocation()
    root=Path(args.project_root).resolve();plan_path=inside(args.plan,root)
    output=plan_path.parent;plan=json.loads(plan_path.read_text())
    spec=json.loads((output/'PILOT.json').read_text())
    if plan['fit_count']!=24 or len(spec['trials'])!=3 or spec['TEST_access'] is not False \
            or plan['pilot_hash']!=digest(spec):
        raise ValueError('This finite supervisor admits only the declared24-fit/3-trial pilot')
    marker=output/'QUEUE_START.json'
    with marker.open('x') as stream:
        json.dump({'UTC':utc(),'pid':os.getpid(),'plan_sha256':__import__('hashlib').sha256(plan_path.read_bytes()).hexdigest(),
                   'existing_training_pids':args.existing_training_pid,'maximum_total_own_workers':2,
                   'no_automatic_retries':True,'full_campaign_GO':False},stream,indent=2)
    bindings=plan['jobs'];jobs=[json.loads(Path(b['job']).read_text()) for b in bindings]
    paths={j['id']:{'job':Path(b['job']),'output':Path(b['output'])} for j,b in zip(jobs,bindings)}
    # Frozen, score-independent order: body0, base, ingredients, remaining I4 bodies.
    order=[]
    for start in range(0,24,8):order.extend(bindings[start+i] for i in (0,4,5,6,7,1,2,3))
    live={};done=[];failed=[];started=time.monotonic()
    while order or live:
        for pid,value in list(live.items()):
            process,binding,log=value;code=process.poll()
            if code is None:continue
            log.close();live.pop(pid)
            terminal=Path(binding['output'])/'RESULT.json'
            result=json.loads(terminal.read_text()) if terminal.is_file() else {}
            item={'id':binding['id'],'pid':pid,'exit_code':code,'status':result.get('status','missing_terminal')}
            done.append(item)
            if code!=0 or result.get('status')!='complete':failed.append(item)
            atomic_json(output/'COSTS.json',cost_manifest(jobs,paths))
        reserved=sum(alive(pid) for pid in args.existing_training_pid)
        while order and not failed and len(live)+reserved<2:
            binding=order.pop(0);folder=inside(binding['output'],root)
            if folder.exists():raise ValueError('Fresh queue must not overwrite/retry a prior fit')
            token=Path(binding['job']).stem
            log=(output/('queue_'+token+'.log')).open('x')
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,'-B','-u',str(Path(__file__).with_name('run_fit.py')),
                     '--project-root',str(root),'--job',binding['job'],'--output',binding['output']]
            process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            live[process.pid]=(process,binding,log)
        progress={'UTC':utc(),'status':'failed_draining' if failed else 'running','elapsed_seconds':time.monotonic()-started,
                  'live':[{'pid':pid,'id':v[1]['id']} for pid,v in live.items()],
                  'completed':done,'queued':len(order),'reserved_older_workers':reserved,
                  'failed':failed,'TEST_accessed':False,'full_campaign_GO':False}
        atomic_json(output/'QUEUE_PROGRESS.json',progress)
        if failed and not live:break
        if order or live:time.sleep(10)
    progress.update(status='failed' if failed else 'complete',finished_UTC=utc())
    if not failed:
        for index,cohort in enumerate(plan['cohorts']):
            members=[str(paths[jid]['output']/'selected_VALID.pt') for jid in cohort['native_fit_ids']]
            code=subprocess.call([sys.executable,'-B',str(Path(__file__).with_name('aggregate_selected.py')),
                  '--project-root',str(root),'--members',*members,'--output',str(output/('I4_'+str(index).zfill(3)))],cwd=root)
            if code:progress.update(status='aggregation_failed',failed_cohort=cohort['id']);break
    atomic_json(output/'COSTS.json',cost_manifest(jobs,paths))
    atomic_json(output/'QUEUE_RESULT.json',progress)
    return 0 if progress['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
