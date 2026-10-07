"""One portable Molhiv family: 8 frozen arms x 3 seeds, no retry or analysis."""
import argparse,datetime,hashlib,importlib.metadata,json,os,resource,signal,socket,subprocess,sys,time
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(temporary,path)


def start_ticks(pid):
    file=Path('/proc')/str(pid)/'stat'
    return int(file.read_text().rsplit(') ',1)[1].split()[19]) if file.exists() else None


def runtime_identity():
    import torch
    return {'python':sys.version.split()[0],'torch':str(torch.__version__),'cuda':torch.version.cuda,
            **{name:importlib.metadata.version(package) for name,package in
               [('numpy','numpy'),('pyg','torch-geometric'),('ogb','ogb'),('torch_sparse','torch-sparse'),('torch_scatter','torch-scatter')]}}


def gpu_observation(device):
    import torch
    chosen=torch.device(device)
    if chosen.type!='cuda':raise ValueError('This queue targets one CUDA device')
    index=0 if chosen.index is None else chosen.index
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    physical=visible.split(',')[index].strip() if visible is not None else str(index)
    values=subprocess.check_output(['nvidia-smi','-i',physical,'--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
    if len(values)!=1:raise ValueError('One selected physical device required')
    uuid,free,total=[field.strip() for field in values[0].split(',')]
    properties=torch.cuda.get_device_properties(index)
    return {'physical_uuid':uuid,'logical_device':'cuda:'+str(index),'selector':physical,
            'CUDA_VISIBLE_DEVICES':visible,'name':properties.name,'capability':[properties.major,properties.minor],
            'total_bytes':int(total)*1024**2,'free_bytes':int(free)*1024**2}


def owned_stop(child,ticks,deadline):
    if child.poll() is not None:return
    if start_ticks(child.pid)!=ticks or os.getpgid(child.pid)!=child.pid:
        raise RuntimeError('Only the exact owned child group may be stopped')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=max(0,min(5,deadline-time.monotonic())))
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        try:child.wait(timeout=max(0,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','source','train','valid','work-check','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--hard-seconds',type=int,default=32400)
    parser.add_argument('--memory-wait-seconds',type=int,default=21600)
    args=parser.parse_args();began=time.monotonic()
    paths=[args.project_root,args.source,args.train,args.valid,args.work_check,args.output]
    if any(not path.is_absolute() for path in paths):raise ValueError('Explicit absolute project paths required')
    project=args.project_root.resolve()
    if any(not path.resolve().is_relative_to(project) for path in paths[1:]):raise ValueError('Keep source/data/check/output inside project root')
    if args.hard_seconds<30 or args.memory_wait_seconds<0:raise ValueError('Finite prospective caps required')
    source=args.source.resolve();manifest=json.loads((source/'MANIFEST.json').read_text());source_sha=sha(source/'MANIFEST.json')
    for row in manifest['files']:
        path=source/row['path']
        if not path.resolve().is_relative_to(source) or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Source seal changed')
    recipe=json.loads((source/'recipes/molhiv.json').read_text())
    roster=[{'task':'molhiv','arm':arm,'seed':seed,'cell':arm+'_'+str(seed)} for seed in recipe['pilot_seeds'] for arm in recipe['arms']]
    if len(roster)!=24:raise ValueError('Frozen 8-arm/3-seed family required')
    data={'train_npz_sha256':sha(args.train),'valid_npz_sha256':sha(args.valid)}
    runtime=runtime_identity();device=gpu_observation(args.device);check=json.loads(args.work_check.read_text())
    if check.get('schema')!='portable-representative-work-v1' or not check.get('complete') or check.get('exit_code')!=0 or not check.get('reaped'):
        raise ValueError('Actual complete representative work receipt required')
    if check['task']!='molhiv' or check['source_manifest_sha256']!=source_sha or check['data']!=data or check['runtime']!=runtime or check['physical_gpu_uuid']!=device['physical_uuid']:
        raise ValueError('Representative source/data/runtime/device differs')
    required={'single','be_init_contrastive','independent4'}
    if not required.issubset({row['arm'] for row in check['cases']}):raise ValueError('Representative single/joint/own branches required')
    if any(not row['complete_TRAIN_update_VALID_snapshot_reload_serving'] for row in check['cases']):raise ValueError('Incomplete representative work')
    peak=check['peak_GPU_bytes']
    if type(peak) is not int or peak<=0:raise ValueError('Measured positive GPU peak required')
    minimum=peak+2*1024**3
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'cells').mkdir()
    owner={'PID':os.getpid(),'start_ticks':start_ticks(os.getpid()),'hostname':socket.gethostname(),
           'python_executable':sys.executable,'project_root':str(project),'source_directory':str(source),
           'source_manifest_sha256':source_sha,'data':data,'runtime':runtime,'physical_device':device,
           'representative_work_sha256':sha(args.work_check),'hard_seconds':args.hard_seconds,
           'active_seconds':args.hard_seconds-10,'cleanup_seconds':10,'minimum_free_GPU_bytes':minimum}
    write(args.output/'OWNER.json',owner);write(args.output/'ROSTER.json',roster)
    rows=[];fatal=None
    try:
        for cell in roster:
            row={**cell,'status':'unlaunched','automatic_retry':False};cell_started=time.monotonic();child=None;ticks=None;deadline=None
            try:
                wait_started=time.monotonic()
                while True:
                    observed=gpu_observation(args.device)
                    if observed['physical_uuid']!=device['physical_uuid'] or runtime_identity()!=runtime:raise RuntimeError('Family device/runtime changed')
                    row['memory_observation']=observed
                    if observed['free_bytes']>=minimum:break
                    if time.monotonic()-wait_started>=args.memory_wait_seconds:raise TimeoutError('Finite memory wait expired')
                    time.sleep(20)
                row['memory_wait_seconds']=time.monotonic()-wait_started
                output=args.output/'cells'/cell['cell']
                command=[sys.executable,'-B',str(source/'train.py'),'--task','molhiv','--arm',cell['arm'],
                         '--seed',str(cell['seed']),'--device',args.device,'--train',str(args.train),'--valid',str(args.valid),'--output',str(output)]
                before=resource.getrusage(resource.RUSAGE_CHILDREN);fit_started=time.monotonic();deadline=fit_started+args.hard_seconds
                with (args.output/'cells'/(cell['cell']+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=project,env=dict(os.environ),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
                    ticks=start_ticks(child.pid);row.update(PID=child.pid,start_ticks=ticks,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    write(args.output/'RUNNING_CELL.json',row)
                    try:code=child.wait(timeout=args.hard_seconds-10)
                    except subprocess.TimeoutExpired:owned_stop(child,ticks,deadline);code=child.poll();row['status']='retained_fit_timeout'
                usage=resource.getrusage(resource.RUSAGE_CHILDREN)
                row.update(exit_code=code,reaped=code is not None,inclusive_fit_seconds=time.monotonic()-fit_started,
                           CPU_user_seconds=usage.ru_utime-before.ru_utime,CPU_system_seconds=usage.ru_stime-before.ru_stime,
                           cumulative_RUSAGE_CHILDREN_peak_RSS_bytes=usage.ru_maxrss*1024,
                           cap_overrun=time.monotonic()-fit_started>args.hard_seconds)
                if row['inclusive_fit_seconds']>args.hard_seconds:row['status']='retained_cap_overrun'
                elif row['status']!='retained_fit_timeout':
                    complete=output/'COMPLETE.json'
                    if code==0 and complete.is_file():
                        record=json.loads(complete.read_text())
                        if not record['complete'] or record['epochs']!=100 or record['steps']!=25800:raise RuntimeError('Incomplete horizon')
                        row.update(status='complete',completion_sha256=sha(complete),selected_sha256=record['selected_sha256'])
                    else:row['status']='retained_fit_failure'
                if code is None:
                    row['status']='retained_unreaped_child';fatal='UnreapedOwnedChild'
            except TimeoutError as error:row.update(status='retained_memory_wait_failure',error=str(error))
            except BaseException as error:
                if child is not None and child.poll() is None:
                    owned_stop(child,ticks,deadline if deadline is not None else time.monotonic()+10)
                    row.update(exit_code=child.poll(),reaped=child.poll() is not None)
                row.update(status='retained_queue_failure',error_type=type(error).__name__,error=str(error));fatal=type(error).__name__
            row['inclusive_cell_seconds']=time.monotonic()-cell_started;rows.append(row)
            write(args.output/'CELL_STATUS.json',rows)
            if fatal:break
    finally:
        completed={row['cell'] for row in rows}
        rows.extend({**cell,'status':'unlaunched_after_queue_failure','automatic_retry':False} for cell in roster if cell['cell'] not in completed)
        write(args.output/'CLOSURE.json',{'family_accounted':len(rows)==24,'all_fits_complete':all(row['status']=='complete' for row in rows),
              'task':'molhiv','cells':rows,'inclusive_family_seconds':time.monotonic()-began,'fatal_error_type':fatal,
              'comparative_analysis_performed':False,'TEST_scoring':False,'automatic_retry_or_resume':False,'owner':owner})
    if fatal:raise SystemExit(1)


if __name__=='__main__':main()
