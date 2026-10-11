"""Finite whole peer-family readout; no partial metric access or model retry."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
PEER_MANIFEST_SHA='f83cb34bae29bda7bd542075bb3009d973fc7ab3bf494bd2cdfcf0801e704cd4'
BLOCKS=(10103,10427,10949)
ARMS=('P1','F1','I4','S4','PA1','IA4','SA4','SM4','SX4','UX4')

def require(ok,message):
    if not ok:raise ValueError(message)

def route():
    require(socket.gethostname()=='anogena-2-0','Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).splitlines()==[UUID],'Excluded wrong sole GPU')
    require(Path.cwd()==REPO,'Run from the authorized allocation repository')

def receipt(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def checked(item):
    require(receipt(item['path'])==item,'Bound metadata receipt differs: '+item['path'])
    return Path(item['path'])

def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def identity(pid):
    path=Path('/proc')/str(pid)/'stat'
    if not path.exists():return None
    fields=path.read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]))

def owner_ok(path,stage,start=None):
    value=json.loads(Path(path).read_text())
    require(value['stage']==stage and value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and value['direct_child_wait'] and value['child_pid_absent'] and value['owned_cuda_pid_absent'] and not value['timed_out'],'Prior owner must close successfully and wait/reap its worker')
    if start is not None:
        require(value['owner']==start['owner'] and value['child']==start['child'] and value['command']==start['command'],'Owner completion must match admitted process identity')
    require(identity(value['child']['pid'])!=value['child'],'Owned worker remains live despite completion receipt')
    return value

def source_check(directory,item):
    manifest=json.loads(checked(item).read_text())
    for rel,fixed in manifest['files'].items():
        actual=receipt(Path(directory)/rel)
        require((actual['bytes'],actual['sha256'])==(fixed['bytes'],fixed['sha256']),'Sealed source differs: '+rel)
    return manifest

def reader_metadata_helpers(source):
    # Both modules are stdlib-only at import. close_panel reads only a complete
    # owner/panel plus checkpoint and label-free export byte receipts.
    spec=importlib.util.spec_from_file_location('runtime',source/'runtime.py')
    runtime=importlib.util.module_from_spec(spec);sys.modules['runtime']=runtime;spec.loader.exec_module(runtime)
    spec=importlib.util.spec_from_file_location('peer_watched_reader',source/'reader.py')
    reader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reader;spec.loader.exec_module(reader)
    return reader

def endpoint_closure(path,base):
    value=json.loads(Path(path).read_text());rows=value['records'];index={(r['block'],r['arm']):r for r in rows}
    require(value['complete'] and value['endpoints']==30 and value['actual_scalar_updates']==15000 and len(rows)==30 and set(index)=={(b,a) for b in BLOCKS for a in ARMS} and value['panel']==base['panel'] and value['role_custody']==base['custody'] and value['DEV_truth_read'] is False,'Complete exact30 endpoints/15000updates before DEV')
    require(all(r['status']=='fixed_endpoint_complete' and r['updates']==500 and r['endpoint']=='latest_tied_minimum_step0_to500' and type(r['selected_step']) is int and 0<=r['selected_step']<=500 and math.isfinite(r['logT']) and -5<=r['logT']<=5 and r['temperature']==math.exp(r['logT']) and r['operator']=='mean_m softmax(member_log_probs_m/T)' for r in rows),'Frozen inside-member temperature and latest-tied endpoint')
    return value

def terminate_owned(proc,birth,owner_output):
    """Bounded cleanup only for this watcher's own stage owner and its worker."""
    cleanup=dict(owner=birth,worker=None,TERM_sent=False,KILL_sent=False)
    start_path=owner_output/'OWNER_START.json'
    if start_path.exists():
        started=json.loads(start_path.read_text())
        if started.get('owner')==birth:cleanup['worker']=started.get('child')
    worker=cleanup['worker']
    if worker and identity(worker['pid'])==worker:
        os.killpg(worker['pgid'],signal.SIGTERM);cleanup['TERM_sent']=True
    # Wait on the direct owner while it reaps its worker; if the owner already
    # exited, also give its identified worker a bounded TERM/KILL cleanup.
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and ((worker and identity(worker['pid'])==worker) or proc.poll() is None):
        time.sleep(.25)
    if worker and identity(worker['pid'])==worker:
        os.killpg(worker['pgid'],signal.SIGKILL);cleanup['KILL_sent']=True
    if proc.poll() is None and identity(proc.pid)==birth:os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if identity(proc.pid)==birth:os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=5)
    cleanup.update(owner_waited=True,owner_pid_absent=identity(proc.pid)!=birth,worker_pid_absent=None if worker is None else identity(worker['pid'])!=worker)
    return cleanup

def run_owner(source,config,path,output):
    write(path,config);stage=config['stage'];cap=config['limits']['wall_seconds'];owner_output=Path(config['owner_output'])
    argv=[str(REPO/'.venv/bin/python'),'-B','-u',str(source/'owner.py'),str(path)]
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    transport=dict(stage=stage,config=receipt(path),command=argv,child_wall_seconds=cap,owner_wait_seconds=cap+65,no_retry=True)
    with (output/(stage+'_owner_transport.log')).open('xb') as log:
        proc=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        birth=identity(proc.pid);require(birth and birth['pgid']==proc.pid,'Owned direct stage owner identity')
        transport['owner']=birth
        try:
            code=proc.wait(timeout=cap+65);transport.update(exit_code=code,owner_waited=True,owner_pid_absent=identity(proc.pid)!=birth)
            require(code==0 and transport['owner_pid_absent'],'Stage owner failed; retain logs and stop')
            started=json.loads((owner_output/'OWNER_START.json').read_text());require(started['owner']==birth,'Actual stage owner start identity')
            owner_ok(owner_output/'OWNER_END.json',stage,started)
            transport.update(complete=True,owner_end=receipt(owner_output/'OWNER_END.json'))
        except BaseException as error:
            transport.update(complete=False,failure=dict(type=type(error).__name__,message=str(error)))
            transport['cleanup']=terminate_owned(proc,birth,owner_output)
            raise
        finally:write(output/(stage+'_OWNER_TRANSPORT.json'),transport)
    return transport

def main():
    route();parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);args=parser.parse_args()
    config_path=args.config.resolve();config=json.loads(config_path.read_text())
    require(config.get('execution_enabled') is True,'Root must enable this exact automatic readout')
    output=Path(config['output_directory']);output.mkdir(parents=True,exist_ok=False)
    stage='admission';started=time.monotonic();result=dict(complete=False,last_stage=stage,partial_metrics_opened=False,partial_owner_logs_opened=False,model_retry=False,readout_retry=False,scientific_gate_used_for_handoff=False,stage_owners=[])
    try:
        root_decision=receipt(checked(config['root_decision']));source_check(Path(__file__).resolve().parent,config['watcher_manifest'])
        require(all(type(config[k]) is int and config[k]>0 for k in ('wait_seconds','poll_seconds','whole_watch_wall_seconds')) and config['poll_seconds']<=60,'Finite wait and bounded polling required')
        for limits in (config['calibrate_limits'],config['score_limits']):
            require(set(limits)=={'wall_seconds','cuda_bytes','rss_bytes'} and all(type(v) is int and v>0 for v in limits.values()),'Positive finite reader/owner caps')
        require(config['whole_watch_wall_seconds']>=config['wait_seconds']+config['calibrate_limits']['wall_seconds']+config['score_limits']['wall_seconds']+180,'Inclusive finite watch envelope')
        def timeout(signum,frame):raise TimeoutError('Finite whole-family watcher wall exceeded')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(config['whole_watch_wall_seconds'])
        acq=json.loads(checked(config['acquisition_config']).read_text())
        require(acq['execution_enabled'] and acq['stage']=='acquire' and acq['blocks']==list(BLOCKS) and acq['arms']==list(ARMS) and acq['source_manifest']['sha256']==PEER_MANIFEST_SHA,'Exact admitted peer V3 family')
        source=Path(acq['source_manifest']['path']).parent;source_check(source,acq['source_manifest']);checked(acq['custody'])
        owner_start=json.loads(checked(config['acquisition_owner_start']).read_text())
        require(owner_start['stage']=='acquire' and owner_start['command'][-1]==config['acquisition_config']['path'] and owner_start['command'][-2]==str(source/'worker.py'),'Actual acquisition start provenance')
        acquisition_owner=Path(acq['owner_output'])/'OWNER_END.json';panel_path=Path(acq['output'])/'COMPLETE_PANEL.json'
        result.update(config=receipt(config_path),root_decision=root_decision,watcher_manifest=config['watcher_manifest'],peer_source_manifest=acq['source_manifest'],acquisition_config=config['acquisition_config'],acquisition_owner_start=config['acquisition_owner_start'],custody=acq['custody'],finite_wall_seconds=config['whole_watch_wall_seconds'],model_qualification_repeated=False)
        stage='waiting_for_whole_acquisition';write(output/'WATCH_START.json',dict(result,pid=os.getpid(),stage=stage))
        deadline=time.monotonic()+config['wait_seconds']
        while not acquisition_owner.exists():
            require(time.monotonic()<deadline,'Finite acquisition closure wait exceeded')
            require(identity(owner_start['owner']['pid'])==owner_start['owner'],'Acquisition owner ended without closure receipt; stop without reading partial results')
            time.sleep(min(config['poll_seconds'],max(.01,deadline-time.monotonic())))
        owner_ok(acquisition_owner,'acquire',owner_start)
        base=dict(execution_enabled=True,blocks=list(BLOCKS),arms=list(ARMS),root_decision=root_decision,finite_external_owner=True,source_manifest=acq['source_manifest'],expected_runtime_versions=acq['expected_runtime_versions'],custody=acq['custody'],panel=receipt(panel_path),acquisition_owner_end=receipt(acquisition_owner))
        reader=reader_metadata_helpers(source);reader.close_panel(base)
        write(output/'WHOLE_FAMILY_CLOSURE.json',dict(complete=True,banks=30,optimizer_trajectories=48,physical_backbones=57,exports=30,panel=base['panel'],owner_end=base['acquisition_owner_end'],source_manifest=base['source_manifest'],no_partial_metric_access=True))
        for stage in ('calibrate','score'):
            source_check(source,acq['source_manifest'])
            limits=config[stage+'_limits'];require(time.monotonic()-started+limits['wall_seconds']+90<config['whole_watch_wall_seconds'],'Insufficient remaining finite watch budget; no stage launch')
            action=dict(base,stage=stage,output=str(output/(stage+'_v1')),owner_output=str(output/(stage+'_owner_v1')),limits=limits)
            if stage=='score':
                owner_ok(output/'calibrate_owner_v1'/'OWNER_END.json','calibrate')
                endpoint_closure(output/'calibrate_v1'/'TEMPERATURES.json',base)
                action.update(temperatures=receipt(output/'calibrate_v1'/'TEMPERATURES.json'),calibration_owner_end=receipt(output/'calibrate_owner_v1'/'OWNER_END.json'))
                write(output/'ALL_CAL_CLOSURE.json',dict(complete=True,endpoints=30,scalar_updates=15000,CAL_owners_closed=[action['calibration_owner_end']],CAL_owner_waited_and_reaped=True,temperatures=action['temperatures'],no_DEV_truth_access_by_watcher=True))
            write(output/'WATCH_PROGRESS.json',dict(stage=stage,whole_family_closed=True,partial_metrics_opened=False))
            transport=run_owner(source,action,output/(stage.upper()+'_CONFIG.json'),output);result['stage_owners'].append(transport['owner_end'])
        stage='complete_report_receipt'
        report=output/'score_v1'/'DEVELOPMENT_REPORT.json';closed=json.loads(report.read_text())
        require(closed['complete'] and len(closed['records'])==30 and closed['calibration_scalar_updates']==15000 and closed['release']['one_whole_release'] and closed['release']['blocks']==list(BLOCKS),'Whole DEV report closed; no survivor subset')
        result.update(complete=True,report=receipt(report),all30banks_48states_57bodies_before30CAL_then_onewholeDEV=True,all_CAL_owners_closed_before_DEV=True,scientific_gate_results_not_used_for_success=True)
    except BaseException as error:
        result['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        signal.alarm(0);result.update(last_stage=stage,seconds=time.monotonic()-started);write(output/'WATCH_END.json',result)
    print(json.dumps(dict(complete=result['complete'],last_stage=stage,watch_end=str(output/'WATCH_END.json'),report=result.get('report'),partial_metrics_opened=False)),flush=True)
    raise SystemExit(0 if result['complete'] else 1)
if __name__=='__main__':main()
