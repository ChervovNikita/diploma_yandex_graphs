"""Finite LM4 whole-acquisition handoff; metadata custody, no scientific gate."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
PHASE=REPO/'experiments_iclr/postsubmission_20260930'
HERE=Path(__file__).resolve().parent
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
SEEDS=(9101,9203,9307)
ARMS=('B1','B4','T16','S64','FS64','LM4','I4[S64]','I4[FS64]')
ENDPOINTS=('F1_b0','F1_b1','F1_b2','F1_b3')+ARMS+('GA4',)
SPEC_SHA='60ec176d581f693633c537be7b82af63d6795af51ec01ede716d3918d7193bf5'
ACQUISITION_MANIFEST_SHA='e2615703f0409d7f38176ef22024f74d48823b2ac393f2218a9e311887794b90'
READER_MANIFEST_SHA='641e9f899402c5c979923ae4fe91f221b66b900eeac7dd2daa396b35188ec100'
LIMITS=dict(wall_seconds=7200,cuda_bytes=34359738368,rss_bytes=17179869184)
WAIT_SECONDS=604800
WHOLE_SECONDS=619380
CLEANUP_RESERVE=20
ROOT_FIELDS=('watcher_manifest','acquisition_config','acquisition_owner_start','acquisition_launch',
    'expected_acquisition_owner','expected_acquisition_child','reader_source_manifest',
    'prepare_config','prepare_owner_start','prepare_owner_end','prepared_custody',
    'qualification_config','qualification_owner_start','qualification_owner_end','qualification',
    'output_directory','wait_seconds','whole_watch_wall_seconds','poll_seconds','calibrate_limits','score_limits')


def require(ok, message):
    if not ok:
        raise ValueError(message)

def route():
    require(socket.gethostname() == 'anogena-2-0', 'Excluded wrong hostname')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        text=True, timeout=15).splitlines() == [UUID], 'Excluded wrong sole GPU')
    require(Path.cwd() == REPO, 'Authorized allocation repository required')

def receipt(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'Receipt outside authorized project')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            digest.update(part)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def identity(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return None
    fields = path.read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]), pgid=int(fields[2]))

def source_check(directory, item):
    directory = Path(directory).resolve()
    require(Path(item['path']).resolve() == directory / 'MANIFEST.json', 'Manifest/source directory linkage')
    manifest = json.loads(checked(item).read_text())
    for rel, fixed in manifest['files'].items():
        actual = receipt(directory / rel)
        require((actual['bytes'], actual['sha256']) == (fixed['bytes'], fixed['sha256']),
            'Sealed source differs: ' + rel)
    return manifest

def terminate_owned(proc, birth, owner_output):
    """Adapted finite cleanup, only for this watcher's own readout processes."""
    cleanup = dict(owner=birth, worker=None, TERM_sent=False, KILL_sent=False)
    start_path = owner_output / 'OWNER_START.json'
    if start_path.exists():
        started = json.loads(start_path.read_text())
        if started.get('owner') == birth:
            cleanup['worker'] = started.get('child')
    worker = cleanup['worker']
    if worker and identity(worker['pid']) == worker:
        os.killpg(worker['pgid'], signal.SIGTERM)
        cleanup['TERM_sent'] = True
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and ((worker and identity(worker['pid']) == worker) or proc.poll() is None):
        time.sleep(.25)
    if worker and identity(worker['pid']) == worker:
        os.killpg(worker['pgid'], signal.SIGKILL)
        cleanup['KILL_sent'] = True
    if proc.poll() is None and identity(proc.pid) == birth:
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if identity(proc.pid) == birth:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
    cleanup.update(owner_waited=True, owner_pid_absent=identity(proc.pid) != birth,
        worker_pid_absent=None if worker is None else identity(worker['pid']) != worker)
    return cleanup


def checked(item):
    require(isinstance(item,dict) and set(item) in ({'path','sha256'},{'path','bytes','sha256'}),'Bound receipt schema')
    actual=receipt(item['path']);require(actual['sha256']==item['sha256'] and ('bytes' not in item or actual['bytes']==item['bytes']),'Bound file differs: '+item['path'])
    return Path(actual['path'])


def worker_command(source,config_path):
    binding=json.loads((source/'SOURCE_BINDINGS.json').read_text())['server_data_and_runtime']['runtime']
    return [str(checked(binding['python'])),'-B','-u',str(source/'worker.py'),str(config_path)]


def start_ok(value,stage,source,config_path,config,expected_owner=None,expected_child=None):
    require(value['stage']==stage and value['command']==worker_command(source,config_path)
        and value['config']==receipt(config_path) and value['source_manifest']==config['source_manifest']
        and value['root_decision']==config['root_decision'] and value['output']==config['output']
        and value['direct_child_wait'] is True and value['retry'] is False
        and value['child_wall_seconds']==config['limits']['wall_seconds']
        and value['external_wall_seconds']==config['limits']['wall_seconds']+60,'Exact producer stage/CONFIG/source/command/finite-owner linkage')
    for key,expected in (('owner',expected_owner),('child',expected_child)):
        actual=value[key];require(set(actual)=={'pid','start_ticks','pgid'} and all(type(v) is int and v>0 for v in actual.values()) and (expected is None or actual==expected),'Exact producer birth identity: '+key)
    require(value['child']['pgid']==value['child']['pid'],'Owned worker process group');return value


def owner_ok(path,stage,start,source,config_path,config):
    require(Path(path)==Path(config['owner_output'])/'OWNER_END.json','Exact producer owner output')
    value=json.loads(Path(path).read_text());start_ok(value,stage,source,config_path,config,start['owner'],start['child'])
    require(value['status']=='COMPLETE_EXIT_0' and value['exit_code']==0 and value['timed_out'] is False
        and value['child_pid_absent'] is True and value['owned_cuda_pid_absent'] is True,'Successful producer direct wait/reap required')
    require(identity(value['child']['pid'])!=value['child'],'Owned worker remains live')
    name=dict(prepare='CUSTODY.json',qualify='QUALIFICATION.json',acquire='PANEL.json',calibrate='TEMPERATURES.json',score='DEVELOPMENT_REPORT.json')[stage]
    require(value['produced_artifact']==receipt(Path(config['output'])/name),'Exact successful owner-produced artifact');return value


def prior_owner(config,stage,source):
    prefix='prepare' if stage=='prepare' else 'qualification'
    path=checked(config[prefix+'_config']);action=json.loads(path.read_text());start_path=checked(config[prefix+'_owner_start'])
    require(action['execution_enabled'] is True and action['stage']==stage and action['source_manifest']['sha256']==ACQUISITION_MANIFEST_SHA
        and start_path==Path(action['owner_output'])/'OWNER_START.json','Exact successful prior producer CONFIG/start')
    start=json.loads(start_path.read_text());start_ok(start,stage,source,path,action)
    end=owner_ok(checked(config[prefix+'_owner_end']),stage,start,source,path,action)
    artifact=config['prepared_custody'] if stage=='prepare' else config['qualification']
    require(end['produced_artifact']==artifact,'Exact prior role/qualifier receipt');checked(artifact);return action


def reader_helpers(source):
    # These sealed top levels are stdlib only. Metadata closure hashes opaque
    # files; it opens neither prediction arrays nor role targets here.
    for name in ('runtime','arithmetic','reader'):
        loaded='LM4_watched_reader' if name=='reader' else name
        spec=importlib.util.spec_from_file_location(loaded,source/(name+'.py'));module=importlib.util.module_from_spec(spec)
        sys.modules[loaded]=module;spec.loader.exec_module(module)
    return sys.modules['LM4_watched_reader']


def fill_template(source,stage,base,output,decision):
    action=json.loads((source/(stage.upper()+'_CONFIG.disabled.json')).read_text())
    require(action['execution_enabled'] is False and action['stage']==stage and action['seeds']==list(SEEDS)
        and action['endpoint_ids']==list(ENDPOINTS) and action['acquisition_source_manifest']==base['acquisition_source_manifest'],'Exact sealed disabled reader template')
    action.update(base,execution_enabled=True,stage=stage,root_decision=decision,limits=dict(LIMITS),output=str(output/(stage+'_v1')),owner_output=str(output/(stage+'_owner_v1')))
    return action


def run_owner(source,config,path,output):
    write(path,config);config_receipt=receipt(path);stage=config['stage'];cap=config['limits']['wall_seconds'];owner_output=Path(config['owner_output'])
    require(not Path(config['output']).exists() and not owner_output.exists(),'Fresh reader/owner outputs')
    argv=[str(REPO/'.venv/bin/python'),'-B','-u',str(source/'owner.py'),str(path)]
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env['PYTHONDONTWRITEBYTECODE']='1'
    transport=dict(stage=stage,config=config_receipt,command=argv,child_wall_seconds=cap,owner_wait_seconds=cap+65,no_retry=True)
    with (output/(stage+'_owner_transport.log')).open('xb') as log:
        proc=subprocess.Popen(argv,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        birth=identity(proc.pid);require(birth and birth['pgid']==proc.pid,'Owned direct readout owner identity');transport['owner']=birth
        try:
            code=proc.wait(timeout=cap+65);require(code==0 and identity(proc.pid)!=birth,'Reader owner failed; no retry')
            require(receipt(path)==config_receipt,'Reader CONFIG changed during stage')
            start_path=owner_output/'OWNER_START.json';start=json.loads(start_path.read_text());start_ok(start,stage,source,path,config,birth)
            owner_ok(owner_output/'OWNER_END.json',stage,start,source,path,config)
            transport.update(complete=True,exit_code=code,owner_waited=True,owner_pid_absent=True,owner_start=receipt(start_path),owner_end=receipt(owner_output/'OWNER_END.json'))
        except BaseException as error:
            transport.update(complete=False,failure=dict(type=type(error).__name__,message=str(error)),cleanup=terminate_owned(proc,birth,owner_output));raise
        finally:write(output/(stage+'_OWNER_TRANSPORT.json'),transport)
    return transport


def temperatures_closed(path,action,config_item,reader,index,prepared):
    value=json.loads(Path(path).read_text());rows=value['records'];endpoints={(r['seed'],r['endpoint_id']):r for r in rows}
    require(value['complete'] is True and value['retry'] is False and value['endpoints']==39
        and len(rows)==len(endpoints)==39 and set(endpoints)==set(index)
        and value['actual_scalar_updates']==value['total_bound_scalar_updates']==19500
        and value['inherited_endpoints']==value['inherited_scalar_updates']==0
        and value['DEV_truth_read'] is False and value['TEST_read'] is False
        and value['panel']==action['panel'] and value['prepared_custody']==action['prepared_custody']
        and value['config']==config_item and value['custody']['stage']=='calibrate'
        and value['custody']['source_manifest']==action['source_manifest'] and value['custody']['root_decision']==action['root_decision'],'Whole39 fresh CAL endpoints bound to owned producer')
    for key,ep in endpoints.items():
        require(ep['inherited'] is False,'No inherited CAL endpoint');reader.endpoint_temperature(ep,index[key]['exports']['cal'],prepared['role_files']['cal'])
    require(sum(ep['updates'] for ep in rows)==19500,'All19500 fresh scalar updates before DEV');return value


def main():
    route();parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);args=parser.parse_args();config_path=args.config.resolve();config=json.loads(config_path.read_text())
    require(config.get('execution_enabled') is True and config.get('stage')=='watch','Root must enable exact whole-acquisition watcher')
    output=Path(config['output_directory']);require(output.is_absolute() and output.resolve().is_relative_to(REPO),'Project-local watcher output');output.mkdir(parents=True,exist_ok=False)
    stage='admission';started=time.monotonic();result=dict(complete=False,partial_metrics_opened=False,partial_owner_logs_opened=False,scientific_gate_used_for_handoff=False,training_or_qualification_launched=False,model_retry=False,readout_retry=False,stage_owners=[])
    try:
        source_check(HERE,config['watcher_manifest']);binding=json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
        require(config['reader_source_manifest']==binding['reader_source_manifest'],'Exact sealed reader binding')
        require(config['wait_seconds']==WAIT_SECONDS and config['whole_watch_wall_seconds']==WHOLE_SECONDS
            and type(config['poll_seconds']) is int and 1<=config['poll_seconds']<=60
            and config['calibrate_limits']==config['score_limits']==LIMITS,'Exact finite watcher/reader envelopes')
        def timeout(signum,frame):raise TimeoutError('Whole watch ended with bounded cleanup reserve')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(WHOLE_SECONDS-CLEANUP_RESERVE)
        decision_item=receipt(checked(config['root_decision']));decision=json.loads(checked(decision_item).read_text())
        require(set(('watch','calibrate','score')).issubset(decision.get('admitted_stages',[])) and decision.get('specification_sha256')==SPEC_SHA,'Explicit root whole-acquisition/readout admission')
        for key in ROOT_FIELDS:require(decision.get(key)==config[key],'Root decision must bind exact handoff: '+key)
        acq_path=checked(config['acquisition_config']);acq=json.loads(acq_path.read_text())
        require(acq['execution_enabled'] is True and acq['stage']=='acquire' and acq['seeds']==list(SEEDS)
            and acq['arms']==list(ARMS) and acq['source_manifest']==binding['acquisition_source_manifest']
            and acq_path.parent==Path(binding['intended_acquisition_root']),'Exact admitted acquisition CONFIG/source/root')
        acquisition_source=Path(acq['source_manifest']['path']).parent;source_check(acquisition_source,acq['source_manifest'])
        reader_source=Path(config['reader_source_manifest']['path']).parent;require(config['reader_source_manifest']['sha256']==READER_MANIFEST_SHA and reader_source!=acquisition_source,'Exact separate sealed reader');source_check(reader_source,config['reader_source_manifest'])
        require(Path(config['prepare_config']['path']).resolve().is_relative_to(Path(binding['activation_root'])),'Root-bound successful v2 preparation inside activation root')
        prepare=prior_owner(config,'prepare',acquisition_source);qualified=prior_owner(config,'qualify',acquisition_source)
        require(acq['prepared_custody']==config['prepared_custody'] and acq['prepare_owner_end']==config['prepare_owner_end']
            and acq['qualification']==config['qualification'] and acq['qualification_owner_end']==config['qualification_owner_end']
            and qualified['prepared_custody']==config['prepared_custody'],'Exact successful prior preparation/qualifier bound by acquisition')
        start_path=checked(config['acquisition_owner_start']);require(start_path==Path(acq['owner_output'])/'OWNER_START.json','Actual acquisition start output')
        start=json.loads(start_path.read_text());owner=config['expected_acquisition_owner'];child=config['expected_acquisition_child'];start_ok(start,'acquire',acquisition_source,acq_path,acq,owner,child)
        launch=json.loads(checked(config['acquisition_launch']).read_text());require(launch['stage']=='acquire' and launch['source_manifest']==acq['source_manifest']
            and launch['config_sha256']==config['acquisition_config']['sha256'] and launch['owner_spawn_pid']==owner['pid'] and launch['owner_start']==start,'Saved actual launch and bound live start agree')
        for key,birth in (('owner_observed',owner),('scientific_child_observed',child)):
            require(all(launch[key][k]==v for k,v in birth.items()),'Saved actual process birth: '+key)
        owner_end=Path(acq['owner_output'])/'OWNER_END.json';panel_path=Path(acq['output'])/'PANEL.json'
        result.update(config=receipt(config_path),root_decision=decision_item,watcher_manifest=config['watcher_manifest'],reader_source_manifest=config['reader_source_manifest'],acquisition_source_manifest=acq['source_manifest'],acquisition_config=config['acquisition_config'],acquisition_owner_start=config['acquisition_owner_start'],acquisition_launch=config['acquisition_launch'],finite_wall_seconds=WHOLE_SECONDS,cleanup_reserve_seconds=CLEANUP_RESERVE)
        stage='waiting_for_whole_acquisition';write(output/'WATCH_START.json',dict(result,pid=os.getpid(),stage=stage));deadline=time.monotonic()+WAIT_SECONDS
        while not owner_end.exists():
            require(time.monotonic()<deadline,'Finite acquisition closure wait exceeded');require(identity(owner['pid'])==owner,'Exact acquisition owner ended without closure; no partial reads')
            time.sleep(min(config['poll_seconds'],max(.01,deadline-time.monotonic())))
        checked(config['acquisition_config']);checked(config['acquisition_owner_start']);owner_ok(owner_end,'acquire',start,acquisition_source,acq_path,acq)
        base=dict(seeds=list(SEEDS),endpoint_ids=list(ENDPOINTS),finite_external_owner=True,source_manifest=config['reader_source_manifest'],expected_runtime_versions=acq['expected_runtime_versions'],acquisition_source_manifest=acq['source_manifest'],acquisition_config=config['acquisition_config'],acquisition_owner_start=config['acquisition_owner_start'],acquisition_owner_end=receipt(owner_end),panel=receipt(panel_path),prepare_config=config['prepare_config'],prepare_owner_start=config['prepare_owner_start'],prepare_owner_end=config['prepare_owner_end'],prepared_custody=config['prepared_custody'])
        reader=reader_helpers(reader_source);panel,index,prepared=reader.close_panel(base)
        require(panel['stage1_closed_before_stage2'] is True and len(panel['groups'])==36 and len(panel['trajectories'])==54 and len(index)==39,'Whole36/54/39 closure')
        write(output/'WHOLE_FAMILY_CLOSURE.json',dict(complete=True,fresh_banks=36,optimizer_trajectories=54,endpoints=39,inherited_fits_or_CAL=0,panel=base['panel'],acquisition_config=base['acquisition_config'],owner_end=base['acquisition_owner_end'],reader_source_manifest=base['source_manifest'],no_partial_metric_access=True))
        cal_transport=cal_action=None
        for stage in ('calibrate','score'):
            source_check(reader_source,config['reader_source_manifest']);require(time.monotonic()-started+LIMITS['wall_seconds']+90<WHOLE_SECONDS-CLEANUP_RESERVE,'Insufficient inclusive watch budget; no stage launch')
            action=fill_template(reader_source,stage,base,output,decision_item)
            if stage=='score':
                checked(cal_transport['config']);cal_start=json.loads(checked(cal_transport['owner_start']).read_text());owner_ok(checked(cal_transport['owner_end']),'calibrate',cal_start,reader_source,Path(cal_transport['config']['path']),cal_action)
                path=Path(cal_action['output'])/'TEMPERATURES.json';temperatures_closed(path,cal_action,cal_transport['config'],reader,index,prepared)
                action.update(temperatures=receipt(path),calibration_config=cal_transport['config'],calibration_owner_start=cal_transport['owner_start'],calibration_owner_end=cal_transport['owner_end'])
                write(output/'ALL_CAL_CLOSURE.json',dict(complete=True,endpoints=39,fresh_scalar_updates=19500,inherited_endpoints=0,inherited_scalar_updates=0,CAL_config=cal_transport['config'],CAL_owner_start=cal_transport['owner_start'],CAL_owner_end=cal_transport['owner_end'],CAL_owner_waited_and_reaped=True,temperatures=action['temperatures'],no_DEV_truth_access_by_watcher=True,score_handoff_unconditional=True))
            write(output/'WATCH_PROGRESS.json',dict(stage=stage,whole_family_closed=True,partial_metrics_opened=False));transport=run_owner(reader_source,action,output/(stage.upper()+'_CONFIG.json'),output);result['stage_owners'].append(transport['owner_end'])
            if stage=='calibrate':cal_transport,cal_action=transport,action
        stage='complete_report_receipt';report=Path(action['output'])/'DEVELOPMENT_REPORT.json';closed=json.loads(report.read_text())
        require(closed['complete'] is True and closed['retry'] is False and closed['survivor_subset'] is False and len(closed['records'])==39 and len(closed['comparisons'])==156
            and closed['calibration_scalar_updates']==19500 and closed['inherited_scalar_updates']==0 and closed['panel']==base['panel'] and closed['temperatures']==action['temperatures']
            and closed['release']['one_whole_release'] is True and closed['release']['seeds']==list(SEEDS)
            and closed['release']['after_fresh_fit_banks']==36 and closed['release']['after_fresh_optimizer_trajectories']==54
            and closed['release']['after_CAL_endpoints']==39 and closed['release']['after_scalar_CAL_updates']==19500
            and closed['release']['calibration_config']==cal_transport['config'] and closed['release']['calibration_owner_end']==cal_transport['owner_end']
            and closed['release']['custody']['stage']=='score' and closed['release']['custody']['root_decision']==decision_item
            and closed['release']['custody']['source_manifest']==config['reader_source_manifest'],'Complete exact whole DEV report from owned score producer')
        result.update(complete=True,report=receipt(report),all_CAL_owners_closed_before_DEV=True,scientific_gate_results_not_used_for_success=True)
    except BaseException as error:result['failure']=dict(type=type(error).__name__,message=str(error))
    finally:
        signal.alarm(0);result.update(last_stage=stage,seconds=time.monotonic()-started);write(output/'WATCH_END.json',result)
    print(json.dumps(dict(complete=result['complete'],last_stage=stage,watch_end=str(output/'WATCH_END.json'),report=result.get('report'),partial_metrics_opened=False)),flush=True);raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
