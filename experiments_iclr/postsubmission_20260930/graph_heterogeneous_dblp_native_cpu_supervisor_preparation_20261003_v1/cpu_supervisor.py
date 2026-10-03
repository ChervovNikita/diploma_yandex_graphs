"""One owned native-v2 CPU training child; bounded transport only, no scoring."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CHILD_BOOTSTRAP='import runpy,sys,torch; assert torch.__version__=="2.1.2+cu118"; torch.set_num_threads(1); torch.set_num_interop_threads(1); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'


class Interrupted(Exception):pass


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):digest.update(data)
    return digest.hexdigest()


def verify(record):
    path=Path(record['path']);require(sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Custody differs: '+str(path));return path


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def preservation(binding,manifest_sha,release_record):
    require(sha(HERE/'MANIFEST.json')==manifest_sha,'Supervisor manifest changed')
    for row in json.loads((HERE/'MANIFEST.json').read_text())['payload']:verify(dict(row,path=str(HERE/row['path'])))
    for row in binding['source_records']+[binding['freeze'],binding['paired_HGT_freeze'],binding['qualification_transport'],binding['qualification_result'],release_record]:verify(row)


def admission(binding,frozen,release,manifest_sha,run_name):
    require(release['execution_authorized'] is True and release['CPU_fixture_passed'] is True
        and release['device']=='cpu' and release['mode']=='native_v2_serial_CPU_supervised' and release['run_name']==run_name
        and release['prepared_manifest_sha256']==binding['native_manifest_sha256']
        and release['study_freeze_sha256']==binding['freeze']['sha256']
        and release['paired_HGT_freeze_sha256']==binding['paired_HGT_freeze']['sha256']
        and release['supervisor_manifest_sha256']==manifest_sha
        and release['qualification_result_sha256']==binding['qualification_result']['sha256']
        and release['qualification_manifest_sha256']==binding['qualification_manifest_sha256'],'Exact root native CPU execution release required')
    for key,value in binding['limits'].items():require(release[key]==value,'Root CPU bound differs: '+key)
    require(frozen['seeds']==binding['seeds'] and frozen['arms']==binding['arms'],'Complete fifteen frozen cases required')
    driver=load('supervisor_exact_native_v2_guard',verify(binding['native_driver']))
    driver.guard(frozen,release,binding['freeze']['sha256'],binding['native_manifest_sha256'],json.loads(verify(binding['native_template']).read_text()))
    q=json.loads(verify(binding['qualification_result']).read_text());transport=json.loads(verify(binding['qualification_transport']).read_text())
    actual=transport['remote_receipt']
    require(transport['exit_code']==0 and actual['exit_code']==0 and actual['status']=='completed'
        and actual['qualification_result']==q and actual['original_sources_and_freezes_preserved'] is True
        and q['status']=='qualified' and q['originals_preserved'] is True and q['device']=='cpu'
        and q['native_study_freeze_sha256']==binding['freeze']['sha256']
        and q['native_prepared_manifest_sha256']==binding['native_manifest_sha256']
        and q['qualification_manifest_sha256']==binding['qualification_manifest_sha256']
        and [r['arm'] for r in q['rows']]==binding['arms'] and all(r['status']=='qualified' for r in q['rows'])
        and q['validation_or_test_scored'] is False and q['training_driver_main_called'] is False,
        'Actual all-three native CPU resource qualification required')


def screen(required):
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):memory['host_available_bytes']=int(line.split()[1])*1024
    try:
        maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if maximum!='max':memory['cgroup_available_bytes']=int(maximum)-int(Path('/sys/fs/cgroup/memory.current').read_text())
    except (OSError,ValueError):pass
    free=min(memory.values());affinity=len(os.sched_getaffinity(0));load=os.getloadavg()[0]
    return dict(status='passed' if free>=required and affinity>=1 and load<=.75*affinity else 'resource_deferred',
        free_bytes=free,required_free_bytes=required,affinity_CPUs=affinity,one_minute_load=load,**memory)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',required=True);parser.add_argument('--run-name',required=True)
    args=parser.parse_args(argv);require(Path(args.run_name).name==args.run_name and args.run_name not in ('','.','..'),'Fresh simple native run name')
    binding=json.loads((HERE/'BINDINGS.json').read_text());manifest_sha=sha(HERE/'MANIFEST.json')
    require(json.loads((HERE/'SEAL.json').read_text())['manifest_sha256']==manifest_sha,'Supervisor seal differs')
    release_path=Path(args.admission).resolve();release_bytes=release_path.read_bytes();release=json.loads(release_bytes)
    release_record=dict(path=str(release_path),sha256=hashlib.sha256(release_bytes).hexdigest(),bytes=len(release_bytes))
    preservation(binding,manifest_sha,release_record);frozen=json.loads(verify(binding['freeze']).read_text())
    admission(binding,frozen,release,manifest_sha,args.run_name)
    native=Path(binding['native_driver']['path']).parent;study=native/'runs'/args.run_name/'STUDY.json'
    require(not list((native/'runs').glob('*/STUDY_STARTED.json')),'No automatic native replacement/restart study')
    require(not list((HERE/'runs').glob('*/SUPERVISOR_STARTED.json')),'One fresh supervisor admission only')
    require(sys.version_info[:3]==(3,11,14),'Exact qualified Python runtime required')
    out=HERE/'runs'/args.run_name;out.mkdir(parents=True,exist_ok=False)
    record=dict(status='preparing',native_output=str(study.parent),root_release=release_record,
        supervisor_manifest_sha256=manifest_sha,native_manifest_sha256=binding['native_manifest_sha256'],
        scientific_functions_or_arm_order_changed=False,successful_subset_scored=False,GPU_computation=False,
        owned_child_pid=None,exit_code=None,canonical_study=None)
    child=None;started=time.monotonic();peak=0;vpeak=0;previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    def interrupt(number,frame):
        for s in previous:signal.signal(s,signal.SIG_IGN)
        raise Interrupted('Supervisor interrupted by signal '+str(number))
    for s in previous:signal.signal(s,interrupt)
    try:
        resource_screen=screen(binding['limits']['required_free_host_bytes']);record['preflight']=resource_screen
        if resource_screen['status']!='passed':record['status']='resource_deferred'
        else:
            write(out/'SUPERVISOR_STARTED.json',dict(root_release=release_record,supervisor_manifest_sha256=manifest_sha))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
            command=[sys.executable,'-B','-c',CHILD_BOOTSTRAP,str(verify(binding['native_driver'])),'--freeze',binding['freeze']['path'],
                '--admission',str(release_path),'--run-name',args.run_name,'--device','cpu']
            record['argv']=command;limit=binding['limits'];started=time.monotonic()
            with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
                mask=signal.pthread_sigmask(signal.SIG_BLOCK,(signal.SIGINT,signal.SIGTERM))
                def bounded():
                    signal.signal(signal.SIGINT,signal.SIG_DFL);signal.signal(signal.SIGTERM,signal.SIG_DFL)
                    signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                    resource.setrlimit(resource.RLIMIT_AS,(limit['address_space_limit_bytes'],)*2)
                try:
                    child=subprocess.Popen(command,cwd=binding['canonical_repo'],env=env,stdout=stdout,stderr=stderr,preexec_fn=bounded)
                    record['owned_child_pid']=child.pid
                    write(out/'OWNED_CHILD.json',dict(pid=child.pid,argv=command,preimport_RLIMIT_AS_bytes=limit['address_space_limit_bytes']))
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                record['status']='running'
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    try:
                        for line in Path('/proc',str(child.pid),'status').read_text().splitlines():
                            if line.startswith('VmRSS:'):peak=max(peak,int(line.split()[1])*1024)
                            if line.startswith('VmHWM:'):peak=max(peak,int(line.split()[1])*1024)
                            if line.startswith('VmPeak:'):vpeak=max(vpeak,int(line.split()[1])*1024)
                    except OSError:pass
                    if elapsed>limit['wall_budget_seconds'] or peak>limit['RSS_limit_bytes']:
                        record['status']='wall_budget_exhausted' if elapsed>limit['wall_budget_seconds'] else 'RSS_budget_exhausted';break
                    time.sleep(.5)
                if child.poll() is not None:record['status']='child_completed'
    except (Exception,KeyboardInterrupt) as error:
        record.update(status='supervisor_failed',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
    finally:
        try:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                record['exit_code']=child.returncode;record['owned_child_reaped']=True
            if study.is_file():record['canonical_study']=dict(path=str(study),sha256=sha(study),bytes=study.stat().st_size)
            try:preservation(binding,manifest_sha,release_record);record['originals_preserved']=True
            except Exception as error:record.update(status='original_preservation_failed',originals_preserved=False,preservation_error=str(error))
            record.update(wall_seconds=time.monotonic()-started,peak_observed_RSS_bytes=peak,peak_observed_virtual_bytes=vpeak,
                root_validates_complete_fifteen_outputs=True,canonical_study_created_or_modified_by_supervisor=False)
            write(out/'SUPERVISOR_RECEIPT.json',record)
        finally:
            for s,handler in previous.items():signal.signal(s,handler)
    print(json.dumps(dict(status=record['status'],exit_code=record['exit_code'],receipt=str(out/'SUPERVISOR_RECEIPT.json')),sort_keys=True))
    return 0 if record['status']=='child_completed' and record['exit_code']==0 and record.get('originals_preserved') is True and record['canonical_study'] is not None else 1


if __name__=='__main__':raise SystemExit(main())
