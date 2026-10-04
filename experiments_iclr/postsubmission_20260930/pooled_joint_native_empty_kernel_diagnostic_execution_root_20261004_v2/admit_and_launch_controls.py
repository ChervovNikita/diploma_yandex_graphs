"""Release four distinct fabricated diagnostic cases; never a qualification."""
from datetime import datetime, timezone
from pathlib import Path
import base64
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
SOURCE=PHASE/'pooled_joint_native_empty_kernel_diagnostic_preparation_20261004_v2'
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
REMOTE_PHASE=REPO/'experiments_iclr/postsubmission_20260930'
REMOTE=REMOTE_PHASE/HERE.name


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def main():
    assert sha(SOURCE/'MANIFEST.json')=='0e75b301e548704aee01cf42a3bfb8f2af5fe944bbe69ea6b50738d47f74da83'
    error=HERE/'owned_monitor01/exact/run01/FIRST_ERROR.json'
    value=json.loads(error.read_text())
    assert value['label']=='aten.addmm.default' and value['event']=='CALL_THROW_ERROR'
    release=json.loads((SOURCE/'ROOT_RELEASE_controls.example.json').read_text())
    release.update(execution_enabled=True,root_authorization_reference=str(REMOTE/'ROOT_CONTROL_ADMISSION.json'),preparation_manifest_sha256=sha(SOURCE/'MANIFEST.json'))
    admission=dict(UTC=datetime.now(timezone.utc).isoformat(),status='APPROVED_FOUR_DISTINCT_FABRICATED_CONTROLS',source_manifest_sha256=sha(SOURCE/'MANIFEST.json'),preserved_exact_error_sha256=sha(error),
        reason='The synchronized failure occurred in nonempty dense addmm; empty-sparse causation is unproven. Controls vary zero versus one query row and depth0 versus native overlap/spmm scope.',
        limits='Each original bounded source supervisor owns one fresh process. No numerical/model/kernel/profile changes, retries, datasets, VALID, TEST, checkpoints or scientific donors.',
        attribution_rule='A failure before native common-support/spmm markers cannot be attributed to empty spmm.',qualification_PASS_allowed=False,fits=0)
    save(HERE/'ROOT_CONTROL_ADMISSION.json',admission)
    save(HERE/'ROOT_RELEASE_controls.json',release)
    payload=[]
    for local in (HERE/'ROOT_CONTROL_ADMISSION.json',HERE/'ROOT_RELEASE_controls.json'):
        payload.append(dict(path=str(REMOTE/local.name),sha256=sha(local),data=base64.b64encode(local.read_bytes()).decode()))
    authority=json.loads((PHASE/'graph_ncNC_predictive_runtime_authority_root_20261003_v1/RUNTIME_AUTHORITY.json').read_text())
    environment=dict(CUDA_VISIBLE_DEVICES=release['cuda_visible_devices'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(REPO/'.gnnm_runtime/buddy_extra_v1/site'))
    jobs=[]
    for row in release['authorized_invocations']:
        command=[authority['interpreter_path'],'-B',str(REMOTE_PHASE/SOURCE.name/'diagnose.py'),'--case',row['case'],'--release',str(REMOTE/'ROOT_RELEASE_controls.json'),'--output',row['output_directory']]
        jobs.append(dict(case=row['case'],command=command,output_directory=row['output_directory']))
    code='from pathlib import Path\nfrom datetime import datetime,timezone\nimport base64,hashlib,json,os,subprocess\n'
    code+='repo=Path('+repr(str(REPO))+');phase=Path('+repr(str(REMOTE_PHASE))+');root=Path('+repr(str(REMOTE))+')\nassert Path.cwd()==repo and os.uname().nodename=="peptide"\n'
    code+='assert hashlib.sha256((phase/'+repr(SOURCE.name+'/MANIFEST.json')+').read_bytes()).hexdigest()=='+repr(sha(SOURCE/'MANIFEST.json'))+'\n'
    code+='jobs='+repr(jobs)+'\nfor job in jobs:assert not Path(job["output_directory"]).exists()\n'
    code+='q=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)\ngpu_rows=[[v.strip() for v in line.split(",")] for line in q.stdout.splitlines()];assert int(next(row[1] for row in gpu_rows if row[0]=='+repr(release['cuda_visible_devices'])+'))>=16384\n'
    code+='rows='+repr(payload)+'\nfor row in rows:\n path=Path(row["path"]);assert path.resolve().is_relative_to(phase);raw=base64.b64decode(row["data"],validate=True);assert hashlib.sha256(raw).hexdigest()==row["sha256"]\n with path.open("xb") as stream:stream.write(raw)\n'
    code+='environment='+repr(environment)+';launches=[]\nfor job in jobs:\n with (root/("DETACHED_"+job["case"]+".stdout")).open("xb") as out,(root/("DETACHED_"+job["case"]+".stderr")).open("xb") as err:\n  child=subprocess.Popen(job["command"],cwd=repo,env=dict(os.environ,**environment),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)\n raw=(Path("/proc")/str(child.pid)/"stat").read_text();v=raw[raw.rfind(")")+2:].split();launches.append(dict(job,supervisor_PID=child.pid,supervisor_start_ticks=int(v[19])))\n'
    code+='result=dict(UTC=datetime.now(timezone.utc).isoformat(),status="FOUR_SEPARATE_FABRICATED_CONTROLS_LAUNCHED",launches=launches,GPU_rows=gpu_rows,environment=environment,fits=0,qualification_PASS_allowed=False)\nwith (root/"CONTROL_DETACHED_LAUNCH.json").open("x") as stream:json.dump(result,stream,indent=2);stream.write("\\n")\nprint(json.dumps(result))\n'
    spec=importlib.util.spec_from_file_location('safe_transport',PHASE/'ncnc_heldout_wrapper_qualification_execution_root_20261004_v1/stage_and_launch_qa.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport);transport.HERE=HERE
    result=transport.run('pooled_native_controls_v2_launch_20261004',code)
    save(HERE/'CONTROL_DETACHED_LAUNCH.json',result)
    print(json.dumps({k:result[k] for k in ('UTC','status','launches')}))


if __name__=='__main__':
    main()
