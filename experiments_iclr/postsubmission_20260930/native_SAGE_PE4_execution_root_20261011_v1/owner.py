"""One finite external owner for the fixed PE qualifier or three-bank fit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
UUID='GPU-44039938-fd82-41d2-fefd-de71514e2fac'

def now():return datetime.now(timezone.utc).isoformat()
def item(p):
    p=Path(p).resolve();assert p.is_relative_to(R)
    b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    assert not p.exists();p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def identity(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return None
    a=p.read_text().rsplit(')',1)[1].split()
    return dict(PID=pid,start_ticks=int(a[19]),state=a[0])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['qualify','fit'],required=True);a=ap.parse_args()
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==[UUID]
    H=Path(__file__).resolve().parent;assert H.is_relative_to(P) and Path.cwd()==R
    S=P/'native_SAGE_PE4_full_TRAIN_integration_source_20261011_v1'
    PE=P/'native_SAGE_PE4_gamma1_source_20261011_v1'
    E=P/'native_SAGE_endpoint_value_execution_root_20261011_v1'
    template=json.loads((PE/'CONFIG_TEMPLATE.json').read_text());ecfg=json.loads((E/'SCIENTIFIC_CONFIG_V1.json').read_text())
    for key in ('native_repo','common_routes','factors','train_npz','valid_npz','device'):template[key]=ecfg[key]
    mode=a.mode;side=H/(mode+'_owner_v1');side.mkdir(exist_ok=False)
    output=H/('qualify_v1' if mode=='qualify' else 'actual_PE_family_v1');assert not output.exists()
    limits=dict(wall_seconds=300 if mode=='qualify' else 7200,cuda_bytes=32*1024**3,rss_bytes=16*1024**3)
    runtime=E/'RUNTIME_FINGERPRINT_V1.json';fp=json.loads(runtime.read_text())
    for row in fp['python_sources'].values():assert item(row['path'])==row
    runtime_versions=json.loads(subprocess.check_output([str(R/'.venv/bin/python'),'-B','-c','import torch,torch_geometric,json;print(json.dumps(dict(torch_version=str(torch.__version__),pyg_version=str(torch_geometric.__version__))))'],text=True,cwd=R,timeout=30))
    source_paths=dict(runner=S/'run.py',contracts=S/'integration_contracts.py',qualifier=S/'qualify_full_train.py',
        F_owner=P/'SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py',
        PE_adapter=PE/'family_adapter.py',PE_packed=PE/'packed_sage.py',PE_protocol=PE/'PROTOCOL.json',
        common_routes=Path(template['common_routes']),factors=Path(template['factors']),
        models=R/'models.py',run_base=R/'run_base.py',run_common=R/'run_common.py',
        sage_conv=R/'.venv/lib/python3.11/site-packages/torch_geometric/nn/conv/sage_conv.py',
        mean_aggregation=R/'.venv/lib/python3.11/site-packages/torch_geometric/nn/aggr/basic.py',
        scatter=R/'.venv/lib/python3.11/site-packages/torch_geometric/utils/_scatter.py')
    sources={k:item(v) for k,v in source_paths.items()}
    roles={'train':item(template['train_npz'])}
    qualification=None;closure=None
    if mode=='fit':
        qpath=H/'qualify_v1/QUALIFICATION.json';q=json.loads(qpath.read_text());assert q['status']=='PASS'
        qend=json.loads((H/'qualify_owner_v1/OWNER_END.json').read_text());assert qend['success']
        endpath=E/'fit_owner_v1/OWNER_END.json';end=json.loads(endpath.read_text())
        assert end['scientific_success'] and end['exit_code']==0 and end['child_pid_absent'] and end['owned_cuda_pid_absent']
        assert (end['banks'],end['optimizer_bundles'],end['native_body_records'])==(15,24,24)
        cpath=H/'ENDPOINT_NATIVE_CLOSURE_V1.json'
        write(cpath,dict(scope='endpoint_VALUE_native_closure',native_fits_complete=True,banks=15,optimizer_bundles=24,native_body_records=24,owner_end=item(endpath),native_complete=item(E/'actual_family_v1/COMPLETE_FAMILY.json'),partial_predictive_metrics_read=False))
        closure=item(cpath);qualification=item(qpath);roles['valid']=item(template['valid_npz'])
        apath=H/'ROOT_FIT_ADMISSION_V1.json'
        write(apath,dict(enabled=True,scope='native_SAGE_PE4_gamma1_BASE_parameter_cap',protocol_sha256=sources['PE_protocol']['sha256'],full_TRAIN_qualification_pass=True,external_finite_supervision_admitted=True,qualification=qualification,endpoint_native_closure=closure,TEST_access=False))
        template['root_fit_admission']=item(apath)
    cfgpath=H/('QUALIFICATION_CONFIG_V1.json' if mode=='qualify' else 'SCIENTIFIC_CONFIG_V1.json');write(cfgpath,template)
    admission_path=H/(mode.upper()+'_HOST_RUNTIME_ADMISSION_V1.json')
    write(admission_path,dict(status='HOST_GPU_SOURCE_ROLES_RUNTIME_ADMITTED',hostname=socket.gethostname(),gpu_uuid=UUID,mode=mode,
        config_sha256=item(cfgpath)['sha256'],source_sha256={k:v['sha256'] for k,v in sources.items()},safe_role_sha256={k:v['sha256'] for k,v in roles.items()},
        runtime_fingerprint_sha256=item(runtime)['sha256'],**runtime_versions,UTC=now()))
    releasepath=H/(mode.upper()+'_RELEASE_V1.json')
    command=[str(R/'.venv/bin/python'),'-B',str(S/'run.py'),'--release',str(releasepath)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',DGLBACKEND='pytorch')
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    barrier="import os,sys;assert sys.stdin.buffer.readline()==b'GO\\n';os.execv(sys.argv[1],sys.argv[1:])"
    started=time.monotonic()
    with (side/'stdout.log').open('x') as out,(side/'stderr.log').open('x') as err:
        child=subprocess.Popen([str(R/'.venv/bin/python'),'-B','-c',barrier,*command],cwd=R,env=env,stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True)
        ident=identity(child.pid)
        ownpath=side/'EXTERNAL_OWNER.json'
        write(ownpath,dict(finite_pending_call_watchdog=True,child_pid=child.pid,supervisor_pid=os.getpid(),child_pgid=child.pid,watchdog_pid=os.getpid(),limits=limits,mode=mode))
        release=dict(enabled=True,mode=mode,config=item(cfgpath),output=str(output),sources=sources,roles=roles,
            host_runtime_admission=item(admission_path),external_owner=item(ownpath),limits=limits,qualification=qualification,
            endpoint_native_closure=closure,historical_cross_runtime_score_reuse=False)
        write(releasepath,release)
        write(side/'OWNER_START.json',dict(UTC=now(),owner=identity(os.getpid()),child=ident,command=command,release=item(releasepath),direct_child_wait=True,finite_seconds=limits['wall_seconds'],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
        child.stdin.write(b'GO\n');child.stdin.close();timed_out=False
        try:code=child.wait(timeout=limits['wall_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    cuda=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
    record=output/('QUALIFICATION.json' if mode=='qualify' else 'CONSUMER_COMPLETE.json')
    result=json.loads(record.read_text()) if record.exists() else {}
    closed=identity(child.pid) is None and str(child.pid) not in [s.strip() for s in cuda]
    success=code==0 and not timed_out and closed and result.get('status')==('PASS' if mode=='qualify' else 'COMPLETE')
    end=dict(UTC=now(),mode=mode,success=success,exit_code=code,timed_out=timed_out,child=ident,child_pid_absent=identity(child.pid) is None,owned_cuda_pid_absent=str(child.pid) not in [s.strip() for s in cuda],direct_child_wait=True,seconds=time.monotonic()-started,record=item(record) if record.exists() else None,TEST_access=False)
    write(side/'OWNER_END.json',end)
    return 0 if success else 1

if __name__=='__main__':raise SystemExit(main())
