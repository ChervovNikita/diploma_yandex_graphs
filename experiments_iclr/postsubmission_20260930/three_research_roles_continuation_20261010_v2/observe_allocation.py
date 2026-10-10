"""Read owned process identities and completion metadata, omitting fit quality."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REMOTE = r'''
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,socket,subprocess
assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P=R/'experiments_iclr/postsubmission_20260930'
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot=='24c315a7-3c08-471f-b550-b9a3e1faf75d'
def ident(pid,ticks):
    q=Path('/proc',str(pid),'stat')
    if not q.is_file():return None
    f=q.read_text().rsplit(')',1)[1].split()
    assert int(f[19])==ticks
    return dict(PID=pid,start_ticks=int(f[19]),state=f[0],pgid=int(f[2]),sid=int(f[3]))
def read(q):return json.loads(q.read_text())
families={}
H=P/'private_cmcl_polyformer_root_preparation_20261010_v2'/'science'
progress=read(H/'FAMILY_PROGRESS.json') if (H/'FAMILY_PROGRESS.json').is_file() else {}
b=progress.get('active_child')
families['private_CMCL18']=dict(owner=ident(594139,6041323646),child=ident(b['PID'],b['start_ticks']) if b else None,
    completed_records=progress.get('completed',[]),current_record=progress.get('current_record'),
    family_complete=(H/'FAMILY_COMPLETE.json').is_file(),family_failure=(H/'FAMILY_FAILURE.json').is_file())
for label,stem,pid,ticks in [
    ('typed_reference24','typed_context_reference_science_activation_source_20261010_v1',601131,6041915125),
    ('typed_candidate18','typed_context_candidate_science_activation_source_20261010_v1',603792,6042126966)]:
    A=P/stem
    launch=read(A/'LAUNCH.json') if (A/'LAUNCH.json').is_file() else None
    release=read(A/'RELEASE.json')
    O=Path(release['output_directory'])
    assert O.resolve().is_relative_to(P)
    terminal=read(A/'TERMINAL.json') if (A/'TERMINAL.json').is_file() else None
    report=read(O/'COHORT_REPORT.json') if (O/'COHORT_REPORT.json').is_file() else {}
    c=launch['child'] if launch else None
    families[label]=dict(owner=ident(pid,ticks),child=ident(c['pid'],c['start_ticks']) if c else None,
        status=report.get('status'),complete=report.get('complete',False),completed_records=len(report.get('runs',[])),
        assembled_banks=len(report.get('ensembles',[])),
        terminal=terminal,launch=launch if label=='typed_candidate18' else None,
        quality_fields_omitted=True)
resources=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),boot_id=boot,families=families,
    aggregate_GPU_resources=resources,partial_quality_opened=False,
    HEAD=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
    canonical_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['RESEARCH_STATE.md','RESEARCH_WORKFLOW.md','PUBLIC_STATUS.md','research_ledger.json']})))
'''

def main():
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
          '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
          'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru','python3 -c '+shlex.quote(REMOTE)]
    r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (HERE/('TRANSPORT_'+tag+'.json')).open('x') as f:
        json.dump(dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),f,indent=2)
    if r.returncode:
        print(r.stderr)
        raise SystemExit(r.returncode)
    data=json.loads(r.stdout)
    with (HERE/('OBSERVATION_'+tag+'.json')).open('x') as f:json.dump(data,f,indent=2)
    print(json.dumps(data))

if __name__=='__main__':main()
