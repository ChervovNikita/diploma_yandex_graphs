"""Start exactly one reviewed reference stage on the authorized allocation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REMOTE=r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,shutil,socket,subprocess,sys
assert socket.gethostname()=='anogena-2-0'
GPU='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()==[GPU]
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
H=P/'pubmed_factorized_I4_reference_source_20261010_v1';A=P/'factorized_I4_reference_root_activation_20261010_v1'
stage=sys.argv[1]
assert stage in ('admission','qualification','science','assembly','comparison')
assert hashlib.sha256((H/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()=='14a36fc9ec515245e6b8b4336d4f9f4eee7cc855deaa1d2c54d3929815761f5e'
target=A/stage;target.mkdir(exist_ok=False)
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
mem={r.split(':')[0]:int(r.split(':')[1].strip().split()[0])*1024 for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r}
assert mem['MemAvailable']>=16*1024**3 and shutil.disk_usage(P).free>=4*1024**3
readiness=H/('READINESS_'+stage+'.json');assert not readiness.exists()
readiness.write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),GPU_inventory=inventory,MemAvailable_bytes=mem['MemAvailable'],normal_host_execution=True,other_jobs_changed=False,overlapping_acquisition=True),indent=2)+'\n')
python=json.loads((H/'DATA_AND_RUNTIME.json').read_text())['runtime']['python']['path']
render=subprocess.run([python,'-B','-P',str(H/'render.py'),'--purpose',stage,'--readiness',str(readiness)],cwd=R,capture_output=True,text=True,timeout=60)
(target/'RENDER.json').write_text(json.dumps(dict(exit_code=render.returncode,stdout=render.stdout,stderr=render.stderr),indent=2)+'\n')
assert render.returncode==0,render.stderr
plan=H/(stage.upper()+'_OWNER_PLAN.json');digest=hashlib.sha256(plan.read_bytes()).hexdigest()
with (target/'owner.stdout').open('xb') as out,(target/'owner.stderr').open('xb') as err:
    owner=subprocess.Popen([python,'-B','-P',str(H/'owner.py'),'--plan',str(plan),'--plan-sha256',digest],cwd=R,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    f=Path('/proc',str(owner.pid),'stat').read_text().rsplit(')',1)[1].split()
    starter=dict(UTC=datetime.now(timezone.utc).isoformat(),stage=stage,PID=owner.pid,start_ticks=int(f[19]),pgid=int(f[2]),sid=int(f[3]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),source_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),source_manifest_sha256='14a36fc9ec515245e6b8b4336d4f9f4eee7cc855deaa1d2c54d3929815761f5e',owner_plan_sha256=digest,automatic_retry=False,normal_host_execution=True,TEST_access=False,other_jobs_changed=False,scientific_fits=9 if stage=='science' else 0)
    assert starter['pgid']==starter['sid']==owner.pid
    (target/'STARTER.json').write_text(json.dumps(starter,indent=2)+'\n')
print(json.dumps(dict(started=starter,retained={str(q.relative_to(P)):json.loads(q.read_text()) for q in (target/'STARTER.json',target/'RENDER.json',readiness,plan)})))
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('admission','qualification','science','assembly','comparison'),required=True)
    args=parser.parse_args()
    receipt=HERE/(args.stage+'_TRANSPORT.json')
    assert not receipt.exists()
    argv=['ssh','-tt','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt',
          '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=yes','-o','UpdateHostKeys=no',
          'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru',
          'python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(args.stage)]
    r=subprocess.run(argv,capture_output=True,text=True,timeout=90)
    receipt.write_text(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    if r.returncode:
        print(r.stderr)
        raise SystemExit(r.returncode)
    row=json.loads(r.stdout)
    for name,data in row['retained'].items():
        q=HERE/'fetched'/name
        assert q.resolve().is_relative_to((HERE/'fetched').resolve())
        q.parent.mkdir(parents=True,exist_ok=True)
        q.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(row['started']))

if __name__=='__main__':main()
