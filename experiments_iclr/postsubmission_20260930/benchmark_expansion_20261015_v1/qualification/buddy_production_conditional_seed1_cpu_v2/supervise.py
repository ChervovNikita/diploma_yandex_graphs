"""One finite production-provider conditional-interaction lifecycle; no retry."""
import argparse,datetime,hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path

def main():
    assert socket.gethostname()=='anogena-2-0'
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.output).resolve();q=Path(__file__).resolve().parent
    assert q.is_relative_to(root) and out.is_relative_to(root) and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for row in json.loads((q/'MANIFEST.json').read_text())['files']:
        raw=(q/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','started_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_limit_seconds':600,'automatic_retries':0,'TEST_accessed':False,'scientific_fits':0,'full_native_epochs':0,'bounds':{'native_constructions':2,'optimizer_updates':2,'full_VALID_calls':3,'native_forwards':32,'reverse_collections':2,'extra_reverse_collections':0}}
    def save():
        part=out/'SUPERVISOR.json.part';part.write_text(json.dumps(report,indent=2)+'\n');part.replace(out/'SUPERVISOR.json')
    save();start=time.perf_counter()
    with (out/'gate.log').open('x') as log:
        try:
            result=subprocess.run([sys.executable,'-B','-u',str(q/'qualify.py'),'--method','interaction_input_conditionalJ','--root',str(root),'--output',str(out/'gate')],stdout=log,stderr=subprocess.STDOUT,timeout=600,check=False)
            report.update(exit_code=result.returncode,status='passed' if result.returncode==0 else 'failed')
        except subprocess.TimeoutExpired:report.update(exit_code=None,status='failed',timeout=True)
    report.update(wall_seconds=time.perf_counter()-start,finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
