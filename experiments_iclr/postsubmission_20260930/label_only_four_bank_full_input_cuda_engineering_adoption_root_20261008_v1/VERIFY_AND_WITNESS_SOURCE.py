from pathlib import Path
import socket,subprocess,json,datetime,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_full_input_cuda_engineering_adoption_root_20261008_v1';E=P/'label_only_four_bank_full_input_cuda_engineering_execution_root_20261008_v1/run01'
assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def observe(pid):
 f=Path('/proc',str(pid),'stat')
 if not f.exists():return None
 s=f.read_text();v=s[s.rfind(')')+2:].split();return dict(pid=pid,start_ticks=int(v[19]),state=v[0])
a=json.loads((E/'OWNER.json').read_text());r=json.loads((E/'ENGINEERING_RESULT.json').read_text());t=json.loads((E/'TERMINAL.json').read_text())
assert a['parent_PID']==533324 and a['parent_start_ticks']==6029072651
assert a['worker_PID']==533325 and a['worker_start_ticks']==6029072661
assert r['engineering_passed'] and all(r['checks'].values()) and t['outcome']=='engineering_passed' and t['return_code']==0 and t['worker_reaped']
assert r['scientific_accuracy_endpoints']==False and r['VALID_truth_scoring']==False and r['selector_performed']==False
current={'parent':observe(533324),'worker':observe(533325)}
assert current['parent'] is None and current['worker'] is None
w=dict(schema='label-four-bank-engineering-closed-absence-witness-v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),hostname=socket.gethostname(),gpu_uuid='GPU-44039938-fd82-41d2-fefd-de71514e2fac',recorded_parent={'pid':533324,'start_ticks':6029072651},recorded_worker={'pid':533325,'start_ticks':6029072661},observed=current,source_commit='53e73cf737b15385ece81d3cb6b12212fb1e5db1',quality_scores_read=False,source_receipts={name:{'relative_path':str((E/name).relative_to(P)),'sha256':hashlib.sha256((E/name).read_bytes()).hexdigest(),'bytes':(E/name).stat().st_size} for name in ('OWNER.json','ENGINEERING_RESULT.json','TERMINAL.json')})
D.mkdir(exist_ok=True)
with (D/'ABSENCE_WITNESS.json').open('x') as h:json.dump(w,h,indent=2);h.write('\n')
print(json.dumps(w))
