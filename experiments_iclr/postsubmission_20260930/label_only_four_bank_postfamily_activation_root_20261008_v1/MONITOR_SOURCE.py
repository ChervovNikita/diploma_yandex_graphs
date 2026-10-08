from pathlib import Path
import json,socket,subprocess,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';D=P/'label_only_four_bank_post_family_prediction_collection_root_20261008_v1';A=P/'label_only_four_bank_postfamily_activation_root_20261008_v1'
assert socket.gethostname()=='anogena-2-0';assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
out={}
f=Path('/proc/537031/stat')
if f.exists():
 v=f.read_text();a=v[v.rfind(')')+2:].split();out['process']=dict(pid=537031,start_ticks=int(a[19]),state=a[0]);assert int(a[19])==6029877103
else:out['process']=None
for n in ('PROGRESS.json','COMPLETE.json','FAILURE.json','COST_TERMINAL.json'):
 f=D/n
 if f.exists():out[n]=json.loads(f.read_text())
if out.get('process') is None or 'FAILURE.json' in out:out['log_tail']=(A/'collector.log').read_text()[-3500:]
out['native_states']=[]
for seed in (6101,6203,6307):
 f=P/'label_only_four_bank_first_screen_execution_root_20261008_v2'/('seed'+str(seed))/'NATIVE_OWN_BEST_ALL_BANKS.pt';h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 out['native_states'].append(dict(seed=seed,path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(out))
