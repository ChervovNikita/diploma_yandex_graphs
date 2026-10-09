from pathlib import Path
import json,socket,subprocess,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_IMDB_paired_independent_reference_activation_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def observed(h):
 f=Path('/proc',str(h['pid']),'stat')
 if not f.exists():return dict(pid=h['pid'],present=False)
 s=f.read_text();v=s[s.rfind(')')+2:].split();assert int(v[19])==h['start_ticks'];return dict(pid=h['pid'],present=True,birth=int(v[19]),state=v[0])
l=json.loads((A/'LAUNCH.json').read_text());starter=json.loads((A/'STARTER.json').read_text());cfg=json.loads((A/'RELEASE.json').read_text())
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=observed(starter['parent']),child=observed(l['child']),actual_child_handle=l['child'],source_commit=l['source_commit'],terminal_present=(A/'TERMINAL.json').exists(),quality_scores_opened=False)
for key in ('family_output_directory','entry_output_directory'):
 q=Path(cfg[key]);v=dict(output=str(q),present=q.exists())
 f=q/('FAMILY_REPORT.json' if key=='family_output_directory' else 'ROOT_REPORT.json')
 if f.exists():
  r=json.loads(f.read_text());v['report']={k:r[k] for k in ('status','complete','failure','pairs','variants','reference_competence_pending','native_and_independent_reference_competence_pending','actual_body_fits') if k in r};v['cells']=[{k:c[k] for k in ('role_seed','base_seed','variant','status','complete') if k in c} for c in r.get('cells',[])]
 d[key]=v
if (A/'TERMINAL.json').exists():d['terminal']=json.loads((A/'TERMINAL.json').read_text())
d['worker_tail']=(A/'worker.stderr').read_text()[-2000:]
print(json.dumps(d))
