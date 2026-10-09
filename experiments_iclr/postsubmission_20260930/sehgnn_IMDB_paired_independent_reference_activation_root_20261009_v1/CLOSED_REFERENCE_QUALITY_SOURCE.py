from pathlib import Path
import json,socket,subprocess,datetime,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';A=P/'sehgnn_IMDB_paired_independent_reference_activation_root_20261009_v1'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
launch=json.loads((A/'LAUNCH.json').read_text());start=json.loads((A/'STARTER.json').read_text());cfg=json.loads((A/'RELEASE.json').read_text())
def present(h):
 f=Path('/proc',str(h['pid']),'stat')
 if not f.exists():return False
 t=f.read_text();v=t[t.rfind(')')+2:].split();assert int(v[19])==h['start_ticks'];return True
def bind(f):return dict(path=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
d=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_live=present(start['parent']),child_live=present(launch['child']),terminal_present=(A/'TERMINAL.json').exists(),quality_read=False)
if d['terminal_present']:
 t=json.loads((A/'TERMINAL.json').read_text());d['terminal']=t
 assert t['exit_code']==0 and t['child_reaped'] and t['original_pid_absent'] and t['stop_reason'] is None and not d['parent_live'] and not d['child_live']
 groups={start['parent']['group'],launch['child']['group']}
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:v=(q/'stat').read_text()
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  assert int(v[v.rfind(')')+2:].split()[2]) not in groups
 for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines():assert row.strip() not in {str(h) for h in groups}
 root=Path(cfg['entry_output_directory'])/'ROOT_REPORT.json';family=Path(cfg['family_output_directory']);report=json.loads(root.read_text());fr=json.loads((family/'FAMILY_REPORT.json').read_text())
 assert report['complete'] and report['complete_groups']==6 and report['completed_actual_body_fits']==24 and fr['complete'] and len(fr['cells'])==6 and all(c['complete'] for c in fr['cells'])
 d['reference_root_binding']=bind(root);d['reference_root']=report;d['family_report']=fr;d['quality_read']=True;d['cells']=[]
 for cell in fr['cells']:
  folder=Path(cell['output']);result=json.loads((folder/'RESULT.json').read_text());summary=json.loads((folder/'SELECTED_VALID_SUMMARY.json').read_text());assert result['complete']
  d['cells'].append(dict(role_seed=cell['role_seed'],variant=cell['variant'],fresh_metrics=result['fresh_metrics'],selected_epochs=result['selected_epochs'],member_F1=summary['full_member_F1'],pool_F1=summary['full_pool_F1'],member_BCE=summary['full_member_BCE'],pool_BCE=summary['full_pool_BCE'],result_binding=bind(folder/'RESULT.json'),summary_binding=bind(folder/'SELECTED_VALID_SUMMARY.json')))
 nr=P/'sehgnn_IMDB_literal_five_seed_native_reference_execution_root_20261009_v2/COHORT_REPORT.json';d['native5_report_binding']=bind(nr);d['native5_report_text']=nr.read_text()
else:
 f=Path(cfg['family_output_directory'])/'FAMILY_REPORT.json';r=json.loads(f.read_text());d['complete_groups']=sum(c['complete'] for c in r['cells']);d['expected_groups']=6
print(json.dumps(d))
