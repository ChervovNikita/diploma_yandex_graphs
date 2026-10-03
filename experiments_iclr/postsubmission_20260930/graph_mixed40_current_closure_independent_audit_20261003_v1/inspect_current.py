"""Authorized mixed40 metadata/source descriptor inspection; no score payloads."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
assert json.loads((HERE/'ROUTE_VERIFICATION.json').read_text())['required_single_gpu_uuid_matches'] is True
CODE=r'''
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930';os.chdir(repo)
def read(p):
 p=Path(p);assert p.is_absolute() and p.is_relative_to(phase) and '..' not in p.parts
 assert not any(q.is_symlink() for q in [p,*p.parents] if q.is_relative_to(phase))
 b=p.read_bytes();return json.loads(b),{'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def metadata(p,fields):
 p=Path(p)
 if not p.is_file():return {'path':str(p),'exists':False}
 d,r=read(p);return dict(r,exists=True,metadata={k:d[k] for k in fields if k in d})
g=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
assert Path.cwd()==repo and g.returncode==0 and g.stdout.strip()=='GPU-44039938-fd82-41d2-fefd-de71514e2fac'
root=phase/'graph_mixed_block_execution_root_20261003_v1/transport/root_full40_run01'
run=phase/'graph_mixed_block_training_preparation_20261003_v2/runs/root_full40_run01'
child,childrec=read(root/'CHILD.json');assert child['PID']==379193
fields=('schema','UTC','start_UTC','terminal_UTC','PID','exit_code','timed_out','unrelated_processes_touched','automatic_restart')
transport=[metadata(root/n,fields) for n in ('STARTED.json','CHILD.json','SUPERVISOR_TERMINAL.json')]
processes=[]
for pid,start,token,parent in [(379192,'5981324833','graph_mixed_block_execution_root_20261003_v1/training_supervisor_run01.py',1),(379193,'5981324843','graph_mixed_block_training_preparation_20261003_v2/train_policies.py',379192)]:
 p=Path('/proc')/str(pid);row={'PID':pid,'expected_starttime_ticks':start,'process_exists':p.exists()}
 if p.exists():
  args=p.joinpath('cmdline').read_bytes().decode('utf-8','replace').split('\0')
  stat=p.joinpath('stat').read_text();tail=stat[stat.rfind(')')+2:].split()
  row.update(state=tail[0],PPID=int(tail[1]),starttime_ticks=tail[19],expected_script_argument_present=any(token in a and '\n' not in a for a in args))
  row['exact_handle_matches']=row['starttime_ticks']==start and row['PPID']==parent and row['expected_script_argument_present']
 processes.append(row)
casefields=('dataset','seed','policy','status','attempted','final_labels_closed','reused','checkpoint_bindings_verified','selected_state_replay','label_payloads_saved','successful_subset_scored','error_type')
rows=[]
for dataset in ('HGB-DBLP','HGB-ACM'):
 for seed in (131,137,139,149,151):
  for policy in ('own/own','pool/pool','pool/own','own/pool'):
   p=run/dataset/('seed'+str(seed))/policy.replace('/','__')/'TERMINAL.json';row={'dataset':dataset,'seed':seed,'policy':policy,'terminal_present':p.is_file()}
   if p.is_file():
    d,r=read(p);row.update({k:d[k] for k in casefields if k in d});row['terminal']=r
   rows.append(row)
study={'path':str(run/'STUDY.json'),'exists':(run/'STUDY.json').is_file()}
if study['exists']:
 d,r=read(run/'STUDY.json');study.update(r);study['metadata']={k:d[k] for k in ('schema','originals_preserved','baseline_reuse_mode','final_labels_closed','CP_gate_dependency') if k in d}
 study['summary_metadata']={k:d.get('summary',{})[k] for k in ('status','all40_terminals','successful_subset_scored') if k in d.get('summary',{})}
 study['case_metadata']=[{k:q[k] for k in casefields if k in q} for q in d.get('rows',[])]
 study['error_types']=[q.get('error_type') for q in d.get('errors',[])]
packet=phase/'graph_mixed_block_closed_family_evaluation_preparation_20261003_v2'
manifest,mr=read(packet/'MANIFEST.json');assert mr['sha256']=='462c7bad84921d4fd7e136cc80af57c7854d57e1fc05634c79db4719ca25df41'
prov,pr=read(packet/'PROVENANCE.json');verified=[]
for row in manifest['payload']:
 p=packet/row['path'];_,rec=read(p) if p.suffix=='.json' else (None,{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
 assert rec['sha256']==row['sha256'] and rec['bytes']==row['bytes'];verified.append(rec)
for row in prov['source_records']+prov['optional_source_records']:
 p=Path(row['path']);p=p if p.is_absolute() else phase/p;b=p.read_bytes()
 assert hashlib.sha256(b).hexdigest()==row['sha256'] and len(b)==row['bytes']
review,rr=read(phase/'mixed_closed_family_evaluation_source_review_20261003_v2/REVIEW.json')
assert review['packet']['manifest_sha256']==mr['sha256'] and review['resolved_finding']['status']=='resolved_in_source'
na,na_rec=read(phase/'graph_heterogeneous_dblp_native15_audit_execution_root_20261003_v1/NATIVE15_AUDIT_run01.json')
assert na_rec['sha256']=='c697a571abcc9bdef6f5d543ba8cda58cb394fa4784405653138d75e690f01ce' and na_rec['bytes']==72174
native={'audit':na_rec,'top_level_keys':list(na),'metadata':{k:v for k,v in na.items() if isinstance(v,(str,bool,int,float)) and not any(t in k.lower() for t in ('loss','nll','f1','score','accuracy','delta','mean','gate','quality'))}}
for key in ('source_manifest','freeze','release','study','graph_schema','cases'):
 if key in na:
  native[key+'_metadata']=na[key] if key!='cases' else [{'seed':q.get('seed'),'arm':q.get('arm'),'keys':list(q)} for q in na[key]]
print(json.dumps({'schema':'mixed40_current_read_only_closure_metadata_v1','UTC':datetime.now(timezone.utc).isoformat(),'repository_pwd':str(Path.cwd()),'required_single_gpu_uuid_matches':True,'exact_processes':processes,'transport':transport,'rows':rows,'terminal_status_counts':dict(Counter(r.get('status','unterminated') for r in rows)),'study':study,'evaluator_manifest':mr,'source_manifest_payloads_verified':len(verified),'required_source_records_verified':len(prov['source_records']),'optional_source_records_verified':len(prov['optional_source_records']),'independent_review':rr,'evaluation_authorities':prov['authorities'],'native15':native,'heldout_labels_metrics_arrays_checkpoints_opened_or_emitted':False,'remote_writes':False,'evaluator_started':False}))
'''
if __name__=='__main__':
 (HERE/'METADATA_REMOTE_CODE.py.txt').write_text(CODE)
 ssh=['ssh','-p','2222','-i','/Users/alex/.ssh/mlspace__private_key_anogena.txt','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','UpdateHostKeys=no','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
 r=subprocess.run([*ssh,shlex.join(['/usr/bin/python3','-I','-S','-B','-c',CODE])],capture_output=True,text=True,timeout=55)
 receipt={'UTC':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode,'stderr':r.stderr,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'remote_code_sha256':hashlib.sha256(CODE.encode()).hexdigest(),'private_key_contents_inspected':False,'remote_write_or_evaluator_or_restart':False}
 (HERE/'METADATA_TRANSPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if r.returncode:print(json.dumps(receipt));raise SystemExit(r.returncode)
 d=json.loads(r.stdout);(HERE/'OBSERVATION.json').write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({'UTC':d['UTC'],'exact_processes':d['exact_processes'],'terminal_status_counts':d['terminal_status_counts'],'study_exists':d['study']['exists'],'evaluator_manifest':d['evaluator_manifest'],'native15_metadata':d['native15']['metadata'],'native15_keys':d['native15']['top_level_keys']}))
