from pathlib import Path
import socket,subprocess,json,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';E=P/'label_only_four_bank_first_screen_execution_root_20261008_v2'
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
 return h.hexdigest()
c=json.loads((E/'FAMILY_CLOSURE.json').read_text());t=json.loads((E/'PARENT_TERMINAL.json').read_text())
assert c['complete'] is True and c['fixed_seeds']==[6101,6203,6307] and c['full_epochs_per_block']==1100 and c['required_correction_records']==12
assert t['family_complete'] is True and t['completed_seed_count']==3 and t['error'] is None
states=[];native=[]
for seed,row in zip((6101,6203,6307),c['completed']):
 assert row['seed']==seed and row['exit_receipt']['exit_code']==0 and row['exit_receipt']['reaped'] is True
 D=E/('seed'+str(seed));f=D/'COMPLETE.json';assert sha(f)==row['completion_sha256'];v=json.loads(f.read_text())
 assert v['complete'] is True and v['epochs']==1100 and v['native_trajectories']==1 and len(v['required_bank_records'])==4
 for arm,b in zip(('C4','S_joint4head','U4_sharedB','S_one_path'),v['required_bank_records']):
  state=D/arm/'selected.pt';assert b['arm']==arm and b['complete'] is True and b['epochs']==1100 and b['selected_sha256']==sha(state)
  states.append(dict(seed=seed,arm=arm,sha256=b['selected_sha256']))
 for f in D.glob('native*.pt'):native.append(dict(seed=seed,path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha(f)))
print(json.dumps(dict(complete=True,closure_sha256=sha(E/'FAMILY_CLOSURE.json'),terminal=t,selected_states=states,native_states=native,execution_commit=c['execution_source_commit'],current_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),numerical_payloads_opened=False)))
