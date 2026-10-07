import json,hashlib,socket,subprocess
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert socket.gethostname()=='anogena-2-0' and Path.cwd()==R
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
A=P/'wikics_private_graph_controls_activation_20261007_v2';cfg=json.loads((A/'CONFIG_ACTIVATED.json').read_text());assert h(A/'CONFIG_ACTIVATED.json')=='fc037fe504a2135002f2c631158374c1299b03127a7968d3499d1d690e1ce61c'
D=P/cfg['execution_root_relative'];rows=[]
for e in cfg['entries']:
 if e['arm']=='DONOR_INDEPENDENT':continue
 out=Path(e['output_directory']);p=out/'FREEZE.json';f=json.loads(p.read_text());exitfile=D/'owner'/'logs'/(e['cell_id']+'.EXIT.json');t=json.loads(exitfile.read_text());job=P/e['job_relative']
 assert h(job)==e['job_sha256'] and f['complete'] and f['arm']==e['arm'] and f['seed']==e['seed'] and f['source_manifest_sha256']==cfg['source_sha256'] and f['job_sha256']==e['job_sha256'] and f['TEST_access'] is False
 assert t['exit_code']==0 and t['reason'] is None and not t['signals_sent'] and t['terminal_wait_observed'] and t['job_sha256']==e['job_sha256'] and not (out/'FAILURE.json').exists()
 identity=t['raw_identity_observation'];q=Path('/proc')/str(identity['PID']);absent=not q.exists()
 if not absent:
  st=(q/'stat').read_text();fields=st[st.rfind(')')+2:].split();assert int(fields[19])!=identity['start_ticks']
 for b in ([f['model_freeze']] if 'model_freeze' in f else []):assert h(P/b['path'])==b['sha256']
 cache_verified=None
 if e['arm'] in ('E_joint','E_own'):
  j=json.loads(job.read_text());cache=json.loads((out/'CACHE_FREEZE.json').read_text());original=json.loads((P/j['frozen_cache']['path']).read_text());assert h(P/j['frozen_cache']['path'])==j['frozen_cache']['sha256'] and cache['donor_selected']==original['donor_selected'];assert h(P/cache['cache']['path'])==cache['cache']['sha256'];assert cache['cache']['sha256']==original['cache']['sha256'];cache_verified=True
 rows.append({'arm':e['arm'],'seed':e['seed'],'complete':True,'source_manifest_sha256':cfg['source_sha256'],'freeze':{'path':str(p.relative_to(P)),'sha256':h(p)},'clean_owned_terminal':{'path':str(exitfile.relative_to(P)),'sha256':h(exitfile)},'old_owned_child_absent':absent,'exact_pre_correction_cache_reuse_verified':cache_verified,'elapsed_seconds':t['elapsed_seconds'],'peak_sampled_owned_GPU_bytes':t['max_sampled_owned_GPU_bytes'],'peak_sampled_owned_RSS_bytes':t['max_sampled_owned_RSS_bytes'],'source_integrated_learning_checks':f['integrated_TRAIN_checks'],'quality_values_exported':False})
print(json.dumps({'complete_fixed_common_controls':len(rows)==12,'rows':rows,'quality_values_exported':False,'I_native_U_stage_still_required':True,'TEST_access':False}))
