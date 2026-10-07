from pathlib import Path
import socket,json,hashlib,datetime
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert socket.gethostname()=='anogena-2-0'
def read(f):return json.loads(f.read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
roots=[('E_stage','wikics_private_graph_comparison_execution_root_20261007_v1',498058),('common','wikics_private_graph_controls_execution_root_20261007_v2',498539),('native17','wikics_independent17_successor_execution_root_20261007_v2',502012)]
rows=[];closures={}
for kind,name,pid in roots:
 D=P/name;assert not (Path('/proc')/str(pid)).exists()
 f=D/('COMPLETE.json' if kind=='native17' else 'owner/COMPLETE.json');c=read(f);assert c.get('all_complete',c.get('complete')) is True
 closures[name]={'path':str(f.relative_to(P)),'sha256':sha(f),'value':c}
 paths=list((D/'runs').glob('*/FREEZE.json'))
 assert len(paths)==({'E_stage':3,'common':12,'native17':2}[kind])
 for f in paths:
  x=read(f);assert x['complete'] is True and x['TEST_access'] is False
  assert x['program_sha256']=='c7d8c8c6ca18985207c089b069bc165fcb7163d5fde4a151e2e2569bb07a166d'
  probability=x.get('probabilities')
  if probability:assert sha(P/probability['path'])==probability['sha256']
  rows.append({'cell':x['arm']+'_'+str(x['seed']),'path':str(f.relative_to(P)),'sha256':sha(f),'value':x,'probability_file_sha_verified':probability is not None})
assert len(rows)==17
print(json.dumps({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'closures':closures,'TEST_access':False,'original_probabilities_hashed':True,'new_training_or_model_forwards':0,'limits':'Original closure and PID-only absence where exact historical ticks are unavailable; metadata readout, not independent checkpoint replay.'}))
