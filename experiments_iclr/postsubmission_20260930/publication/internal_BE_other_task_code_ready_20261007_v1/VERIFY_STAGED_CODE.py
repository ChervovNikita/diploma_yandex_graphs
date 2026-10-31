import os,socket,json,hashlib,subprocess
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';os.chdir(R);assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
D=P/'internal_be_collab_molhiv_launch_preparation_20261007_v2';sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert sha(D/'MANIFEST.json')=='04fc9d1db54afa0a87c03f3710312fb0785000a5b38d71d030da79f3130d34b2'
records={}
for task in ['collab','molhiv']:
 plan=json.loads((D/(task+'.json')).read_text())
 for key in ['config','data_manifest','data_export_review','qualifier_source_review']:
  row=plan[key];q=P/row['path'];assert q.resolve().is_relative_to(P) and sha(q)==row['sha256']
 for row in plan['source_review_evidence']:assert sha(P/row['path'])==row['sha256']
 for folder,pin in [(plan['suite_directory'],plan['source_manifest_sha256']),(plan['qualifier_directory'],plan['qualifier_manifest_sha256'])]:
  S=P/folder;assert sha(S/'MANIFEST.json')==pin
  for row in json.loads((S/'MANIFEST.json').read_text())['files']:
   q=S/row['path'];assert q.resolve().is_relative_to(P) and sha(q)==row['sha256'] and q.stat().st_size==row['bytes']
 result=subprocess.run(['/usr/bin/python3','-I','-S','-B',str(D/'family.py'),'--task',task,'--list'],cwd=R,capture_output=True,text=True,check=True)
 cells=json.loads(result.stdout);assert len(cells)==24
 records[task]={'code_available':True,'metadata_roster':len(cells),'bound_sources_and_approvals_available':True,'new_training_started':False}
print(json.dumps({'tasks':records,'source_manifest_verified':True,'model_import_or_scientific_execution':False,'TEST_access':False}))
