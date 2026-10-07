import json,socket,subprocess,hashlib
from pathlib import Path
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930';assert Path.cwd()==R and socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).split()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
D=P/'wikics_independent17_successor_execution_root_20261007_v1';S=P/'wikics_independent17_successor_activation_20261007_v1';C=P/'wikics_private_graph_controls_execution_root_20261007_v2';r={'successor_job_files':[str(p.relative_to(D)) for p in (D/'jobs').glob('*')],'successor_run_directories':[p.name for p in (D/'runs').glob('*')],'stderr':(S/'owner.stderr.log').read_text(),'failure':json.loads((D/'FAILURE.json').read_text()),'prefix_terminal':{},'original_owner_complete':{}}
for p,target in [(C/'owner/logs/DONOR_INDEPENDENT_17.EXIT.json','prefix_terminal'),(C/'owner/COMPLETE.json','original_owner_complete')]:
 a=json.loads(p.read_text());r[target]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**{k:a[k] for k in ('exit_code','reason','signals_sent','terminal_wait_observed','all_complete','fits') if k in a}}
print(json.dumps(r))
