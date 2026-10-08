from pathlib import Path
import subprocess,socket,json
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
exact=P/'wikics_official_acquisition_gpu77_root_20261007_v1/available/wikics_split0.pt'
r=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*wikics*.pt','-g','*WikiCS*','-g','train.npz','-g','valid.npz'],cwd=R,capture_output=True,text=True,timeout=20)
rows=[x for x in r.stdout.splitlines() if 'wikics' in x.lower()]
A=P/'internal_BE_molhiv18_family_waiting_launcher_activation_root_20261007_v1'
print(json.dumps({'exact_available_payload_exists':exact.is_file(),'WikiCS_project_file_candidates':rows[:40],'candidate_count':len(rows),'Mol18_activation_files':[x.name for x in A.iterdir() if x.is_file()],'numeric_arrays_opened':False}))
