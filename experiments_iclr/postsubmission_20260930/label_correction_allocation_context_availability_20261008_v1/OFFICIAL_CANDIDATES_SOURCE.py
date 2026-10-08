from pathlib import Path
import socket,json,subprocess,hashlib
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
rows=[]
for f in P.glob('wikics_official*/AVAILABLE_MANIFEST.json'):
 x=json.loads(f.read_text());row=x.get('available');q=P/row['path'] if row else None
 rows.append({'manifest':str(f.relative_to(P)),'payload':row,'payload_exists':q.is_file() if q else False})
print(json.dumps({'official_WikiCS_candidates':rows,'numeric_arrays_opened':False}))
