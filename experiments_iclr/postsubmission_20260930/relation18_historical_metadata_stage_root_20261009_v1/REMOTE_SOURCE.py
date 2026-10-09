
from pathlib import Path
import socket,subprocess,json,hashlib,os
assert socket.gethostname()=='anogena-2-0'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
R=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');P=R/'experiments_iclr/postsubmission_20260930'
E='Wiki24_selected_analysis_after_closure_execution_root_20261007_v1';CPU='Wiki24_analysis_after_closure_execution_root_20261007_v1'
att=json.loads((P/'relation18_original_allocation_custody_attestation_root_20261009_v2/ATTESTATION.json').read_text());gate=json.loads((P/att['original_gate']['path']).read_text())
paths=[E+'/'+name for name in ('COLLECT_RELEASE.json','TERMINAL_EVIDENCE.json','CHILD_OWNER.json','CHILD_TERMINAL.json')]+[CPU+'/extraction/'+name for name in ('METADATA_EXPORT.json','EXTRACTION_COST.json')]+[gate['closure']['path'],gate['owner']['path']]
rows=[]
for rel in sorted(set(paths)):
 p=P/rel;assert p.resolve().is_relative_to(P) and p.is_file() and not p.is_symlink();b=p.read_bytes();r='experiments_iclr/postsubmission_20260930/'+rel
 rows.append(dict(source_relative=r,target_relative=r,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
print(json.dumps(rows))
