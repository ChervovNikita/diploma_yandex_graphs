from pathlib import Path
import json,base64,hashlib,socket,subprocess
P=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/experiments_iclr/postsubmission_20260930')
assert socket.gethostname()=='anogena-2-0' and subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
assert json.loads((P/'staged_posterior_complete_readout_execution_root_20261009_v1/ROUTINE_TERMINAL.json').read_text())['complete']
assert json.loads((P/'staged_posterior_matched_CS_reference_execution_root_20261009_v1/COST_TERMINAL.json').read_text())['complete']
rows=[]
for rel in ['staged_posterior_complete_readout_execution_root_20261009_v1/FROZEN_GATE.json', 'staged_posterior_complete_readout_execution_root_20261009_v1/RECONSTRUCTION_VALIDATION.json', 'staged_posterior_complete_readout_execution_root_20261009_v1/DESCRIPTIVE_ERRORS.json', 'staged_posterior_complete_readout_execution_root_20261009_v1/DESCRIPTIVE_SEED_INTERVALS.json', 'staged_posterior_complete_readout_execution_root_20261009_v1/ROUTINE_TERMINAL.json', 'staged_posterior_matched_CS_reference_execution_root_20261009_v1/RESULT.json', 'staged_posterior_matched_CS_reference_execution_root_20261009_v1/COST_TERMINAL.json', 'staged_posterior_full12_execution_root_20261009_v1/FAMILY_CLOSURE.json', 'staged_posterior_full12_execution_root_20261009_v1/PARENT_TERMINAL.json']:
 f=P/rel;assert f.resolve().is_relative_to(P) and f.stat().st_size<500000
 b=f.read_bytes();rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))
print(json.dumps(rows))
