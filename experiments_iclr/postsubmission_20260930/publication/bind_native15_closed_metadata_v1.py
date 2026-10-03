"""Bind the completed native15 family without opening tensor payloads or labels."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

PHASE = Path(__file__).resolve().parents[1]
CODE = r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
assert subprocess.run(['git','rev-parse','--show-toplevel'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()==str(repo)
assert subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
phase=repo/'experiments_iclr/postsubmission_20260930'
native=phase/'graph_heterogeneous_dblp_native_challengers_preparation_20261003_v2'
supervisor=phase/'graph_heterogeneous_dblp_native_cpu_supervisor_preparation_20261003_v1'
name='root_native_serial_CPU_run01'
run=native/'runs'/name
sr=supervisor/'runs'/name
def desc(p):
 p=p.resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
receipt=json.loads((sr/'SUPERVISOR_RECEIPT.json').read_text())
assert receipt['status']=='child_completed' and receipt['exit_code']==0
assert receipt['owned_child_reaped'] is True and receipt['originals_preserved'] is True
study_desc=desc(run/'STUDY.json')
assert receipt['canonical_study']==study_desc
assert receipt['root_release']==desc(supervisor/'admissions'/f'{name}.json')
study=json.loads((run/'STUDY.json').read_text())
seeds=[131,137,139,149,151];arms=['native_GAT','native_Simple_HGN','native_SeHGNN']
assert [(r['seed'],r['arm']) for r in study['rows']]==[(s,a) for s in seeds for a in arms]
assert study['summary']['status']=='complete_development_challengers'
assert study['summary']['all_frozen_terminals'] is True
assert study['summary']['TEST_label_reads']==study['summary']['TEST_diagnostics']==0
assert study['summary']['successful_subset_scored'] is False
assert study['original_inputs_unchanged'] is True and study['preservation_error'] is None
cases=[]
for row in study['rows']:
 assert row['status']=='selected' and row['selected_state_replay'] is True
 assert row['TEST_label_reads']==row['TEST_diagnostics']==0
 cell=run/f"seed{row['seed']}"/row['arm']
 assert json.loads((cell/'SELECTION.json').read_text())==row
 cases.append(dict(seed=row['seed'],arm=row['arm'],artifacts={n:desc(cell/n) for n in ['SELECTION.json','TRACE.jsonl','selected.pt','selected_validation_logits.pt']}))
owned=json.loads((sr/'OWNED_CHILD.json').read_text())
assert owned['pid']==receipt['owned_child_pid'] and owned['argv']==receipt['argv']
assert not Path('/proc',str(owned['pid'])).exists(),'Original native child PID remains live'
print(json.dumps(dict(UTC=datetime.now(timezone.utc).isoformat(),status='closed_selected15_root_metadata_verified',study=study_desc,graph_schema=desc(run/'GRAPH_SCHEMA.json'),supervisor_receipt=desc(sr/'SUPERVISOR_RECEIPT.json'),supervisor_record=receipt,cases=cases,all15_selected=True,training_child_reaped=True,root_observed=True,tensor_payloads_deserialized=False,development_labels_opened=False,heldout_labels_opened=False,numerical_scores_disclosed=False,independent_replay_audit_complete=False)))
'''


def main():
    out = PHASE/'graph_heterogeneous_dblp_native15_audit_execution_root_20261003_v1'
    out.mkdir(exist_ok=False)
    (out/'CLOSURE_REMOTE_CODE.txt').write_text(CODE)
    ssh = ['ssh', '-p', '2222', '-i', '/Users/alex/.ssh/mlspace__private_key_anogena.txt',
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'UpdateHostKeys=no',
           '-o', 'StrictHostKeyChecking=yes', 'anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru']
    r = subprocess.run([*ssh, shlex.join(['/usr/bin/python3', '-I', '-S', '-B', '-c', CODE])],
                       capture_output=True, text=True, timeout=45)
    receipt = dict(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr,
                   destination=ssh[-1], source_sha256=hashlib.sha256(CODE.encode()).hexdigest())
    (out/'CLOSURE_TRANSPORT_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if r.returncode:
        print(json.dumps(receipt))
        return r.returncode
    value = json.loads(r.stdout)
    (out/'CLOSED_SELECTED15_ROOT_BINDINGS.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ('UTC','status','all15_selected','training_child_reaped','numerical_scores_disclosed','independent_replay_audit_complete')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
