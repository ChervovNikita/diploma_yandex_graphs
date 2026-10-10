"""Exact copied-six source/recipe/selector/cost/terminal metadata only."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
WRAPPER=PHASE/'gpu77_connection_recovery_v1/run_gpu77_v3.py'
ID='INDEPENDENT_SCORER_COPIED6_EXACT_METADATA_20261010_v1'
REMOTE=r'''
import hashlib,json,socket,subprocess
from pathlib import Path
R=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git');P=R/'experiments_iclr/postsubmission_20260930'
assert socket.gethostname()=='peptide' and Path.cwd().resolve()==R
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
assert gpus==['GPU-98aa0f2e-3dd1-5cd8-f001-f259f707a998','GPU-5dcf7db7-a450-3ca8-41b2-6c5316128ced']
def b(path):
 data=path.read_bytes();return dict(path=str(path.relative_to(P)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def read(path):return json.loads(path.read_text())
owner=P/'graph_relation_private_credit_full12_owner_execution_root_20261008_v1'
closure_path=owner/'FAMILY_CLOSURE.json';closure=read(closure_path)
terminal_path=P/'relation18_selected_readout_activation_root_20261010_v2/TERMINAL_EVIDENCE.json';terminal=read(terminal_path)
assert b(closure_path)==terminal['relation_family_closure'] and terminal['complete'] and terminal['all_owners_and_children_terminal']
rows=[]
for seed in (6101,6203,6307):
 for policy in ('alphaF','relationJ'):
  cell=str(seed)+'_'+policy
  release_path=P/'graph_relation_private_credit_full12_activation_root_20261008_v1/cells'/(cell+'.json');release=read(release_path)
  output=P/release['output'];complete_path=output/'COMPLETE.json';complete=read(complete_path)
  exit_path=owner/'logs'/(cell+'.EXIT.json');ex=read(exit_path)
  child=next(row for row in terminal['relation_children'] if row['cell_id']==cell)
  assert child['source_receipt']==b(exit_path) and child['child_absent'] and child['child_no_CUDA_rows'] and child['wait_and_reap_observed']
  assert ex['exit_code']==0 and ex['terminal_wait_observed'] and not ex['signals_sent'] and ex['reason'] is None
  assert complete['complete'] and complete['epochs']==complete['steps']==1100 and complete['seed']==seed
  assert release['policy']==policy and release['epochs']==1100 and release['risk_beta']==.5 and release['auxiliary'] is False
  assert release['physical_gpu_uuid']==gpus[0 if seed==6203 else 1]
  assert complete['source_manifest_sha256']==release['source_manifest_sha256']=='bd4031d735aedf6c7f2e555bc3768239d3bec4dde18e1a3485be30209749c61f'
  selected=output/'selected.pt'
  rows.append(dict(seed=seed,policy=policy,physical_gpu_uuid=release['physical_gpu_uuid'],release=b(release_path),completion=b(complete_path),actual_exit=b(exit_path),original_cost=dict(source='Original COMPLETE inclusive successor seconds and existing supervised EXIT',inclusive_successor_seconds=complete['successor_inclusive_seconds'],original_complete_seconds=complete.get('seconds'),actual_active_and_cleanup_seconds=ex['elapsed_seconds'],owned_GPU_bytes=ex['max_sampled_owned_GPU_bytes'],owned_RSS_bytes=ex['max_sampled_owned_RSS_bytes'],output_bytes=ex['max_sampled_own_output_bytes'],log_bytes=ex['max_sampled_own_log_bytes']),selected_checkpoint=dict(path=str(selected.relative_to(P)),bytes=selected.stat().st_size,sha256=complete['selected_sha256'],hash_authority='Exact original COMPLETE selected_sha256; payload not reopened'),source_manifest=dict(path='graph_relation_private_credit_source_20261008_v2/SOURCE_MANIFEST.json',bytes=3944,sha256=release['source_manifest_sha256']),recipe=dict(epochs=1100,local_epochs=100,risk_beta=.5,policy=policy,auxiliary=False,train=release['train'],valid=release['valid'],polynormer=release['polynormer']),selector=dict(source=b(P/'portable_internal_be_public_interface_20261007_v2/train.py'),rule='Original strict-first maximum complete VALID accuracy; coherent whole-model/Adam selected local restore; live member dropout streams retained; local selection allowed'),terminal_custody=child,checkpoint_reuse_for_training=False))
report=dict(schema='independent-native-local-scorer-copied-six-exploratory-custody-v1',complete=True,all_original12_closed=True,root_observed=False,records=rows,source_manifest=rows[0]['source_manifest'],relation_family_closure=b(closure_path),relation_terminal_evidence=b(terminal_path),original_relation_owner=terminal['relation_owner'],original_family_closure_non_score_summary={k:v for k,v in closure.items() if isinstance(v,(int,float,bool,str)) or v is None},raw_or_selected_bytes_on_Mac=False,checkpoint_reuse_for_training=False,VALID_scores_read=False,TEST_access=False,original_outcomes_unchanged=True)
print(json.dumps(report,sort_keys=True))
'''

def main():
 assert hashlib.sha256(WRAPPER.read_bytes()).hexdigest()=='035b740ceb50eefcfa3cec2aef6dbd1294c769d133e2bc4cf88fb52b088421ff'
 command='cd '+shlex.quote('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')+' && /usr/bin/python3 -I -S -B -c '+shlex.quote(REMOTE)
 file=WRAPPER.parent/(ID+'_COMMAND.txt');file.write_text(command)
 result=subprocess.run(['python3','-B',str(WRAPPER),'--id',ID,'--command-file',str(file)],capture_output=True,text=True)
 with (HERE/'ANCHOR_METADATA_TRANSPORT.json').open('x') as f:json.dump(dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),f,indent=2);f.write('\n')
 assert result.returncode==0
 outer=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')][-1]
 assert outer['exit_code']==0
 report=next(json.loads(line) for line in outer['stdout'].splitlines() if line.startswith('{'))
 with (HERE/'COPIED_ANCHOR_CUSTODY.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(complete=report['complete'],anchors=len(report['records']),costs=[dict(seed=r['seed'],policy=r['policy'],cost=r['original_cost']) for r in report['records']]),indent=2))

if __name__=='__main__':main()
