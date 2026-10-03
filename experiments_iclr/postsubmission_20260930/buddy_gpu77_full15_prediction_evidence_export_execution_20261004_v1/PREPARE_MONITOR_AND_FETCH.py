"""Fetch only export receipts; candidate/logit tensors stay on the authorized server."""
from pathlib import Path
import argparse,ast,hashlib,json,shlex

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
old=PHASE/'buddy_gpu77_postfamily_heldout_evaluation_execution_20261004_v1/PREPARE_MONITOR_AND_FETCH.py'
node=next(node for node in ast.parse(old.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE' for t in node.targets))
CODE=ast.literal_eval(node.value)
for old,new in [('root_eval_supervision_20261004_v1','root_predictions_supervision_20261004_v1'),
 ('root_eval_v1','root_predictions_v1'),('ROOT_EVALUATION_ADMISSION.json','ROOT_PREDICTION_EXPORT_ADMISSION.json'),
 ('run_heldout_evaluation_once.py','run_prediction_export_once.py'),
 ('48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c','3057f8b7cac62278ec4dfcf8a149cc41d84671cc98157c0af5ffcdbacfc88623'),
 ('f74a141b9e4d8302b0d3456aa232255228d9ef29c3e704076c87bd6f3c365d06','78943faec665b260d2077123df24c2cfdf6e1665e60a8719e213e22f1f3a225a'),
 ('EVALUATION_RECEIPT.json','PREDICTIONS_RECEIPT.json'),('EVALUATION_CLAIM.json','EXPORT_CLAIM.json')]:CODE=CODE.replace(old,new)
begin=CODE.index('scalar_paths=')
end=CODE.index('rows=[];missing=[];oversized=[]',begin)
block=r'''
completed_count=len(receipt.get('completed_exports',[])) if receipt else None
all15=bool(physical and physical['exit_code']==0 and receipt and receipt['status']=='all15_prediction_evidence_replays_exported_once' and completed_count==15 and receipt['extra_forwards']==15)
server_descriptors=[];server_issues=[]
def tensor_descriptor(path):
 safe(path)
 if not path.is_file():return None
 h=hashlib.sha256();count=0
 with path.open('rb') as stream:
  while chunk:=stream.read(1024*1024):h.update(chunk);count+=len(chunk)
 return dict(path=str(path.relative_to(REPO)),bytes=count,sha256=h.hexdigest(),tensor_fetched=False)
if all15:
 assert {(row['arm'],row['seed']) for row in receipt['completed_exports']}=={(row['arm'],row['seed']) for row in lock['runs']}
 for item in receipt['completed_exports']:
  path=EVAL/f"{item['arm']}_seed{item['seed']}.pt";observed=tensor_descriptor(path)
  if not observed or observed['bytes']!=item['bytes'] or observed['sha256']!=item['file_sha256']:server_issues.append(str(path))
  if observed:server_descriptors.append(observed)
 order=tensor_descriptor(EVAL/'CANDIDATE_ORDER.pt')
 if not order or order['sha256']!=receipt['candidate_order_file_sha256']:server_issues.append(str(EVAL/'CANDIDATE_ORDER.pt'))
 if order:server_descriptors.append(order)
if physical:
 paths += [terminal,SUP/'CHILD.stdout.log',SUP/'CHILD.stderr.log',SUP/'RUNNER.stdout.log',SUP/'RUNNER.stderr.log',
           EVAL/'EXPORT_CLAIM.json',EVAL/'PREDICTIONS_RECEIPT.json']
'''
CODE=CODE[:begin]+block+CODE[end:]
CODE=CODE.replace("final_scalar_files_present=completed_count", "completed_export_rows=completed_count,server_only_export_descriptors=server_descriptors,server_export_retention_issues=server_issues")
CODE=CODE.replace('scalar_results_fetched_only_after_all15_physical_and_stage_closure=True', 'export_receipts_fetched_after_physical_closure=True,large_tensor_payloads_fetched=False')
CODE=CODE.replace('prediction_export_executed=False,evaluation_relaunched=False', 'export_relaunched=False')
CODE=CODE.replace('buddy77_owned_heldout_evaluation_readonly_monitor_fetch_v1','buddy77_owned_prediction_export_readonly_monitor_fetch_v1')
parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=int,required=True);args=parser.parse_args();assert 1<=args.sequence<=99
transport=PHASE/'gpu77_connection_recovery_v1/commands/buddy_v4_prediction_export_launch_20261004_v1/RECEIPT.json'
value=json.loads(transport.read_text());assert value['exit_code']==0
launch=json.loads(value['stdout'].strip());assert launch['status']=='LAUNCHED_ONCE'
code=CODE.replace('__HANDLE__',repr(launch['runner_handle'])).replace('__SEQUENCE__',repr(args.sequence))
compile(code,'read_only_export_receipt_fetch','exec')
with (HERE/f'MONITOR_REMOTE_{args.sequence:02d}.py.txt').open('x') as h:h.write(code)
command=PHASE/f'gpu77_connection_recovery_v1/buddy_v4_prediction_export_monitor_20261004_v{args.sequence}_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+code])+'\n')
print(json.dumps(dict(command_file=str(command.relative_to(PHASE)),sequence=args.sequence,
 command_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),remote_executed=False,large_tensor_payloads_fetched=False)))
