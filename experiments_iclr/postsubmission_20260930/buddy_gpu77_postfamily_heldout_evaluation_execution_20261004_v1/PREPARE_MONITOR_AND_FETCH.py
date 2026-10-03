"""Prepare one read-only exact-handle observation; scalar retrieval requires all15 closure."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
REPO='/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git'
CODE=r'''
from pathlib import Path
from datetime import datetime,timezone
import os,base64,hashlib,json,subprocess
REPO=Path('/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git')
HERE=REPO/'experiments_iclr/postsubmission_20260930/buddy_gpu77_postfamily_eval_preparation_v4'
SUP=HERE/'root_eval_supervision_20261004_v1'
EVAL=HERE/'root_eval_v1'
assert Path.cwd().resolve()==REPO
def safe(path):
 assert path.resolve().is_relative_to(REPO) and not path.is_symlink()
 for parent in path.parents:
  if parent==REPO:break
  assert not parent.is_symlink()
 return path
def raw(path):
 safe(path);assert path.is_file() and path.stat().st_size<=2*1024*1024
 data=path.read_bytes();data.decode('utf-8');assert b'\0' not in data
 return data
def read(path):return json.loads(raw(path))
def handle(value):
 pid=value['pid'];stat=Path(f'/proc/{pid}/stat')
 if not stat.exists():return dict(**value,observation='absent')
 text=stat.read_text();fields=text[text.rfind(')')+2:].split();ticks=int(fields[19])
 if ticks!=value['start_ticks']:return dict(**value,observation='PID_reused',observed_start_ticks=ticks)
 return dict(**value,observation='same_identity',state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
 observed_argv=[p.decode() for p in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if p],
 observed_cwd=os.readlink(f'/proc/{pid}/cwd'),user_ticks=int(fields[11]),system_ticks=int(fields[12]),RSS_pages=int(fields[21]))
launch=read(SUP/'DETACHED_LAUNCH.json')
assert launch['status']=='LAUNCHED_ONCE' and launch['runner_handle']==__HANDLE__
assert launch['admission']['sha256']=='48c4eb06863688d73e8b4c68458b2d494db1b615c7612486794c46013510125c'
assert hashlib.sha256(raw(HERE/'ROOT_EVALUATION_ADMISSION.json')).hexdigest()==launch['admission']['sha256']
assert hashlib.sha256(raw(SUP/'run_heldout_evaluation_once.py')).hexdigest()=='f74a141b9e4d8302b0d3456aa232255228d9ef29c3e704076c87bd6f3c365d06'
paths=[HERE/'ROOT_EVALUATION_ADMISSION.json',SUP/'run_heldout_evaluation_once.py',SUP/'DETACHED_LAUNCH.json',
 SUP/'RUNNER_STARTED.json',SUP/'CHILD_STARTED.json']
terminal=SUP/'PHYSICAL_TERMINAL.json'
physical=read(terminal) if terminal.exists() else None
receipt=read(EVAL/'EVALUATION_RECEIPT.json') if physical and (EVAL/'EVALUATION_RECEIPT.json').exists() else None
lock=read(HERE/'root_lock_v1/FAMILY_LOCK.json')
assert hashlib.sha256(raw(HERE/'root_lock_v1/FAMILY_LOCK.json')).hexdigest()=='4d4041ad0d02a36c94bd9112f4e01029fb722eee75339a5743a0259435c33fd9'
assert len(lock['runs'])==15
scalar_paths=[Path(row['run_directory'])/'final_test.json' for row in lock['runs']]
completed_count=sum(path.exists() for path in scalar_paths)
all15=bool(physical and physical['exit_code']==0 and receipt and receipt['status']=='all15_locked_cells_scored_once' and len(receipt['final_results'])==15 and completed_count==15)
if physical:
 paths += [terminal,SUP/'CHILD.stdout.log',SUP/'CHILD.stderr.log',SUP/'RUNNER.stdout.log',SUP/'RUNNER.stderr.log',
           EVAL/'EVALUATION_CLAIM.json',EVAL/'EVALUATION_RECEIPT.json',EVAL/'score.log']
if all15:
 paths += scalar_paths+[REPO/'experiments_iclr/postsubmission_20260930/buddy_complete_data_cache_preparation_v3/root_run_77_v1/cache/test_manifest.json']
rows=[];missing=[];oversized=[]
for path in paths:
 safe(path);relative=str(path.relative_to(REPO))
 if not path.exists():missing.append(relative);continue
 if path.stat().st_size>2*1024*1024:oversized.append(dict(path=relative,bytes=path.stat().st_size));continue
 data=raw(path);rows.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),data_base64=base64.b64encode(data).decode()))
handles=[handle(launch['runner_handle'])]
if (SUP/'CHILD_STARTED.json').exists():handles.append(handle(read(SUP/'CHILD_STARTED.json')['identity']))
status='ALL15_PHYSICALLY_COMPLETE' if all15 else ('PHYSICALLY_TERMINATED_WITHOUT_ALL15_SUCCESS' if physical else 'RUNNING_OR_TERMINAL_RECEIPT_PENDING')
print(json.dumps(dict(schema='buddy77_owned_heldout_evaluation_readonly_monitor_fetch_v1',UTC=datetime.now(timezone.utc).isoformat(),
 status=status,sequence=__SEQUENCE__,all15_closure_established=all15,final_scalar_files_present=completed_count,
 physical_exit_code=physical['exit_code'] if physical else None,stage_status=receipt['status'] if receipt else None,
 files=rows,missing=missing,oversized=oversized,owned_handles=handles,
 scalar_results_fetched_only_after_all15_physical_and_stage_closure=True,
 scientific_binary_payloads_fetched=False,prediction_export_executed=False,evaluation_relaunched=False,remote_writes=False)))
'''

parser=argparse.ArgumentParser();parser.add_argument('--sequence',type=int,required=True);args=parser.parse_args()
assert 1<=args.sequence<=99
transport=PHASE/'gpu77_connection_recovery_v1/commands/buddy_v4_heldout_evaluation_launch_20261004_v1/RECEIPT.json'
value=json.loads(transport.read_text());assert value['exit_code']==0
launch=json.loads(value['stdout'].strip());assert launch['status']=='LAUNCHED_ONCE'
code=CODE.replace('__HANDLE__',repr(launch['runner_handle'])).replace('__SEQUENCE__',repr(args.sequence))
compile(code,'buddy_evaluation_readonly_fetch','exec')
with (HERE/f'MONITOR_REMOTE_{args.sequence:02d}.py.txt').open('x') as h:h.write(code)
command=PHASE/f'gpu77_connection_recovery_v1/buddy_v4_heldout_evaluation_monitor_20261004_v{args.sequence}_command.txt'
with command.open('x') as h:h.write(shlex.join(['/usr/bin/python3','-I','-S','-B','-c','import os;os.chdir('+repr(REPO)+');\n'+code])+'\n')
print(json.dumps(dict(command_file=str(command.relative_to(PHASE)),sequence=args.sequence,
 command_sha256=hashlib.sha256(command.read_bytes()).hexdigest(),remote_executed=False)))
