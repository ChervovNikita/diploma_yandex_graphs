"""Run the trusted-root-approved existing history byte wrapper once; gather two metadata files."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess,time,zlib
HERE=Path(__file__).resolve().parent;P=HERE.parent
REQUEST=P/'shared_private_transfer_complete39_history_inventory_root_request_preparation_20261006_v1'
REPO='/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs'
def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(n,v):
 with (HERE/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
 (HERE/n).chmod(0o444)
def bind(p):
 b=p.read_bytes();return {'path':str(p.relative_to(P)),'bytes':len(b),'sha256':sha(b)}
root_instruction='Root approved and created exact readonly ROOT_HISTORY_INVENTORY_RELEASE.json 786B/SHA d84424c9… in your request packet. Execute the exact reviewed once-only byte inventory using your bounded staging and request argv, freeze/gather its two outputs and actual terminal/cost, preserve failures. Then prepare and send the final real-inventory D2 release/request immediately (no new science changes; unchanged gates and selectors). Prioritize actual comparison, no further optional packet bureaucracy. Do not launch D2 until I adopt actual bindings. Allocation only; no77 access.'
write('AUTHORIZATION_RECEIPT.json',{'schema':'trusted_parent_history_BYTE_once_only_authorization_receipt_v1','UTC':now(),'sender':'/root','text':root_instruction,
 'saved_as_received_instruction_not_root_generated_authority_file':True,'one_BYTE_inventory_invocation_authorized':True,'D2_launch_authorized':False})
request=json.loads((REQUEST/'ROOT_APPROVAL_REQUEST.json').read_text());plan=json.loads((REQUEST/'STAGING_INVENTORY.json').read_text())
release=REQUEST/'ROOT_HISTORY_INVENTORY_RELEASE.json';body=release.read_bytes();assert len(body)==786 and sha(body)=='d84424c9a4746c24d26a1c96f9bc686fe9ed6038b3d575e2339b244bd6905c78'
assert release.stat().st_mode&0o222==0 and json.loads(body)['root_history_inventory_approved'] is True
payload=base64.b64encode(zlib.compress(json.dumps({'release_base64':base64.b64encode(body).decode()},sort_keys=True).encode(),9))+b'\n'
assert len(payload)==plan['expected_stdin_bytes'] and sha(payload)==plan['expected_stdin_sha256']
write('STAGING_COMMAND.json',{'UTC':now(),'argv':plan['argv'],'stdin_sha256':sha(payload),'stdin_bytes':len(payload),'authorization':bind(HERE/'AUTHORIZATION_RECEIPT.json')})
t=time.monotonic();s=subprocess.run(plan['argv'],input=payload.decode(),capture_output=True,text=True,timeout=55)
stage={'UTC':now(),'exit_code':s.returncode,'elapsed_seconds':time.monotonic()-t,'stderr':s.stderr,'stdout':s.stdout}
if s.returncode==0:stage['result']=json.loads(s.stdout)
write('STAGING_RECEIPT.json',stage);assert s.returncode==0,'No inventory launch; preserve failed staging, no automatic retry.'
write('INVENTORY_LAUNCH_INTENT.json',{'UTC':now(),'argv':request['remote_invocation_argv'],'inventory_argv':request['inventory_argv'],
 'release':bind(release),'request':bind(REQUEST/'ROOT_APPROVAL_REQUEST.json'),'invocation_count_limit':1,'retry_or_resume':False})
t=time.monotonic();started=now();proc=subprocess.Popen(request['remote_invocation_argv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
write('LOCAL_SSH_PROCESS_STARTED.json',{'UTC':started,'PID':proc.pid,'argv':request['remote_invocation_argv'],'monotonic_started_seconds':t})
print(json.dumps({'stage_exit':0,'inventory_invoked_once':True,'local_ssh_PID':proc.pid}),flush=True)
stdout,stderr=proc.communicate()
ex={'UTC':now(),'started_UTC':started,'PID':proc.pid,'exit_code':proc.returncode,'terminal_wait_observed':True,'elapsed_seconds':time.monotonic()-t,
 'stdout':stdout,'stderr':stderr,'invocations':1,'timeout_signals_retry_or_resume':False}
write('INVENTORY_EXIT.json',ex);print(json.dumps({'inventory_exit':proc.returncode,'elapsed_seconds':ex['elapsed_seconds']}),flush=True)
assert proc.returncode==0,'Actual inventory launch failed/uncertain; preserve same output, never rerun.'
source=r'''
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,socket,subprocess
repo=Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs');phase=repo/'experiments_iclr/postsubmission_20260930'
assert Path.cwd().resolve()==repo and socket.gethostname()=='anogena-2-0'
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).split();assert uuids==['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
root=phase/OUTPUT_TEXT;names=['HISTORY_INVENTORY_RELEASE.json','VALID_HISTORY_INVENTORY.json']
assert {p.name for p in root.iterdir()}==set(names) and root.resolve(strict=True).is_relative_to(phase) and not root.is_symlink()
value=json.loads((root/'VALID_HISTORY_INVENTORY.json').read_text())
assert value['schema']=='retrospective_allocation_replication_full39_history_inventory_v1' and value['complete'] is True and value['selected_logical_scientific_fits']==39 and value['new_physical_fits']==26
assert value['created_after_all39_terminal_source_artifact_checks'] is True and value['history_or_FREEZE_JSON_parsed'] is False and value['scores_read'] is False and value['pre_fit_authority'] is False
assert value['complete39_registry']==REGISTRY_TEXT and len(value['records'])==39
assert value['inventory_manifest_sha256']=='c29ccad053cfcaa4c11499854fdaca962ade48a910efa82435241d7d2f3e2ec6'
assert hashlib.sha256((root/'HISTORY_INVENTORY_RELEASE.json').read_bytes()).hexdigest()=='d84424c9a4746c24d26a1c96f9bc686fe9ed6038b3d575e2339b244bd6905c78'
files=[]
for name in names:
 p=root/name;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root) and p.stat().st_size<1_000_000
 b=p.read_bytes();before=hashlib.sha256(b).hexdigest();oldmode=oct(p.stat().st_mode&0o777);p.chmod(0o444)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==before and p.stat().st_mode&0o777==0o444
 files.append({'path':str(p.relative_to(phase)),'bytes':len(b),'sha256':before,'mode_before':oldmode,'mode':'0444','utf8':b.decode()})
assert hashlib.sha256((phase/'shared_private_transfer_allocation_replication_preparation_20261005_v2/ATTEMPT_HISTORY.json').read_bytes()).hexdigest()=='e3c70e8b22f71075078f376a82ec2534cb89d07cd508c4e2a2361dc2c0484970'
print(json.dumps({'UTC':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),'GPU_UUIDs':uuids,'files':files,'complete39_history_inventory':{'path':str((root/'VALID_HISTORY_INVENTORY.json').relative_to(phase)),'sha256':files[1]['sha256']},'history_JSON_or_scores_parsed':False,'no77_access_or_GPU_actions':True,'original_UNKNOWN_preserved':True}))
'''.replace('OUTPUT_TEXT',repr(request['output_phase_relative'])).replace('REGISTRY_TEXT',repr(request['actual_complete39_registry']))
with (HERE/'METADATA_GATHER_REMOTE_SOURCE.py').open('x') as f:f.write(source)
(HERE/'METADATA_GATHER_REMOTE_SOURCE.py').chmod(0o444)
compressed=base64.b64encode(zlib.compress(source.encode(),9)).decode()+'\n'
bootstrap="import sys,base64,zlib;exec(compile(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())), '<history-inventory-metadata-gather>', 'exec'))"
argv=request['remote_invocation_argv'][:-1]+['cd '+shlex.quote(REPO)+' && exec /usr/bin/python3 -I -S -B -c '+shlex.quote(bootstrap)]
t=time.monotonic();g=subprocess.run(argv,input=compressed,capture_output=True,text=True,timeout=55)
gather={'UTC':now(),'exit_code':g.returncode,'elapsed_seconds':time.monotonic()-t,'stderr':g.stderr,'argv':argv,'remote_source_sha256':sha(source.encode())}
if g.returncode==0:gather['result']=json.loads(g.stdout)
else:gather['stdout']=g.stdout
write('METADATA_GATHER_RECEIPT.json',gather);assert g.returncode==0,'Inventory succeeded; gather failed. Do not repeat inventory.'
for row in gather['result']['files']:
 b=row['utf8'].encode();assert len(b)==row['bytes'] and sha(b)==row['sha256'];local=HERE/'emitted_metadata'/Path(row['path']).name;local.parent.mkdir(exist_ok=True)
 with local.open('xb') as f:f.write(b)
 local.chmod(0o444)
inventory=json.loads((HERE/'emitted_metadata/VALID_HISTORY_INVENTORY.json').read_text())
registry=json.loads((P/'shared_private_transfer_complete39_collection_operation_20261006_v1/emitted_metadata/COLLECTION_FREEZE.json').read_text())
assert [r['cell_id'] for r in inventory['records']]==[r['cell_id'] for r in registry['completed']]
write('COMPLETION.json',{'schema':'actual_all39_history_BYTE_inventory_completion_v1','UTC':now(),'status':'SUCCESS_STOPPED_AFTER_ACTUAL_HISTORY_BYTE_INVENTORY',
 'actual_inventory':gather['result']['complete39_history_inventory'],'local_inventory':bind(HERE/'emitted_metadata/VALID_HISTORY_INVENTORY.json'),
 'actual_registry':request['actual_complete39_registry'],'source_sha256':request['inventory_source']['sha256'],'exit_code':0,'terminal_wait_observed':True,
 'inventory_invocations':1,'raw_history_files_hashed_as_bytes':39,'raw_history_copies':0,'metadata_outputs_0444':2,
 'history_score_FREEZE_CONFIG_semantics_tensors_held_D2_fit_model_GPU_or77_actions':False,'original_UNKNOWN_and_excluded20_preserved':True,
 'cost_seconds':{'staging':stage['elapsed_seconds'],'inventory':ex['elapsed_seconds'],'metadata_gather':gather['elapsed_seconds']}})
print(json.dumps({'status':'SUCCESS_STOPPED_AFTER_ACTUAL_HISTORY_BYTE_INVENTORY','actual_inventory':gather['result']['complete39_history_inventory'],
 'inventory_bytes':(HERE/'emitted_metadata/VALID_HISTORY_INVENTORY.json').stat().st_size,'inventory_elapsed_seconds':ex['elapsed_seconds']}),flush=True)
