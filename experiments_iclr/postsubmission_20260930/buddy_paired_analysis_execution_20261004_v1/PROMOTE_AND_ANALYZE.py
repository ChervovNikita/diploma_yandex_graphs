"""Promote exact authentic JSON custody inputs and run only the unchanged paired analyzer."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,resource,subprocess,time

HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def descriptor(path):return dict(path=str(path.relative_to(PHASE)),bytes=path.stat().st_size,sha256=sha(path))
def read(path):return json.loads(path.read_text())
def safe(path):
 assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
 for parent in path.parents:
  if parent==PHASE:break
  assert not parent.is_symlink()
 return path
source=PHASE/'buddy_paired_analysis_v1/analyze.py'
assert sha(source)=='7487df7e16ac8d4781ef736c1cd92df0c84b205914931e4c57ef010138232e5c'
plan=source.with_name('PLAN.md');plan_sha=sha(plan)
transport_path=PHASE/'gpu77_connection_recovery_v1/commands/buddy_v4_original_family_terminal_fetch_20261004_v1/RECEIPT.json'
transport=read(transport_path);assert transport['exit_code']==0
fetched=json.loads(transport['stdout'].strip());assert len(fetched['files'])==1 and fetched['remote_writes'] is False
row=fetched['files'][0];raw=base64.b64decode(row['data_base64'],validate=True)
assert len(raw)==row['bytes']==4521 and hashlib.sha256(raw).hexdigest()==row['sha256']=='156eb215ab775e8f45d4922536196e77da4adf9891c3ca804cd16708b7fd9e0e'
terminal=HERE/'ORIGINAL_FAMILY_LAUNCH_RECEIPT.json'
with terminal.open('xb') as h:h.write(raw)
heldout=PHASE/'buddy_gpu77_postfamily_heldout_evaluation_execution_20261004_v1'
monitor=read(heldout/'MONITOR_RESULT_01.json')
assert monitor['status']=='ALL15_PHYSICALLY_COMPLETE'
promotions=[]
for item in monitor['files']:
 relative=Path(item['path']).relative_to('experiments_iclr/postsubmission_20260930')
 if relative.name=='final_test.json' or str(relative)=='buddy_gpu77_postfamily_eval_preparation_v4/root_eval_v1/EVALUATION_RECEIPT.json' or relative.name=='test_manifest.json':
  promotions.append((heldout/item['local_path'],PHASE/relative,item['sha256']))
old=PHASE/'buddy_gpu77_postfamily_lock_audit_execution_20261004_v1/fetched_registered_outputs_v1/root_lock_v1'
for name,digest in [('FAMILY_LOCK.json','4d4041ad0d02a36c94bd9112f4e01029fb722eee75339a5743a0259435c33fd9'),
 ('LOCK_AUDIT.json','2c9d8b13759c88cfb657782c3e6a22938388e91f171e8a67ab49616c9b392b38')]:
 promotions.append((old/name,PHASE/'buddy_gpu77_postfamily_eval_preparation_v4/root_lock_v1'/name,digest))
promotions.append((terminal,PHASE/'buddy_gpu77_resource_family_launcher_v3/root_family_v2/FAMILY_LAUNCH_RECEIPT.json',row['sha256']))
assert len(promotions)==20 and len({str(target) for _,target,_ in promotions})==20
# Inspect every collision before making the first canonical write.
for original,target,digest in promotions:
 safe(original);safe(target)
 assert original.suffix==target.suffix=='.json' and original.stat().st_size<2*1024*1024 and sha(original)==digest
 if target.exists():assert target.is_file() and target.read_bytes()==original.read_bytes(),str(target)
cost_start=time.monotonic();rows=[]
for original,target,digest in promotions:
 raw=original.read_bytes();safe(target);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  assert target.read_bytes()==raw;action='existing_identical_bytes_preserved'
 else:
  with target.open('xb') as h:h.write(raw)
  action='exclusive_authentic_metadata_promotion'
 assert sha(target)==digest
 rows.append(dict(source=descriptor(original),target=descriptor(target),action=action))
custody=dict(schema='buddy77_authentic_scalar_analysis_input_promotion_custody_v1',UTC=datetime.now(timezone.utc).isoformat(),
 rows=rows,files=20,only_authentic_JSON_metadata_promoted=True,existing_differing_bytes_overwritten=False,
 source_sha256=sha(source),plan_sha256=plan_sha,terminal_fetch_transport=descriptor(transport_path),
 promotion_wall_seconds=time.monotonic()-cost_start,scientific_binary_payloads_fetched=False)
with (HERE/'CANONICAL_METADATA_CUSTODY.json').open('x') as h:json.dump(custody,h,indent=2);h.write('\n')
output=HERE/'report';assert not output.exists()
argv=['/usr/bin/python3','-I','-S','-B',str(source),'--output',str(output.relative_to(PHASE))]
before=resource.getrusage(resource.RUSAGE_CHILDREN);began=datetime.now(timezone.utc).isoformat();start=time.monotonic()
result=subprocess.run(argv,cwd=PHASE,capture_output=True,text=True)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
record=dict(schema='buddy77_predeclared_paired_analysis_execution_v1',start_UTC=began,terminal_UTC=datetime.now(timezone.utc).isoformat(),
 argv=argv,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,
 physical_subprocess_wall_seconds=time.monotonic()-start,owned_descendant_CPU_user_seconds=after.ru_utime-before.ru_utime,
 owned_descendant_CPU_system_seconds=after.ru_stime-before.ru_stime,
 owned_descendant_peak_RSS_bytes=after.ru_maxrss,
 peak_RSS_scope='macOS RUSAGE_CHILDREN ru_maxrss is bytes; this fresh wrapper launches exactly one analyzer child.',
 source=descriptor(source),plan=descriptor(plan),fixed_paired_contrasts=3,full_family_cells=15,
 labels_logits_checkpoints_or_models_read=False,additional_model_forwards=0,source_or_plan_changed=False)
assert sha(source)==record['source']['sha256'] and sha(plan)==plan_sha
if result.returncode==0:
 report=read(output/'REPORT.json')
 assert report['source_sha256']==sha(source) and report['plan_sha256']==plan_sha
 assert len(report['inputs'])==20 and len(report['contrasts'])==3
 assert report['additional_model_forwards']==0 and report['original_scores_changed'] is False
 record['reports']=[descriptor(output/name) for name in ('REPORT.json','REPORT.md')]
 record['status']='PASS_FULL15_THREE_FIXED_CONTRASTS'
else:record['status']='FAILED_PRESERVED_NO_AUTOMATIC_RETRY'
with (HERE/'ANALYSIS_EXECUTION_RECEIPT.json').open('x') as h:json.dump(record,h,indent=2);h.write('\n')
print(json.dumps(record))
raise SystemExit(result.returncode)
