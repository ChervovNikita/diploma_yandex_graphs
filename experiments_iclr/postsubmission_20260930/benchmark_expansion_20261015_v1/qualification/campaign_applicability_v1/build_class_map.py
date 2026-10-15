"""Map exact staged task IDs to finite functional/resource applicability classes.

Metadata only. No PASS gate, admission, claim, runtime, model or dataset payload
is created. Uses the existing host staging receipts, including server51's actual
allocation staging paths rather than pretending target paths already exist.
"""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
GATES=('source_context','provider_split_evaluator','native_epoch_controller','selected_export_resume_rng',
 'mature_target_resource','reporting_statistics_access','source_reference_policy_or_matched_controller','campaign_scope_bridge')
def require(ok,message):
 if not ok:raise ValueError(message)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def pin(path):
 path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read_pinned(path,sha):
 actual=pin(path);require(actual['sha256']==sha,'Pinned input changed');return json.loads(Path(path).read_text()),actual

def build(stage,read_staged,facts_pin):
 release=read_staged(stage['release']);candidates=read_staged(stage['candidates']);by_id={r['id']:r for r in candidates}
 require(len(by_id)==len(candidates)==len(release['tasks']),'Exact stage candidate/task coverage required')
 classes={};tasks=[];inherited=Counter();gate_map={g:[] for g in GATES}
 for task in release['tasks']:
  candidate=by_id[task['id']];is_fit=task['kind']=='fit';job=read_staged(candidate['job_descriptor']) if is_fit else None
  key={'worker':task['worker'],'cell_id':task['cell_id'],'kind':task['kind'],'family':task.get('family'),
   'scientific_method_id':task.get('scientific_method_id'),'protocol_track':task.get('protocol_track'),
   'runtime_entrypoint_sha256':candidate['runtime_entrypoint']['sha256']}
  if job is not None:
   key.update(callback_sha256=job['callback_sha256'],session_transform_sha256=(job.get('session_transform') or {}).get('sha256'),
    link_policy_sha256=job.get('link_policy_sha256'),source_rng_audit_sha256=(job.get('source_rng_emulation') or {}).get('audit_sha256'),
    normalization_source_sha256=(job.get('normalization_composition') or {}).get('source_manifest_sha256'),
    normalization_policy=(job.get('normalization_composition') or {}).get('policy'),controller_sha256=digest(job['policy']))
  cid=digest(key);entry=classes.setdefault(cid,{'class_id':cid,'source_and_execution_signature':key,'task_ids':[],'split_indices':[],
   'status':'requires_reviewed_applicability_not_a_native_PASS','actual_shared_evidence_catalogue':facts_pin,
   'required_gate_decisions':list(GATES),'resource_profile_pending':True})
  entry['task_ids'].append(task['id'])
  if task.get('split') not in entry['split_indices']:entry['split_indices'].append(task.get('split'))
  for old in task.get('qualification_blocks',[]):inherited[old]+=1
  for gate in GATES:gate_map[gate].append({'task_id':task['id'],'class_id':cid,'science_contract_hash':task['science_contract_hash']})
  tasks.append({'id':task['id'],'class_id':cid,'worker':task['worker'],'canonical_runtime_job_hash':task.get('canonical_runtime_job_hash'),
   'execution_runtime_job_hash':task.get('runtime_job_hash'),'canonical_science_contract_hash':task['science_contract_hash'],
   'execution_science_contract_hash':task.get('execution_science_contract_hash'),'execution_candidate':candidate,
   'source_attachment':task.get('functional_evidence_attachment'),'dependencies':task['dependencies'],'output_relative':task['output_relative']})
 counts={'tasks':len(tasks),'fits':sum(t['kind']=='fit' for t in release['tasks']),'I4_exports':sum(t['kind']!='fit' for t in release['tasks']),
  'review_classes':len(classes),'new_native_runs':0,'PASS_gates_created':0,'claims_created':0}
 return {'schema':'campaign6730-task-component-applicability-map-v1','host':release['host'],'release':stage['release'],
  'source_manifest':release['server51_successor']['source_manifest'],'evidence_catalogue':facts_pin,'counts':counts,
  'classes':list(classes.values()),'tasks':tasks,'gate_applications':gate_map,
  'inherited_dormant_blocks_as_history_not_fresh_failures':dict(inherited),'human_GO':False,'native_work':0}

def main():
 p=argparse.ArgumentParser()
 for field in ('staging-receipts','staging-receipts-sha256','output'):p.add_argument('--'+field,required=True)
 args=p.parse_args();stage,stage_pin=read_pinned(args.staging_receipts,args.staging_receipts_sha256)
 files={row['execution']['path']:row for row in stage['staged_files']}
 def read_staged(receipt):
  require(receipt['path'] in files,'Artifact absent from exact staging receipts')
  row=files[receipt['path']];require(row['execution']==receipt,'Execution byte receipt differs')
  path=row['staging']['path'];actual=pin(path)
  require(actual==row['staging'] and all(actual[k]==receipt[k] for k in ('bytes','sha256')),'Actual staging bytes differ')
  return json.loads(Path(path).read_text())
 result=build(stage,read_staged,pin(HERE/'ACTUAL_EVIDENCE_SNAPSHOT.json'));result['actual_staging_receipts']=stage_pin
 output=Path(args.output).resolve();require(output.parent==HERE and not output.exists(),'Fresh direct application-plan output required')
 with output.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps({'output':pin(output),'counts':result['counts'],'human_GO':False},sort_keys=True))
if __name__=='__main__':main()
