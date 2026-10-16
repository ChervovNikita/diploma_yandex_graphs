"""Materialize the existing 6,730 canonical tasks into two dormant host releases.

Run on the allocation only. This reads frozen JSON/source bytes, writes a fresh
package beneath this successor, and never imports numerical libraries, starts
workers, claims tasks, or grants qualification/GO. Server51 paths are explicit
intended deployment paths, with separately retained actual staging receipts.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from descriptor_binding import (derive,digest,json_bytes,byte_receipt,relocate_path,require)
from host_binding import (HERE,WORKERS,ALLOCATION_ROOT,PHASE_RELATIVE,config_path,load_config,guard)
from ownership import CAMPAIGN,validate as validate_ownership
from reporting_projection import project_profile
from repair_bindings import repair_sources

E=HERE.parent
CANONICAL_SHA='be7bfe7c3beb5e5c9e639b53404455ee8b92b20f4dc038b05224fe141782ff64'
CANONICAL_BYTES=27035931
QUEUE_PINS={
 'allocation/gpu0':(3125790,'b65d11cb2b55a06fa4c7664ba151ce9e1497e52cf399b4c8ad562a93586f2ea4'),
 'server51/gpu0':(3125199,'d22eb84930f15f337e3b990449c99a9babf470d375338e7f087393dfc0ae5549'),
 'server51/gpu1':(3122401,'657c22c04e98fed8732d3aa42e63d37d8a3a261381b170efb317560fc03178cd')}
REPORT_SHA='cd2bc64047e630452eb7b4ce13a0829b205327876c5183d7e28853f6d4445db6'
RELEASE_SHA='d5c09b12cc5d73ec7207a9d8f2417e88fa37821ca1fdcaefdb23bc77546500a9'
CHANGES={'worker','canonical_runtime_job_hash','runtime_job_hash','runtime_binding','source_descriptor',
 'execution_science_contract_hash','canonical_task_sha256','binding','execution_status'}

def pin(path):
 path=Path(path).resolve();return byte_receipt(path,path.read_bytes())
def read_exact(path,expected=None):
 path=Path(path).resolve();actual=pin(path)
 if expected is not None:require(all(actual[k]==expected[k] for k in ('bytes','sha256') if k in expected),'Frozen input differs: '+str(path))
 return json.loads(path.read_text()),actual

def verify_canonical_plan(release,skeleton):
 actual=pin(release['canonical_skeleton']['path'])
 require(actual['sha256']==CANONICAL_SHA and actual['bytes']==CANONICAL_BYTES,'Use exact existing normalized canonical bytes')
 require(release['canonical_plan_origin']['sha256']==CANONICAL_SHA,'Canonical origin changed')
 require(skeleton['schema']=='normalization-portable-canonical-plan-v1' and len(skeleton['tasks'])==6730,'Fixed canonical inventory required')
 require(sum(t['kind']=='fit' for t in skeleton['tasks'])==6460,'Fixed6,460fits required');return True

def validate_projected_task(task,original,worker):
 require(task['worker']==worker and task['canonical_task_sha256']==digest(original),'Assignment/canonical task join changed')
 require({k:v for k,v in task.items() if k not in CHANGES}=={k:v for k,v in original.items() if k not in CHANGES},'Canonical task/science/controls/dependencies changed')
 require(task['science_contract_hash']==original['science_contract_hash'],'Canonical science identity changed')
 require(task.get('canonical_runtime_job_hash')==original.get('runtime_job_hash'),'Canonical runtime identity changed')
 if original['kind']=='fit':
  require(task.get('runtime_binding') and task.get('execution_science_contract_hash'),'Explicit execution binding required')
  require(all(task['source_descriptor'][k]==original['source_descriptor'][k] for k in ('bytes','sha256') if k in original['source_descriptor']),'Canonical descriptor copy is not the original bytes')
  require(task['runtime_binding']['canonical_descriptor_origin']==original['source_descriptor'],'Canonical descriptor origin changed')
 else:require(task['runtime_binding'] is None,'Native I4 remains an export-only task')
 return True

def assignment_map(plan,queue_dir):
 index={t['id']:t for t in plan['tasks']};require(len(index)==6730,'Fixed unique tasks required')
 owners={};order={};pins={}
 for worker in WORKERS:
  n,sha=QUEUE_PINS[worker];q,p=read_exact(Path(queue_dir)/('QUEUE_'+worker.replace('/','_')+'.json'),{'bytes':n,'sha256':sha})
  require(q['worker']==worker and q['launch_admitted'] is False and q['dispatch_supported'] is False,'Only exact dormant queues accepted')
  require(q['canonical_plan']['sha256']==CANONICAL_SHA,'Wrong queue campaign');order[worker]=[];pins[worker]=p
  for row in q['tasks']:
   fields=row['canonical_task'];fid=fields['id'];require(fid in index and fid not in owners,'Unknown or duplicate assigned task')
   require(all(index[fid].get(k)==v for k,v in fields.items()),'Compact queue fields changed canonical task')
   require(row['canonical_task_sha256']==digest(index[fid]),'Queue row changed canonical task bytes')
   require(all(dep in order[worker] for dep in fields['dependencies']),'Dependency absent, later or assigned elsewhere')
   owners[fid]=worker;order[worker].append(fid)
 require(set(owners)==set(index),'Dormant queues omit canonical tasks')
 cohorts={}
 for task in index.values():
  require(cohorts.setdefault(task['group_id'],owners[task['id']])==owners[task['id']],'Whole original cohort must remain co-located')
 return owners,order,pins

class Stager:
 def __init__(self,output,host):
  self.output=Path(output);self.host=host;self.root=Path(host['project_root']);self.files={}
  self.folder=self.output/host['host'];self.folder.mkdir()
  self.intended=self.root/host['phase_relative']/'benchmark_expansion_20261015_v1/server51_campaign_port_v2'/self.output.name/host['host']
 def raw(self,relative,raw):
  path=self.folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():require(path.read_bytes()==raw,'Staging artifact collision')
  else:
   with path.open('xb') as stream:stream.write(raw)
  intended=self.intended/relative;value=byte_receipt(intended,raw)
  self.files[str(intended)]={'staging':pin(path),'execution':value,'destination_bytes_verified':False};return value
 def json(self,relative,value):return self.raw(relative,json_bytes(value))
 def source(self,path,expected=None):
  """Verify source bytes here; retain explicit intended target receipt separately."""
  value=str(path);sr=PurePosixPath(ALLOCATION_ROOT);tr=PurePosixPath(self.host['project_root']);p=PurePosixPath(value)
  if p.is_relative_to(sr):source=Path(value);target=relocate_path(value,ALLOCATION_ROOT,PHASE_RELATIVE,str(tr),self.host['phase_relative'])
  elif p.is_relative_to(tr):
   target=value;source=Path(relocate_path(value,str(tr),self.host['phase_relative'],ALLOCATION_ROOT,PHASE_RELATIVE))
  else:raise ValueError('Source pin escapes explicit repos: '+value)
  actual=pin(source)
  if expected is not None:require(all(actual[k]==expected[k] for k in ('bytes','sha256') if k in expected),'Actual source bytes differ: '+str(source))
  return dict(actual,path=target)

def prepare(canonical_path,queue_dir,report_index_path,parent_release_path,output):
 output=Path(output).resolve();require(output.parent==HERE and not output.exists(),'Fresh direct successor child output required')
 plan,plan_pin=read_exact(canonical_path,{'bytes':CANONICAL_BYTES,'sha256':CANONICAL_SHA})
 report,report_pin=read_exact(report_index_path,{'sha256':REPORT_SHA})
 parent,parent_pin=read_exact(parent_release_path,{'sha256':RELEASE_SHA})
 require(parent['all_tasks_qualified'] is False and parent['human_GO'] is False,'Never transfer old admission')
 owners,order,queues=assignment_map(plan,queue_dir)
 configs={w:load_config(config_path(w))[0] for w in WORKERS}
 hosts={host:{'host':host,'project_root':configs[next(w for w in WORKERS if w.startswith(host+'/'))]['project_root'],
  'phase_relative':PHASE_RELATIVE,'maximum_children_per_GPU':2} for host in ('allocation','server51')}
 assignments=[{'id':t['id'],'worker':owners[t['id']],'canonical_runtime_job_hash':t.get('runtime_job_hash'),
  'science_contract_hash':t['science_contract_hash'],'dependencies':copy.deepcopy(t['dependencies']),
  'output_relative':t['output_relative'],'cohort':t['group_id']} for t in plan['tasks']]
 ownership={'schema':'portable-global-ownership-v2','campaign':CAMPAIGN,'hosts':hosts,'stop_issuer':'reviewed_campaign_operator','assignments':assignments}
 validate_ownership(ownership,plan['tasks'])
 roster=json.loads((E/'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
 profiles=[(read_exact(p['path'],p)[0],p) for p in report['preset_files']]
 require(len(profiles)==27,'Keep all27report profiles')
 index={t['id']:t for t in plan['tasks']};output.mkdir();results={}
 for host in hosts.values():
  stage=Stager(output,host);canonical_copy=stage.raw('CANONICAL_PLAN.json',Path(canonical_path).read_bytes());owner_pin=stage.json('OWNERSHIP.json',ownership)
  tasks=[];candidates={}
  ids=[fid for worker in WORKERS if worker.startswith(host['host']+'/') for fid in order[worker]]
  for fid in ids:
   original=index[fid];task=copy.deepcopy(original);task.update(worker=owners[fid],canonical_runtime_job_hash=original.get('runtime_job_hash'),canonical_task_sha256=digest(original),runtime_binding=None)
   if original['kind']=='fit':
    job,origin_pin=read_exact(original['source_descriptor']['path'],original['source_descriptor'])
    require(digest(job)==original['runtime_job_hash'],'Canonical job differs')
    canonical_job=stage.raw('canonical_jobs/'+hashlib.sha256(fid.encode()).hexdigest()+'.json',Path(origin_pin['path']).read_bytes())
    parent_job=parent_raw=parent_copy=None;exec_parent=None
    if job.get('normalization_composition'):
     old=job['normalization_composition']['parent_descriptor'];parent_job,_=read_exact(old['path'],old);parent_raw=Path(old['path']).read_bytes()
     parent_copy=stage.raw('canonical_parents/'+old['sha256']+'.json',parent_raw)
     exec_parent=stage.intended/'execution_parents'/(hashlib.sha256(fid.encode()).hexdigest()+'.json')
    bound,parent_bytes,binding=derive(job,ALLOCATION_ROOT,PHASE_RELATIVE,host['project_root'],host['phase_relative'],
     parent=parent_job,parent_raw=parent_raw,execution_parent_path=exec_parent,expansion=E,roster=roster)
    binding['canonical_descriptor_origin']=copy.deepcopy(original['source_descriptor']);binding['canonical_descriptor_copy']=canonical_job
    if parent_copy:
     binding['canonical_parent_copy']=parent_copy
     require(stage.raw('execution_parents/'+exec_parent.name,parent_bytes)==binding['execution_parent_descriptor'],'Actual staged rebound parent bytes differ')
    descriptor=stage.json('jobs/'+hashlib.sha256(fid.encode()).hexdigest()+'.json',bound)
    task.update(source_descriptor=canonical_job,runtime_job_hash=digest(bound),runtime_binding=binding,execution_science_contract_hash=bound['expanded_contract_hash'])
    runtime=HERE/('composed/runtime' if bound.get('normalization_composition') else 'runtime_link' if bound['family']=='link' else 'runtime_node_molecular')/'run_fit.py'
   else:descriptor=None;bound=None;runtime=HERE/'runtime_I4/aggregate_selected.py'
   task.update(binding=None,execution_status='unadmitted_explicit_destination_binding')
   validate_projected_task(task,original,owners[fid]);tasks.append(task)
   candidates[fid]={'id':fid,'worker':owners[fid],'job_descriptor':descriptor,'runtime_entrypoint':stage.source(runtime),
    'canonical_runtime_job_hash':original.get('runtime_job_hash'),'runtime_job_hash':task.get('runtime_job_hash'),
    'science_contract_hash':original['science_contract_hash'],'execution_science_contract_hash':task.get('execution_science_contract_hash'),
    'admitted':False,'_job':bound}
  projected_profiles=[]
  by_id={t['id']:t for t in tasks}
  scorer=stage.source(HERE/'scoring/MANIFEST.json')
  for profile,parent_profile_pin in profiles:
   value=project_profile(profile,parent_profile_pin,by_id,candidates,host,stage.source);value['selected_serving_source']=scorer
   projected_profiles.append(stage.json('reporting/'+Path(parent_profile_pin['path']).name,value))
  projected_index=copy.deepcopy(report);projected_index.update(preset_files=projected_profiles,selected_serving_source=scorer,
   campaign_output_root=str(Path(host['project_root'])/host['phase_relative']/'benchmark_expansion_20261015_v1/campaign_outputs'/CAMPAIGN),
   canonical_registry=report_pin,canonical_counts=copy.deepcopy(report['counts']))
  projected_index['host_projection']={'schema':'allocation-server51-reporting-registry-projection-v1','host':host['host'],
   'canonical_registry':report_pin,'observer_source':stage.source(HERE/'reporting/observe_exports.py'),
   'collector_source':stage.source(HERE/'reporting/collect_campaign.py'),'runtime_entry_source':stage.source(HERE/'reporting/runtime_entry.py'),
   'composed_authority_source':stage.source(HERE/'reporting/composed_authority.py'),
   'native_qualification_granted':False,'human_GO':False,'source_reference_counts_remain_global':True}
  projected_index['parent_reporter_source']=stage.source(report['parent_reporter_source']['path'],report['parent_reporter_source'])
  registry=stage.json('reporting/INDEX.json',projected_index)
  hook={'bridge':stage.source(HERE/'reporting/backend_bridge.py'),'entrypoint':stage.source(HERE/'reporting/runtime_entry.py'),
   'registry':registry,'manifest':stage.source(HERE/'reporting/MANIFEST.json')}
  closure={}
  for old in parent['source_files']:
   value=stage.source(old['path'],old);closure[value['path']]=value
  # These finite source closures are reviewed separately from the frozen parent.
  dependencies=repair_sources()
  for relative in ('composed/SOURCE_BINDINGS.json','scoring/SOURCE_BINDINGS.json','TRANSITIVE_SOURCE_CLOSURE_ADDITIONS.json'):
   dependencies+=json.loads((HERE/relative).read_text())['files']
  for row in dependencies:
   value=stage.source(E.parent/row['path_relative_to_phase'],row);closure[value['path']]=value
  package=json.loads((HERE/'MANIFEST.json').read_text())
  for path in [HERE/'MANIFEST.json',*[HERE/row['path'] for row in package['files']]]:
   value=stage.source(path);closure[value['path']]=value
  for value in projected_profiles+[registry]:closure[value['path']]=value
  release=copy.deepcopy(parent);release.update(canonical_skeleton=canonical_copy,canonical_plan_origin=plan_pin,
   ownership=owner_pin,host=host['host'],tasks=tasks,source_files=list(closure.values()),source_bundle_sha256=digest({p['path']:p['sha256'] for p in closure.values()}),
   output_root=projected_index['campaign_output_root'],host_config_pins={w:stage.source(config_path(w)) for w in WORKERS if w.startswith(host['host']+'/')},
   legacy_supervisor_sources={k:stage.source(v['path'],v) for k,v in parent['legacy_supervisor_sources'].items()},
   all_tasks_qualified=False,human_GO=False,TEST_access=False,reporting=hook)
  release['server51_successor']={'source_manifest':stage.source(HERE/'MANIFEST.json'),'parent_release':parent_pin,
   'dormant_queue_pins':queues,'canonical_registry_origin':report_pin,'canonical_source_preserved':True,
   'actual_admissions':0,'claims_created':0,'source_scoring_repair_parent':stage.source(E/'composable_normalization_final_scoring_v3/MANIFEST.json'),
   'assignment_target_phase_is_provenance_only':True,'execution_phase':host['phase_relative']}
  release_pin=stage.json('RELEASE_UNADMITTED.json',release)
  for worker in WORKERS:
   if not worker.startswith(host['host']+'/'):continue
   stage.json('RUN_SPEC_DRAFT_'+worker.replace('/','_')+'.json',{'schema':'portable-reviewed-run-v2','worker':worker,
    'host_config':stage.source(config_path(worker)),'release':release_pin,'ownership':owner_pin,'go':None,'readiness':None,'reporting':hook,
    'preparation_only':True,'launch_admitted':False})
  candidates_pin=stage.json('CANDIDATES.json',[{k:v for k,v in row.items() if k!='_job'} for row in candidates.values()])
  result={'schema':'allocation-server51-host-stage-v1','host':host,'release':release_pin,'candidates':candidates_pin,
   'counts':{'tasks':len(tasks),'fits':sum(t['kind']=='fit' for t in tasks),'exports':sum(t['kind']!='fit' for t in tasks)},
   'staged_files':list(stage.files.values()),'destination_bytes_verified':False,'human_GO':False,'claims_created':0,'children_started':0}
  stage.json('STAGING_RECEIPTS.json',result);results[host['host']]={k:v for k,v in result.items() if k!='staged_files'}
 require(sum(r['counts']['fits'] for r in results.values())==6460,'Fit count changed')
 require('torch' not in sys.modules and 'numpy' not in sys.modules,'Metadata preparation imported numerical libraries')
 summary={'schema':'allocation-server51-campaign-preparation-v1','source_manifest':pin(HERE/'MANIFEST.json'),
  'canonical_plan':plan_pin,'hosts':results,'total_fits':6460,'total_exports':270,'human_GO':False,'claims_created':0,'native_work':0}
 with (output/'PREPARATION.json').open('x') as stream:json.dump(summary,stream,indent=2,sort_keys=True);stream.write('\n')
 return summary

def main():
 c,_=load_config(config_path('allocation/gpu0'));guard(c)
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('canonical-plan','queue-dir','reporting-index','parent-release','output','source-manifest-sha256'):parser.add_argument('--'+name,required=True)
 args=parser.parse_args();require(pin(HERE/'MANIFEST.json')['sha256']==args.source_manifest_sha256,'Reviewed successor source pin required')
 for row in json.loads((HERE/'MANIFEST.json').read_text())['files']:
  actual=pin(HERE/row['path']);require(all(actual[k]==row[k] for k in ('bytes','sha256')),'Successor source member differs: '+row['path'])
 result=prepare(args.canonical_plan,args.queue_dir,args.reporting_index,args.parent_release,args.output)
 print(json.dumps({k:v for k,v in result.items() if k!='hosts'},sort_keys=True))
if __name__=='__main__':main()
