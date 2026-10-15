"""Host-local observer profiles with unchanged canonical scientific identities."""
import copy
import json
from pathlib import Path
from descriptor_binding import digest, require
RUNTIME_NAMES=('run_fit.py','adapters.py','bookkeeping.py','common.py','controllers.py','source_rng.py')
COMMON=('native_source','data','controller','evaluator')

def project_profile(parent,parent_pin,tasks,candidates,host,source_pin):
 """Callbacks supplied by the metadata stager pin real staged/source bytes.

source_pin(path) returns target-local source bytes verified from the actual
source checkout. No runtime/qualification authority is inferred from this.
"""
 target=copy.deepcopy(parent);target['fit_specs']={};target['I4_specs']={}
 for fid,spec in parent['fit_specs'].items():
  if fid not in tasks:continue
  task=tasks[fid];candidate=candidates[fid];job=candidate['_job'];value=copy.deepcopy(spec)
  value['canonical_job_hash']=spec['job_hash'];value['canonical_descriptor']=copy.deepcopy(spec['descriptor'])
  value['canonical_wrapper_authority']=copy.deepcopy(spec['wrapper_authority'])
  value.update(job_hash=task['runtime_job_hash'],descriptor=copy.deepcopy(candidate['job_descriptor']),runtime_entrypoint=copy.deepcopy(candidate['runtime_entrypoint']))
  value['wrapper_authority']['descriptor']=copy.deepcopy(candidate['job_descriptor'])
  if job.get('normalization_composition'):value['wrapper_authority']['normalization_composition']=copy.deepcopy(job['normalization_composition'])
  observed=[source_pin(p['path'],p) for p in spec['observed_input_files']]
  if job.get('normalization_composition'):
   base=Path(candidate['runtime_entrypoint']['path']).parent.parent
   expansion=base.parent.parent
   helper=expansion/'portable_dispatcher_v3'/('runtime_link' if job['family']=='link' else 'runtime_node_molecular')
   runtime_pins=[source_pin(base/'runtime'/name) for name in RUNTIME_NAMES]
   helper_pins=[source_pin(helper/name) for name in RUNTIME_NAMES]
   producer=source_pin(base/'MANIFEST.json');normalization=source_pin(expansion/'composable_normalization_v2/MANIFEST.json')
   pins=[*runtime_pins,*helper_pins,producer,normalization,source_pin(base/'runtime_bridge.py')]
   authority={'schema':'normalization-reporting-runtime-authority-v1','project_root':host['project_root'],
    'producer_manifest':producer,'entrypoint':candidate['runtime_entrypoint'],'normalization_source':normalization,
    'normalization_composition':copy.deepcopy(job['normalization_composition']),
    'runtime_sha256':{Path(p['path']).name:p['sha256'] for p in runtime_pins},
    'helper_runtime_sha256':{Path(p['path']).name:p['sha256'] for p in helper_pins},
    'callback_sha256':job['callback_sha256'],'extension_sha256':(job.get('session_transform') or {}).get('sha256'),
    'policy_sha256':job.get('link_policy_sha256') if job['family']=='link' else None,'source_files':list({p['path']:p for p in pins}.values())}
   value['composed_runtime_authority']=authority;observed+=authority['source_files']
  else:value.pop('composed_runtime_authority',None)
  # Keep canonical comparison hashes; runtime bytes and host paths have separate explicit authority.
  require(value['identity']==spec['identity'] and value['bindings']==spec['bindings'],'Scientific reporting identity changed')
  value['observed_input_files']=list({p['path']:p for p in observed}.values())
  value['execution_binding']=copy.deepcopy(task['runtime_binding']);target['fit_specs'][fid]=value
 for tid,spec in parent['I4_specs'].items():
  if tid not in tasks:continue
  require(all(dep in tasks for dep in spec['component_fit_ids']),'I4 components cannot cross host')
  value=copy.deepcopy(spec);value['canonical_aggregator_source']=copy.deepcopy(spec['aggregator_source'])
  value['aggregator_source']=copy.deepcopy(candidates[tid]['runtime_entrypoint']);target['I4_specs'][tid]=value
 require(target['registry']==parent['registry'] and target['contrast_plan']==parent['contrast_plan'],'Canonical hypotheses/contrasts changed')
 target['canonical_profile']=parent_pin
 target['host_projection']={'schema':'allocation-server51-reporting-profile-projection-v1','host':host['host'],
  'parent_profile_sha256':digest(parent),'local_tasks':len(target['fit_specs'])+len(target['I4_specs']),
  'canonical_scientific_identities_preserved':True,'missing_other_host_results_are_not_zero':True,
  'native_qualification_granted':False,'human_GO':False}
 return target
