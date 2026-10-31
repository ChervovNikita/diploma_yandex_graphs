"""Pure, typed canonical→execution bindings, including actual composed parents.

Canonical descriptors and their hashes remain immutable history. Execution
parents are serialized first; new parent byte receipts and composition hashes
are derived from those bytes. No old receipt is relabeled as new parent bytes.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath

# The composed facade loads this module by absolute source path in a fresh worker.
_HERE=Path(__file__).resolve().parent
_repair_spec=importlib.util.spec_from_file_location('_server51_port5_repair_bindings',_HERE/'repair_bindings.py')
_repair_module=importlib.util.module_from_spec(_repair_spec);_repair_spec.loader.exec_module(_repair_module)
rebind_buddy=_repair_module.rebind_buddy

RUNTIME_FIELDS=('/callback_module','/data_root','/recipe/_repository_root','/recipe/_redesign_root',
 '/recipe/redesign_root','/link_policy_module','/source_rng_emulation/audit','/session_transform/path',
 '/extension_factory/path','/session_extension/path','/method_settings/context_manifest','/recipe/_native_callback',
 '/parent_compiler_descriptor/path','/normalization_composition/parent_descriptor/path')
PROVENANCE={'expanded_contract','descriptor_preparation','source_plan_entry'}
DATA_FIELDS=('graph','train','valid','test')
NORMALIZATION_SHA='cedb1bb2125f6d5ea78ea136d26ce17ac4ebbd33bd806b88545d270989c9eb5d'
BANKS_SHA='d90a74bf98be32408cad306f35ab5ebd1ec80153981a0b7f92cba07e7f447fea'
def require(ok,message):
 if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def byte_receipt(path,raw):return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def normalized_absolute(value):
 require(isinstance(value,str) and value.startswith('/'),'Absolute POSIX path required')
 p=PurePosixPath(value);require(str(p)==value and '..' not in p.parts,'Normalized POSIX path required');return p
def leaves(value,path=''):
 if isinstance(value,dict):
  for k,v in value.items():yield from leaves(v,path+'/'+k.replace('~','~0').replace('/','~1'))
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from leaves(v,path+'/'+str(i))
 else:yield path,value
def is_provenance(pointer):return pointer.split('/')[1] in PROVENANCE
def allowed(pointer):return pointer in RUNTIME_FIELDS or pointer in tuple('/recipe/data_files/'+k for k in DATA_FIELDS)
def set_field(value,pointer,replacement):
 fields=[x.replace('~1','/').replace('~0','~') for x in pointer.lstrip('/').split('/')]
 for k in fields[:-1]:value=value[int(k)] if isinstance(value,list) else value[k]
 value[fields[-1]]=replacement
def audit_runtime_paths(job,root,phase_relative):
 root=normalized_absolute(root);phase=PurePosixPath(phase_relative)
 require(not phase.is_absolute() and '..' not in phase.parts,'Safe phase relative path required')
 rows=[]
 for pointer,value in leaves(job):
  if is_provenance(pointer) or not isinstance(value,str) or not value.startswith('/'):continue
  require(allowed(pointer),'Unclassified absolute runtime field: '+pointer)
  require(normalized_absolute(value).is_relative_to(root),'Runtime path escapes host root: '+pointer)
  rows.append({'pointer':pointer,'path':value})
 require(job['recipe']['_repository_root']==str(root),'Recipe root differs')
 require(job['recipe']['_redesign_root']==str(root/phase/'reference_wrapper_redesign_20261013_v1'),'Recipe phase differs')
 if job.get('expanded_contract'):require(digest(job['expanded_contract'])==job['expanded_contract_hash'],'Expanded contract hash differs')
 return rows
def relocate_path(value,source_root,source_phase,target_root,target_phase):
 p=normalized_absolute(value);sr=normalized_absolute(source_root);tr=normalized_absolute(target_root)
 require(p.is_relative_to(sr),'Runtime path escapes source root')
 sp=sr/source_phase;tp=tr/target_phase
 return str(tp/p.relative_to(sp)) if p.is_relative_to(sp) else str(tr/p.relative_to(sr))
def rebind_plain(original,source_root,source_phase,target_root,target_phase):
 require('normalization_composition' not in original,'Composed job requires a materialized rebound parent')
 audit_runtime_paths(original,source_root,source_phase)
 result=copy.deepcopy(original);changes=[]
 for pointer,value in leaves(original):
  if is_provenance(pointer) or not allowed(pointer) or not isinstance(value,str):continue
  replacement=relocate_path(value,source_root,source_phase,target_root,target_phase)
  if replacement!=value:set_field(result,pointer,replacement);changes.append({'pointer':pointer,'from':value,'to':replacement})
 audit_runtime_paths(result,target_root,target_phase)
 for field in PROVENANCE:require(result.get(field)==original.get(field),'Scientific/provenance field changed')
 return result,changes
def composition_contract(expansion):
 path=Path(expansion)/'composable_normalization_v2/contracts.py'
 spec=importlib.util.spec_from_file_location('_server51_composition_contracts',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def derive(original,source_root,source_phase,target_root,target_phase,*,parent=None,parent_raw=None,execution_parent_path=None,expansion=None,roster=None):
 """Return execution job, serialized parent bytes, and an explicit identity join."""
 audit_runtime_paths(original,source_root,source_phase)
 if 'normalization_composition' not in original:
  require(parent is None and execution_parent_path is None,'Plain binding cannot smuggle a parent')
  bound,changes=rebind_plain(original,source_root,source_phase,target_root,target_phase);raw=None;exec_parent=None
  bound,source_repair=rebind_buddy(bound,target_root,target_phase)
 else:
  declaration=original['normalization_composition'];require(parent is not None and parent_raw is not None and execution_parent_path is not None,'Actual canonical parent bytes and intended execution path required')
  require(set(declaration)=={'schema','policy','source_manifest_sha256','banks_sha256','parent_descriptor','parent_job_sha256','matched_control_method_id'},'Exact composed declaration required')
  require(declaration['source_manifest_sha256']==NORMALIZATION_SHA and declaration['banks_sha256']==BANKS_SHA,'Frozen normalization/bank law required')
  old_pin=byte_receipt(declaration['parent_descriptor']['path'],parent_raw)
  require(old_pin==declaration['parent_descriptor'] and json.loads(parent_raw)==parent and digest(parent)==declaration['parent_job_sha256'],'Canonical parent bytes/identity differ')
  contracts=composition_contract(expansion);require(original==contracts.migrated_job(parent,declaration,roster),'Canonical composed child differs from exact original migration')
  rebound,changes=rebind_plain(parent,source_root,source_phase,target_root,target_phase)
  rebound,source_repair=rebind_buddy(rebound,target_root,target_phase)
  require(normalized_absolute(str(execution_parent_path)).is_relative_to(normalized_absolute(target_root)),'Execution parent must be inside target repo')
  raw=json_bytes(rebound);exec_parent=byte_receipt(execution_parent_path,raw)
  new_decl=copy.deepcopy(declaration);new_decl['parent_descriptor']=exec_parent;new_decl['parent_job_sha256']=digest(rebound)
  bound=contracts.migrated_job(rebound,new_decl,roster);audit_runtime_paths(bound,target_root,target_phase)
  require(bound['id']==original['id'],'Canonical fit ID cannot change')
 return bound,raw,{'schema':'canonical-to-host-execution-binding-v2','id':original['id'],
  'canonical_runtime_job_hash':digest(original),'bound_runtime_job_hash':digest(bound),
  'canonical_science_contract_hash':original.get('expanded_contract_hash'),
  'execution_science_contract_hash':bound.get('expanded_contract_hash'),'field_changes':changes,
  'source_binding_repair':source_repair,
  'canonical_normalization_declaration':copy.deepcopy(original.get('normalization_composition')),
  'execution_parent_descriptor':exec_parent,'scientific_settings_changed':False,
  'execution_metadata_inside_composed_contract_rederived':exec_parent is not None,
  'destination_source_or_data_bytes_verified':False,'launch_admitted':False}
