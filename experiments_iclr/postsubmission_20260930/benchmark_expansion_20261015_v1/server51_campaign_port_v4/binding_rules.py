"""Recompute every execution descriptor from immutable canonical metadata."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from descriptor_binding import derive, digest, json_bytes, require, NORMALIZATION_SHA, BANKS_SHA
from host_binding import ALLOCATION_ROOT, PHASE_RELATIVE
HERE=Path(__file__).resolve().parent
E=HERE.parent

def pin(path):
 path=Path(path).resolve();raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def json_pin(value,root):
 path=Path(value['path']).resolve();require(path.is_relative_to(Path(root).resolve()),'Confined JSON required')
 require(pin(path)==value,'Metadata receipt differs');return json.loads(path.read_text())
def expected_entrypoint(task,root):
 if task['kind']!='fit':return HERE/'runtime_I4/aggregate_selected.py'
 job=json_pin(task['source_descriptor'],root)
 if job.get('normalization_composition'):return HERE/'composed/runtime/run_fit.py'
 return HERE/('runtime_link' if job['family']=='link' else 'runtime_node_molecular')/'run_fit.py'
def validate_job_binding(task,job,root):
 """Actual parent bytes are re-read; neither canonical nor execution receipts may drift."""
 original=json_pin(task['source_descriptor'],root)
 provenance=task['runtime_binding'];parent=parent_raw=None
 if original.get('normalization_composition'):
  cp=provenance['canonical_parent_copy'];parent=json_pin(cp,root);parent_raw=Path(cp['path']).read_bytes()
  json_pin(provenance['execution_parent_descriptor'],root)
 roster=json.loads((E/'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
 bound,raw,receipt=derive(original,ALLOCATION_ROOT,PHASE_RELATIVE,str(root),PHASE_RELATIVE,
  parent=parent,parent_raw=parent_raw,execution_parent_path=(provenance.get('execution_parent_descriptor') or {}).get('path'),expansion=E,roster=roster)
 receipt['canonical_descriptor_origin']=copy.deepcopy(provenance['canonical_descriptor_origin'])
 receipt['canonical_descriptor_copy']=copy.deepcopy(task['source_descriptor'])
 if parent is not None:
  receipt['canonical_parent_copy']=copy.deepcopy(provenance['canonical_parent_copy'])
  require(Path(provenance['execution_parent_descriptor']['path']).read_bytes()==raw,'Execution parent bytes differ from actual typed derivation')
 require(provenance==receipt,'Execution binding receipt differs from independent derivation')
 require(digest(original)==task['canonical_runtime_job_hash'],'Canonical descriptor hash changed')
 require(task['science_contract_hash']==original['expanded_contract_hash'],'Canonical science identity changed')
 require(job==bound and digest(job)==task['runtime_job_hash'],'Execution JOB is not exact typed binding')
 require(job['expanded_contract_hash']==task['execution_science_contract_hash'],'Execution metadata hash changed')
 return original
