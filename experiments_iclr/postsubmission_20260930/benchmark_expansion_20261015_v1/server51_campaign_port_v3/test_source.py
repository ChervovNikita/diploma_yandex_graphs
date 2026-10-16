"""Stdlib engineering checks: no models, data payloads, numerical libraries or jobs."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from descriptor_binding import derive,digest,json_bytes,byte_receipt,audit_runtime_paths
from host_binding import load_config,config_path,check_observed_identity,ALLOCATION_ROOT,SERVER51_ROOT,PHASE_RELATIVE,WORKERS
from host_ledger import admissible
from backend import memory_available
from reporting_projection import project_profile
from prepare_projection import validate_projected_task
from engine import Engine

class SourceChecks(unittest.TestCase):
 def test_syntax_without_imports(self):
  for path in HERE.rglob('*.py'):
   if '.source_test_' not in str(path):ast.parse(path.read_text(),filename=str(path))
 def test_exact_three_hosts_and_restricted_route_rejected(self):
  for worker in WORKERS:
   c,_=load_config(config_path(worker));check_observed_identity(c,c['expected_hostname'],c['expected_gpu_uuids_by_index'])
   with self.assertRaises(ValueError):check_observed_identity(c,'anogena-0',['GPU-other']*7)
   with self.assertRaises(ValueError):check_observed_identity(c,c['expected_hostname'],list(reversed(c['expected_gpu_uuids_by_index']))+['extra'])
  with self.assertRaises(ValueError):config_path('server77/gpu0')
 def test_composed_parent_materialization_and_canonical_identity(self):
  E=HERE.parent;roster=json.loads((E/'expanded_study_design_v1/PROPOSED_ROSTER.json').read_text())
  cell=next(c for c in roster['cells'] if c['task_family']=='ogb')
  method=next(m for m in cell['shared_method_ids'] if m not in ('gnnm_boundary_native','gnnm_private_bn_state_v3'))
  parent={'schema':'native-full-epoch-fit-v1','id':'engineering-source-only','family':'ogb','recipe_id':cell['id'],
   'scientific_method_id':method,'TEST_access':False,'device':'cuda:0','callback_module':ALLOCATION_ROOT+'/'+PHASE_RELATIVE+'/callback.py',
   'data_root':ALLOCATION_ROOT+'/data','policy':{'track':'matched_extended'},'recipe':{'_repository_root':ALLOCATION_ROOT,
    '_redesign_root':ALLOCATION_ROOT+'/'+PHASE_RELATIVE+'/reference_wrapper_redesign_20261013_v1','learning_rate':0.001},
   'expanded_contract':{'id':'engineering-source-only','method':method,'learning_rate':0.001}}
  parent['expanded_contract_hash']=digest(parent['expanded_contract']);raw=json_bytes(parent)
  from descriptor_binding import NORMALIZATION_SHA,BANKS_SHA,composition_contract
  decl={'schema':'native-route-BN-composition-v1','policy':'private_native_running_state_shared_affine_v1',
   'source_manifest_sha256':NORMALIZATION_SHA,'banks_sha256':BANKS_SHA,'parent_descriptor':byte_receipt(ALLOCATION_ROOT+'/parent.json',raw),
   'parent_job_sha256':digest(parent),'matched_control_method_id':'gnnm_private_bn_state_v3'}
  original=composition_contract(E).migrated_job(parent,decl,roster);before=copy.deepcopy(original)
  path=SERVER51_ROOT+'/'+PHASE_RELATIVE+'/execution-parent.json'
  bound,parent_bytes,binding=derive(original,ALLOCATION_ROOT,PHASE_RELATIVE,SERVER51_ROOT,PHASE_RELATIVE,
   parent=parent,parent_raw=raw,execution_parent_path=path,expansion=E,roster=roster)
  self.assertEqual(original,before);self.assertEqual(bound['id'],original['id']);self.assertEqual(bound['recipe']['learning_rate'],0.001)
  self.assertEqual(binding['canonical_runtime_job_hash'],digest(original));self.assertNotEqual(digest(bound),digest(original))
  self.assertEqual(bound['normalization_composition']['parent_descriptor'],byte_receipt(path,parent_bytes))
  self.assertNotEqual(hashlib.sha256(parent_bytes).hexdigest(),hashlib.sha256(raw).hexdigest())
  self.assertNotEqual(binding['execution_science_contract_hash'],binding['canonical_science_contract_hash'])
  with self.assertRaises(ValueError):derive(original,ALLOCATION_ROOT,PHASE_RELATIVE,SERVER51_ROOT,PHASE_RELATIVE,parent=parent,parent_raw=raw+b' ',execution_parent_path=path,expansion=E,roster=roster)
  malicious=copy.deepcopy(parent);malicious['unknown_absolute']=ALLOCATION_ROOT+'/forbidden'
  with self.assertRaises(ValueError):audit_runtime_paths(malicious,ALLOCATION_ROOT,PHASE_RELATIVE)
 def test_normal_two_slots_and_external_capacity(self):
  resources={'profile':'same','qualified_peer_profiles':['same'],'max_concurrent_owned_children':2,'vram_bytes':10,'ram_bytes':10,'disk_bytes':10,'cpu_slots':1}
  budget={'gpu_headroom_bytes':2,'host_ram_headroom_bytes':2,'disk_headroom_bytes':2,'host_cpu_slots':8}
  probe={'untracked_own_gpu_pids':[9999],'free_vram_by_uuid':{'gpu':100},'available_ram_bytes':100,'free_disk_bytes':100,'allocated_vram_by_pid':{},'rss_by_pid':{}}
  self.assertEqual(admissible(resources,'gpu',[],probe,budget),(True,'admitted'))
  active=[{'gpu_uuid':'gpu','resources':resources,'child_identity':{'pid':1}}]
  self.assertTrue(admissible(resources,'gpu',active,probe,budget)[0])
  self.assertEqual(admissible(resources,'gpu',active*2,probe,budget),(False,'two_slot_limit'))
  probe['free_vram_by_uuid']['gpu']=11
  self.assertEqual(admissible(resources,'gpu',[],probe,budget),(False,'VRAM_reservation'))
  result=memory_available(1800,[{'path':'/sys/fs/cgroup','maximum':'260','current':246,'stats':{'inactive_file':240,'shmem':2}}])
  self.assertEqual(result['available_ram_bytes'],252)
 def test_reporting_retains_canonical_science(self):
  identity={'fit_id':'x','bindings':{k:k+'-hash' for k in ('native_source','data','controller','evaluator','readout')}}
  profile={'fit_specs':{'x':{'descriptor':{'path':'old'},'job_hash':'old-hash','identity':identity,
   'bindings':identity['bindings'],'wrapper_authority':{'descriptor':{'path':'old'}},'observed_input_files':[]}},
   'I4_specs':{},'registry':{'expected_predictors':[identity]},'contrast_plan':{'planned':['fixed']}}
  task={'x':{'runtime_job_hash':'new-hash','runtime_binding':{'explicit':'join'}}}
  candidates={'x':{'_job':{'family':'node'},'job_descriptor':{'path':'new'},'runtime_entrypoint':{'path':'new-runtime'}}}
  result=project_profile(profile,{'path':'canonical-profile'},task,candidates,{'host':'server51'},lambda p,expected=None:{'path':str(p)})
  self.assertEqual(result['registry'],profile['registry']);self.assertEqual(result['contrast_plan'],profile['contrast_plan'])
  self.assertEqual(result['fit_specs']['x']['identity'],identity);self.assertEqual(result['fit_specs']['x']['canonical_job_hash'],'old-hash')
  self.assertEqual(result['fit_specs']['x']['job_hash'],'new-hash')
 def test_resource_wait_does_not_fail_or_start_early(self):
  with tempfile.TemporaryDirectory(prefix='.source_test_',dir=HERE) as value:
   root=Path(value);(root/'stops').mkdir()
   class Ledger:pass
   class Backend:
    def __init__(self):self.ledger=Ledger();self.ledger.root=root;self.calls=0;self.rechecks=0
    def admit(self,task,active):
     self.calls+=1;return (False,'VRAM_reservation') if self.calls==1 else (True,'reserved')
    def recheck(self,task):self.rechecks+=1
    def peer_activity(self):return False
    def cancel_pending(self,reason):pass
   class SourceEngine(Engine):
    def establish(self):self.control.mkdir();self.stops=root/'stops';self.baseline_stops=set()
    def reconcile(self,task):self.states[task['id']]={'status':'pending'}
    def launch(self,task):self.states[task['id']]={'status':'complete'}
   backend=Backend();task={'id':'fake-source-task','kind':'fit','worker':'allocation/gpu0','dependencies':[],'output_relative':'fits/a','binding':{'resources':{}}}
   engine=SourceEngine([task],root/'output','source-hash','allocation/gpu0',backend,[],allowed_root=root)
   (root/'output').mkdir()
   with patch('engine.time.sleep') as wait:result=engine.run()
   self.assertEqual(result['status'],'complete');self.assertEqual(backend.calls,2);self.assertEqual(backend.rechecks,1)
   self.assertEqual(result['children_started_this_invocation'],0);wait.assert_called_once()
 def test_no_numerical_library_import(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
