"""Synthetic configuration/registry-state checks only; no scientific data/imports."""
import copy, json, unittest
from pathlib import Path
from campaign_compiler import compile_campaign
from convergence_policy import LongerTrackSelector
from scheduler_policy import admissible

def fixture():
 selection={'components':[{'metric':'accuracy','direction':'max'}],'ties':'first','checkpoint0':False}
 strict={'unit':'epochs','maximum':100,'optimizer':{'name':'Adam','lr':.001},'selection':selection,'native_schedule':{'name':'constant'},'source_stop':'fixed100'}
 long={**copy.deepcopy(strict),'maximum':1000,'schedule_adaptations':['constant schedule unchanged; longer cap and gated patience are declared adaptations'],'stopping':{'minimum_units':200,'patience_validation_checks':3,'required_gates':['source_transition_complete']}}
 recipe={'id':'metadata_fixture_only','dataset':'new-example','source_bindings':{'fixture':'no files or model loaded'},'native_factory':'disabled:never_called','native_fixed_config':{'width':64},'official_split_bindings':{'0':{'fixture':'no arrays opened'}},'official_evaluator':{'fixture':'never_called'},'native_seed_offsets':[0,1009,2018,3027],'seeds':[1,2,3],'split_ids':[0],'published_native_training':strict,'tracks':{'native_replica':strict,'long_convergence':long},'wrappers':[{'id':'shared4','factory':'disabled:never_called','tracks':['native_replica','long_convergence'],'native_operator_preservation':'declared fixture; never constructed','training_objective':'mean own CE','members':4}],'resources':{'vram_mib':40,'ram_mib':10,'disk_mib':10}}
 return {'schema':'reference-wrapper-roster-v1','scientific_launch_enabled':False,'recipes':[recipe]}
class Checks(unittest.TestCase):
 def test_count_and_exact_member0_reuse(self):
  p,s=compile_campaign(fixture());self.assertEqual(s['fits'],30);self.assertEqual(len(p['groups']),6)
  for bank in p['logical_banks']:
   if bank['id'].endswith('native-I4'):
    single=next(b for b in p['logical_banks'] if b['id']==bank['id'].replace('native-I4','native-M1'))
    self.assertEqual(single['fit_ids'][0],bank['fit_ids'][0]);self.assertEqual(len(set(bank['fit_ids'])),4)
  self.assertEqual(sum(s['fits_per_worker'].values()),30)
 def test_changed_native_replica_and_HPO_rejected(self):
  r=fixture();r['recipes'][0]['tracks']['native_replica']=copy.deepcopy(r['recipes'][0]['tracks']['native_replica']);r['recipes'][0]['tracks']['native_replica']['maximum']=200
  with self.assertRaises(ValueError):compile_campaign(r)
  r=fixture();r['recipes'][0]['native_fixed_config']['selected_config_from']='old_hpo'
  with self.assertRaises(ValueError):compile_campaign(r)
 def test_two_slots_and_combined_reservations(self):
  j={'id':'j','dependencies':[],'resources':{'vram_mib':40,'ram_mib':10,'disk_mib':10}};w='allocation/gpu0';caps={w:{'vram_mib':70,'ram_mib':100,'disk_mib':100}};a=[{'worker':w,'resources':dict(j['resources'])}]
  self.assertEqual(admissible(j,w,a,caps,{'j':'registered'}),(False,'combined_vram_mib_reservation'))
  caps[w]['vram_mib']=100;self.assertEqual(admissible(j,w,a*2,caps,{'j':'registered'}),(False,'two_slot_limit'))
 def test_warmup_and_transition_gates(self):
  r=fixture()['recipes'][0]['tracks']['long_convergence'];s=LongerTrackSelector(r['selection'],r['stopping'])
  self.assertFalse(s.observe(1,{'accuracy':.8},{})['stop_requested'])
  for unit in (198,199,200,201):self.assertFalse(s.observe(unit,{'accuracy':.7},{})['stop_requested'])
  for unit in (202,203):self.assertFalse(s.observe(unit,{'accuracy':.7},{'source_transition_complete':True})['stop_requested'])
  self.assertTrue(s.observe(204,{'accuracy':.7},{'source_transition_complete':True})['stop_requested'])
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 receipt={'schema':'synthetic-metadata-only-campaign-QA-v1','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'model_or_dataset_constructed':False,'scientific_fits':0,'optimizer_updates':0,'resource_probes':0,'fixture_counts_are_not_scientific_campaign_counts':True}
 Path(__file__).with_name('METADATA_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if not result.wasSuccessful():raise SystemExit(1)
