"""Synthetic configuration/state checks only; never imports a learning package."""
import copy,json,unittest
from pathlib import Path
from campaign_compiler import compile_campaign
from convergence_policy import LongerTrackSelector
from scheduler_policy import admissible

def fixture():
 selection={'components':[{'metric':'accuracy','direction':'max'}],'ties':'first','checkpoint0':False}
 strict={'unit':'epochs','maximum':100,'optimizer':{'name':'Adam','lr':.001},'selection':selection,'native_schedule':{'name':'constant'},'source_stop':'fixed100'}
 long={**copy.deepcopy(strict),'maximum':1000,'schedule_adaptations':['constant schedule unchanged; longer cap and gated patience declared'],'stopping':{'minimum_units':200,'patience_validation_checks':3,'required_gates':['source_transition_complete']}}
 wrappers=[{'id':arm,'factory':'disabled:never_called','tracks':['long_convergence'],'native_operator_preservation':'metadata fixture; never constructed','training_objective':'mean own CE','members':4} for arm in ('global','private')]
 recipe={'id':'metadata_fixture_only','dataset':'new-example','source_bindings':{'fixture':'no files or model loaded'},'native_factory':'disabled:never_called','native_fixed_config':{'width':64},'official_split_bindings':{str(i):{'fixture':'no arrays opened'} for i in range(3)},'official_evaluator':{'fixture':'never_called'},'native_seed_offsets':[0,1009,2018,3027],'cohorts_by_track':{'native_replica':[{'split':0,'seed':1},{'split':1,'seed':2}],'long_convergence':[{'split':0,'seed':1},{'split':1,'seed':2},{'split':2,'seed':3}]},'source_seed_fidelity':{'status':'configured_source_recipe_with_unresolved_global_rng','source_global_rng_known':False,'source_seed_policy_known':True,'winner_dropout_known':False,'unresolved':['global RNG and winner dropout unresolved in fixture']},'published_native_training':strict,'tracks':{'native_replica':strict,'long_convergence':long},'wrappers':wrappers,'resources':{'vram_mib':40,'ram_mib':10,'disk_mib':10},'fit_relative_cost_weights':{'long_convergence/'+arm:{'value':4.,'basis':'four route calls; synthetic metadata annotation','measured':False} for arm in ('global','private')}}
 return {'schema':'reference-wrapper-roster-v2','scientific_launch_enabled':False,'recipes':[recipe]}
class Checks(unittest.TestCase):
 def test_count_paired_cohorts_member0_and_statistical_endpoints(self):
  p,s=compile_campaign(fixture());self.assertEqual(s['fits'],20);self.assertEqual(s['fits_by_track'],{'native_replica':2,'long_convergence':18});self.assertEqual(len(p['groups']),5);self.assertEqual(s['logical_banks_by_track'],{'native_replica':2,'long_convergence':12})
  self.assertTrue(all(b['id'].endswith('native-M1') and len(b['fit_ids'])==1 for b in p['logical_banks'] if b['track']=='native_replica'))
  for bank in p['logical_banks']:
   if bank['id'].endswith('native-I4'):
    single=next(b for b in p['logical_banks'] if b['id']==bank['id'].replace('native-I4','native-M1'));self.assertEqual(single['fit_ids'][0],bank['fit_ids'][0]);self.assertEqual(len(set(bank['fit_ids'])),4)
  self.assertEqual(sum(s['fits_per_worker'].values()),20);self.assertEqual(sum(s['relative_planning_load_per_worker'].values()),38.)
  self.assertTrue(all(j['contract']['statistical_only'] and not j['contract']['imports_or_executes_native_model'] for j in p['jobs'] if j['kind']=='evaluate_roster'))
  self.assertTrue(all(not j['contract']['whole_paper_replication_claim'] for j in p['jobs']))
 def test_changed_native_track_HPO_and_false_fidelity_rejected(self):
  r=fixture();r['recipes'][0]['tracks']['native_replica']=copy.deepcopy(r['recipes'][0]['tracks']['native_replica']);r['recipes'][0]['tracks']['native_replica']['maximum']=200
  with self.assertRaises(ValueError):compile_campaign(r)
  r=fixture();r['recipes'][0]['native_fixed_config']['selected_config_from']='old_hpo'
  with self.assertRaises(ValueError):compile_campaign(r)
  r=fixture();r['recipes'][0]['source_seed_fidelity']['status']='whole_paper_protocol_verified'
  with self.assertRaises(ValueError):compile_campaign(r)
 def test_GPU_slots_and_host_shared_RAM_disk(self):
  j={'id':'j','dependencies':[],'resources':{'vram_mib':40,'ram_mib':60,'disk_mib':60}};w='server77/gpu0';caps={'gpu':{'server77/gpu0':{'vram_mib':100},'server77/gpu1':{'vram_mib':100}},'host':{'server77':{'ram_mib':100,'disk_mib':200}}};a=[{'worker':'server77/gpu1','resources':dict(j['resources'])}]
  self.assertEqual(admissible(j,w,a,caps,{'j':'registered'}),(False,'combined_ram_mib_reservation'))
  caps['host']['server77']['ram_mib']=200;caps['host']['server77']['disk_mib']=100;self.assertEqual(admissible(j,w,a,caps,{'j':'registered'}),(False,'combined_disk_mib_reservation'))
  a=[{'worker':w,'resources':dict(j['resources'])}]*2;self.assertEqual(admissible(j,w,a,caps,{'j':'registered'}),(False,'two_slot_limit'))
 def test_warmup_and_transition_gates(self):
  r=fixture()['recipes'][0]['tracks']['long_convergence'];s=LongerTrackSelector(r['selection'],r['stopping']);self.assertFalse(s.observe(1,{'accuracy':.8},{})['stop_requested'])
  for unit in (198,199,200,201):self.assertFalse(s.observe(unit,{'accuracy':.7},{})['stop_requested'])
  for unit in (202,203):self.assertFalse(s.observe(unit,{'accuracy':.7},{'source_transition_complete':True})['stop_requested'])
  self.assertTrue(s.observe(204,{'accuracy':.7},{'source_transition_complete':True})['stop_requested'])
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));receipt={'schema':'synthetic-metadata-only-campaign-QA-v2','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'model_or_dataset_constructed':False,'scientific_fits':0,'optimizer_updates':0,'resource_probes':0,'fixture_counts_are_not_scientific_campaign_counts':True};Path(__file__).with_name('METADATA_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
 if not result.wasSuccessful():raise SystemExit(1)
