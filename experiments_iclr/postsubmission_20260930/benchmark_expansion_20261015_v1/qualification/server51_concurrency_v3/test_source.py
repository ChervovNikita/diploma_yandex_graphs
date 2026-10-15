import ast,copy,sys,unittest
from pathlib import Path
from qualifier_contracts import HERE,WARMUP_EPOCHS,TIMED_EPOCHS,PHASES,cache_aware_available
from summarize import summary
class Checks(unittest.TestCase):
 def test_syntax_no_native_imports(self):
  for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
  self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
 def test_fixed_finite_protocol_and_cache_aware_headroom(self):
  self.assertEqual((WARMUP_EPOCHS,TIMED_EPOCHS),(10,100));self.assertEqual(list(PHASES),['native_alone','structured_alone','concurrent'])
  self.assertEqual(PHASES['concurrent'],['native','structured'])
  result,_=cache_aware_available(1800,[{'path':'/sys/fs/cgroup','maximum':'260','current':246,'stats':{'inactive_file':240,'shmem':2}}])
  self.assertEqual(result,252)
 def test_summary_exact_cost_overlap_and_no_quality_claim(self):
  def row(start,seconds):
   return {'status':'PASS','TEST_accessed':False,'TRAIN_epochs_completed':110,'VALID_calls_completed':110,'optimizer_calls':{'attempts':[110],'completed':[110]},
    'measurement_start_monotonic_ns':start,'measurement_end_monotonic_ns':start+int(seconds*1e9),'startup_seconds':3,
    'epochs':[{'stage':'timed','seconds':seconds/100,'start_monotonic_ns':start+int(i*seconds/100*1e9),'end_monotonic_ns':start+int((i+1)*seconds/100*1e9)} for i in range(100)]}
  rows={'native_alone':{'native':row(0,2)},'structured_alone':{'structured':row(0,8)},'concurrent':{'native':row(0,3),'structured':row(0,9)}}
  result=summary(rows);self.assertEqual(result['completed_optimizer_updates'],440);self.assertAlmostEqual(result['mixed_workload_speedup'],10/9)
  self.assertEqual(result['concurrent_overlap_seconds'],3);self.assertEqual(result['fully_overlapping_epoch_times']['structured']['count'],33)
  self.assertFalse(result['prediction_quality_claim']);self.assertFalse(result['automatic_generalization_to_other_profiles'])
  broken=copy.deepcopy(rows);broken['concurrent']['native']['optimizer_calls']['completed']=[109]
  with self.assertRaises(ValueError):summary(broken)
class TerminalCostChecks(unittest.TestCase):
 def test_external_kill_stale_result_and_authority_mismatch_are_unknown(self):
  from terminal_cost import complete_terminal_cost
  expected={'child':{'pid':42},'phase':'concurrent','arm':'native','admission':{'sha256':'a'}}
  row={**expected,'status':'PASS','TRAIN_epochs_completed':110,'VALID_calls_completed':110,'optimizer_calls':{'attempts':[110],'completed':[110]},'invocation_wall_seconds':30}
  self.assertTrue(complete_terminal_cost(row,0,expected=expected))
  stale=copy.deepcopy(row);stale['status']='running'
  self.assertFalse(complete_terminal_cost(stale,-9,expected=expected))
  self.assertFalse(complete_terminal_cost(stale,0,expected=expected))
  self.assertFalse(complete_terminal_cost(row,-9,expected=expected))
  self.assertFalse(complete_terminal_cost(row,0,expected=expected,forced_KILL=True))
  self.assertFalse(complete_terminal_cost(row,0,expected=expected,cleanup_error='uncertain'))
  mismatched=copy.deepcopy(row);mismatched['phase']='native_alone'
  self.assertFalse(complete_terminal_cost(mismatched,0,expected=expected))
if __name__=='__main__':unittest.main(verbosity=2)
