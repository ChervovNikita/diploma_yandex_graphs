import ast,sys,unittest
from pathlib import Path
from build_class_map import HERE,build,GATES
class Checks(unittest.TestCase):
 def test_two_seed_tasks_share_scope_without_pass(self):
  tasks=[];candidates=[];objects={}
  for seed in (0,1):
   fid='fit'+str(seed);descriptor={'path':'job'+str(seed)};job={'callback_sha256':'callback','policy':{'track':'matched_extended'}};objects[descriptor['path']]=job
   tasks.append({'id':fid,'kind':'fit','worker':'allocation/gpu0','cell_id':'cell','family':'node','scientific_method_id':'native_single','protocol_track':'matched_extended','science_contract_hash':'science'+str(seed),'split':seed,'dependencies':[],'output_relative':'fits/'+fid,'qualification_blocks':['old_pending']})
   candidates.append({'id':fid,'job_descriptor':descriptor,'runtime_entrypoint':{'sha256':'runtime'}})
  release={'tasks':tasks,'host':'allocation','server51_successor':{'source_manifest':{'sha256':'source'}}};objects.update(release=release,candidates=candidates)
  result=build({'release':{'path':'release'},'candidates':{'path':'candidates'}},lambda r:objects[r['path']],{'sha256':'facts'})
  self.assertEqual(result['counts']['review_classes'],1);self.assertEqual(result['counts']['PASS_gates_created'],0);self.assertEqual(result['counts']['new_native_runs'],0)
  self.assertEqual(len(result['classes'][0]['task_ids']),2);self.assertEqual(len(result['gate_applications']),len(GATES));self.assertFalse(result['human_GO'])
 def test_no_numerical_import_or_syntax_error(self):
  for p in HERE.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('torch',sys.modules);self.assertNotIn('numpy',sys.modules)
if __name__=='__main__':unittest.main(verbosity=2)
