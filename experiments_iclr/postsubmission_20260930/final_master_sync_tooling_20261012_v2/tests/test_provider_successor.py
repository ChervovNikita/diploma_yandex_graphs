"""Exact provider-key/scope closure fixtures; no native providers imported."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE/'runtime'))
import runtime_requirements as requirements
from sync_common import receipt
class ProviderSuccessorTests(unittest.TestCase):
 def observation(self,paths,names):
  versions={'torch':'2.1.2','torch-householder':'1.0','setuptools':'68.0.0','ninja':'1.11.1','datasketch':'1.6.5','dgl':'2.4.0'}
  return {'search_paths':paths,'distributions':{requirements.normalized(n):{'name':n,'version':versions[n]} for n in names},'missing_distribution_metadata':[],
   'dependency_version_mismatches':[],'scientific_imports':0,'CUDA_initialized':False,'package_installs':0}
 def plan(self):return {'runtime_requirements':{'packages':{'torch':'2.1.2'},'python_major_minor':'3.11','platform_system':'Linux','machine':'x86_64'}}
 def binding(self,root):
  (root/'provider/toolchain_setuptools68').mkdir(parents=True);(root/'author').mkdir();(root/'provider/native.so').write_bytes(b'never import synthetic native provider')
  return {'householder_provider_relative_root':'provider','author_directory':'author','provider':{'files':[receipt(root,'provider/native.so')]}}
 def contracts(self,root,runs,**kwargs):
  with patch.object(requirements,'metadata_environment',side_effect=self.observation),patch.object(requirements,'source_imports',return_value={'module_distribution_candidates':{}}):
   return requirements.contracts(root,self.plan(),[{'study_id':'fixture','runs':runs}],[],**kwargs)
 def test_legacy_and_bayesian_keys_preserve_one_exact_scoped_native_contract(self):
  with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
   root=Path(tmp);b=self.binding(root);runs=[{'kind':'model_fit','run_id':'old','config':{'bsnn_sources':b}}, {'kind':'model_fit','run_id':'new','config':{'bayesian_bsnn_sources':copy.deepcopy(b)}}]
   value=self.contracts(root,runs);providers=value['provider_contracts'];self.assertEqual(len(providers),1);p=providers[0]
   self.assertEqual(p['source_binding'],b);self.assertEqual(p['source_binding_keys'],['bayesian_bsnn_sources','bsnn_sources']);self.assertEqual(p['run_keys'],['fixture/old','fixture/new'])
   self.assertEqual(p['compiled_provider_receipts'],b['provider']['files']);self.assertFalse(p['native_import_or_ABI_probe_performed']);self.assertTrue(p['target_host_native_provider_qualification_required'])
 def test_conflicting_dual_keys_and_missing_compiled_provider_rejected(self):
  with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
   root=Path(tmp);b=self.binding(root);other=copy.deepcopy(b);other['author_directory']='different'
   with self.assertRaisesRegex(ValueError,'Conflicting'):self.contracts(root,[{'kind':'model_fit','run_id':'bad','config':{'bsnn_sources':b,'bayesian_bsnn_sources':other}}])
   b['provider']['files']=[]
   with self.assertRaisesRegex(ValueError,'compiled provider'):self.contracts(root,[{'kind':'model_fit','run_id':'bad','config':{'bayesian_bsnn_sources':b}}])
 def test_BUDDY_isolated_cache_provider_scoped_with_receipts_and_shared_hash_cache(self):
  with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
   root=Path(tmp);directory=root/'buddy/providers/datasketch';directory.mkdir(parents=True);(root/'buddy/source').mkdir();(root/'buddy/source/cache_builder.py').write_text('# no imports')
   (directory/'provider.py').write_text('# isolated synthetic datasketch source')
   manifest={'distribution':'datasketch1.6.5','files':[receipt(root,'buddy/providers/datasketch/provider.py')],
    'runtime_loader':'BUDDY native cached fitting does not import datasketch; provider retained for exact cache regeneration'}
   (root/'buddy/PROVIDER_MANIFEST.json').write_text(json.dumps(manifest));calls=[];cache={}
   def cached(r,p):
    if p not in cache:cache[p]=receipt(r,p);calls.append(p)
    return cache[p]
   binding={'cache_builder':receipt(root,'buddy/source/cache_builder.py')};runs=[{'kind':'model_fit','run_id':n,'config':{'buddy_sources':binding}} for n in ('one','two')]
   p=self.contracts(root,runs,receipt_function=cached)['provider_contracts'][0]
   self.assertEqual(p['kind'],'isolated_BUDDY_cache_regeneration_provider');self.assertFalse(p['scientific_fit_runtime_dependency']);self.assertEqual(p['declared_versions'],{'datasketch':'1.6.5'})
   self.assertEqual(p['provider_receipts'],manifest['files']);self.assertEqual(p['run_keys'],['fixture/one','fixture/two']);self.assertEqual(calls,['buddy/PROVIDER_MANIFEST.json'])
   self.assertEqual(p['observed_metadata_closure']['search_paths'],[str(directory)]);self.assertEqual(p['provider_imports'],0);self.assertEqual(p['package_installs'],0)
 def test_existing_isolated_author_environment_contract_kept(self):
  with tempfile.TemporaryDirectory(dir=HERE/'tests') as tmp:
   root=Path(tmp);(root/'author-overlay').mkdir();cli={'private_dependency_dir':'author-overlay','environment':{'PYTHONPATH':str(root/'author-overlay'),'DGLBACKEND':'pytorch'},
    'runtime_versions':{'dgl':'2.4.0'},'python_executable':'/synthetic/python','source_files':[],'raw_files':[],'runtime_provider_files':[]}
   p=self.contracts(root,[{'kind':'author_cli_group','run_id':'author','cli':cli}])['provider_contracts'][0]
   self.assertEqual(p['kind'],'isolated_author_child_environment');self.assertEqual(p['environment'],cli['environment']);self.assertTrue(p['environment_applied_in_child_only']);self.assertFalse(p['native_import_or_ABI_probe_performed'])
if __name__=='__main__':unittest.main()
