"""Small stdlib/source/JSON/synthetic scalar checks, never native execution."""
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import extension as ex
import roles


class Checks(unittest.TestCase):
    def test_frozen_source_authorities(self):
        sources=json.loads((HERE/'SOURCES.json').read_text());prefix=sources['project_root']+'/experiments_iclr/postsubmission_20260930/'
        for pin in sources['source_pins']:
            if not pin['path'].startswith(prefix):continue  # remote installed encoder is metadata only here
            path=ex.E.parent/pin['path'][len(prefix):]
            self.assertEqual(ex.sha(path),pin['sha256'])
        self.assertEqual((HERE/'BASE_CASE.json').read_bytes(),(ex.E/'pilot_20261015_residual_only_private_bn_v1/JOB_seed101.json').read_bytes())
        for path in HERE.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node,ast.Import):self.assertFalse(any(a.name.startswith(('torch','numpy')) for a in node.names))
                if isinstance(node,ast.ImportFrom):self.assertFalse((node.module or '').startswith(('torch','numpy')))

    def test_exact_native_and_shared_descriptors(self):
        sources=json.loads((HERE/'SOURCES.json').read_text());base=json.loads((HERE/'BASE_CASE.json').read_text())
        sys.path.insert(0,str(ex.E/'runtime'))
        runner=ex.load(ex.E/'runtime/run_fit.py','_pair_original_runner_checks')
        rows=[]
        for seed in (101,202,303):
            native=[ex.descriptor(sources['project_root'],seed,'native_common',body) for body in range(4)]
            self.assertEqual([j['seed'] for j in native],[seed+x for x in ex.OFFSET])
            self.assertEqual([j['body'] for j in native],[0,1,2,3])
            self.assertTrue(all(j['procedure']==j['base_procedure']=='native_single' for j in native))
            self.assertEqual(native[0]['seed'],seed)  # M1 reuses exact body0 descriptor
            rows+=native+[ex.descriptor(sources['project_root'],seed,arm) for arm in ('common','structured','permuted')]
        self.assertEqual(len({j['id'] for j in rows}),21)
        for job in rows:
            runner.validate_job(job)
            for key in ('recipe','policy','callback_module','callback_sha256','data_root','split','device','torch_threads','TEST_access'):
                self.assertEqual(job[key],base[key])
            self.assertEqual(job['method_settings']['pair_coefficient'],.5)
        with self.assertRaises(ValueError):ex.descriptor(sources['project_root'],101,'native_common')

    def test_fractional_ties_and_label_independent_rule(self):
        frozen=roles.rule([0.,0.,0.,1.,2.,3.])
        self.assertEqual(roles.membership(frozen,0.),(.5,.5,0.,0.))
        sums=[sum(roles.membership(frozen,s)[m] for s in [0.,0.,0.,1.,2.,3.]) for m in range(4)]
        self.assertEqual(sums,[1.5]*4)
        self.assertEqual(roles.weights(roles.rule([0.]*6),0.),(1.,)*4)
        ids=list(range(6));hist=[[1.,0.],[0.,1.],[1.,1.],[0.,0.],[1.,2.],[2.,1.]]
        a=roles.Table(ids,hist,[1,0,1,0,1,None],variant='structured',seed=101)
        b=roles.Table(ids,hist,[0,1,0,1,None,1],variant='structured',seed=101)
        self.assertEqual(a.frozen,b.frozen)
        self.assertEqual(a.metadata()['all_input_pair_support'],15)
        self.assertEqual(a.metadata()['observed_opposite_pair_support'],6)
        self.assertEqual(a.metadata()['zero_bond_inputs'],1)
        for m in a.metadata()['all_input_structured_route_mass']:self.assertAlmostEqual(m,15.)

    def test_symmetric_whole_vector_permutation_and_mask(self):
        ids=[9,3,5,1,7,11];hist=[[1.,0.],[0.,1.],[1.,1.],[0.,0.],[1.,2.],[2.,1.]];labels=[1,0,1,0,1,None]
        a=roles.Table(ids,hist,labels,variant='structured',seed=101)
        b=roles.Table(ids,hist,labels,variant='permuted',seed=101)
        c=roles.Table(ids,hist,labels,variant='common',seed=101)
        pairs,original=a.batch(ids,labels);other,permuted=b.batch(ids,labels)
        self.assertEqual(pairs,other);self.assertEqual(Counter(original),Counter(permuted))
        self.assertEqual(a.metadata()['opposite_label_route_mass'],b.metadata()['opposite_label_route_mass'])
        self.assertEqual(b.metadata()['assigned_pair_code_sha256'],roles.Table(ids,hist,labels,variant='permuted',seed=101).metadata()['assigned_pair_code_sha256'])
        for (i,j),vector in zip(pairs,permuted):
            self.assertEqual(vector,b.vector(ids[j],ids[i]));self.assertAlmostEqual(sum(vector),4.)
            self.assertTrue(all(.5<=v<=2.5 for v in vector))
            self.assertEqual(c.vector(ids[i],ids[j]),(1.,)*4)
        self.assertEqual(a.batch([ids[0],ids[2]],[1,1]),([],[]))
        with self.assertRaises(ValueError):a.batch([ids[0],ids[1]],[0,0])
        with self.assertRaises(ValueError):a.vector(ids[0],ids[0])

    def test_update_and_bounded_inspection_contract(self):
        source=(HERE/'extension.py').read_text();qualifier=(HERE/'qualify.py').read_text()
        self.assertEqual(source.count('loss.backward()'),1)
        self.assertNotIn('autograd.grad',source)
        self.assertIn('if not pairs:return own',source)
        self.assertIn('pair_loss=(torch.nn.functional.softplus(-differences)*weight).mean(1).mean()',source)
        self.assertIn('return own+.5*pair_loss',source)
        inspect=next(n for n in ast.parse(qualifier).body if isinstance(n,ast.FunctionDef) and n.name=='inspect_train')
        text=ast.get_source_segment(qualifier,inspect)
        self.assertNotIn('create_adapter',text);self.assertNotIn('create_session',text)
        self.assertNotIn('get_idx_split',text);self.assertIn("native._RoleView(dataset,ids,torch,'TRAIN')",text)
        self.assertIn("fresh_restore_VALID_bit_exact=True",qualifier)
        method=json.loads((HERE/'METHOD.json').read_text())
        self.assertFalse(method['scientific_admission']);self.assertFalse(method['new_fit_launcher'])
        for name in ('INSPECT_ADMISSION_TEMPLATE.json','LIFECYCLE_ADMISSION_TEMPLATE.json'):
            value=json.loads((HERE/name).read_text());self.assertFalse(value['approved']);self.assertIsNone(value['manifest_sha256'])
        manifest=HERE/'MANIFEST.json'
        if manifest.exists():
            for row in json.loads(manifest.read_text())['files']:
                p=HERE/row['path'];self.assertEqual(p.stat().st_size,row['bytes']);self.assertEqual(ex.sha(p),row['sha256'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report={'schema':'molbace-pairwise-source-checks-v1','status':'passed' if result.wasSuccessful() else 'failed',
        'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_and_synthetic_scalar_only':True,
        'new_native_models_or_forwards_or_optimizer_steps':0,'scientific_fits':0,'data_or_tensor_execution':False,
        'SSH_Git_GPU_operations':0,'TEST_accessed':False}
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
