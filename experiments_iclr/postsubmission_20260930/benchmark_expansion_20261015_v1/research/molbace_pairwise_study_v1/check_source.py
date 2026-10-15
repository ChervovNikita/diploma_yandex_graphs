"""Source/JSON/synthetic scalar gates only; no data/tensor/model execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import study as s
SPEC=json.loads((HERE/'SPEC.json').read_text())


def local(pin):
    prefix=SPEC['project_root']+'/experiments_iclr/postsubmission_20260930/'
    s.require(pin['path'].startswith(prefix),'Local source pin outside phase')
    return s.PHASE/pin['path'][len(prefix):]


class Checks(unittest.TestCase):
    def test_source_and_completed_CPU_descriptor_join(self):
        rows=s.jobs(SPEC)
        for pin in SPEC['source_pins']:
            path=local(pin);self.assertEqual(path.stat().st_size,pin['bytes']);self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin['sha256'])
        for pin in SPEC['completed_CPU_lifecycles']:
            path=local(pin['receipt']);self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin['receipt']['sha256'])
            value=json.loads(path.read_text())['result'];row=next(r for r in rows if r['cohort_seed']==101 and r['kind']==pin['arm'])
            self.assertEqual(value['planned_job'],row['job']);self.assertEqual(value['status'],'passed')
            self.assertEqual(value['optimizer_calls'],{'attempts':[2],'completed':[2]});self.assertEqual(value['VALID_completed'],3)
            self.assertTrue(value['fresh_restore_VALID_bit_exact']);self.assertFalse(value['TEST_accessed'])

    def test_exact21_existing_runner_native_I4(self):
        rows=s.jobs(SPEC);sys.path.insert(0,str(s.E/'runtime'));runner=s.load(s.E/'runtime/run_fit.py','_pair_scientific_original_runner_checks')
        self.assertEqual(len(rows),21);self.assertEqual(len({r['job_hash'] for r in rows}),21)
        for seed in s.SEEDS:
            selected=[r for r in rows if r['cohort_seed']==seed]
            self.assertEqual([r['kind'] for r in selected],['native_common']*4+['common','structured','permuted'])
            self.assertEqual([r['job']['body'] for r in selected[:4]],[0,1,2,3])
            self.assertEqual([r['job']['seed'] for r in selected[:4]],[seed+x for x in (0,100003,200003,300007)])
        for row in rows:runner.validate_job(row['job']);self.assertEqual(row['job']['policy'],SPEC['selection'])
        baseline=json.loads((HERE/'BASELINE.json').read_text())
        self.assertEqual(len(baseline['banks']),12);self.assertEqual(len(baseline['fits']),18)
        for seed in s.SEEDS:
            lookup={r['bank']:r for r in baseline['banks'] if r['seed']==seed}
            self.assertEqual(lookup['BCE_M1']['owners'],lookup['BCE_I4']['owners'][:1])

    def test_all21_terminal_gate(self):
        rows=s.jobs(SPEC);out=[{'id':r['id'],'status':'complete','returncode':0,'RESULT':{'sha256':'a'*64}} for r in rows]
        def gate(v):s.terminal_gate({'status':'complete','TEST_accessed':False,'outcomes':v},v,rows)
        gate(out)
        for bad in (out[:-1],list(reversed(out)),[dict(out[0],status='started')]+out[1:],out[:-1]+[dict(out[-1],timed_out=True)],out[:-1]+[dict(out[-1],RESULT={})]):
            with self.assertRaises(ValueError):gate(bad)

    def test_all9_collection_gate(self):
        rows=[r for r in s.jobs(SPEC) if r['kind']!='native_common'];out=[{'id':r['id'],'status':'complete_with_sanity_warning','returncode':0,'RESULT':{'sha256':'a'*64}} for r in rows]
        s.collection_gate({'status':'complete','TEST_accessed':False,'outcomes':out},rows)
        for bad in (out[:-1],list(reversed(out)),[dict(out[0],returncode=1)]+out[1:]):
            with self.assertRaises(ValueError):s.collection_gate({'status':'complete','TEST_accessed':False,'outcomes':bad},rows)

    def test_ties_credit_repairs_and_alternatives(self):
        d=s.diagnostics();old=[-1.,0.,1.];new=[1.,-1.,0.]
        value=d.compare_pairs(old,new);alternative=s.alternatives([(-1.,-2.),(0.,0.),(1.,-1.)],[(1.,2.),(-1.,-2.),(0.,-1.)],old,new)
        self.assertEqual(value['strict_repairs'],1);self.assertEqual(value['strict_harms'],0)
        self.assertEqual(alternative['tie_aware_repair_credit'],1.);self.assertEqual(alternative['tie_aware_harm_credit'],1.)
        self.assertEqual(alternative['acquired_correct_alternatives'],1);self.assertEqual(alternative['lost_correct_alternatives'],1)
        self.assertEqual(alternative['reference_common_reversal_served_repairs'],1)
        self.assertEqual(value['delta_AUC_pp'],100*(alternative['tie_aware_repair_credit']-alternative['tie_aware_harm_credit'])/3)

    def test_frozen_decisions_and_three_seed_summary(self):
        levels={'BCE_M1':.5,'BCE_I4':.68,'pair_I4':.69,'residual_F':.59,'common':.6,'permuted':.65,'structured':.7}
        rows=[{'bank':name,'seed':seed,'pool_AUC':value} for seed in s.SEEDS for name,value in levels.items()]
        self.assertTrue(s.decision(rows)['main_goal_advancement_screen_passes'])
        tied=[dict(r,pool_AUC=.7) if r['bank']=='common' else r for r in rows]
        self.assertFalse(s.decision(tied)['fixed_structural_recipe_passes'])
        with self.assertRaises(ValueError):s.decision(rows[:-1])
        with self.assertRaises(ValueError):s.diagnostics().summarize(rows[:-1],'bank',['pool_AUC'])

    def test_closed_templates_source_only_and_manifest(self):
        for p in HERE.glob('*.py'):
            tree=ast.parse(p.read_text())
            for n in tree.body:
                if isinstance(n,ast.Import):self.assertFalse(any(a.name.startswith(('torch','numpy','ogb')) for a in n.names))
                if isinstance(n,ast.ImportFrom):self.assertFalse((n.module or '').startswith(('torch','numpy','ogb')))
        analyzer=(HERE/'analyze.py').read_text();driver=(HERE/'driver.py').read_text()
        self.assertLess(analyzer.index('s.collection_gate('),analyzer.index('import torch'))
        self.assertNotIn('create_adapter',analyzer);self.assertNotIn('create_session',analyzer)
        self.assertIn('s.serial()',driver);self.assertIn('analysis/selected_molecular.py',driver)
        self.assertFalse(SPEC['scientific_admission']);self.assertFalse(SPEC['TEST_access'])
        for name in ('FITS_ADMISSION_TEMPLATE.json','COLLECTION_ADMISSION_TEMPLATE.json'):
            value=json.loads((HERE/name).read_text());self.assertFalse(value['approved']);self.assertFalse(value['resource_admission_approved'])
            self.assertIsNone(value['maximum_wall_seconds_per_child']);self.assertIsNone(value['manifest_sha256'])
        path=HERE/'MANIFEST.json'
        if path.exists():
            for pin in json.loads(path.read_text())['files']:
                p=HERE/pin['path'];self.assertEqual(p.stat().st_size,pin['bytes']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),pin['sha256'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report={'schema':'pairwise21-scientific-source-checks-v1','status':'passed' if result.wasSuccessful() else 'failed',
        'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_JSON_synthetic_scalar_only':True,
        'new_models_or_data_or_tensor_execution':False,'SSH_Git_GPU_operations':0,'scientific_fits_launched':0,'TEST_accessed':False}
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
