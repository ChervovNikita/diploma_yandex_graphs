"""Source/JSON and synthetic scalar checks; no model/data/tensor execution."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import study as s
SPEC = json.loads((HERE/'SPEC.json').read_text())


def local(path):
    prefix = SPEC['project_root']+'/experiments_iclr/postsubmission_20260930/'
    s.require(path.startswith(prefix),'Source pin outside phase')
    return s.PHASE/path[len(prefix):]


class Checks(unittest.TestCase):
    def test_exact24_native_and_residual_descriptors(self):
        rows,cohorts = s.jobs(SPEC)
        self.assertEqual(len(rows),24);self.assertEqual(len(cohorts),3)
        ext = s.load(s.COMPOSITION/'extension.py','_pubmed_descriptor_checks')
        sys.path.insert(0,str(s.E/'runtime'));runner = s.load(s.E/'runtime/run_fit.py','_pubmed_runner_checks')
        for seed in s.SEEDS:
            selected = [r for r in rows if r['cohort_seed'] == seed]
            self.assertEqual([r['kind'] for r in selected],['native']*4+['G0','uniform','structured','permuted'])
            self.assertEqual([r['job']['seed'] for r in selected[:4]],[seed+x for x in [0,100003,200003,300007]])
            cohort = next(c for c in cohorts if c['M1_fit_id'] == selected[0]['id'])
            self.assertEqual(cohort['native_fit_ids'],[r['id'] for r in selected[:4]])
            for row in selected:
                runner.validate_job(row['job']);self.assertEqual(row['job']['policy'],ext.policy())
                if row['kind'] in ('uniform','structured','permuted'):
                    expected = ext.descriptor(SPEC['project_root'],seed,row['kind'],SPEC['composition_manifest']['sha256'])
                    expected['recipe']['_repository_root'] = SPEC['project_root']
                    expected['session_transform']['path'] = row['job']['session_transform']['path']
                    self.assertEqual(expected,row['job'])
                else:self.assertNotIn('session_transform',row['job'])
        self.assertEqual(len({r['id'] for r in rows}),24)

    def test_source_pins_and_completed_CPU_receipts(self):
        rows,_ = s.jobs(SPEC)
        for receipt in SPEC['source_pins']:
            p = local(receipt['path']);self.assertEqual(p.stat().st_size,receipt['bytes'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),receipt['sha256'])
        for receipt in SPEC['completed_CPU_qualifications']:
            p = local(receipt['authority']['path']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),receipt['authority']['sha256'])
            v = json.loads(p.read_text())['native'];self.assertEqual(v['status'],'passed')
            expected = next(r for r in rows if r['cohort_seed'] == 101 and r['kind'] == receipt['variant'])
            self.assertEqual(v['planned_job_hash'],expected['job_hash'])
            self.assertEqual(v['optimizer_calls'],{'attempts':[2],'completed':[2]})
            self.assertEqual(v['VALID_completed'],3);self.assertTrue(v['fresh_restore_VALID_bit_exact'])

    def test_all24_terminal_gate(self):
        rows,_ = s.jobs(SPEC)
        outcomes = [{'id':r['id'],'status':'complete','returncode':0,'RESULT':{'sha256':'a'*64}} for r in rows]
        def gate(v,status='complete',test=False):s.terminal_gate({'status':status,'outcomes':v,'TEST_accessed':test},v,rows)
        gate(outcomes)
        for bad in (outcomes[:-1],list(reversed(outcomes)),[dict(outcomes[0],status='running')]+outcomes[1:],
            outcomes[:-1]+[dict(outcomes[-1],returncode=1)],outcomes[:-1]+[dict(outcomes[-1],timed_out=True)],
            outcomes[:-1]+[dict(outcomes[-1],RESULT={})]):
            with self.assertRaises(ValueError):gate(bad)
        with self.assertRaises(ValueError):gate(outcomes,'running')
        with self.assertRaises(ValueError):gate(outcomes,test=True)

    def test_all12_collection_gate(self):
        rows = [r for r in s.jobs(SPEC)[0] if r['kind'] != 'native']
        outcomes = [{'id':r['id'],'status':'complete_with_sanity_warning','returncode':0,'RESULT':{'sha256':'a'*64}} for r in rows]
        s.collection_gate({'status':'complete','TEST_accessed':False,'outcomes':outcomes},rows)
        for changed in (outcomes[:-1],list(reversed(outcomes)),[dict(outcomes[0],status='failed')]+outcomes[1:]):
            with self.assertRaises(ValueError):s.collection_gate({'status':'complete','TEST_accessed':False,'outcomes':changed},rows)

    def test_native_argmax_ties_and_class_macro_NLL(self):
        logits = [[1.,1.,-1.],[2.,1.,0.],[0.,2.,1.]];labels = [0,1,1]
        self.assertEqual(s.predictions(logits),[0,0,1])
        m = s.metrics(logits,labels)
        self.assertAlmostEqual(m['accuracy'],2/3);self.assertAlmostEqual(m['macro_accuracy'],.75)
        self.assertEqual([r['support'] for r in m['per_class']],[1,2,0])
        self.assertIsNone(m['per_class'][2]['accuracy']);self.assertIsNone(m['per_class'][2]['NLL'])
        expected = sum(math.log(sum(math.exp(v-max(z)) for v in z))+max(z)-z[y] for z,y in zip(logits,labels))/3
        self.assertAlmostEqual(m['NLL'],expected)
        self.assertAlmostEqual(s.metrics([[1e30,1e30,1e30]],[0])['NLL'],math.log(3))
        with self.assertRaises(ValueError):s.metrics([[float('nan'),0.,1.]],[0])

    def test_all_member_wrong_does_not_imply_common_false_rival(self):
        members = [[[0.,1.,-2.]],[[0.,-2.,1.]]];pool = [[0.,-.5,-.5]]
        m = s.errors(members,pool,[0])
        self.assertEqual((m['no_correct_member'],m['strict_common_false_rival'],m['pool_errors']),(1,0,0))
        m = s.errors([[[0.,1.,2.]]]*4,[[0.,1.,2.]],[0])
        self.assertEqual(m['strict_common_false_rival'],1)

    def test_signed_repairs_harms_and_common_alternatives(self):
        labels = [0,1,2]
        ref = {'pool':[[0.,2.,1.],[0.,2.,1.],[2.,0.,1.]],'members':[[[0.,2.,1.],[0.,2.,1.],[2.,0.,1.]]]*4}
        new = {'pool':[[2.,0.,1.],[2.,0.,1.],[0.,1.,2.]],'members':[[[2.,0.,1.],[2.,0.,1.],[0.,1.,2.]]]*4}
        m = s.repairs(ref,new,labels)
        self.assertEqual((m['repairs'],m['harms']),(2,1))
        self.assertEqual((m['new_correct_member_alternatives'],m['lost_correct_member_alternatives']),(2,1))
        self.assertEqual((m['reference_common_rival_served_repairs'],m['new_strict_common_false_rivals']),(2,1))
        self.assertEqual([r['support'] for r in m['per_class']],[1,1,1])
        self.assertAlmostEqual(s.metrics(new['pool'],labels)['accuracy']-s.metrics(ref['pool'],labels)['accuracy'],(m['repairs']-m['harms'])/3)

    def test_summary_needs_all_three_seeds(self):
        rows = [{'seed':seed,'bank':'one','accuracy':seed} for seed in s.SEEDS]
        self.assertEqual(s.summarize(rows,'bank',['accuracy'])[0]['summary']['accuracy']['values'],s.SEEDS)
        with self.assertRaises(ValueError):s.summarize(rows[:-1],'bank',['accuracy'])

    def test_source_only_import_and_saved_authority(self):
        for p in HERE.glob('*.py'):
            tree = ast.parse(p.read_text())
            imports = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
            self.assertFalse(any(isinstance(n,ast.Import) and any(a.name.startswith(('torch','numpy')) for a in n.names) for n in imports))
        collector = (HERE/'collect.py').read_text();analyzer = (HERE/'analyze.py').read_text()
        self.assertLess(collector.index('s.terminal_gate('),collector.index('import torch'))
        self.assertLess(analyzer.index('s.collection_gate('),analyzer.index('import torch'))
        self.assertIn('full.mean(0)[ids]',collector)
        self.assertIn("saved['raw_logits'].tolist()",analyzer)
        self.assertNotIn('create_adapter',analyzer)
        self.assertNotIn('test_mask',collector)
        self.assertNotIn('selector.observe',collector)
        self.assertIn('forbidden_update',collector);self.assertIn('forbidden_step',collector)

    def test_minor_class_and_closed_admission_templates(self):
        self.assertEqual(SPEC['provider']['VALID_class_counts'],[72,1919,1952])
        self.assertEqual(SPEC['provider']['source_split_seed'],1941488137)
        for name in ('FITS_ADMISSION_TEMPLATE.json','COLLECTION_ADMISSION_TEMPLATE.json'):
            a = json.loads((HERE/name).read_text())
            self.assertFalse(a['approved']);self.assertFalse(a['resource_admission_approved'])
            self.assertIsNone(a['maximum_wall_seconds_per_child']);self.assertIsNone(a['manifest_sha256'])

    def test_manifest_if_present(self):
        p = HERE/'MANIFEST.json'
        if p.exists():
            for row in json.loads(p.read_text())['files']:
                f = HERE/row['path'];self.assertEqual(f.stat().st_size,row['bytes'])
                self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),row['sha256'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = {'schema':'pubmed-cheb-transfer-source-checks-v1','status':'passed' if result.wasSuccessful() else 'failed',
        'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'native_model_or_data_or_tensor_execution':False,
        'remote_Git_GPU_operations':0,'scientific_fits_launched':0,'TEST_accessed':False}
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
