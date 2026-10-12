"""Focused local stdlib custody/count/graph tests; no target-array readers."""
import copy
import gzip
import json
from pathlib import Path
import sys
import unittest

from adapters.private_rotation import source_build
from compile_rotation_q21_successor import append_study,METHODS,ADDED_EXPECTED
from prove_q21_rotation_overlap import prove
from spec import fingerprint

PACKET=Path(__file__).resolve().parents[2];ROOT=PACKET.parent


class RotationMetadata(unittest.TestCase):
    def setUp(self):
        self.prior=json.loads((PACKET/'lineage/prior_q21/matrix/PLAN.json').read_text())
        self.bindings=json.loads((PACKET/'SOURCE_BINDINGS.json').read_text())
    def studies(self):
        return [json.loads(gzip.decompress((PACKET/'lineage/prior_q21/matrix/studies'/Path(row['path']).name).read_bytes())) for row in self.prior['studies']]

    def test_exact_provider_three_anchors_and_original_train_branch(self):
        source,paths,v=source_build(ROOT,self.bindings,7,3)
        self.assertEqual(v['provider_changes'],['features','classes','stem_sign_features'])
        self.assertTrue(v['copied_full_bank_or_scoped_single_unit']);self.assertFalse(v['source_legacy_loader_selector_owner_executed'])
        bad=copy.deepcopy(self.bindings);bad['family']['sha256']='0'*64
        with self.assertRaises(ValueError):source_build(ROOT,bad,7,3)

    def test_mapping_complete_static_finite_unit_proof(self):
        value=prove(ROOT,PACKET)
        self.assertEqual(len(value['cohorts']),12);self.assertTrue(value['normalized_unweighted_bank_core_AST_identical'])
        self.assertEqual(value['new_bank_fits_for_mapped_three'],0);self.assertFalse(value['whole_legacy_trajectory_equivalence'])

    def test_old_rows_roles_weighted_and_original_gnnm_exact_with_finite_increment(self):
        counts={k:0 for k in ADDED_EXPECTED};gnnm=0
        for original in self.studies():
            value,n,banks,p=append_study(original,self.bindings,[],PACKET.name,{})
            self.assertEqual(value['runs'][:len(original['runs'])],original['runs']);self.assertEqual(value['data'],original['data'])
            self.assertEqual(value['comparisons'][:-1],original['comparisons']);self.assertEqual(n['atomic_fit_groups'],594 if value['dataset']=='wikics' else 324)
            self.assertEqual(n['all_run_records'],1497 if value['dataset']=='wikics' else 777);gnnm+=p['original_GNNM_fits']
            for k in counts:counts[k]+=n[k]
        self.assertEqual(counts,ADDED_EXPECTED);self.assertEqual(gnnm,252)

    def test_full_cohorts_genuine_i4_exact_m1_body0_and_new_selector_custody(self):
        for original in self.studies():
            value,n,banks,p=append_study(original,self.bindings,[],PACKET.name,{})
            runs={r['run_id']:r for r in value['runs']};cohorts={(i,s) for i in range(20 if value['dataset']=='wikics' else 10) for s in (77201,77203,77207)}
            for method in METHODS.values():
                self.assertEqual({(r['split'],r['seed']) for r in value['runs'] if r['kind']=='ensemble_assembly' and r['method']==method},cohorts)
            for bank in banks:
                if bank['genuine_I4']:
                    members=[runs[p] for p in bank['fit_parents']];self.assertEqual([r['body'] for r in members],[0,1,2,3])
                    self.assertTrue(all(r['config']['members']==1 for r in members));self.assertEqual(len({r['config']['initialization_seed'] for r in members}),4)
                    m1=next(b for b in banks if b['fit_parents']==[members[0]['run_id']])
                    self.assertEqual(m1['fit_parents'][0],bank['fit_parents'][0])
                    config=[{k:v for k,v in r['config'].items() if k!='initialization_seed'} for r in members]
                    self.assertTrue(all(c==config[0] for c in config));self.assertEqual(len(members[0]['parents']),1)

    def test_registered_packet_controller_ingestion_and_namespaced_factory(self):
        sys.path.insert(0,str(ROOT/'training_pipeline_v2_20261012'))
        from compile_master_plan import load_packet
        from run_study import load_factory
        entry=json.loads((PACKET/'MASTER_INPUT_ENTRY.json').read_text());v=load_packet(ROOT,entry)
        self.assertEqual(len(v['studies']),4);self.assertEqual(v['summary']['counts']['atomic_fit_groups'],7020)
        factory=load_factory(ROOT,PACKET.name+'/runtime/adapters/private_rotation.py:make_adapter')
        self.assertEqual(factory.__name__,'make_adapter');self.assertTrue(factory.__module__.startswith('gnnm_packet_'))
        self.assertEqual(v['summary']['added_original_GNNM_fits'],0)


if __name__=='__main__':unittest.main()
