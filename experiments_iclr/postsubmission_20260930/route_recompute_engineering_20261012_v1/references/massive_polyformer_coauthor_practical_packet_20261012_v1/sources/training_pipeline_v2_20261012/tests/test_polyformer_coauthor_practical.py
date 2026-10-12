"""Concrete Coauthor domain/reference metadata checks; zero model execution."""
import copy
from pathlib import Path
import unittest
from test_polyformer_matrix import fixture_inputs
from compile_polyformer_matrix import make_study
from compile_polyformer_boundary_matrix import append_boundary
from compile_polyformer_coauthor_practical_matrix import rebuild_coauthor
from compile_polyformer_practical_matrix import COAUTHOR_POLICY, practical_domain
from adapters.polyformer_boundary import M4, M1


class CoauthorPracticalPolicyTests(unittest.TestCase):
    def test_native_shapes_grouped_LRs_nine_points_all_references_and_genuine_I4(self):
        for dataset in ('coauthor-cs','coauthor-physics'):
            custody,recipe,source,profile=fixture_inputs(dataset);root=Path.cwd()
            custody['splits']=[{**copy.deepcopy(custody['splits'][0]),'split_id':i} for i in range(10)]
            base,c,b,d=make_study(root,custody,root/'tiny',recipe,source,profile,'native_fixed','old/runtime')
            original,fc,fb,fd=append_boundary(base,{'counts':c,'served_banks':b,'search_domains':d},'v2/runtime',[])
            immutable=copy.deepcopy(original)
            value,total,served,domains=rebuild_coauthor(original,custody,root/'tiny',recipe,
                {'counts':fc,'served_banks':fb,'search_domains':fd},root,'new/runtime',[],COAUTHOR_POLICY)
            self.assertEqual(original,immutable)
            self.assertEqual(value['data']['evidence']['split_roster'],list(range(10)))
            self.assertEqual(total['optimizer_starts_by_stage'],{'search':648,'comparative':2790})
            self.assertEqual(total['initial_training_segments'],3438)
            self.assertEqual(total['all_run_records'],8424)
            self.assertEqual(total['genuine_I4_banks'],570)
            self.assertEqual(total['full_budget_optimizer_updates'],6876000)
            self.assertEqual(len(domains),36)
            self.assertEqual({r['split'] for r in value['runs']},set(range(10)))
            for method,domain in domains.items():
                points=domain['parameter_points'];self.assertEqual(len(points),9)
                self.assertEqual(sum(p['source_anchor'] for p in points),1)
                for point in points:
                    p=point['parameters'];native=p['native']
                    self.assertEqual(native['hidden'],128);self.assertEqual(native['nlayer'],1)
                    self.assertEqual(native['K'],recipe['operator']['polynomial_order_K'])
                    self.assertEqual(native['n_head'],recipe['locks']['heads']);self.assertEqual(native['d_ffn'],recipe['locks']['d_ffn'])
                    self.assertAlmostEqual(p['attn_lr']/p['learning_rate'],recipe['native_point']['learning_rate_groups']['attnmodule']/recipe['native_point']['learning_rate_groups']['other'])
                    self.assertEqual(p['weight_decay'],0 if method in (M4,M1) else recipe['native_point']['weight_decay_groups']['other'])
                    self.assertEqual(p['attn_weight_decay'],0 if method in (M4,M1) else recipe['native_point']['weight_decay_groups']['attnmodule'])
            original_fits=[r for r in value['runs'] if r['kind']=='model_fit' and r['method']==M4]
            self.assertEqual(len(original_fits),48)
            self.assertTrue(all(r['config']['is_original_gnnm'] and r['config']['members']==4 and r['config']['fusion']=='mean_logits' for r in original_fits))
            self.assertTrue(all(M4 in c['references'] and M1+'_I4' in c['references'] for c in value['comparisons']))
            groups={r['config']['independence_group'] for r in value['runs'] if r['kind']=='model_fit' and r['config'].get('independence_group')}
            self.assertEqual(len(groups),570)
            self.assertFalse(value['source']['coauthor_coverage_policy']['author_score_protocol_equality'])

    def test_policy_changes_rejected_and_frozen_native_anchor_checked(self):
        _,recipe,_,_=fixture_inputs('coauthor-cs')
        altered=copy.deepcopy(COAUTHOR_POLICY);altered['datasets']['coauthor-cs']['dropout']=[0,.2,.4]
        with self.assertRaisesRegex(ValueError,'frozen'):practical_domain(recipe,M4,altered)
        altered=copy.deepcopy(recipe);altered['native_point']['learning_rate_groups']['attnmodule']=.001
        with self.assertRaisesRegex(ValueError,'anchor'):practical_domain(altered,M4,COAUTHOR_POLICY)


if __name__=='__main__':unittest.main()
