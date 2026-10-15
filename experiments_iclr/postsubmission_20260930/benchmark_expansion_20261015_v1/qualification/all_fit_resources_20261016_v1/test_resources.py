"""Safety tests for actual finite source scopes and scheduler reservation API."""
import copy
import json
import unittest
import resources as r


def task(preset,method='gnnm_boundary_native',family='node'):
    return{'kind':'fit','cell_id':preset,'scientific_method_id':method,'family':family,'worker':'allocation/gpu0'}


class ResourceAPI(unittest.TestCase):
    def test_exact_budget_and_ignored_seed(self):
        t=task('polyformer-coauthor-physics-mono')
        value=r.reserve_for(t)
        self.assertEqual(set(value),set(r.FIELDS))
        self.assertEqual(value['vram_bytes'],69*1024**3)
        t.update(seed=31233,split=4,id='fresh-smoke-id')
        self.assertEqual(value,r.reserve_for(t))
        self.assertFalse(r.can_pair(value,r.reserve_for(task('polyformer-citeseer-cheb','native_single'))))

    def test_compatible_pairs_and_budget_tampering(self):
        small=r.reserve_for(task('ogb-ogbg-molhiv-gin-virtual-drop0.5',family='ogb'))
        link=r.reserve_for(task('collab-ncnc',family='link'))
        self.assertTrue(r.can_pair(small,link))
        self.assertTrue(r.can_pair(link,small))
        self.assertFalse(r.can_pair(link,link))
        forged=dict(small,vram_bytes=1)
        self.assertFalse(r.can_pair(forged,link))

    def test_old_dataset_wrong_method_shape_threads_rejected(self):
        with self.assertRaises(ValueError):r.reserve_for(task('cora'))
        with self.assertRaises(ValueError):r.reserve_for(task('polyformer-citeseer-cheb','unreviewed_new_model'))
        t=task('ogb-ogbg-molhiv-gin-virtual-drop0.5',family='ogb')
        config=copy.deepcopy(r._catalogue()['presets'][t['cell_id']]['native_fixed_config']);config['hidden']=600
        with self.assertRaises(ValueError):r.reserve_for(t,{'recipe':{'native_fixed_config':config}})
        with self.assertRaises(ValueError):r.reserve_for(t,{'torch_threads':8})

    def test_actual_descriptor_alias_and_I4(self):
        t=task('ogb-ogbg-molhiv-gin-virtual-drop0.5','interaction_input_conditionalJ','ogb')
        job={'recipe_id':t['cell_id'],'family':'ogb','scientific_method_id':t['scientific_method_id'],
             'procedure':'source_frontend_alias','torch_threads':2}
        self.assertEqual(r.reserve_for(t),r.reserve_for(t,job))
        value=r.reserve_for({'kind':'aggregate_I4_exports','cell_id':'collab-buddy'})
        self.assertEqual(value['vram_bytes'],2*1024**3)
        self.assertEqual(value['cpu_slots'],2)

    def test_all_actual_classes_retain_reviewed_reservations(self):
        old=r.HERE.parent/'campaign_resources_v1/review_snapshot_v1/RESOURCE_CLASS_PROPOSALS.json'
        source=json.loads(old.read_text())
        for row in source['classes']:
            s=row['signature']
            expected={k:row['resources'][k]for k in r.FIELDS}
            self.assertEqual(r.reserve_for(s),expected)


if __name__=='__main__':unittest.main()
