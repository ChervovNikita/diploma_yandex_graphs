"""Synthetic CPU resource-helper proofs; never CUDA or predictive fits."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from qa.staged_random_resource_helpers import build_proxy,measure,role_paths,local
from adapters.retained_pt_routes import receipt
from adapters.gru_memory import prepare_tour_capsule
from adapters.landmark_memory import prepare_graph_capsule,SOURCE_PATHS

PT_PATHS={'pt':'native_meanSAGE_private_continuation_PT_core_source_20261011_v2/core.py','native':'native_meanSAGE_attribute_peer_feedback_core_source_20261011_v1/core.py','original':'common_wrapper_native_source_20261010_v1/models.py','original_driver':'common_wrapper_native_source_20261010_v1/run_common.py'}
GRU_PATHS={**{k:v for k,v in PT_PATHS.items() if k!='pt'},'gru':'native_meanSAGE_private_GRU_core_source_20261011_v1/core.py','backend':'native_meanSAGE_private_GRU_core_source_20261011_v1/gru_backend.py','tours':'native_meanSAGE_tour_scan_preparation_source_20261011_v1/tours.py'}


class StagedRandomResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.t=torch; torch.set_num_threads(1); cls.reference=Path(os.environ['GNNM_REFERENCE_PROJECT'])
        spec=importlib.util.spec_from_file_location('denial_cap_CPU_fixture',Path(__file__).resolve().parents[1]/'qa/practical_zero_update_resources.py'); cls.h=importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.h)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name); self.sources={}
        for family,paths in [('pt',PT_PATHS),('gru',GRU_PATHS),('landmark',SOURCE_PATHS)]:
            self.sources[family]={}
            for key,rel in paths.items():
                path=self.root/rel; path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists(): shutil.copyfile(self.reference/rel,path)
                self.sources[family][key]=receipt(self.root,path)
        t=self.t; x=t.arange(360,dtype=t.float32).reshape(72,5)/360; links=[(i,i+1) for i in range(71)]; edges=[(i,i) for i in range(72)]+links+[(v,u) for u,v in links]; inputs={}
        for role,value in [('graph',{'x':x,'edge_index':t.tensor(edges,dtype=t.long).T.contiguous()}),('train',{'ids':t.arange(36),'y':t.arange(36)%3})]:
            path=self.root/(role+'.pt'); t.save(value,path); inputs[role]=receipt(self.root,path)
        self.inputs=inputs; self.capsules={}
        for family,prepare in [('gru',prepare_tour_capsule),('landmark',prepare_graph_capsule)]:
            capsule=prepare(t,self.root,inputs['graph'],self.sources[family],10103); path=self.root/(family+'_capsule.pt'); t.save(capsule,path); self.capsules[family]={'capsule':receipt(self.root,path),'payload_hash':capsule['payload_hash']}
    def case(self,family,arm,bodies=1):
        return {'architecture_family':family,'arm':arm,'dropout':.5 if family=='landmark' else .4,'runtime_package':'.','GRAPH_TRAIN_bindings':self.inputs,
            'source_operator_pins':self.sources[family],'prepared_GRAPH_capsule':self.capsules.get(family),'declared_dimensions':{'width':256 if family=='landmark' else 32,'depth':2,'classes':3},
            'random_seed':10103,'paired_seed':10103,'prototype_bodies':bodies}
    def probe(self,family,arm,bodies=1):
        output=self.root/(family+'_'+arm+'_'+str(bodies)); output.mkdir(); allowed=[self.root/v['path'] for v in self.inputs.values()]
        if family in self.capsules: allowed.append(self.root/self.capsules[family]['capsule']['path'])
        with self.h.OptimizerDenial(self.t) as denial,self.h.RoleIO(self.t,self.root,allowed,[self.root/'FORBIDDEN_SELECT.pt'],output) as io:
            proxy=build_proxy(self.t,self.root,self.case(family,arm,bodies)); result=measure(self.t,self.h,proxy,denial,output,1)
            self.assertEqual(result['actual_optimizer_updates'],0); self.assertEqual(result['actual_optimizer_kernel_calls'],0); self.assertEqual(result['parameters_sha256_before'],result['parameters_sha256_after']); self.assertEqual(result['moments_sha256_before'],result['moments_sha256_after'])
            self.assertEqual(result['scores_emitted'],0); self.assertTrue(result['disposable_serialization']['removed']); self.assertFalse((output/'DISPOSABLE_RANDOM_ENGINEERING_STATE.pt').exists()); self.assertEqual(io.denied,0)
            self.assertEqual(set(io.tensor_load_paths),{str(p.resolve()) for p in allowed if family!='pt' or not p.name.endswith('_capsule.pt')})
        return proxy,result
    def test_pure_metadata_boundaries(self):
        self.assertEqual(role_paths({'graph':{'path':'graph.pt','bytes':1,'sha256':'a'},'train':{'path':'train.pt','bytes':1,'sha256':'b'}}),{'graph.pt','train.pt'})
        with self.assertRaises(ValueError): local(self.root,'../escape')
    def test_PT_direct_random_source_and_separable_ownership(self):
        for arm in ('factor_s4','mixed_single','separable4'):
            proxy,result=self.probe('pt',arm); self.assertFalse(result['trained_or_paid_ancestry_qualified']); self.assertIsNone(result['all_fit_ETA'])
            self.assertEqual(result['source_operator_counts_by_cycle'][0]['route_predictions'],16 if arm=='separable4' else 4)
    def test_GRU_random_H0_and_distinct_live_GN4(self):
        for arm in ('GF1','GN4'):
            proxy,result=self.probe('gru',arm); self.assertEqual(len(result['random_cache_hashes']),0 if arm=='GN4' else 1); self.assertEqual(result['source_operator_counts_by_cycle'][0]['route_predictions'],4)
        proxy,result=self.probe('gru','GF1',4); self.assertEqual(result['optimizer_objects'],4); self.assertEqual(result['source_operator_counts_by_cycle'][0]['route_predictions'],16)
    def test_landmark_random_raw_H_U_private_views_and_four_own_prototypes(self):
        for arm in ('LM4','FS64'):
            proxy,result=self.probe('landmark',arm); self.assertEqual(len(result['random_cache_hashes']),1); self.assertEqual(set(result['random_cache_hashes'][0]),{'random_H','random_U'}); self.assertEqual(result['source_operator_counts_by_cycle'][0]['common_projection_banks'],1)
        proxy,result=self.probe('landmark','S64',4); self.assertEqual(result['optimizer_objects'],4); self.assertEqual(result['source_operator_counts_by_cycle'][0]['route_predictions'],4)


if __name__=='__main__': unittest.main()
