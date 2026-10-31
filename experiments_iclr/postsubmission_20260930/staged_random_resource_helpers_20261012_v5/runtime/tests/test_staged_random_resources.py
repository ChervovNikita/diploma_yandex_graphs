"""Synthetic CPU resource-helper proofs; never CUDA or predictive fits."""
import hashlib
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import json
import tempfile
import unittest

from qa.staged_random_resource_helpers import build_proxy,measure,measure_train_export,role_paths,local,physics_custody,PHYSICS,PHYSICS_DIR,READOUT_DIR,READOUT_WATCHER
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


class TerminalCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name).resolve()
        self.start={'stage':'acquire','child_wall_seconds':604800,'external_wall_seconds':604860,'direct_child_wait':True,'retry':False,**copy.deepcopy(PHYSICS),
            'command':['python','-B','-u','source/worker.py','root/CONFIG.json']}
        self.start_path=self.root/PHYSICS_DIR/'OWNER_START.json'; self.write(self.start_path,self.start); self.start_binding=receipt(self.root,self.start_path)
        self.end_path=self.start_path.with_name('OWNER_END.json')
        self.end={**copy.deepcopy(self.start),'status':'COMPLETE_EXIT_0','exit_code':0,'timed_out':False,'seconds':12.5,'child_pid_absent':True,'owned_cuda_pid_absent':True}; self.write(self.end_path,self.end)
        watcher_root=self.root/READOUT_DIR; self.watch_start={'complete':False,'pid':READOUT_WATCHER['pid'],'stage':'waiting_for_whole_acquisition'}
        for key in ('partial_metrics_opened','partial_owner_logs_opened','scientific_gate_used_for_handoff','training_or_qualification_launched','model_retry','readout_retry'): self.watch_start[key]=False
        for key in ('root_decision','watcher_manifest','reader_source_manifest','acquisition_source_manifest','acquisition_config','acquisition_launch','confidence_clarification'): self.watch_start[key]={'path':key+'.json','bytes':1,'sha256':'a'*64}
        self.watch_start['config']={'path':str(watcher_root/'CONFIG.json'),'bytes':1,'sha256':'b'*64}; self.watch_start['acquisition_owner_start']={**self.start_binding,'path':str(self.start_path)}
        self.watch_start.update(finite_wall_seconds=619380,cleanup_reserve_seconds=20)
        launch={'watcher_observed':{**READOUT_WATCHER,'state':'S'},'watcher_pid':READOUT_WATCHER['pid'],'watch_start':self.watch_start,'config_sha256':'b'*64,
            'no_model_retry':True,'no_readout_retry':True,'partial_predictions_opened':False,'launch_exit':None,
            'command':['python','-B','-u',str(self.root/'private_GRU_whole_readout_watcher_source_20261011_v1/watcher.py'),str(watcher_root/'CONFIG.json')]}
        self.launch_path=watcher_root/'LAUNCH.json'; self.watch_start_path=watcher_root/'watch_v1/WATCH_START.json'; self.write(self.launch_path,launch); self.write(self.watch_start_path,self.watch_start)
        self.watcher={'launch':receipt(self.root,self.launch_path),'watch_start':receipt(self.root,self.watch_start_path),'expected_identity':READOUT_WATCHER}
    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,sort_keys=True)+'\n')
    def observer(self,pid):
        return {**READOUT_WATCHER,'state':'S'} if pid==READOUT_WATCHER['pid'] else None
    def check(self,observer=None): return physics_custody(self.root,self.start_binding,self.watcher,process_reader=observer or self.observer)
    def test_terminal_forged_end_is_rejected(self):
        variants=[{'owner':{**PHYSICS['owner'],'start_ticks':1}},{'child':{**PHYSICS['child'],'pgid':1}},{'command':['forged']},{'status':'forged'},
            {'child_pid_absent':False},{'owned_cuda_pid_absent':False},{'status':'FAILED_RETAINED'},{'timed_out':True},{'exit_code':1},{'extra':1}]
        for change in variants:
            with self.subTest(change=change):
                self.write(self.end_path,{**self.end,**change})
                with self.assertRaises(ValueError): self.check()
        self.write(self.end_path,self.end)
    def test_terminal_still_present_old_child_or_owner_is_rejected(self):
        for key in ('child','owner'):
            for state in ('R','Z'):
                with self.subTest(key=key,state=state):
                    def observer(pid): return {**PHYSICS[key],'state':state} if pid==PHYSICS[key]['pid'] else self.observer(pid)
                    with self.assertRaisesRegex(ValueError,'Old Physics owner/child identity'): self.check(observer)
    def test_terminal_correct_original_watcher_custody_and_PID_reuse(self):
        result=self.check(); self.assertFalse(result['old_identities_present']); self.assertEqual(result['watcher']['mode'],'original_readout_watcher_exact_process_live')
        self.assertFalse(result['scientific_outcome_read'])
        self.write(self.end_path,{**self.end,'status':'FAILED_RETAINED','exit_code':1})
        def reused(pid): return {**PHYSICS['child'],'start_ticks':PHYSICS['child']['start_ticks']+1,'state':'R'} if pid==PHYSICS['child']['pid'] else self.observer(pid)
        self.assertFalse(self.check(reused)['old_identities_present'])
        watch_end={k:v for k,v in self.watch_start.items() if k not in ('pid','stage')}; watch_end.update(complete=False,last_stage='waiting_for_whole_acquisition',seconds=20.,failure={'type':'ValueError'})
        end_path=self.watch_start_path.with_name('WATCH_END.json'); self.write(end_path,watch_end)
        with self.assertRaisesRegex(ValueError,'Old readout watcher'): self.check()
        terminal=self.check(lambda pid:None); self.assertEqual(terminal['watcher']['mode'],'original_readout_watcher_terminal_metadata_verified')
        self.write(end_path,{**watch_end,'acquisition_owner_start':{'path':'forged'}})
        with self.assertRaisesRegex(ValueError,'watcher terminal lifecycle'): self.check(lambda pid:None)


class ExportSupplementTests(StagedRandomResourceTests):
    def test_sequential_export_supplement(self):
        output=self.root/'new_export_logic'; output.mkdir(); allowed=[self.root/v['path'] for v in self.inputs.values()]+[self.root/self.capsules['gru']['capsule']['path']]
        with self.h.OptimizerDenial(self.t) as denial,self.h.RoleIO(self.t,self.root,allowed,[self.root/'FORBIDDEN_SELECT.pt'],output) as guard:
            proxy=build_proxy(self.t,self.root,self.case('gru','GF1',4)); self.h.allocate_adam_moments(self.t,proxy)
            self.assertFalse(any('fixed_H' in k for k in proxy.model.state_dict()))
            bank,counts=proxy.model.export_train_logits()
            self.assertEqual(bank['logits'].shape,(16,36,3)); self.assertEqual(bank['view_body_ids'].tolist(),sum(([i]*4 for i in range(4)),[]))
            self.assertEqual(set(bank),{'ids','logits','view_body_ids','engineering_random_untrained_unselected','scientific_parent_receipt'})
            self.assertEqual(counts['full_graph_body_forwards'],4); self.assertEqual(counts['prototype_bodies'],4); self.assertEqual(counts['maximum_simultaneously_live_full_graph_views'],4)
            self.assertEqual(counts['maximum_live_full_graph_logit_bytes'],4*72*3*4)
            result=measure_train_export(self.t,self.h,proxy,output); self.assertEqual(result['served_views'],16); self.assertEqual(result['parameters_sha256_before'],result['parameters_sha256_after']); self.assertEqual(result['moments_sha256_before'],result['moments_sha256_after'])
            self.assertFalse((output/'DISPOSABLE_RANDOM_TRAIN_LOGITS.pt').exists()); self.assertTrue(result['TRAIN_only_disposable_logit_serialization']['removed'])
            self.assertEqual(denial.denied_step_calls,0); self.assertEqual(denial.denied_functional_calls,0); self.assertEqual(guard.denied,0)


if __name__=='__main__': unittest.main()
