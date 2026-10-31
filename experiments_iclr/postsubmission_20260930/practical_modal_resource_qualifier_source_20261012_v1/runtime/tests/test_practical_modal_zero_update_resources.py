"""Synthetic CPU/resource custody checks only; never benchmark role data."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from qa import practical_zero_update_resources as h
from qa.practical_modal_zero_update_resources import (engineering_clone, measure_epochs,
    synthetic_link_select, synthetic_molecular_select, terminal_failure)


class ModalMetadataTests(unittest.TestCase):
    def test_wrapped_OOM_is_deferred_and_incomplete_update_proof_is_unknown(self):
        inner=type('OutOfMemoryError',(RuntimeError,),{})('synthetic memory refusal')
        outer=RuntimeError('synthetic adapter envelope');outer.__cause__=inner
        result=terminal_failure(h,outer,{'pid':123,'start_ticks':456},time.monotonic()-.01)
        self.assertEqual(result['status'],'RESOURCE_DEFERRED')
        self.assertEqual(result['failure_chain_types'],['RuntimeError','OutOfMemoryError'])
        self.assertIsNone(result['actual_optimizer_update_count_verified'])
        self.assertFalse(result['zero_update_equality_proof_complete']);self.assertFalse(result['automatic_retry'])


class ModalSyntheticCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.output=self.root/'engineering';self.output.mkdir()

    def tearDown(self):self.tmp.cleanup()

    def test_unused_molecule_is_wholly_synthetic_valid_and_disjoint(self):
        from adapters.molecular import validate_pack
        pack=validate_pack(self.t,synthetic_molecular_select(self.t),1,'complete_binary')
        self.assertEqual(pack['ids'].tolist(),[2**62]);self.assertEqual(pack['y'].tolist(),[[0.]])
        self.assertEqual(pack['x'].shape,(1,9));self.assertEqual(pack['edge_index'].shape,(2,0))

    def test_link_interface_uses_only_public_nonTRAIN_pairs_and_source_schema(self):
        from adapters.link import validate_inputs, pair_keys, intersects
        t=self.t;positive=t.tensor([[0,1],[1,2],[0,1]],dtype=t.long)
        graph={'x':t.zeros((12,3)),'raw_edge_index':t.cat([positive,positive.flip(1)]).T}
        train={'positive':positive};select=synthetic_link_select(t,graph,train,3)
        config={'support':{'record_duplicate_policy':'native_record_multiset','select_overlap':'reject'},'metric':{'hits_k':3}}
        result=validate_inputs(t,graph,train,select,config)
        self.assertTrue(result['train_only_support']);self.assertEqual(result['select_negative'],3)
        self.assertFalse(intersects(t,pair_keys(t,positive,12),pair_keys(t,t.cat(list(select.values())),12)))

    def test_clone_does_not_read_or_mutate_original_molecular_SELECT(self):
        t=self.t;train=self.root/'TRAIN.pt';select=self.root/'SELECT.pt'
        t.save(synthetic_molecular_select(t),train);t.save({'protected':'synthetic-test-only'},select)
        inputs={k:h.receipt(self.root,p) for k,p in [('train',train),('select',select)]}
        run={'kind':'model_fit','run_id':'synthetic','config':{'molecular_inputs':inputs}}
        study={'stage':'development','data':{'roles':inputs},'runs':[run]};before=copy.deepcopy(study)
        protected_hash=h.receipt(self.root,select)
        with h.RoleIO(t,self.root,[train],[select],self.output) as guard:
            clone,resource_run,stub=engineering_clone(t,h,self.root,study,{'family':'molecular','declared_source_run':run},self.output)
            guard.allowed.add(stub.resolve());self.assertEqual(t.load(stub,weights_only=True)['ids'].tolist(),[2**62])
            self.assertEqual(guard.denied,0);self.assertEqual(clone['stage'],'engineering')
            self.assertEqual(resource_run['config']['molecular_inputs']['select'],clone['data']['roles']['select'])
            with self.assertRaises(h.ForbiddenRoleAccess):select.read_bytes()
        self.assertEqual(study,before);self.assertEqual(h.receipt(self.root,select),protected_hash)

    def test_full_epoch_attempts_never_update_or_persist_returned_losses(self):
        t=self.t
        class Adapter:
            def __init__(self):
                self.model=t.nn.Linear(3,1);self.optimizer=t.optim.Adam(self.model.parameters(),lr=.001)
            def train_step(self,epoch):
                for batch in range(2):
                    self.optimizer.zero_grad(set_to_none=True);self.model(t.ones((4,3))).square().mean().backward();self.optimizer.step()
                return {'objective':123.456,'ce':123.456,'operation_counts':{'optimizer_updates':2,'backward_calls':2,'supervised_examples':8}}
            def validate(self,*a):raise AssertionError('Should be replaced before any observation')
        with h.OptimizerDenial(t) as denied:
            adapter=Adapter();value=measure_epochs(t,h,adapter,denied,self.output,warm_iterations=2)
        self.assertEqual(value['denied_optimizer_step_calls'],6);self.assertEqual(value['actual_optimizer_updates'],0)
        self.assertEqual(value['parameters_sha256_before'],value['parameters_sha256_after'])
        self.assertEqual(value['moments_sha256_before'],value['moments_sha256_after'])
        self.assertEqual(value['actual_source_TRAIN_operation_counts_without_nominal_updates'],[{'backward_calls':2,'supervised_examples':8}]*3)
        self.assertNotIn('objective',value);self.assertNotIn('123.456',json.dumps(value));self.assertEqual(value['scores_emitted'],0)
        with self.assertRaises(h.ForbiddenRoleAccess):adapter.validate(1)


if __name__=='__main__':unittest.main()
