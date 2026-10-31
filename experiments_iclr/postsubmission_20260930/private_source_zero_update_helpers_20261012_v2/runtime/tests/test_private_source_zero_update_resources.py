"""Pure source callback interception and equality; only synthetic CPU inputs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from qa import private_source_zero_update_resources as resource
from qa import practical_zero_update_resources as h
import test_private_source_packets as source_tests
NUMERIC=source_tests.NUMERIC


class PrivateResourceMetadataTests(unittest.TestCase):
    def test_typed_laws_charge_replay_VJP_and_each_decoder_or_encoder_owner(self):
        cmcl={'architecture':'private_cmcl','cmcl_condition':'private_cmcl','source_optimizer_count':1}
        core={'architecture':'masked_core','core_condition':'untied4_shared_decoder_core','source_optimizer_count':5}
        self.assertEqual(resource.operator_law(cmcl),{'native_forwards':8,'autograd_grad_VJP_calls':8,'Tensor_backward_calls':0,'intercepted_optimizer_attempts':1})
        self.assertEqual(resource.operator_law(core),{'native_forwards':8,'autograd_grad_VJP_calls':0,'Tensor_backward_calls':4,'intercepted_optimizer_attempts':5})
        core['source_optimizer_count']=1
        with self.assertRaisesRegex(ValueError,'owner'):resource.operator_law(core)


@unittest.skipUnless(NUMERIC,'Allocation interpreter required for synthetic source CPU checks')
class PrivateResourceSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.fixture=source_tests.PrivateSourceSyntheticTests('test_genuine_separate_controls_and_actual_boundary');self.fixture.setUp();self.t=self.fixture.t
    def tearDown(self):self.fixture.tearDown()

    def test_intrinsic_parameter_moment_view_and_data_writes_denied_grad_fields_live(self):
        t=self.t;p=t.nn.Parameter(t.ones(3));moment=t.zeros(3);p.grad=t.ones(3)
        with resource.TensorCommitDenial(t,[p],[moment]) as guard:
            for operation in (lambda:p.data.add_(1),lambda:p.detach().copy_(t.zeros(3)),lambda:moment[0].fill_(1),lambda:setattr(p,'data',t.zeros(3))):
                with self.assertRaises(resource.ProtectedTensorCommit):operation()
            p.grad.div_(4);self.assertTrue(t.equal(p.grad,t.full((3,),.25)))
            self.assertEqual(guard.denied,4)
        self.assertTrue(t.equal(p,t.ones(3)));self.assertTrue(t.equal(moment,t.zeros(3)))

    def test_every_source_gradient_law_and_multi_owner_reaches_denied_commit_same_state(self):
        cases=[('cmcl',name) for name in resource.CMCL]+[('adam','normalized')]+[('core',name) for name in resource.CORE]
        for family,name in cases:
            output=self.fixture.root/('zero-'+family+'-'+name);output.mkdir()
            with h.OptimizerDenial(self.t) as denied:
                adapter=self.fixture.make(family,name)
                value=resource.measure_source(self.t,h,[adapter],denied,output,warm_iterations=1)
            expected=resource.operator_law(adapter.provenance['config'])
            self.assertEqual(value['actual_source_operator_counts_by_cycle'],[expected,expected])
            self.assertEqual(value['denied_optimizer_step_calls'],2*adapter.provenance['config']['source_optimizer_count'])
            self.assertEqual(value['actual_optimizer_updates'],0);self.assertTrue(value['zero_update_equality_proof_complete'])
            for field in ('parameters','moments','buffers','immutable_GRAPH_TRAIN_stub','source_owned_state'):
                self.assertEqual(value[field+'_sha256_before'],value[field+'_sha256_after'])
            self.assertNotIn('objective',json.dumps(value));self.assertNotIn('accuracy',value)
            self.assertFalse((output/'DISPOSABLE_RANDOM_ENGINEERING_STATE.pt').exists())

    def test_exact_factory_clone_denies_original_SELECT_and_resets_CMCL_custody(self):
        f=self.fixture;t=self.t;row=f.row('cmcl','private_cmcl');row['split']=0
        row['adapter']='runtime/adapters/private_cmcl.py:make_adapter'
        package=Path(__file__).resolve().parents[1]
        for name in ('run_study.py','registry.py','spec.py','engine.py','metrics.py','adapters/__init__.py','adapters/private_cmcl.py','adapters/private_polyformer_source.py','adapters/polyformer.py','adapters/classifier.py','adapters/node_losses.py'):
            path=f.root/'runtime'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((package/name).read_bytes())
        study={**f.study,'stage':'development','runs':[row],
            'data':{**f.study['data'],'roles':{'select':{'0':f.inputs['select']},'train':{'0':f.inputs['train']}}}}
        case={'run':row,'runtime_package':'runtime','prototype_bodies':1}
        # The real loader is used against the local synthetic runtime path.
        before=copy.deepcopy(study);protected=h.receipt(f.root,f.root/'select.pt');output=f.root/'clone-resource';output.mkdir()
        with h.OptimizerDenial(t) as denied,h.RoleIO(t,f.root,[f.root/'graph.pt',f.root/'train.pt'],[f.root/'select.pt'],output) as guard:
            adapters,stub=resource.engineering_adapters(t,h,f.root,case,study,output,guard,'cpu')
            self.assertEqual(adapters[0].source_session.identity['VALID_custody_sha256'],h.receipt(f.root,stub)['sha256'])
            result=resource.measure_source(t,h,adapters,denied,output,warm_iterations=1)
            self.assertEqual(guard.denied,0);self.assertEqual(result['scores_emitted'],0)
            with self.assertRaises(h.ForbiddenRoleAccess):(f.root/'select.pt').read_bytes()
        self.assertEqual(study,before);self.assertEqual(h.receipt(f.root,f.root/'select.pt'),protected)

    def test_four_actual_M1_factories_have_four_owners_and_no_selected_or_paid_payload(self):
        output=self.fixture.root/'four-owners';output.mkdir()
        with h.OptimizerDenial(self.t) as denied:
            adapters=[self.fixture.make('control','factorized_m1',body) for body in range(4)]
            value=resource.measure_source(self.t,h,adapters,denied,output,warm_iterations=1)
        self.assertEqual(value['optimizer_objects'],4);self.assertEqual(value['denied_optimizer_step_calls'],8)
        self.assertEqual([v['denied_step_calls'] for v in value['owners']],[2]*4)
        self.assertTrue(value['disposable_serialization']['removed']);self.assertTrue(value['disposable_serialization']['never_selected_checkpoint'])


if __name__=='__main__':unittest.main()
