"""Source CPU fixtures for boundary maps, sparse bias law, costs and restore."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapters.link_boundary import BOUNDARY_SOURCE,CONTRACT,be_class,make_adapter,validate_config
from compile_link_boundary_matrix import domain,make_study
import test_link as source_tests
from test_link import config,run,study,LINK_RUNTIME,torch

PROJECT=Path(os.environ.get('GNNM_REFERENCE_PROJECT',Path(__file__).resolve().parents[2]))
BE_PATH='common_wrapper_native_source_20261010_v1/models.py'


def enrich(row,source,kind='original_boundary',members=4):
    c=row['config'];seed=c['initialization_seed']
    c.update(boundary={'kind':kind,'source':source},members=members,source_optimizer_count=1,
        train_minibatches_per_epoch=2,is_original_gnnm=members==4,shuffle_seed=seed,
        dropout_seed=seed+3000007,graphmodelsource=copy.deepcopy(c['link_source']))
    return row


class MetadataTests(unittest.TestCase):
    def test_pinned_original_class_and_finite126_roster(self):
        raw=(PROJECT/BE_PATH).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),BOUNDARY_SOURCE['sha256'])
        nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='BEBlock']
        self.assertEqual(len(nodes),1);self.assertEqual(len(domain('NCN')),6);self.assertEqual(len(domain('NCNC')),6)
        template={'data':{'task':'link_prediction','preprocessing':'synthetic fixture','prior_exposure':'development','roles':{'train':config()['link_inputs']['train'],'select':config()['link_inputs']['select'],
            'cal':{'absent':True},'evaluate':{'fixture':True}},'evidence':{'endpoint_inputs':{'evaluate_queries':{'path':'query','bytes':1,'sha256':'a'*64},
            'evaluate_targets':{'path':'target','bytes':1,'sha256':'b'*64}}}},'runs':[run()]}
        source={'git_commit':'fixture','files':[{'path':'source','sha256':'c'*64}]}
        profile={'reservation_mib':32768,'wall_seconds':250000000,'estimated_seconds_per_epoch':1800,
            'estimated_selection_seconds':60,'estimated_export_seconds':7200,'evidence':'provisional'}
        result,counts,banks=make_study(template,{'path':BE_PATH,**BOUNDARY_SOURCE},source,profile,'runtime')
        self.assertEqual((len(result['runs']),counts['model_fits'],counts['search_model_fits'],counts['comparative_model_fits']),(192,126,72,54))
        self.assertEqual((counts['registered_optimizer_updates'],counts['registered_member_backward_calls']),(214200,367200))
        self.assertEqual(sum(b['genuine_I4'] for b in banks),12);self.assertEqual(sum(b['body0_reused'] for b in banks),12)
        from compile_master_plan import reference_matches,bank_roster
        self.assertTrue(all(r['matched'] for r in reference_matches(result)))
        self.assertEqual(sum(b['genuine_I4'] for b in bank_roster(result)),12)

    def test_guard_precedes_metadata_and_test_fit_inputs_rejected(self):
        import compile_link_boundary_matrix as compiler
        argv=['program']
        for name in ('project-root','template-study','boundary-source','runtime-package','qa-receipt','resources','output'):
            argv+=['--'+name,'missing']
        with patch('sys.argv',argv),patch.object(compiler,'verify_allocation',side_effect=RuntimeError('guard')),patch.object(Path,'read_text') as read:
            with self.assertRaisesRegex(RuntimeError,'guard'):compiler.main()
            read.assert_not_called()
        row=enrich(run(),{'path':BE_PATH,**BOUNDARY_SOURCE});row['config']['link_inputs']['test']={'path':'test','bytes':1,'sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'exactly GRAPH'):validate_config(row['config'],row)


@unittest.skipUnless(LINK_RUNTIME,'CPU link dependencies unavailable')
class SourceCPUChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def fixture(self,root,predictor='NCN',kind='original_boundary',members=4,arm='native_single'):
        row=source_tests.LinkCPUFixtureTests().fixture(root,predictor,arm)
        raw=(PROJECT/BE_PATH).read_bytes();(root/'boundary.py').write_bytes(raw)
        return enrich(row,{'path':'boundary.py','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},kind,members)

    def test_exact_BE_initialization_and_native_afterpropagation_affine(self):
        from adapters.link import make_adapter as native_factory
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row=self.fixture(root)
            reference=copy.deepcopy(row);reference['config']={k:v for k,v in reference['config'].items() if k not in __import__('adapters.link_boundary',fromlist=['EXTRA']).EXTRA}
            plain=native_factory(torch=torch,project_root=root,study=study(reference),run=reference,device='cpu')
            a=make_adapter(torch=torch,project_root=root,study=study(row),run=row,device='cpu')
            first=a.model.encoder.convs[0];last=a.model.decoder.lin[-1].block
            self.assertIsNone(first.native.bias)
            b=first.native.lin.block
            self.assertTrue(torch.all(b.R.abs()==1));self.assertTrue(torch.all(b.S==1));self.assertTrue(torch.all(b.B==0))
            self.assertTrue(torch.all(last.R==1));self.assertTrue(torch.all(last.S==1));self.assertTrue(torch.all(last.B==0))
            self.assertEqual(sum(isinstance(m,torch.nn.Linear) for m in a.model.decoder.modules()),sum(isinstance(m,torch.nn.Linear) for m in plain.model.decoder.modules()))
            with torch.no_grad():
                b.W.weight.copy_(plain.model.encoder.convs[0].lin.weight);b.R.fill_(1);b.S.fill_(1);b.B.copy_(plain.model.encoder.convs[0].bias.expand(4,-1))
                last.W.weight.copy_(plain.model.decoder.lin[-1].weight);last.B.copy_(plain.model.decoder.lin[-1].bias.expand(4,-1))
            a.model.eval();plain.model.eval()
            for key in ('normalize','add_self_loops','cached','improved','flow'):
                self.assertEqual(getattr(first.native,key),getattr(plain.model.encoder.convs[0],key))
            with a.member_scope(0,training=False):
                h=a.model.encoder(a.graph['x'],a.full_graph)
                expected=plain.model.encoder(plain.graph['x'],plain.full_graph)
                self.assertTrue(torch.equal(h,expected))
                pairs=a.select['positive'].T
                self.assertTrue(torch.equal(a.model.decoder(h,a.full_graph,pairs),plain.model.decoder(expected,plain.full_graph,pairs)))
            # A nonzero B distinguishes placement after normalized propagation.
            with torch.no_grad():b.B[0].copy_(torch.arange(b.B.shape[1],dtype=torch.float32))
            with a.member_scope(0,training=False):
                actual=first(a.graph['x'],a.full_graph)
                expected=plain.model.encoder.convs[0](plain.graph['x'],plain.full_graph)+b.B[0]
                self.assertTrue(torch.equal(actual,expected))

    def test_sequential_BCE_gradient_streams_costs_and_selected_restore(self):
        from engine import run_trajectory,restore_checkpoint
        for predictor in ('NCN','NCNC'):
            with self.subTest(predictor=predictor),tempfile.TemporaryDirectory() as d:
                root=Path(d);row=self.fixture(root,predictor)
                a=make_adapter(torch=torch,project_root=root,study=study(row),run=row,device='cpu')
                contract={'fit_id':row['run_id'],'budget_steps':1,'budget_unit':'epochs','validation_interval':1,
                    'validate_initial':False,'prefixes':[],'inherited_fit_ids':[],'adapter_spec':{}}
                result=run_trajectory(torch,a,contract,root/'qa')
                self.assertEqual((result['operation_counts']['optimizer_updates'],result['operation_counts']['backward_calls']),(2,8))
                self.assertEqual(result['operation_counts']['native_encoder_calls'],12) #8 TRAIN+4 SELECT
                self.assertEqual(result['operation_counts']['dropped_positive_tail'],1)
                # The engine clears gradients when restoring its selected state.
                a.train_step(2)
                self.assertTrue(all(p.grad is not None for n,p in a.model.named_parameters() if '.ptlin.' not in n))
                first=a.model.encoder.convs[0].native.lin.block
                self.assertTrue(all(first.R.grad[m].abs().sum()>0 for m in range(4)))
                restore_checkpoint(torch,root/'qa/selected.pt',a,expected_fit_id=row['run_id'])
                state=copy.deepcopy(a.capture_state());weights=copy.deepcopy(a.model.state_dict());opt=copy.deepcopy(a.optimizer.state_dict())
                nxt=a.train_step(2)
                a.model.load_state_dict(weights);a.optimizer.load_state_dict(opt);a.restore_state(state)
                self.assertEqual(nxt,a.train_step(2))
                scores=a.predict_queries(a.select['positive'],a.select['negative'])
                outputs=[]
                with torch.no_grad():
                    for m in range(4):
                        with a.member_scope(m,training=False):
                            h=a.model.encoder(a.graph['x'],a.full_graph)
                            chunks=[a.model.decoder(h,a.full_graph,a.select['positive'][ids].T).flatten()
                                for ids in a.utils.PermIterator(a.device,len(a.select['positive']),a.config['training']['eval_batch_size'],False)]
                            outputs.append(torch.cat(chunks))
                self.assertTrue(torch.equal(scores[0],torch.stack(outputs).mean(0)))

    def test_genuine_I4_disjoint_states_and_partial_failure_receipts(self):
        from engine import inspect_independence,AdapterStepError
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);template=self.fixture(root,members=1,arm='native_independent4_body');rows=[]
            for body in range(4):
                row=copy.deepcopy(template);row['body']=body;row['run_id']='body-'+str(body)
                c=row['config'];c['initialization_seed']+=1000003*body;c['shuffle_seed']=c['initialization_seed'];c['dropout_seed']=c['initialization_seed']+3000007;rows.append(row)
            declaration=study(rows[0],rows)
            adapters=[make_adapter(torch=torch,project_root=root,study=declaration,run=r,device='cpu') for r in rows]
            self.assertEqual(inspect_independence(adapters)['independent_adapters'],4)
            second=adapters[1].member_rng[0]['torch_cpu'].clone();adapters[0].train_step(1)
            self.assertTrue(torch.equal(second,adapters[1].member_rng[0]['torch_cpu']))
            def fail(*args,**kwargs):raise RuntimeError('synthetic decoder failure')
            adapters[2].model.decoder.multidomainforward=fail
            with self.assertRaises(AdapterStepError) as caught:adapters[2].train_step(1)
            self.assertEqual(caught.exception.operation_counts['optimizer_updates'],0)
            self.assertEqual(caught.exception.operation_counts['native_encoder_calls'],1)


if __name__=='__main__':unittest.main()
