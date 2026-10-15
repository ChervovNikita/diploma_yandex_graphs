"""Source/JSON, native-source arithmetic and mock ownership checks only."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_pubmed_cheb_source_checks',HERE/'extension.py')
ext = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ext; spec.loader.exec_module(ext)


class Checks(unittest.TestCase):
    def test_source_and_frozen_protocol_pins(self):
        for row in json.loads((HERE/'SOURCES.json').read_text())['files']:
            actual = ext.pin(ext.PHASE/row['path_relative_to_phase'])
            self.assertEqual((actual['bytes'],actual['sha256']),(row['bytes'],row['sha256']))
        if (HERE/'MANIFEST.json').exists():
            ext.source_bindings(ext.PHASE.parent,ext.pin(HERE/'MANIFEST.json')['sha256'])

    def test_exact_descriptor_preset_and_native_runner_schema(self):
        sys.path.insert(0,str(ext.E/'runtime'))
        runner = ext.load(ext.E/'runtime/run_fit.py')
        base = json.loads((HERE/'BASE_CASE.json').read_text())
        protocol = json.loads((ext.E/'research/second_setting_protocol_v1/PLAN.json').read_text())
        selected = next(r for r in protocol['settings'] if r.get('selected_for_next_source_preparation'))
        self.assertEqual(selected['preset'],'polyformer-pubmed-cheb')
        self.assertEqual(base['recipe']['native_fixed_config'],selected['native_config'])
        jobs = [ext.descriptor(ext.PHASE.parent,s,v,'a'*64) for s in (101,202,303) for v in ext.METHODS.values()]
        self.assertEqual(len({j['id'] for j in jobs}),9)
        for j in jobs:
            runner.validate_job(j)
            self.assertEqual(j['recipe']['id'],'polyformer-pubmed-cheb')
            self.assertEqual(j['recipe']['official_split_bindings']['0']['source_split_seed'],1941488137)
            self.assertEqual(j['method_settings'],{'manifest_sha256':'a'*64})
            self.assertEqual(j['policy'],ext.policy())
        for seed,variant in ((1,'uniform'),(101,'other'),(True,'uniform')):
            with self.assertRaises(ValueError): ext.descriptor(ext.PHASE.parent,seed,variant,'a'*64)

    def test_parameter_formula_from_exact_source_config(self):
        p = json.loads((HERE/'BINDING.json').read_text())['parameter_accounting']
        # Native Linear has weight+bias; Q/K have no bias; both LayerNorms affine.
        D,H,C,K,heads,ffn,multi = 500,128,3,7,16,64,2
        linear = lambda a,b: a*b+b
        native = linear(D,H)+linear(H,H)+linear(H,C)+K*(linear(H,H*multi)+linear(H*multi,H))
        native += 2*H*H+heads*K+2*(2*H)+linear(H,ffn)+linear(ffn,H)
        private = 4*2*H*8
        self.assertEqual((native,private,native+private),(592435,8192,600627))
        self.assertEqual((p['native_parameters'],p['private_parameters'],p['total_parameters']),(native,private,native+private))
        self.assertEqual(p['removed_BE_coordinates'],4*(D+2*H+H+2*C))
        self.assertEqual((p['extra_FP32_parameter_bytes'],p['extra_dense_Adam_moment_bytes']),(32768,65536))
        self.assertEqual(p['weight_FP64_CPU_bytes'],4*11829*8)

    def test_optimizer_contract_all_options_and_restore_dict_identity(self):
        helper = ext.load(ext.LEGACY)
        a,b = object(),object()
        opt = types.SimpleNamespace(param_groups=[{'params':[a,b],'lr':.0005,'weight_decay':.001,'option':None}],
            defaults={'lr':.001,'eps':1e-8,'foreach':None})
        expected = helper.optimizer_contract(opt); old = tuple(opt.param_groups)
        helper.check_optimizer_contract(opt,expected,old)
        opt.param_groups = [dict(opt.param_groups[0])]
        with self.assertRaises(ValueError): helper.check_optimizer_contract(opt,expected,old)
        fresh = helper.check_optimizer_contract(opt,expected,old,after_parent_restore=True)
        helper.check_optimizer_contract(opt,expected,fresh)
        for kind in ('option','parameter','defaults'):
            groups,defaults = copy.copy(opt.param_groups),dict(opt.defaults)
            opt.param_groups = [dict(opt.param_groups[0])]
            if kind == 'option': opt.param_groups[0]['option'] = False
            elif kind == 'parameter': opt.param_groups[0]['params'] = [object(),b]
            else: opt.defaults['eps'] = 1e-7
            with self.assertRaises(ValueError): helper.check_optimizer_contract(opt,expected,fresh,after_parent_restore=True)
            opt.param_groups,opt.defaults = groups,defaults

    def test_exhaustive_residual_partition_rejects_alias_or_extra_owner(self):
        class Parameter:
            requires_grad = True
            def __init__(self, shape, size): self.shape,self.size = shape,size
            def numel(self): return self.size
            def count_nonzero(self): return types.SimpleNamespace(item=lambda:0)
        native = Parameter((1,),592435); down = Parameter((4,8,128),4096); up = Parameter((4,128,8),4096)
        model = types.SimpleNamespace(native=types.SimpleNamespace(named_parameters=lambda:[('native',native)]),
            input_residual=types.SimpleNamespace(down=down,up=up),private_parameters=lambda:(down,up),
            parameters=lambda:(native,down,up))
        opt = types.SimpleNamespace(param_groups=[{'params':[native,down,up]}])
        view = ext.partition(model,opt,initial=True)
        self.assertEqual(view.private_parameters(),(down,up))
        for pars in ((native,down), (native,down,up,up), (native,down,up,Parameter((1,),1))):
            opt.param_groups[0]['params'] = pars
            with self.assertRaises(ValueError): ext.partition(model,opt)

    def test_uniform_is_the_exact_same_F_object_and_weighted_uses_native_mean(self):
        class Logits:
            shape = (19717,3)
            def __getitem__(self,key): return 'TRAIN-only'
        sentinel = types.SimpleNamespace(new_zeros=lambda shape:'zero')
        stack = lambda values: types.SimpleNamespace(mean=lambda:sentinel)
        CE = lambda values,targets,**kwargs: (values,targets,kwargs)
        calls = []
        def weighted(t,records,weights,**kwargs):
            calls.append((weights,kwargs)); return 'weighted'
        session = types.SimpleNamespace(torch=types.SimpleNamespace(stack=stack,
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=CE))),targets='TRAIN-labels',
            train_ids='TRAIN-IDs',device='cpu',rules=types.SimpleNamespace(weighted_native_own_loss=weighted))
        routes = [types.SimpleNamespace(raw_logits=Logits()) for _ in range(4)]
        weights = types.SimpleNamespace(to=lambda device:'fixed-weights')
        F,W,_ = ext.terms(session,routes,weights,uniform=True)
        self.assertIs(F,W); self.assertIs(F,sentinel); self.assertEqual(calls,[])
        F,W,_ = ext.terms(session,routes,weights,uniform=False)
        self.assertIs(F,sentinel); self.assertEqual(W,'weighted'); self.assertEqual(calls[0][0],'fixed-weights')
        self.assertEqual(calls[0][1]['native_regularizer'],'zero')

    def test_degree_and_exact_fractional_tie_and_permutation_primitives(self):
        table = ext.load(ext.E/'structural_class_responsibility_v1/table.py')
        self.assertEqual(table.native_incoming_counts(3,[0,0,1,2],[1,1,1,2],[0,1,2]),(0,3,1))
        ids = list(range(12)); labels = [0]*6+[1]*6; metrics = [1,1,2,3,4,4]*2
        a = table.make_table(ids,labels,metrics,role='TRAIN')
        b = table.make_table(ids,labels,metrics,role='TRAIN',permutation_seed=101)
        self.assertEqual(a.weights[0][0],a.weights[0][1])
        for label in (0,1):
            positions = [i for i,v in enumerate(labels) if v == label]
            for m in range(4):
                self.assertEqual(sum(a.weights[m][i] for i in positions),len(positions))
                self.assertEqual(sorted(a.weights[m][i] for i in positions),sorted(b.weights[m][i] for i in positions))
        self.assertNotEqual(a.digest(),b.digest())

    def test_stdlib_import_no_tensor_execution_or_launcher(self):
        for filename in ('extension.py','qualify.py','check_source.py'):
            tree = ast.parse((HERE/filename).read_text())
            top = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
            self.assertFalse(any(isinstance(n,ast.Import) and any(a.name.startswith(('torch','numpy')) for a in n.names) for n in top))
        text = (HERE/'extension.py').read_text()
        self.assertIn('if uniform: return F, F, losses',text)
        self.assertIn('self._routes(True)',text)
        self.assertIn('step_private_responsibility',text)
        self.assertNotIn('torch.optim.',text)
        self.assertNotIn('test_mask',text)
        self.assertNotIn('lin2.register_',text)
        self.assertFalse((HERE/'run.py').exists())

    def test_completed_provider_support_retained(self):
        provider = json.loads((HERE/'BINDING.json').read_text())['provider']
        self.assertEqual(provider['TRAIN_class_counts'],[3943]*3)
        self.assertEqual(provider['VALID_class_counts'],[72,1919,1952])
        self.assertEqual(provider['source_split_seed'],1941488137)
        self.assertLess(72/3943,.02)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    report = {'schema':'pubmed-cheb-composition-source-checks-v1','status':'passed' if result.wasSuccessful() else 'failed',
        'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'local_model_dataset_tensor_execution':False,'remote_or_Git_operations':0,'new_native_forwards':0,
        'new_optimizer_updates':0,'numerical_qualification_performed':False,'TEST_accessed':False}
    (HERE/'SOURCE_CHECKS.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
