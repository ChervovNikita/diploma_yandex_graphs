"""Focused stdlib accounting/lineage/controller checks; numerical proof inherited."""
import copy
import gzip
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adapters import polyformer_recompute as adapter_code
from compile_coauthor_recompute_successor import transform_study
from engine import _counts
from run_study import resolve_selected_config,load_factory


def report(logical,replay,*,failure=False):
    return {'counts':{'logical_checkpoint_route_calls':logical,'completed_logical_route_forwards':logical,
      'recomputed_route_forward_attempts':replay+int(failure),'completed_recomputed_route_forwards':replay,
      'bypassed_route_calls':0,'failed_route_invocations':int(failure)},
      'records':[{'RNG_snapshot_bytes':5072} for _ in range(logical)],
      'host_route_call_seconds':{'logical_route_calls':.1,'recomputed_route_calls':.2,'bypassed_route_calls':0.},
      'token_bank_copies_created':0}


class SourceStandIn:
    """No tensor or optimizer; isolates integration custody from inherited source algebra."""
    def __init__(self,logical=8,error=None):
        self.t=None;self.model=object();self.logical=logical;self.error=error;self.calls=[]
    def train_step(self,step):
        self.calls.append(step)
        if self.error is not None:raise self.error
        return {'objective':1.,'ce':.9,'regularization':.1,
          'operation_counts':{'optimizer_updates':1,'model_forwards':self.logical,'route_predictions':self.logical,
                              'backward_calls':1,'supervised_examples':80},'source_component':{'retained':True}}
    def validate(self,step):return {'eager_source_validation':step}
    def capture_state(self):return {'source_state':True}
    def restore_state(self,state):self.restored=state


class FocusedIntegrationTests(unittest.TestCase):
    def handle_factory(self,work):
        handles=[]
        class Handle:
            def __init__(self,*unused):self.closed=False;handles.append(self)
            def __enter__(self):return self
            def __exit__(self,*unused):self.closed=True
            def report(self):return copy.deepcopy(work)
        return Handle,handles

    def test_native_counts_preserved_replays_charged_and_eval_state_eager(self):
        option=adapter_code.execution_option('polyformer_shared4_paired_class_full')
        source=SourceStandIn();expected=source.train_step(1);source.calls=[]
        factory,handles=self.handle_factory(report(8,8));costs=adapter_code.PaidReplayCosts({'fixture':1})
        result=adapter_code.install(source,option,costs,checkpoint_factory=factory).train_step(1)
        for key in ('objective','ce','regularization','source_component'):self.assertEqual(result[key],expected[key])
        for key,value in expected['operation_counts'].items():self.assertEqual(result['operation_counts'][key],value)
        self.assertEqual(_counts(result['operation_counts'],positive_updates=True)['route_recompute_full_replay_forwards'],8)
        self.assertEqual(result['operation_counts']['TRAIN_route_forwards_including_replay'],16)
        self.assertEqual(source.validate(2),{'eager_source_validation':2})
        self.assertEqual(source.capture_state(),{'source_state':True});source.restore_state({'restore':1})
        self.assertEqual(costs.value['counts']['completed_recomputed_route_forwards'],8)
        self.assertEqual(source.calls,[1]);self.assertTrue(handles[0].closed)

    def test_atomic_compact_paid_sidecar_continues_across_restore_and_reconstruction(self):
        identity={'fixture':'paid_work'}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fit/route_recompute_paid_costs.json';costs=adapter_code.PaidReplayCosts(identity,path)
            self.assertFalse(path.exists())  # Engine's fresh-directory admission stays intact.
            source=SourceStandIn(4);factory,_=self.handle_factory(report(4,4))
            adapter_code.install(source,adapter_code.execution_option('polyformer_shared4_gncl_supcon'),costs,checkpoint_factory=factory)
            source.train_step(1);selected=source.capture_state();source.train_step(2);source.restore_state(selected)
            self.assertEqual(costs.value['train_units_attempted'],2)
            resumed=adapter_code.PaidReplayCosts(identity,path);resumed.record(2,report(4,4),source_completed=True,error=None)
            value=json.loads(path.read_text());self.assertEqual(value['train_units_attempted'],3)
            self.assertEqual(value['counts']['completed_recomputed_route_forwards'],12)
            self.assertNotIn('records',value['last_unit']['work']);self.assertEqual(list(path.parent.glob('*.tmp')),[])
            self.assertLess(path.stat().st_size,4000)
            with self.assertRaisesRegex(ValueError,'another execution'):adapter_code.PaidReplayCosts({'other':1},path)

    def test_original_source_exception_keeps_optimizer_partial_unknown_and_partial_replay_known(self):
        error=RuntimeError('synthetic interrupted source callback');source=SourceStandIn(4,error)
        factory,handles=self.handle_factory(report(4,2,failure=True));costs=adapter_code.PaidReplayCosts({'fixture':2})
        adapter_code.install(source,adapter_code.execution_option('polyformer_shared4_gncl_supcon'),costs,checkpoint_factory=factory)
        with self.assertRaises(RuntimeError) as caught:source.train_step(3)
        self.assertIs(caught.exception,error);self.assertFalse(hasattr(error,'operation_counts'))
        self.assertEqual(costs.value['counts']['completed_recomputed_route_forwards'],2)
        self.assertEqual(costs.value['counts']['recomputed_route_forward_attempts'],3)
        self.assertEqual(costs.value['train_units_failed'],1);self.assertTrue(handles[0].closed)

    def test_disabled_option_delegates_and_enabled_factory_requires_controller_context(self):
        self.assertIn('execution_context',inspect.signature(adapter_code.make_adapter).parameters)
        run={'method':'polyformer_shared4_gncl_supcon','config':{'execution_engineering':adapter_code.execution_option('polyformer_shared4_gncl_supcon',enabled=False)}}
        with patch.object(adapter_code,'source_make_adapter',return_value='source_adapter') as source:
            self.assertEqual(adapter_code.make_adapter(torch=None,project_root=Path.cwd(),study={},run=run,device='cpu'),'source_adapter')
            self.assertEqual(source.call_count,1)
        run['config'].update(execution_engineering=adapter_code.execution_option(run['method']),members=4,polyformer_mode='polyformer_internal_be')
        with patch.object(adapter_code,'source_make_adapter') as source:
            with self.assertRaisesRegex(ValueError,'execution context'):
                    adapter_code.make_adapter(torch=None,project_root=Path.cwd(),study={'dataset':'coauthor-physics'},run=run,device='cpu',execution_context={})
            source.assert_not_called()

    def test_context_free_export_reconstruction_is_eager_and_cannot_start_TRAIN(self):
        source=SourceStandIn(4)
        run={'method':'polyformer_shared4_gncl_supcon','config':{'execution_engineering':adapter_code.execution_option('polyformer_shared4_gncl_supcon'),
              'members':4,'polyformer_mode':'polyformer_internal_be'}}
        with patch.object(adapter_code,'source_make_adapter',return_value=source):
            restored=adapter_code.make_adapter(torch=None,project_root=Path.cwd(),study={'dataset':'coauthor-physics'},run=run,device='cpu')
        self.assertIs(restored,source);self.assertEqual(restored.capture_state(),{'source_state':True})
        restored.restore_state({'selected':1});self.assertEqual(restored.validate(3),{'eager_source_validation':3})
        self.assertEqual(restored.restored,{'selected':1})
        with self.assertRaisesRegex(RuntimeError,'execution context'):restored.train_step(1)
        self.assertEqual(source.calls,[]);self.assertFalse(hasattr(source,'paid_route_recompute_costs'))

    def test_all_records_original_baselines_and_selected_parameters_preserved(self):
        packet=Path(__file__).resolve().parents[2];prior=packet/'lineage/prior_coauthor_packet'
        plan=json.loads((prior/'matrix/PLAN.json').read_text());prefix=plan['immutable_source_bundle']['package']
        enabled=0;extra=0;original_baselines=0
        for path in sorted((prior/'matrix/studies').glob('*.gz')):
            original=json.loads(gzip.decompress(path.read_bytes()));saved=copy.deepcopy(original)
            value,work=transform_study(original,old_prefix=prefix,new_prefix='new/runtime',runtime_files=[],lineage={'path':'fixture','bytes':1,'sha256':'0'*64})
            self.assertEqual(original,saved);self.assertEqual(len(value['runs']),8424)
            self.assertEqual(value['data'],original['data']);self.assertEqual(value['comparisons'],original['comparisons'])
            enabled+=len(work['enabled_fits']);extra+=work['additional_planned_replay_forwards']
            old={r['run_id']:r for r in original['runs']};new={r['run_id']:r for r in value['runs']}
            for run in value['runs']:
                if run['kind']=='model_fit' and run['config'].get('is_original_gnnm'):
                    original_baselines+=1;self.assertEqual(run['config'],old[run['run_id']]['config'])
                if 'execution_engineering' in run['config'] and run['config']['stage']=='comparative':
                    selector=new[run['config']['selected_config_from']];choice=selector['config']['configurations'][-1]
                    registry=SimpleNamespace(run=lambda *unused:{'state':'complete','contract':selector,
                      'result':{'selected_configuration':choice,'ranking_source':'fixture','choice_hash':'0'*64}})
                    resolved=resolve_selected_config(value,run,registry)
                    self.assertEqual(resolved['config']['execution_engineering'],run['config']['execution_engineering'])
                    for key,val in choice['parameters'].items():self.assertEqual(resolved['config'][key],val)
        self.assertEqual((enabled,extra,original_baselines),(96,1152000,96))

    def test_dynamic_factory_uses_successor_package_namespace(self):
        packet=Path(__file__).resolve().parents[2];root=packet.parent
        factory=load_factory(root,str((packet/'runtime/adapters/polyformer_recompute.py').relative_to(root))+':make_adapter')
        self.assertIn('gnnm_packet_',factory.__module__)
        self.assertIn('execution_context',inspect.signature(factory).parameters)


if __name__=='__main__':unittest.main()
