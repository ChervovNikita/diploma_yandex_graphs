"""Synthetic CPU and metadata checks only; no benchmark role data or GPU work."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from qa.practical_zero_update_resources import (ForbiddenOptimizerKernel, ForbiddenRoleAccess,
    OptimizerDenial, RoleIO, allocate_adam_moments, digest, engineering_clone,
    measure_adapter, prepare_manifest, receipt, representative_candidates, tensor_digest,
    allocator_cap, configure_cuda_allocator, ResourceDeferred, ResourceTimeout, terminal_failure)


class ResourceMetadataTests(unittest.TestCase):
    def test_allocator_cap_preserves_fresh_headroom_and_forty_percent_ceiling(self):
        gib=1024**3
        for free,expected in [(75*gib,32*gib),(20*gib,16*gib)]:
            cap=allocator_cap(free,80*gib)
            self.assertEqual(cap['allocator_cap_bytes'],expected)
            self.assertLessEqual(cap['allocator_fraction'],.4)
            self.assertLessEqual(cap['allocator_cap_bytes'],free-4*gib)
        with self.assertRaises(ResourceDeferred):allocator_cap(4*gib,80*gib)

    def test_failure_keeps_owner_elapsed_and_does_not_claim_completed_zero_update_audit(self):
        owner={'pid':123,'start_ticks':45,'pgid':123}
        value=terminal_failure(ResourceTimeout('synthetic deadline'),owner,time.monotonic()-.1)
        self.assertEqual(value['owner'],owner);self.assertEqual(value['status'],'RESOURCE_DEFERRED')
        self.assertGreater(value['elapsed_wall_seconds'],.05)
        self.assertFalse(value['automatic_retry']);self.assertFalse(value['zero_update_equality_proof_complete'])
        self.assertIsNone(value['actual_optimizer_update_count_verified'])

    def test_deferred_cap_keeps_external_usage_and_fresh_free_total_before_factory(self):
        gib = 1024**3
        cuda = SimpleNamespace(device_count=lambda: 1, set_device=mock.Mock(),
                               mem_get_info=lambda device: (4*gib, 80*gib),
                               set_per_process_memory_fraction=mock.Mock())
        observations = [{'external_used_bytes': 7*gib}, {'external_used_bytes': 8*gib}]
        with tempfile.TemporaryDirectory() as directory, mock.patch(
                'qa.practical_zero_update_resources.GPU_process_usage', side_effect=observations):
            output = Path(directory)
            with self.assertRaises(ResourceDeferred):
                configure_cuda_allocator(SimpleNamespace(cuda=cuda), output)
            saved = json.loads((output/'ALLOCATOR_CAP_AND_INITIAL_GPU_USAGE.json').read_text())
        self.assertEqual(saved['before_init'], observations[0])
        self.assertEqual(saved['after_init'], observations[1])
        self.assertEqual(saved['fresh_free_bytes'], 4*gib)
        self.assertEqual(saved['stage'], 'fresh_free_total_observed')
        cuda.set_per_process_memory_fraction.assert_not_called()

    def test_frontier_keeps_actual_source_anchor_and_maximum_shapes(self):
        values = []
        for width, dropout, anchor in [(128, .1, False), (128, .4, False), (256, .2, True)]:
            values.append({'config_id': str(width)+str(dropout), 'run': {'config': {'width': width, 'depth': 2, 'dropout': dropout}}, 'source_anchor': anchor})
        chosen = representative_candidates(values)
        self.assertEqual({(r['run']['config']['width'], r['run']['config']['dropout']) for r in chosen}, {(128, .4), (256, .2)})

    def test_manifest_uses_custody_counts_and_retains_unqualified_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            custody = {'schema': 'node-provider-role-custody-v1', 'dataset_id': 'synthetic-node',
                       'splits': [{'split_id': i, 'role_counts': {'train': n}} for i, n in [(0, 2), (1, 3), (2, 2)]]}
            p = root/'CUSTODY.json';p.write_text(json.dumps(custody));source = {'files': [receipt(root, p)]}
            searches = [{'run_id': 'search-'+str(i), 'kind': 'model_fit', 'adapter': 'source/node.py:make_adapter', 'method': 'ordinary_native_single',
                         'seed': 1, 'body': 0, 'split': 0,
                         'config': {'stage': 'search', 'width': width, 'depth': 2, 'dropout': drop,
                                    'node_inputs': {'graph': 'graph.pt', 'train': 'train0.pt', 'select': 'select0.pt'}}}
                        for i, (width, drop) in enumerate([(128, .1), (128, .4), (256, .2)])]
            selection = {'run_id': 'select-single', 'kind': 'selection', 'config': {'configurations':
                         [{'config_id': str(i), 'run_ids': [r['run_id']], 'source_anchor': i == 2} for i, r in enumerate(searches)]}}
            comparative = [{'run_id': 'compare'+str(i), 'kind': 'model_fit', 'body': 0, 'seed': 2, 'split': i,
                            'config': {'selected_config_from': 'select-single', 'node_inputs':
                                       {'graph': 'graph.pt', 'train': 'train'+str(i)+'.pt', 'select': 'select'+str(i)+'.pt'}}}
                           for i in range(3)]
            for r in comparative:r['adapter'] = 'source/node.py:make_adapter'
            study = {'study_id': 'synthetic-node', 'dataset': 'synthetic-node', 'data': {'task': 'node_classification'},
                     'source': source, 'runs': searches+[selection]+comparative}
            other = {'study_id': 'synthetic-link', 'data': {'task': 'link_prediction'}, 'runs': []}
            entries = []
            for index, s in enumerate([study, other]):
                p = root/(str(index)+'.json.gz');p.write_bytes(gzip.compress(json.dumps(s).encode()))
                entries.append({'path': p.name, 'spec_hash': digest(s)})
            plan = {'scientific_launch_enabled': False, 'studies': entries};p=root/'PLAN.json';p.write_text(json.dumps(plan))
            manifest = prepare_manifest(root, p, digest(plan))
            self.assertEqual(len(manifest['cases']), 2)
            self.assertTrue(all(c['representative_mask'] == 1 and c['actual_TRAIN_rows'] == 3 for c in manifest['cases']))
            self.assertEqual(len(manifest['unqualified_studies']), 1)
            self.assertFalse(manifest['original_selector_or_endpoint_qualification'])


class ResourceSyntheticCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name).resolve()
        self.output = self.root/'engineering';self.output.mkdir()
        self.graph = self.root/'graph.pt';self.train=self.root/'train.pt';self.heldout=self.root/'select.pt'
        torch.save({'x': torch.arange(18,dtype=torch.float32).reshape(6,3)/18, 'edge_index': torch.tensor([[0,1],[1,0]])}, self.graph)
        torch.save({'ids':torch.tensor([0,1,2,3]), 'y':torch.tensor([0,1,0,1])}, self.train)
        torch.save({'ids':torch.tensor([4,5]), 'y':torch.tensor([1,0])}, self.heldout)

    def tearDown(self):
        self.tmp.cleanup()

    def toy_adapter(self, two=False):
        t=self.t
        from adapters.classifier import ClassifierAdapter
        from adapters.coordinate import OptimizerBundle
        model = t.nn.ModuleList([t.nn.Linear(3,2) for _ in range(2 if two else 1)])
        x=t.arange(18,dtype=t.float32).reshape(6,3)/18
        # ModuleList does not expose a forward; give this synthetic module a bank forward.
        model.forward = lambda: t.stack([body(x) for body in model])
        opts=[t.optim.AdamW(body.parameters(),lr=.001) for body in model]
        optimizer = OptimizerBundle(opts) if two else opts[0]
        return ClassifierAdapter(t,t.nn.functional,model,optimizer,logits_fn=lambda m,training,step:m(),
            train_ids=t.tensor([0,1,2,3]), train_y=t.tensor([0,1,0,1]), select_ids=t.tensor([4]),select_y=t.tensor([0]),
            regularizer=lambda m:x.new_zeros(()), fusion='mean_probabilities',
            selector={'metric':'accuracy','direction':'max','ties':'first','patience_steps':10,'save_every':1},
            provenance={'engineering_only':True})

    def test_global_concrete_and_functional_actions_never_apply_updates(self):
        t=self.t;p=t.nn.Parameter(t.ones(2));before=p.detach().clone()
        with OptimizerDenial(t) as denied:
            optimizer=t.optim.AdamW([p],lr=.1);p.grad=t.ones_like(p);optimizer.step()
            self.assertTrue(t.equal(p,before));self.assertEqual(denied.denied_step_calls,1)
            with self.assertRaises(ForbiddenOptimizerKernel):
                import torch.optim._functional as functional
                functional.adamw([], [], [], [], [], [])
            self.assertEqual(denied.actual_optimizer_kernels,0)

    def test_role_guard_denies_original_tensor_and_hash_reads(self):
        t=self.t
        with RoleIO(t,self.root,[self.graph,self.train],[self.heldout],self.output) as guarded:
            t.load(self.train,weights_only=True)
            for operation in [lambda:t.load(self.heldout,weights_only=True),lambda:self.heldout.read_bytes()]:
                with self.assertRaises(ForbiddenRoleAccess):operation()
            self.assertEqual(len(guarded.tensor_load_paths),1)
            self.assertEqual(guarded.denied,2)

    def test_engineering_SELECT_stub_keeps_approved_study_and_role_files_unchanged(self):
        t=self.t;source={'files':[receipt(self.root,p) for p in [self.graph,self.train,self.heldout]]}
        run={'run_id':'source-only','kind':'model_fit','split':0,'config':{'node_inputs':{'graph':'graph.pt','train':'train.pt','select':'select.pt'}}}
        study={'stage':'development','source':source,'runs':[run], 'data':{'roles':{'train':{'0':'train.pt'},'select':{'0':'select.pt'}}}}
        before=copy.deepcopy(study);held_hash=receipt(self.root,self.heldout)
        with RoleIO(t,self.root,[self.graph,self.train],[self.heldout],self.output) as guarded:
            clone,resource_run,stub=engineering_clone(t,self.root,study,{'run':run},self.output)
            guarded.allowed.add(stub.resolve());synthetic=t.load(stub,weights_only=True)
            self.assertEqual(synthetic['y'].tolist(),[0]);self.assertEqual(synthetic['ids'].tolist(),[4])
            self.assertNotEqual(resource_run['config']['node_inputs']['select'],'select.pt')
            self.assertEqual(clone['stage'],'engineering');self.assertEqual(guarded.denied,0)
        self.assertEqual(study,before);self.assertEqual(receipt(self.root,self.heldout),held_hash)

    def test_full_train_proxy_keeps_parameters_moments_and_reports_no_scores(self):
        t=self.t
        with OptimizerDenial(t) as denied:
            adapter=self.toy_adapter();result=measure_adapter(t,adapter,denied,self.output,warm_iterations=2)
            self.assertEqual(result['denied_optimizer_step_calls'],3)
            self.assertEqual(result['parameters_sha256_before'],result['parameters_sha256_after'])
            self.assertEqual(result['moments_sha256_before'],result['moments_sha256_after'])
            self.assertEqual(result['actual_optimizer_updates'],0);self.assertEqual(result['scores_emitted'],0)
            self.assertGreater(result['gradient_bytes'],0);self.assertGreater(result['optimizer_moment_bytes'],0)
            self.assertIsNone(result['all_fit_ETA']);self.assertIsNone(result['SELECT_CAL_TEST_endpoint_timing'])
            for operation in [lambda:adapter.validate(1),lambda:adapter.observe_validation(1,{})]:
                with self.assertRaises(ForbiddenRoleAccess):operation()

    def test_multi_optimizer_bundle_is_blocked_and_accounted(self):
        t=self.t
        with OptimizerDenial(t) as denied:
            adapter=self.toy_adapter(two=True);result=measure_adapter(t,adapter,denied,self.output,warm_iterations=1)
            self.assertEqual(result['optimizer_objects'],2)
            self.assertEqual(result['denied_optimizer_step_calls'],4)
            self.assertEqual(result['actual_optimizer_kernel_calls'],0)

    def test_random_checkpoint_proxy_has_no_selected_validation_or_ancestor_schema(self):
        t=self.t
        from engine import capture_checkpoint
        with OptimizerDenial(t) as denied:
            result=measure_adapter(t,self.toy_adapter(),denied,self.output,warm_iterations=1,capture=capture_checkpoint)
            self.assertFalse(result['serialization']['selected_checkpoint'])
            self.assertGreater(result['serialization']['bytes'],0)
            self.assertFalse((self.output/'RANDOM_ENGINEERING_SNAPSHOT.pt').exists())


if __name__ == '__main__':
    unittest.main()
