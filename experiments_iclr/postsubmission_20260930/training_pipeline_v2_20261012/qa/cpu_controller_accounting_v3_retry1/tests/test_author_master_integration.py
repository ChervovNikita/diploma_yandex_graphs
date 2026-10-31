"""Native CLI custody/gates/counting with the real common controller schema."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from compile_master_plan import (CONTROLLER_FILES, assemble, controller_bindings,
                                  readiness, receipt)
from launcher import worker_script
from partition import DEFAULT_WORKERS
from phase_closure import collect, write_host_receipts
from registry import Registry
from spec import canonical, fingerprint, validate
from tests.test_author_cli import fixture, SUCCESS_SOURCE
from tests.test_master_plan import study, PACKAGE


class AuthorMasterIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.author = fixture(self.root, SUCCESS_SOURCE)
        runtime = self.root / 'controller'
        runtime.mkdir()
        for name in CONTROLLER_FILES:
            shutil.copyfile(PACKAGE / name, runtime / name)
        self.bindings = controller_bindings(self.root, 'controller')

    def tearDown(self):
        self.temporary.cleanup()

    def packet(self, name, value):
        folder = self.root / name
        folder.mkdir()
        (folder/'study.json.gz').write_bytes(gzip.compress(canonical(value).encode(),mtime=0))
        plan = {'schema_version':1,'slots_per_gpu':2,'scientific_launch_enabled':False,
                'human_GO_required':True,'studies':[{'path':name+'/study.json.gz','spec_hash':fingerprint(value)}],
                'groups':[{'id':value['study_id'],'run_keys':[value['study_id']+'/'+r['run_id'] for r in value['runs']],
                           'estimated_seconds':1000,'maximum_cuda_bytes':4*1024**2,'gpu_assignment':'allocation/gpu0'}]}
        for path, payload in [('PLAN.json',plan),('SUMMARY.json',{'plan_hash':fingerprint(plan),'GO_receipt_created':False})]:
            (folder/path).write_text(canonical(payload))
        return {'id':name,'plan':receipt(self.root,name+'/PLAN.json'),'summary':receipt(self.root,name+'/SUMMARY.json')}

    def master(self):
        entries = [self.packet('node_packet',study('node')), self.packet('author_packet',self.author)]
        packets,ready = readiness(self.root,{'schema':'disabled-master-input-index-v1','packets':entries})
        self.assertEqual(ready['invalid_inputs'],[])
        return assemble(self.root,packets,ready,'controller',self.bindings,
                        {'worker_capacities_cuda_bytes':{w:100*1024**2 for w in DEFAULT_WORKERS}},'master',[])

    def test_shared_validator_registry_and_dispatch_preserve_native_protocol(self):
        validate(self.author)
        registry = Registry(self.root/'registry.sqlite')
        try:
            registry.register(self.author)
            self.assertEqual(registry.run(self.author['study_id'],'synthetic-cli')['contract']['kind'],'author_cli_group')
        finally:
            registry.close()
        mixed = copy.deepcopy(self.author)
        mixed['runs'].append(study('node')['runs'][0])
        with self.assertRaises(ValueError):
            validate(mixed)
        self.assertEqual(worker_script('author_cli_group'),'run_author_cli.py')
        self.assertEqual(worker_script('model_fit'),'run_study.py')
        with self.assertRaises(ValueError):
            worker_script('invented_cli')

    def test_author_counts_gates_worker_source_and_assignment_stay_explicit(self):
        plan,specs,summary = self.master()
        totals = summary['counts']['totals']
        self.assertEqual(totals['atomic_fit_run_groups'],3)
        self.assertEqual(totals['paid_optimizer_starts'],3)
        self.assertEqual(totals['author_cli_groups'],1)
        self.assertEqual(totals['author_sequential_repeat_trajectories'],10)
        self.assertEqual(totals['author_registered_maximum_optimizer_updates'],20000)
        self.assertEqual(totals['known_registered_optimizer_step_calls'],6)
        self.assertEqual(plan['global_gates']['whole_search']['all_packet_source_run_keys'],['node/search','node/select'])
        self.assertIn('author_cli',plan['global_gates']['whole_search']['required_for_stages'])
        self.assertEqual(len(plan['global_gates']['whole_author']['all_packet_source_run_keys']),1)
        author = next(s for s in specs if s['study_id']==self.author['study_id'])
        self.assertEqual(author['source']['worker_entrypoint'],'controller/run_author_cli.py')
        assigned = next(g for g in plan['groups'] if author['study_id'] in g['study_ids'])
        self.assertEqual(assigned['gpu_assignment'],'allocation/gpu0')
        self.assertEqual(author['runs'][0]['cli'],self.author['runs'][0]['cli'])
        self.assertEqual(set(author['data']['roles']),{'train','select','evaluate'})
        self.assertFalse(plan['scientific_launch_enabled'])

    def test_failed_author_group_closes_only_terminal_cost_custody_gate(self):
        plan,specs,_ = self.master()
        records = {s['study_id']+'/'+r['run_id']:{'study':s['study_id'],'run':r} for s in specs for r in s['runs']}
        placement = {key:{'worker':g['gpu_assignment']} for g in plan['groups'] for key in g['run_keys']}
        registry = Registry(self.root/'registry.sqlite')
        try:
            for value in specs:registry.register(value)
            registry.claim(self.author['study_id'],'synthetic-cli',{'pid':1,'hostname':'synthetic-engineering'})
            registry.finish(self.author['study_id'],'synthetic-cli',{'status':'failed','actual_updates':None},success=False)
            digest=fingerprint(plan)
            write_host_receipts(self.root,plan,digest,records,placement,registry,'allocation')
            result=collect(self.root,plan,digest,records,placement,'whole_author')
            self.assertTrue(result['terminal_roster'])
            self.assertFalse(result['all_successful'])
            self.assertFalse((self.root/plan['global_gates']['whole_search']['path']).exists())
        finally:
            registry.close()

if __name__=='__main__':unittest.main()
