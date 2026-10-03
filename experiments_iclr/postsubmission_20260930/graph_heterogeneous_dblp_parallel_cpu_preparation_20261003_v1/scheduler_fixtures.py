"""Stdlib-only scheduler checks: synthetic files/processes, no Torch or real labels."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


scheduler=load('fixture_parallel_scheduler',HERE/'parallel_cpu.py')
driver=load('fixture_exact_v2_driver',HERE.parent/'graph_heterogeneous_dblp_training_preparation_20261003_v2/train_dblp.py')
FAKE=r'''
import hashlib,json,os,pathlib,sys,time
seed=int(sys.argv[1]);root=pathlib.Path(sys.argv[2]);mode=sys.argv[3];binding=sys.argv[4]
out=root/f'seed{seed}';out.mkdir()
arms=('native_HGT','global_BE','shared_relation','CP','unrestricted','untied_HGT','wider_BE')
rows=[]
for index,arm in enumerate(arms):
 if mode=='partial' and index==3:break
 case=out/arm;case.mkdir()
 row=dict(seed=seed,arm=arm,status='selected',selected_state_replay=True,checkpoint_bindings_verified=True,
  checkpoint_binding_sha256=binding,selection=dict(validation_NLL=1+seed/10000+index/1000),final_labels_closed=True)
 for key,name in (('selected_checkpoint','selected.pt'),('selected_logits','selected_member_logits.pt'),
                  ('selection_receipt','SELECTION.json'),('training_trace','TRACE.jsonl')):
  data=(json.dumps({key:row[key] for key in ('seed','arm','status','selection','selected_state_replay','final_labels_closed')},sort_keys=True)+'\n').encode() if name=='SELECTION.json' else f'SYNTHETIC TEXT ONLY {seed} {arm} {name}\n'.encode()
  (case/name).write_bytes(data)
  row[key]=dict(path=str(case/name),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
 rows.append(row)
 with (out/'ARM_TERMINALS.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
result=dict(status='completed',originals_preserved=mode!='preservation_failed',seed=seed)
(out/'WORKER_RESULT.json').write_text(json.dumps(result)+'\n')
sys.exit(7 if mode=='nonzero' else 0)
'''


class CountingDriver:
    def __init__(self):self.calls=0
    def paired_development(self,rows,arms):
        self.calls+=1;return driver.paired_development(rows,arms)


class SchedulerChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.out=Path(self.temp.name).resolve()
        self.binding={'study_checkpoint_binding_sha256':'synthetic-custody'}
        self.spy=CountingDriver()

    def tearDown(self):self.temp.cleanup()

    def launch(self,modes=None):
        modes=modes or {};commands=[(s,[sys.executable,'-I','-S','-B','-c',FAKE,str(s),str(self.out),modes.get(s,'complete'),
            self.binding['study_checkpoint_binding_sha256']]) for s in scheduler.SEEDS]
        records=scheduler.run_workers(commands,self.out,10,8*2**30,dict(os.environ,CUDA_VISIBLE_DEVICES=''),limit_resources=False)
        rows=[r for s in scheduler.SEEDS for r in scheduler.collect(s,self.out/f'seed{s}',records[s])]
        return records,rows

    def assert_closed(self,rows,permitted=True):
        result=scheduler.closure(rows,self.spy,self.binding,self.out,comparison_permitted=permitted)
        self.assertEqual(result['status'],'incomplete');self.assertIsNone(result['comparison'])
        self.assertTrue(result['all35_terminals']);self.assertFalse(result['successful_subset_scored']);self.assertEqual(self.spy.calls,0)
        return result

    def test_five_process_complete_order_and_exact_summary(self):
        records,rows=self.launch()
        self.assertEqual(set(records),set(scheduler.SEEDS))
        self.assertTrue(all(r['status']=='completed' and r['exit_code']==0 for r in records.values()))
        self.assertEqual([(r['seed'],r['arm']) for r in rows],[(s,a) for s in scheduler.SEEDS for a in scheduler.ARMS])
        result=scheduler.closure(rows,self.spy,self.binding,self.out)
        self.assertEqual(result['comparison'],driver.paired_development(rows,list(scheduler.ARMS)));self.assertEqual(self.spy.calls,1)
        self.assertTrue(result['all35_selected_fits']);self.assertFalse(result['comparison']['heldout_opened'])

    def test_partial_nonzero_and_failed_fit_close_comparison(self):
        records,rows=self.launch({131:'partial',137:'nonzero'})
        self.assertEqual(records[137]['exit_code'],7);self.assertEqual(len(rows),35)
        self.assertEqual(sum(r['status']=='failed' for r in rows),4);self.assert_closed(rows)
        rows[0]['status']='resource_deferred';self.assert_closed(rows)

    def test_duplicate_order_shape_and_nonterminal_receipts(self):
        seed=131;out=self.out/f'seed{seed}';out.mkdir();path=out/'ARM_TERMINALS.jsonl'
        normal=[dict(seed=seed,arm=a,status='failed') for a in scheduler.ARMS]
        mutations=[normal+[normal[0]],list(reversed(normal)),[dict(normal[0],status='running')],
            [dict(normal[0],seed=137)],['bad shape'],[dict(normal[0],arm='unknown')]]
        for rows in mutations:
            with self.subTest(rows=rows):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                actual=scheduler.collect(seed,out,dict(status='completed',exit_code=0))
                self.assertEqual(len(actual),7);self.assertTrue(all(r['status']=='failed' for r in actual))
        path.write_text(json.dumps(normal[0])+'\n{partial')
        actual=scheduler.collect(seed,out,dict(status='wall_budget_exhausted',exit_code=-15))
        self.assertEqual(actual[0],normal[0]);self.assertEqual(sum(r['status']=='resource_deferred' for r in actual),6)
        path.write_text('{broken\n'+json.dumps(normal[0])+'\n')
        self.assertTrue(all(r['status']=='failed' for r in scheduler.collect(seed,out,dict(status='completed',exit_code=0))))
        (out/'ARM_STARTED.jsonl').write_text(json.dumps(dict(seed=seed,arm='CP'))+'\n')
        path.unlink();actual=scheduler.collect(seed,out,dict(status='controller_aborted',exit_code=-15))
        self.assertTrue(next(r for r in actual if r['arm']=='CP')['attempted'])
        self.assertFalse(actual[0]['attempted'])

    def test_spawn_failure_timeout_and_missing_worker_have_terminals(self):
        commands=[(131,['/definitely/absent/synthetic-worker']),
            (137,[sys.executable,'-I','-S','-B','-c','import time; time.sleep(30)'])]
        records=scheduler.run_workers(commands,self.out,.05,8*2**30,os.environ.copy(),limit_resources=False)
        self.assertEqual(records[131]['status'],'spawn_failed');self.assertEqual(records[137]['status'],'wall_budget_exhausted')
        for seed,record in records.items():
            self.assertTrue((self.out/f'worker{seed}.exit.json').is_file())
            self.assertTrue(all(r['status']=='resource_deferred' for r in scheduler.collect(seed,self.out/f'seed{seed}',record)))

    def test_each_artifact_hash_and_bytes_are_verified_in_partial_cohort(self):
        records,rows=self.launch();rows[-1]['status']='failed'
        for key in ('selected_checkpoint','selected_logits','selection_receipt','training_trace'):
            with self.subTest(artifact=key):
                data=Path(rows[0][key]['path']).read_bytes();Path(rows[0][key]['path']).write_bytes(data+b'tampered')
                with self.assertRaises(ValueError):scheduler.closure(rows,self.spy,self.binding,self.out,comparison_permitted=False)
                Path(rows[0][key]['path']).write_bytes(data)
                altered=copy.deepcopy(rows);altered[0][key]['bytes']+=1
                with self.assertRaises(ValueError):scheduler.closure(altered,self.spy,self.binding,self.out,comparison_permitted=False)
                altered=copy.deepcopy(rows);del altered[0][key]['bytes']
                with self.assertRaises(ValueError):scheduler.closure(altered,self.spy,self.binding,self.out)
        self.assertEqual(self.spy.calls,0);self.assert_closed(rows,permitted=False)

    def test_artifact_scope_binding_and_replay_guards(self):
        records,rows=self.launch()
        altered=copy.deepcopy(rows);altered[0]['selected_checkpoint']=copy.deepcopy(rows[1]['selected_checkpoint'])
        with self.assertRaises(ValueError):scheduler.closure(altered,self.spy,self.binding,self.out)
        altered=copy.deepcopy(rows);altered[0]['checkpoint_binding_sha256']='wrong'
        with self.assertRaises(ValueError):scheduler.closure(altered,self.spy,self.binding,self.out)
        altered=copy.deepcopy(rows);altered[0]['selection']['validation_NLL']+=1
        with self.assertRaises(ValueError):scheduler.closure(altered,self.spy,self.binding,self.out)
        artifact=Path(rows[0]['selected_logits']['path']);outside=self.out/'outside';artifact.rename(outside);artifact.symlink_to(outside)
        with self.assertRaises(ValueError):scheduler.closure(rows,self.spy,self.binding,self.out)
        artifact.unlink();outside.rename(artifact)
        altered=copy.deepcopy(rows);altered[0]['checkpoint_bindings_verified']=False;self.assert_closed(altered)
        self.assert_closed(rows,permitted=False)

    def test_actual_release_guard_and_resource_proof(self):
        frozen=json.loads((HERE.parent/'graph_heterogeneous_dblp_execution_root_v1/FROZEN_STUDY.json').read_text())
        binding=json.loads((HERE/'BINDINGS.json').read_text());release=json.loads((HERE/'EXECUTION_RELEASE_TEMPLATE.json').read_text())
        release.update(execution_authorized=True,run_name='fixture',scheduler_manifest_sha256='fixture-manifest',worker_wall_budget_seconds=1)
        qualified=dict(status='qualified',device='cpu',originals_preserved=True,paired_CP_global_untied_geometry_verified=True,
            study_freeze_sha256=binding['freeze']['sha256'],rows=[dict(arm=a,status='qualified',full_state_rng_custody=True) for a in scheduler.ARMS])
        proof=self.out/'RESOURCE.json';proof.write_text(json.dumps(qualified))
        transport=self.out/'TRANSPORT.json'
        actual=dict(exit_code=0,helper_sha256=binding['resource_transport']['wrapper']['sha256'],
            remote_receipt=dict(exit_code=0,status='completed',preflight=dict(memory_limit_bytes=8*2**30,CPU_threads=1,
                CUDA_VISIBLE_DEVICES='',GPU_computation=False),qualification_result=qualified))
        transport.write_text(json.dumps(actual))
        def fake_verify(row):return proof if row==binding['resource_qualification'] else transport
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(scheduler,'verify',side_effect=fake_verify):
            accepted=scheduler.admission_guard(binding,frozen,driver,release,'fixture-manifest','fixture')
            self.assertEqual(accepted['parallel_workers'],5)
            invalid={'execution_authorized':False,'prepared_manifest_sha256':'wrong','study_freeze_sha256':'wrong',
                'scheduler_manifest_sha256':'wrong','resource_qualification_sha256':'wrong','parallel_workers':4,
                'threads_per_worker':2,'device':'cuda:0','worker_address_space_limit_bytes':9*2**30,
                'required_free_host_bytes':39*2**30,'worker_wall_budget_seconds':0,'run_name':'different'}
            for key,value in invalid.items():
                with self.subTest(release=key),self.assertRaises(ValueError):
                    scheduler.admission_guard(binding,frozen,driver,dict(release,**{key:value}),'fixture-manifest','fixture')
            for key in ('status','device','originals_preserved','paired_CP_global_untied_geometry_verified','study_freeze_sha256'):
                broken=copy.deepcopy(qualified);broken[key]='invalid';proof.write_text(json.dumps(broken))
                with self.subTest(proof=key),self.assertRaises(ValueError):scheduler.admission_guard(binding,frozen,driver,release,'fixture-manifest','fixture')
            broken=copy.deepcopy(qualified);broken['rows'][0]['full_state_rng_custody']=False;proof.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):scheduler.admission_guard(binding,frozen,driver,release,'fixture-manifest','fixture')
            broken=copy.deepcopy(qualified);broken['rows'].reverse();proof.write_text(json.dumps(broken))
            with self.assertRaises(ValueError):scheduler.admission_guard(binding,frozen,driver,release,'fixture-manifest','fixture')

    def test_transport_false_release_stays_local_and_stages_exact_inventory(self):
        helper=load('fixture_cpu_deploy_helper',HERE/'deploy_cpu_remote.py')
        payload=self.out/'example.txt';payload.write_text('SYNTHETIC PAYLOAD ONLY\n')
        manifest=dict(payload=[dict(path=payload.name,sha256=scheduler.sha(payload),bytes=payload.stat().st_size)])
        (self.out/'MANIFEST.json').write_text(json.dumps(manifest)+'\n');digest=scheduler.sha(self.out/'MANIFEST.json')
        (self.out/'SEAL.json').write_text(json.dumps(dict(manifest_sha256=digest))+'\n')
        release=self.out/'admission.json';release.write_text(json.dumps(dict(execution_authorized=False,
            scheduler_manifest_sha256=digest,device='cpu',mode='parallel_CPU_five_seeds')))
        receipt=self.out/'transport.json'
        with patch.object(helper,'HERE',self.out),patch.object(sys,'argv',['deploy','--admission',str(release),'--receipt',str(receipt)]),\
            patch.object(helper.subprocess,'run') as ssh:
            with self.assertRaises(ValueError):helper.main()
            ssh.assert_not_called();self.assertFalse(receipt.exists())
        remote=dict(status='sealed_CPU_scheduler_staged')
        with patch.object(helper,'HERE',self.out),patch.object(sys,'argv',['deploy','--stage-only','--receipt',str(receipt)]),\
            patch.object(helper.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(remote)+'\n','')) as ssh:
            self.assertEqual(helper.main(),0)
            request=json.loads(ssh.call_args.kwargs['input'])
            self.assertEqual({r['path'] for r in request['files']},{'example.txt','MANIFEST.json','SEAL.json'})
            self.assertTrue(request['stage_only']);self.assertNotIn('release_data',request)
            self.assertEqual(json.loads(receipt.read_text())['remote_receipt'],remote)
        compile(helper.REMOTE,'CPU_remote_bootstrap','exec')

    def run_controller(self,mode='complete',preservation_error=False,interrupt=False):
        packet=self.out/'packet';packet.mkdir();phase=self.out/'original';phase.mkdir()
        binding=dict(self.binding,source_records=[],freeze={},resource_qualification={},canonical_phase=str(phase))
        frozen=dict(archive={},development_labels={},splits=[])
        release=self.out/'release.json';release.write_text(json.dumps(dict(required_free_host_bytes=40*2**30,worker_wall_budget_seconds=10)))
        original_run=scheduler.run_workers;preservation_calls=[]
        def fake_run(commands,out,wall,memory,env):
            actual=[(s,[sys.executable,'-I','-S','-B','-c',FAKE,str(s),str(out.resolve()),mode,self.binding['study_checkpoint_binding_sha256']]) for s,_ in commands]
            completed=original_run(actual,out,wall,memory,env,limit_resources=False)
            if interrupt:raise KeyboardInterrupt('synthetic controller cancellation')
            return completed
        def preserve(expected):
            preservation_calls.append(expected)
            if preservation_error:raise ValueError('synthetic source changed')
        with patch.object(scheduler,'PACKET',packet),patch.object(scheduler,'packet_guard',return_value=(binding,frozen,self.spy,'fixture-manifest')),\
            patch.object(scheduler,'admission_guard',return_value={'scheduler_manifest_sha256':'fixture-manifest'}),\
            patch.object(scheduler,'verify'),patch.object(scheduler,'scheduler_preservation',side_effect=preserve),\
            patch.object(scheduler,'resource_screen',return_value={'status':'passed'}),patch.object(scheduler,'run_workers',side_effect=fake_run):
            code=scheduler.main(['--admission',str(release),'--run-name','fixture'])
        result=json.loads((packet/'runs/fixture/PARALLEL_STUDY.json').read_text())
        result['canonical_study_fixture']=json.loads((packet/'runs/fixture/STUDY.json').read_text())
        return code,result,preservation_calls

    def test_controller_complete_uses_preservation_before_and_after_summary(self):
        code,result,calls=self.run_controller()
        self.assertEqual(code,0);self.assertEqual(result['status'],'complete_development_summary')
        self.assertEqual(calls,['fixture-manifest','fixture-manifest']);self.assertEqual(self.spy.calls,1);self.assertEqual(len(result['rows']),35)
        canonical=result['canonical_study_fixture'];self.assertEqual(canonical['rows'],result['rows'])
        self.assertEqual(canonical['summary'],result['closure']['comparison']);self.assertTrue(canonical['original_inputs_verified_unchanged'])
        self.assertTrue(canonical['final_labels_closed']);self.assertIn('root_release_sha256',canonical['admission'])

    def test_controller_worker_preservation_failure_retains35_and_no_summary(self):
        code,result,calls=self.run_controller(mode='preservation_failed')
        self.assertEqual(code,1);self.assertEqual(result['status'],'incomplete');self.assertEqual(len(result['rows']),35)
        self.assertFalse(result['closure']['comparison_permitted']);self.assertEqual(self.spy.calls,0)
        self.assertEqual(result['canonical_study_fixture']['summary']['status'],'incomplete')
        self.assertFalse(result['canonical_study_fixture']['original_inputs_verified_unchanged'])

    def test_controller_own_preservation_failure_retains35_and_no_summary(self):
        code,result,calls=self.run_controller(preservation_error=True)
        self.assertEqual(code,1);self.assertEqual(result['status'],'original_preservation_failed');self.assertEqual(len(result['rows']),35)
        self.assertFalse(result['originals_preserved']);self.assertIsNone(result['closure']);self.assertEqual(self.spy.calls,0)
        self.assertEqual(result['canonical_study_fixture']['summary']['status'],'incomplete')
        self.assertFalse(result['canonical_study_fixture']['original_inputs_verified_unchanged'])

    def test_controller_keyboard_interrupt_recovers35_and_closes_summary(self):
        code,result,calls=self.run_controller(interrupt=True)
        self.assertEqual(code,1);self.assertEqual(result['status'],'controller_failed');self.assertEqual(len(result['rows']),35)
        self.assertEqual(len(result['worker_exits']),5);self.assertIsNone(result['closure']);self.assertEqual(self.spy.calls,0)
        self.assertEqual(result['canonical_study_fixture']['summary']['status'],'incomplete')

    def test_actual_SIGTERM_reaps_owned_child_and_keeps_exit_receipt(self):
        bootstrap=r'''
import importlib.util,json,pathlib,signal,sys
spec=importlib.util.spec_from_file_location('synthetic_signal_scheduler',sys.argv[1]);scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler)
out=pathlib.Path(sys.argv[2]);previous=scheduler.install_controller_signals()
worker='import os,pathlib,sys,time;pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(30)'
try:scheduler.run_workers([(131,[sys.executable,'-I','-S','-B','-c',worker,str(out/'child.pid')])],out,60,8*2**30,dict(__import__('os').environ),limit_resources=False)
except scheduler.ControllerInterrupted:(out/'INTERRUPTED.json').write_text(json.dumps(dict(status='controlled_interruption')))
finally:scheduler.restore_controller_signals(previous)
'''
        parent=subprocess.Popen([sys.executable,'-I','-S','-B','-c',bootstrap,str(HERE/'parallel_cpu.py'),str(self.out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+5
            while not (self.out/'child.pid').exists() and parent.poll() is None and time.monotonic()<deadline:time.sleep(.01)
            self.assertTrue((self.out/'child.pid').exists());pid=int((self.out/'child.pid').read_text())
            parent.send_signal(signal.SIGTERM);stdout,stderr=parent.communicate(timeout=10)
            self.assertEqual(parent.returncode,0,stderr)
            self.assertTrue((self.out/'INTERRUPTED.json').exists())
            self.assertEqual(json.loads((self.out/'worker131.exit.json').read_text())['status'],'controller_aborted')
            with self.assertRaises(ProcessLookupError):os.kill(pid,0)
        finally:
            if parent.poll() is None:parent.terminate();parent.wait(timeout=10)


def main():
    require_no_torch='torch' not in sys.modules
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SchedulerChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(schema='DBLP_parallel_CPU_stdlib_scheduler_fixtures_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        synthetic_files_processes_only=True,Torch_imported='torch' in sys.modules,real_labels_or_tensors_read=False,
        numeric_functions_rewritten=False,exact_v2_paired_development_used=True)
    assert require_no_torch and 'torch' not in sys.modules
    print(json.dumps(receipt,sort_keys=True))
    if '--receipt' in sys.argv:
        path=Path(sys.argv[sys.argv.index('--receipt')+1]);scheduler.write(path,receipt)
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
