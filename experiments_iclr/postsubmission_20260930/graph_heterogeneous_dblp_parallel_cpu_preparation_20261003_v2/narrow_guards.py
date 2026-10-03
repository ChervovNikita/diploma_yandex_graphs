"""Source-only runtime-cap/admission checks; no cancellation repetition or Torch."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v1'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok,message):
    if not ok:raise AssertionError(message)


def record(path):return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)


def dump(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n');return record(path)


def source_checks():
    old=(PREVIOUS/'parallel_cpu.py').read_text();new=(HERE/'parallel_cpu.py').read_text()
    old_prefix=old[:old.index('def admission_guard(')]
    new_prefix=new[:new.index('def prior_failure_guard(')]
    check(old_prefix==new_prefix,'Original module/signal/source guards changed')
    old_suffix=old[old.index('def resource_screen('):]
    new_suffix=new[new.index('def resource_screen('):]
    new_suffix=new_suffix.replace(
        'def run_workers(commands,out,wall_seconds,memory_limit_bytes,env,limit_resources=True,rss_limit_bytes=None):',
        'def run_workers(commands,out,wall_seconds,memory_limit_bytes,env,limit_resources=True):')
    new_suffix=new_suffix.replace(
        "elif child['peak_RSS_bytes']>(memory_limit_bytes if rss_limit_bytes is None else rss_limit_bytes):status='RSS_budget_exhausted'",
        "elif child['peak_RSS_bytes']>memory_limit_bytes:status='RSS_budget_exhausted'")
    new_suffix=new_suffix.replace(
        "run_workers(commands,out,release['worker_wall_budget_seconds'],16*2**30,env,rss_limit_bytes=14*2**30)",
        "run_workers(commands,out,release['worker_wall_budget_seconds'],8*2**30,env)")
    check(old_suffix==new_suffix,'Changes outside admission/runtime cap permitted lines')
    old_deploy=(PREVIOUS/'deploy_cpu_remote.py').read_text()
    new_deploy=(HERE/'deploy_cpu_remote.py').read_text()
    new_deploy=new_deploy.replace(
        "base=phase/'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v2'",
        "base=phase/'graph_heterogeneous_dblp_parallel_cpu_preparation_20261003_v1'")
    new_deploy=new_deploy.replace("binding['source_records']+[binding['freeze']]",
        "binding['source_records']+[binding['freeze'],binding['resource_qualification']]")
    check(new_deploy==old_deploy,'Deployment changed outside packet/admission paths')
    evidence=json.loads((HERE/'REUSED_CANCELLATION_EVIDENCE.json').read_text())
    check(evidence['original_scheduler_sha256']==digest(PREVIOUS/'parallel_cpu.py'),
        'Reused cancellation evidence source differs')
    check(evidence['v1_scheduler_fixture_receipt_sha256']==digest(PREVIOUS/'SCHEDULER_FIXTURE_RECEIPT.json'),
        'Reused fixture receipt changed')
    for name in ('parallel_cpu.py','deploy_cpu_remote.py','narrow_guards.py'):
        compile((HERE/name).read_text(),str(HERE/name),'exec')


def main():
    source_checks()
    spec=importlib.util.spec_from_file_location('runtime_amendment_scheduler',HERE/'parallel_cpu.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Driver:
        def freeze_guard(self,*args):pass  # Existing exact function is separately byte-bound.
    tests=[]
    with tempfile.TemporaryDirectory(prefix='hgt_runtime_admission_') as temporary:
        root=Path(temporary).resolve();original=root/'original';original.mkdir()
        rows=[dict(seed=s,arm=a,status='selected' if a=='native_HGT' else 'resource_deferred')
              for s in module.SEEDS for a in module.ARMS]
        dump(original/'PARALLEL_STUDY.json',dict(status='incomplete',originals_preserved=True,
             final_labels_closed=True,rows=rows,closure=dict(comparison=None)))
        dump(original/'STUDY.json',dict(rows=rows,summary=dict(status='incomplete',successful_subset_scored=False)))
        inventory=dict(schema='HGT35_closed_failure_inventory_v1',root_observed=True,status='closed_incomplete',
             original_run_directory=str(original),outcome_metrics_requested=False,prior_selected_checkpoints_reused=False,
             files=[record(p) for p in sorted(original.iterdir())])
        inventory_record=dump(root/'inventory.json',inventory)
        qualification=dict(status='qualified',device='cpu',originals_preserved=True,
             paired_CP_global_untied_geometry_verified=True,study_freeze_sha256='freeze',qualification_only=True,
             training_driver_main_called=False,validation_or_test_scored=False,model_selection=False,
             scientific_study_outcomes=False,
             rows=[dict(arm=a,status='qualified',optimizer_updates=6,full_state_rng_custody=True,
                 per_update_memory=[dict(update=i,**{phase:dict(VmSize_bytes=10*2**30,VmPeak_bytes=10*2**30,VmRSS_bytes=6*2**30) for phase in ('before_TRAIN','after_TRAIN','after_eval_and_checkpoint')}) for i in range(1,7)],validation_or_test_scored=False,model_selection=False)
                 for a in module.ARMS])
        qualification_record=dump(root/'qualification.json',qualification)
        wrapper=dump(root/'wrapper.json',dict(synthetic=True));remote_code=dump(root/'remote_code.json',dict(synthetic=True))
        def transport(q):
            return dict(exit_code=0,helper_sha256=wrapper['sha256'],remote_receipt=dict(exit_code=0,status='completed',
                preflight=dict(memory_limit_bytes=16*2**30,RSS_limit_bytes=14*2**30,CPU_threads=1,
                     CUDA_VISIBLE_DEVICES='',GPU_computation=False,wall_limit_seconds=1200),qualification_result=q))
        transport_record=dump(root/'transport.json',transport(qualification))
        binding=dict(freeze=dict(sha256='freeze'),training_manifest_sha256='training',
             original_failed_run_directory=str(original),source_records=[],original_failed_run_inventory=inventory_record,
             qualified_resource_wrapper_sha256=wrapper['sha256'])
        frozen=dict(seeds=list(module.SEEDS),arms=list(module.ARMS))
        release=dict(run_name='fresh',device='cpu',mode='parallel_CPU_five_seeds',scheduler_manifest_sha256='scheduler',
             parallel_workers=5,threads_per_worker=1,worker_address_space_limit_bytes=16*2**30,
             worker_RSS_limit_bytes=14*2**30,worker_wall_budget_seconds=1,required_free_host_bytes=96*2**30,
             runtime_repair_only=True,fresh_complete35_rerun=True,resume_from_prior_selected_checkpoints=False,
             root_observed_resource_qualification=True,resource_qualification=qualification_record,
             resource_qualification_sha256=qualification_record['sha256'],
             resource_transport=dict(receipt=transport_record,wrapper=wrapper,remote_code=remote_code),
             original_failed_run_inventory=inventory_record)
        previous_cuda=module.os.environ.get('CUDA_VISIBLE_DEVICES');module.os.environ['CUDA_VISIBLE_DEVICES']=''
        def admit(r):
            bound=copy.deepcopy(binding);bound['original_failed_run_inventory']=r['original_failed_run_inventory']
            return module.admission_guard(bound,frozen,Driver(),r,'scheduler','fresh')
        def rejected(name,r):
            try:admit(r)
            except (ValueError,KeyError):tests.append(dict(name=name,status='PASS'));return
            raise AssertionError('Unexpected admission: '+name)
        try:
            admitted=admit(copy.deepcopy(release));check(admitted['fresh_complete35_rerun'] is True,'Fresh release not admitted')
            tests.append(dict(name='synthetic_six_update_fresh_release',status='PASS'))
            for key,value in [('worker_address_space_limit_bytes',8*2**30),('worker_RSS_limit_bytes',16*2**30),
                 ('required_free_host_bytes',80*2**30),('resume_from_prior_selected_checkpoints',True),
                 ('root_observed_resource_qualification',False)]:
                changed=copy.deepcopy(release);changed[key]=value;rejected(key,changed)
            changed_q=copy.deepcopy(qualification);changed_q['rows'][0]['optimizer_updates']=1
            changed=copy.deepcopy(release);changed['resource_qualification']=dump(root/'one_update.json',changed_q)
            changed['resource_qualification_sha256']=changed['resource_qualification']['sha256']
            changed['resource_transport']['receipt']=dump(root/'one_update_transport.json',transport(changed_q))
            rejected('one_update_qualification',changed)
            changed_q=copy.deepcopy(qualification);changed_q['rows'][0]['per_update_memory'].pop()
            changed=copy.deepcopy(release);changed['resource_qualification']=dump(root/'missing_memory.json',changed_q)
            changed['resource_qualification_sha256']=changed['resource_qualification']['sha256']
            changed['resource_transport']['receipt']=dump(root/'missing_memory_transport.json',transport(changed_q))
            rejected('missing_update_memory',changed)
            changed_inventory=copy.deepcopy(inventory);changed_inventory['files'].pop()
            changed=copy.deepcopy(release);changed['original_failed_run_inventory']=dump(root/'omitted_file_inventory.json',changed_inventory)
            rejected('omitted_original_file',changed)
            altered=json.loads((original/'STUDY.json').read_text());altered['summary']['successful_subset_scored']=True
            dump(original/'STUDY.json',altered)
            changed_inventory=copy.deepcopy(inventory);changed_inventory['files']=[record(p) for p in sorted(original.iterdir())]
            changed=copy.deepcopy(release);changed['original_failed_run_inventory']=dump(root/'scored_inventory.json',changed_inventory)
            rejected('original_subset_scored',changed)
        finally:
            if previous_cuda is None:module.os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:module.os.environ['CUDA_VISIBLE_DEVICES']=previous_cuda
    receipt=dict(schema='HGT35_runtime_amendment_narrow_guards_v1',status='PASS',tests=tests,
        tests_run=len(tests),source_identity_checks='PASS',Torch_imported='torch' in sys.modules,
        real_data_models_or_selected_states_opened=False,cancellation_suite_repeated=False,
        scheduler_source_sha256=digest(HERE/'parallel_cpu.py'),deployment_source_sha256=digest(HERE/'deploy_cpu_remote.py'))
    print(json.dumps(receipt,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
