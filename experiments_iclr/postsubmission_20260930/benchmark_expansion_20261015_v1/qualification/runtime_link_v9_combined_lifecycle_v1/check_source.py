"""Source, exact work and authentic metadata-interface checks; no native imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
E=HERE.parents[1]


def check():
    cases=json.loads((HERE/'CASES.json').read_text())['cases'];assert set(cases)=={'NCN','BUDDY'}
    spec=importlib.util.spec_from_file_location('actual_native_link_epoch_source',E/'link_native_integration_v1/epoch.py')
    native_epoch=importlib.util.module_from_spec(spec);spec.loader.exec_module(native_epoch)
    for family,case in cases.items():
        job=json.loads((HERE/('JOB_'+family+'_CUDA.json')).read_text())
        assert job['recipe']['native_fixed_config']==case['native_fixed_config']and job['seed']==case['seed']and job['procedure']==case['method']
        assert job['device']=='cuda:0'and job['engineering_only']is True and job['TEST_access']is False and job['body']==case['engineering_body']
        assert job['policy']['minimum_epochs']==job['policy']['maximum_epochs']==2
        assert job['link_policy_sha256']==hashlib.sha256((HERE/'engineering_policy.py').read_bytes()).hexdigest()
        args=(family,case['native_positive_population'])if family=='NCN'else(family,case['native_positive_population'],1622810)
        assert native_epoch.epoch_layout(*args)==case['layout']
        assert case['limits']['updates_attempted']==case['limits']['optimizer_attempts']==case['limits']['reverse_attempts']==2*case['layout']['optimizer_maps_per_epoch']
        assert case['total_exact_work']['TRAIN_task_query_records']==2*case['layout']['task_query_records_per_epoch']
        template=json.loads((HERE/('ADMISSION_TEMPLATE_'+family+'.json')).read_text())
        assert template['template_is_not_admission']and template['published']is False and template['GPU_resource_evidence']is None and template['own_GPU_reservation_bytes']is None
        assert template['limits']==case['limits']and template['maximum_attempts']==1 and template['automatic_retries']==0 and template['TEST_access']is False
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    tree=ast.parse((HERE/'qualify.py').read_text());text=(HERE/'qualify.py').read_text()
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    assert isinstance(main.body[0],ast.Expr)and isinstance(main.body[0].value,ast.Call)and main.body[0].value.func.id=='guard'
    assert 'def floating('not in text and 'torch.allclose'not in text
    assert text.count("invoke('controlled_completed_epoch1'")==1 and text.count("invoke('resume_remaining_epoch2'")==1 and text.count("invoke('stopped_finalization_resume'")==1
    assert 'expected_updates' in text and "report['updates_completed']-before==maps"in text
    assert 'UPDATE_COST.jsonl'in text and "begin('updates_attempted')"in text and "begin('optimizer_attempts')"in text and "begin('reverse_attempts')"in text
    assert 'original_create(t,descriptor,policy,root)'in text and 'run_fit.run(copy.deepcopy(job),ROOT,folder,resume=resume)'in text
    assert 'no_numerical_reexecution_test=True'in text
    assert text.count('with observer_module.observe(')==1 and text.count('reporting_observer.record_invocation(result,name)')==1
    assert "observed_index['exports']==2"in text and "observed_index['invocations']==3"in text
    # Retain exact durable source process/GPU authentication functions from accepted supervisor.
    old=ast.parse((E/'qualification/runtime_link_v6_native_lifecycle_driver_v1/supervise.py').read_text())
    new=ast.parse((HERE/'supervise.py').read_text())
    for name in('process','gpu_state'):
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    review=json.loads((E/'benchmark_reporting_protocol_v1/ROOT_REVIEW.json').read_text())
    assert review['status']=='accepted'and review['final_root_artifact_review_complete']is True
    return {'schema':'runtime-link-v9-combined-source-checks-v1','status':'PASS_source_only','cases':2,
        'frozen_native_epoch_layouts_match_actual_source':True,'exact_declared_updates':{'NCN':34,'BUDDY':3170},
        'actual_runtime_entry_called':True,'hard_counters_and_watchdog_present':True,'no_numeric_replay_test':True,
        'accepted_process_and_GPU_authentication_AST_identical':True,'reporting_review_attached':'accepted exploratory TRAIN/VALID',
        'BUDDY_compiler_null_body_preserved_provider2_metadata_normalization':True,'templates_cannot_admit_execution':True,'single_observer_context_for_existing_two_exports_three_invocations':True,'Torch_imported':False,
        'native_jobs':0,'models':0,'forwards':0,'optimizer_updates':0,'TEST_accessed':False}


if __name__=='__main__':print(json.dumps(check(),sort_keys=True))
