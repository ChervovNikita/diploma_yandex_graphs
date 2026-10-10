"""Disabled fixed18 PubMed entry using existing pooled/own fit loops and owner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
CONDITIONS = ('shared4_own','private_missinghop','full_aux','common_nonfull','allblock_missinghop','factor1_allview')
SEEDS = (9101,9203,9307)
M4_SELECTOR = 'first strict maximum factual mean-member-probability pooled VALID accuracy'
M1_SELECTOR = 'first strict maximum own raw-logit max(1)[1] VALID accuracy'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    loader = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[name] = module
    loader.loader.exec_module(module)
    return module


def qualification_surface():
    return load('_private_hop_qualified_entry_route',PHASE/'private_hop_credit_pubmed_qualification_source_20261010_v2/run.py')


def admit(args):
    route = qualification_surface();route.route()
    path = args.release.resolve(strict=True)
    if not path.is_relative_to(HERE) or sha(path) != args.release_sha256:
        raise ValueError('Exact new root release required')
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'private-hop-fullfit-release-v1' or spec.get('purpose') != 'science':
        raise ValueError('Fixed representative science release required')
    for key in ('enabled','root_authorized','source_review_approved','complete_roster_frozen','qualification_verified','finite_owner_verified','fresh_resource_readiness_confirmed','ordinary_runtime_confirmed','VALID_access','science_enabled'):
        if spec.get(key) is not True:raise ValueError('Disabled admission: '+key)
    for key in ('TEST_access','automatic_retry','resume','HPO','paper_score_recalculation','confirmation_claim'):
        if spec.get(key) is not False:raise ValueError('Closed scope: '+key)
    if any(key in spec for key in ('test_bundle','test_ids','test_y','full_y','split_ids_bundle')):
        raise ValueError('No TEST/full-label surface')
    condition,seed = spec['condition'],spec['seed']
    if condition not in CONDITIONS or seed not in SEEDS or spec['record_id'] != f'seed{seed}__{condition}':
        raise ValueError('Fixed six-arm/three-seed identity required')
    selector = M1_SELECTOR if condition == 'factor1_allview' else M4_SELECTOR
    if spec['selector'] != selector or spec['max_epochs'] != 2000 or spec['patience'] != 250 or spec['lambda'] != .5:
        raise ValueError('Unchanged selector/full horizon/lambda required')
    if spec['source_manifest_sha256'] != sha(HERE/'SOURCE_MANIFEST.json') or spec['roster_sha256'] != sha(HERE/'ROSTER.json'):
        raise ValueError('Exact reviewed source/roster required')
    for row in json.loads((HERE/'SOURCE_MANIFEST.json').read_text())['files']:
        target = (HERE/row['path']).resolve(strict=True)
        if not target.is_relative_to(HERE) or sha(target) != row['sha256']:raise ValueError('New source changed')
    for row in json.loads((HERE/'SOURCE_BINDINGS.json').read_text())['files']:
        target = (PHASE/row['path']).resolve(strict=True)
        if not target.is_relative_to(PHASE) or sha(target) != row['sha256']:raise ValueError('Reused source changed')
    base = load('_private_hop_existing_reference_surface',PHASE/'pubmed_factor1_controls_source_20261010_v1/run.py')
    review = json.loads(base.bind(spec['root_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256') != spec['source_manifest_sha256']:
        raise ValueError('Exact root source review required')
    base.bind(spec['resource_readiness_evidence'])
    qualification = json.loads((HERE/'QUALIFICATION_BINDING.json').read_text())
    for key in ('qualification_complete','qualification_terminal','qualification_custody'):
        if spec[key] != qualification[key]:raise ValueError('Exact already successful qualification required')
    qualifier = json.loads(base.bind(spec['qualification_complete']).read_text())
    if (qualifier.get('schema') != 'private-hop-all5-TRAIN-qualification-complete-v1' or
            qualifier.get('complete') is not True or qualifier.get('conditions') != list(route.CONDITIONS) or
            qualifier.get('genuine_optimizer_updates') != 5 or qualifier.get('shared_data_gradient_routing_verified') is not True or
            qualifier.get('all_factual_replays_verified') is not True or qualifier.get('VALID_access') is not False or
            qualifier.get('TEST_access') is not False or qualifier.get('providers') != spec['frozen_providers']):
        raise ValueError('Actual complete five-condition TRAIN qualifier required')
    terminal = json.loads(base.bind(spec['qualification_terminal']).read_text())
    if (terminal.get('complete') is not True or terminal.get('directly_waited') is not True or
            terminal.get('child_exit_code') != 0 or terminal.get('cap_or_owner_failure') is not None or
            terminal.get('owned_absence',{}).get('owned_process_absence_verified') is not True or
            terminal.get('owned_absence',{}).get('owned_CUDA_absence_verified') is not True):
        raise ValueError('Actual directly waited qualifier owner/absence required')
    base.terminal(spec['qualification_terminal'],qualifier['release_sha256'])
    custody = json.loads(base.bind(spec['qualification_custody']).read_text())
    if (qualifier.get('record_id') != qualification['record_id'] or
            qualifier.get('source_manifest_sha256') != qualification['source_manifest_sha256'] or
            qualifier.get('release_sha256') != qualification['release_sha256'] or
            custody.get('record_id') != qualifier['record_id'] or custody.get('actual') is not True or
            custody.get('complete_sha256') != spec['qualification_complete']['sha256'] or
            custody.get('raw_owner_terminal') != spec['qualification_terminal'] or
            custody.get('source_manifest_sha256') != qualifier['source_manifest_sha256'] or
            custody.get('release_sha256') != qualifier['release_sha256'] or
            custody.get('owned_process_absence_verified') is not True or
            custody.get('owned_CUDA_absence_verified') is not True):
        raise ValueError('Qualifier completion/source/release/owner custody differ')
    data = json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for key in ('train_bundle','split_custody','valid_bundle','validation_custody','runtime','frozen_providers'):
        if spec[key] != data[key]:raise ValueError('Exact current native roles/runtime required')
    for key in ('split_identity','split_seed','edge_shape'):
        if spec[key] != data[key]:raise ValueError('Exact frozen complete split/graph required')
    spec['_train_custody'] = json.loads(base.bind(spec['split_custody']).read_text())
    spec['_validation_custody'] = json.loads(base.bind(spec['validation_custody']).read_text())
    base.bind(spec['train_bundle']);base.bind(spec['valid_bundle'])
    if spec['limits'] != base.LIMITS:raise ValueError('Existing finite full-fit caps required')
    owner = json.loads(base.bind(spec['external_owner_release']).read_text())
    if (owner.get('enabled') is not True or owner.get('record_id') != spec['record_id'] or
            owner.get('limits') != spec['limits'] or owner.get('automatic_retry') is not False):
        raise ValueError('New exact finite owner release required')
    for key in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(key) is not True:raise ValueError('Owner fact missing: '+key)
    if base.bind(owner['owner_source']) != HERE/'queue.py':raise ValueError('Exact current owner wrapper required')
    owner_review = json.loads(base.bind(owner['owner_review']).read_text())
    if owner_review.get('approved') is not True or owner_review.get('owner_sha256') != sha(HERE/'queue.py'):
        raise ValueError('Exact current owner review required')
    python = Path(os.path.abspath(spec['runtime']['python']['path']))
    if (not python.is_relative_to(PHASE) or sha(python) != spec['runtime']['python']['sha256'] or
            python.resolve() != Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','') != spec['runtime']['PYTHONPATH']):
        raise ValueError('Exact current native Python/PYTHONPATH required')
    output = Path(spec['output']).resolve()
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Fresh root-confined record output required')
    spec['_release_sha256'] = args.release_sha256
    return base,spec,output


def execute(base,spec,output,started):
    with base.reference_surface() as (engine,metrics,_unused_factory):
        hop = load('_fixed18_hop',PHASE/'private_hop_credit_source_20261010_v1/adapter.py')
        hooks = load('_fixed18_hooks',PHASE/'private_hop_credit_pubmed_entry_source_20261010_v1/hooks.py')
        def factory(method,condition,seed,tensor):
            if condition == 'shared4_own':
                return method.Session(condition,seed,tensor['x'],tensor['edge_index'],tensor['train_ids'],tensor['train_y'],device='cuda:0')
            return hooks.fresh(method,hop,condition,seed,tensor,numerical_admitted=True)
        def work(session,updates,evaluations):
            if session.name != 'shared4_own':return hooks.expected_work(session,updates,evaluations)
            return dict(updates=updates,factual_forwards=4*(updates+evaluations),masked_forwards=0,
                        backwards=4*updates,optimizer_steps=updates,serving_forwards=4*evaluations,preprocessing_banks=1)
        if spec['condition'] != 'factor1_allview':
            fit = load('_fixed18_original_M4_loop',HERE/'m4_fit.py')
            return fit.fit(spec,output,started,factory,work,engine.serving_benchmark)
        output.mkdir(parents=True,exist_ok=False);timings={};session=None
        try:
            tick=time.monotonic()
            np,torch,providers,tensor,roles = engine.runtime_and_data(spec,valid=True)
            engine.add(timings,'native_runtime_and_exact_roles',dict(wall_seconds=time.monotonic()-tick))
            method = sys.modules['source'].load_v3(numerical=True)['method']
            fit = load('_fixed18_original_M1_loop',HERE/'m1_fit.py')
            fit.timed,fit.add,fit.check_bounds = engine.timed,engine.add,engine.check_bounds
            body = dict(body_id='body0',initialization_seed=spec['seed'],factual_dropout_seeds=[spec['seed']+300001],
                        auxiliary_dropout_seeds=[spec['seed']+300001+1009*k+500009 for k in (1,2,3)],
                        max_epochs=2000,patience=250,selector=M1_SELECTOR,fresh_full_native_fit=True,reuse_other_record=False)
            session,logits,_state,row = fit.fit_body(spec,body,tensor,roles,output/'body0',started,timings,
                                                   lambda condition,seed,t:factory(method,condition,seed,t),work)
            probabilities = logits.softmax(-1).mean(0)
            tick=time.monotonic()
            readouts = {name:dict(classification=metrics.classification(logits,probabilities,*role),
                                 repair=metrics.repair_diagnostics(logits,probabilities,*role),
                                 prediction_signatures=metrics.signatures(logits,probabilities,*role)) for name,role in roles.items()}
            engine.add(timings,'selected_factual_readouts',dict(wall_seconds=time.monotonic()-tick))
            benchmark = engine.serving_benchmark(session,readouts['VALID']['prediction_signatures'],roles,timings)
            if session.counters != work(session,row['epochs_executed'],row['epochs_executed']+14):
                raise ValueError('All-view M1 complete fit/restore/serving counters differ')
            tick=time.monotonic()
            payload = output/'SELECTED_MEMBER_LOGITS.npz';np.savez_compressed(payload,factual_member_logits=logits.detach().cpu().numpy())
            engine.add(timings,'selected_logit_payload_write',dict(wall_seconds=time.monotonic()-tick))
            engine.check_bounds(spec,output,started)
            result = dict(schema='private-hop-fullfit-complete-v1',complete=True,record_id=spec['record_id'],
                          condition=spec['condition'],seed=spec['seed'],source_manifest_sha256=spec['source_manifest_sha256'],
                          release_sha256=spec['_release_sha256'],providers=providers,split_identity=spec['split_identity'],split_seed=190111,
                          selector=M1_SELECTOR,max_epochs=2000,patience=250,epochs_executed=row['epochs_executed'],
                          selected_epoch=row['selected_epoch'],selected_VALID=readouts['VALID']['classification'],
                          selected_readouts=readouts,selected_prediction_signatures=readouts['VALID']['prediction_signatures'],
                          body=row,counters=dict(session.counters),serving_benchmark=benchmark,timings=timings,
                          predictor_parameters=sum(p.numel() for p in session.bodies[0].parameters()),decoder_parameters=0,
                          complete_saved_member_logits=dict(path=payload.name,sha256=sha(payload),bytes=payload.stat().st_size,
                              keys=['factual_member_logits'],factual_shape=[1,19717,3],contains_labels_or_role_ids=False,server_only=True),
                          train_bundle_sha256=spec['train_bundle']['sha256'],valid_bundle_sha256=spec['valid_bundle']['sha256'],
                          stopped_by=row['stopped_by'],selected_checkpoint_writes=row['selected_checkpoint_writes'],
                          inclusive_seconds=time.monotonic()-started,CPU_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                          CPU_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          peak_CUDA_bytes=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                          TEST_access=False,TEST_scored=False,HPO=False,automatic_retry=False,whole_eighteen_comparison_required=True)
            base.write(output/'COMPLETE.json',result);return result
        except BaseException as error:
            base.write(output/'FAILURE.json',dict(complete=False,record_id=spec['record_id'],error_type=type(error).__name__,
                       error=str(error),inclusive_seconds=time.monotonic()-started,timings=timings,TEST_access=False,automatic_retry=False))
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['science'],required=True)
    parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True)
    base,spec,output=admit(parser.parse_args());result=execute(base,spec,output,time.monotonic())
    print(json.dumps(dict(complete=result['complete'],record_id=result['record_id'],TEST_access=False)))


if __name__ == '__main__':
    main()
