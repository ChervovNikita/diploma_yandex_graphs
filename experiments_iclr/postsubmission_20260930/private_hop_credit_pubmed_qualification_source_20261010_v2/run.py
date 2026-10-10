"""Five finite TRAIN-only one-update qualifiers. Disabled until exact root review."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
REPO = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
SERVER_PHASE = REPO/'experiments_iclr/postsubmission_20260930'
GPU = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
CONDITIONS = ('private_missinghop', 'full_aux', 'common_nonfull', 'allblock_missinghop', 'factor1_allview')
RECORD = 'seed9101__private_hop_all5_TRAIN_qualification'
ATOL = RTOL = 2e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def route(owner=False):
    if socket.gethostname() != 'anogena-2-0' or PHASE != SERVER_PHASE or Path.cwd().resolve() != REPO:
        raise ValueError('Exact allocation hostname/phase/cwd required before metadata')
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines() != [GPU]:
        raise ValueError('Exact sole GPU required')
    if not owner and os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('Existing owner must expose the exact sole GPU')


def admit(args):
    route()
    path = args.release.resolve(strict=True)
    if not path.is_relative_to(HERE) or sha(path) != args.release_sha256:
        raise ValueError('Exact root-confined release required')
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'private-hop-TRAIN-qualification-release-v1' or spec.get('purpose') != 'engineering':
        raise ValueError('This entry only qualifies TRAIN; no science mode')
    for key in ('enabled','root_authorized','source_review_approved','ordinary_runtime_confirmed','external_hard_bound_confirmed','fresh_resource_readiness_confirmed'):
        if spec.get(key) is not True:
            raise ValueError('Disabled numerical admission: '+key)
    if (spec.get('conditions') != list(CONDITIONS) or spec.get('seed') != 9101 or
            spec.get('record_id') != RECORD or spec.get('updates_per_condition') != 1):
        raise ValueError('Exactly five fresh one-update qualifiers required')
    if any(spec.get(key) is not False for key in ('VALID_access','TEST_access','science_enabled','automatic_retry','resume','HPO','paper_score_recalculation')):
        raise ValueError('TRAIN-only engineering scope required')
    if any(key in spec for key in ('valid_bundle','validation_custody','test_bundle','full_y','test_ids','test_y')):
        raise ValueError('No VALID/TEST/full-label input surface')
    manifest = HERE/'SOURCE_MANIFEST.json'
    if sha(manifest) != spec['source_manifest_sha256']:
        raise ValueError('Exact reviewed source manifest required')
    for row in json.loads(manifest.read_text())['files']:
        target = (HERE/row['path']).resolve(strict=True)
        if not target.is_relative_to(HERE) or sha(target) != row['sha256']:
            raise ValueError('Qualification source changed')
    bindings = json.loads((HERE/'SOURCE_BINDINGS.json').read_text())
    for row in bindings['files']:
        target = (PHASE/row['path']).resolve(strict=True)
        if not target.is_relative_to(PHASE) or sha(target) != row['sha256']:
            raise ValueError('Reused source changed')
    base = load('_private_hop_existing_reference_entry', PHASE/'pubmed_factor1_controls_source_20261010_v1/run.py')
    review = json.loads(base.bind(spec['root_review']).read_text())
    if review.get('approved') is not True or review.get('source_manifest_sha256') != spec['source_manifest_sha256']:
        raise ValueError('Exact root source review required')
    data = json.loads((HERE/'DATA_AND_RUNTIME.json').read_text())
    for key in ('train_bundle','split_custody','runtime','frozen_providers'):
        if spec[key] != data[key]:
            raise ValueError('Current native data/runtime binding required')
    spec['_train_custody'] = json.loads(base.bind(spec['split_custody']).read_text())
    base.bind(spec['train_bundle'])
    if spec['limits'] != base.ENGINEERING_LIMITS:
        raise ValueError('Reuse the existing finite engineering limits')
    owner = json.loads(base.bind(spec['external_owner_release']).read_text())
    if (owner.get('enabled') is not True or owner.get('record_id') != RECORD or
            owner.get('limits') != spec['limits'] or owner.get('automatic_retry') is not False):
        raise ValueError('Reviewed existing finite owner required')
    for key in ('separate_process_group','direct_wait_required','resource_caps_enforced','output_and_log_caps_enforced'):
        if owner.get(key) is not True:
            raise ValueError('Finite owner fact missing: '+key)
    base.bind(owner['owner_source']); base.bind(owner['owner_review'])
    python = Path(os.path.abspath(spec['runtime']['python']['path']))
    if (not python.is_relative_to(PHASE) or sha(python) != spec['runtime']['python']['sha256'] or
            python.resolve() != Path(sys.executable).resolve() or os.environ.get('PYTHONPATH','') != spec['runtime']['PYTHONPATH']):
        raise ValueError('Exact current qualified native Python/PYTHONPATH required')
    output = Path(spec['output']).resolve()
    if not output.is_relative_to(HERE) or output.exists():
        raise ValueError('Fresh owner-confined output required')
    spec['_release_sha256'] = args.release_sha256
    return base, spec, output


def streams(session):
    return {key:{name:value.clone() if value is not None else None for name,value in state.items()}
            for key,state in session.streams.states.items()}


def shared_factual_audit(session, gathered, initial_streams):
    """Replay only factual shared derivatives at the unchanged parameter state.

    Four audit forwards/VJPs are charged separately. They are not a second
    optimizer update and do not advance the actual factual/aux stream history.
    """
    credit = session._private_hop_credit
    torch = credit.torch
    if credit.condition != 'private_missinghop' or credit.members != 4:
        raise ValueError('This new routing check targets the private-hop shared field')
    shared = tuple(p for p,private in zip(credit.parameters,credit.is_private) if not private)
    actual = tuple(g for g,private in zip(gathered[0],credit.is_private) if not private)
    expected = [torch.zeros_like(p) for p in shared]
    after = streams(session)
    try:
        session.streams.states = initial_streams
        for member in range(4):
            with credit._guard_plain_state():
                with session.factors.member_context(credit.body,member):
                    with session.streams.use(member,'factual'):
                        logits = credit.body(session.factual)
            ce = torch.nn.functional.cross_entropy(logits[session.train_ids],session.train_y)
            grad = torch.autograd.grad(ce,shared,create_graph=False,retain_graph=False,allow_unused=False)
            for target,value in zip(expected,grad):
                target.add_(value,alpha=.25)
        maximum = max(float((a-e).abs().max()) for a,e in zip(actual,expected))
        for a,e in zip(actual,expected):
            if not torch.isfinite(a).all() or not torch.isfinite(e).all() or not torch.allclose(a,e,atol=ATOL,rtol=RTOL):
                raise ValueError('Private-hop shared field differs from same-state factual mean CE')
        return dict(passed=True,shared_parameter_tensors=len(shared),max_abs_gradient_difference=maximum,
                    atol=ATOL,rtol=RTOL,numerical_verification_only=True,
                    audit_fullgraph_forwards=4,audit_shared_gradient_VJPs=4,optimizer_steps=0,
                    actual_post_gather_streams_restored=True,auxiliary_shared_credit_excluded=True)
    finally:
        session.streams.states = after


def execute(base, spec, output, started):
    output.mkdir(parents=True,exist_ok=False)
    timings, rows = {}, []
    session = replay = None
    try:
        with base.reference_surface() as (engine,metrics,_unused_native_factor_factory):
            tick = time.monotonic()
            np,torch,providers,tensor,roles = engine.runtime_and_data(spec,valid=False)
            engine.add(timings,'native_runtime_and_exact_TRAIN_input',dict(wall_seconds=time.monotonic()-tick))
            if set(roles) != {'TRAIN'}:
                raise ValueError('Only exact TRAIN role admitted')
            method = sys.modules['source'].load_v3(numerical=True)['method']
            hop = load('_qualified_existing_private_hop',PHASE/'private_hop_credit_source_20261010_v1/adapter.py')
            hooks = load('_qualified_existing_private_hop_hooks',PHASE/'private_hop_credit_pubmed_entry_source_20261010_v1/hooks.py')
            for condition in CONDITIONS:
                engine.check_bounds(spec,output,started)
                session,clock = engine.timed(torch,lambda:hooks.fresh(method,hop,condition,9101,tensor,numerical_admitted=True))
                engine.add(timings,'construction_preprocessing_factor_Adam',clock)
                routing = None
                if condition == 'private_missinghop':
                    incoming = streams(session)
                    with hooks._admitted(hop,True):
                        gathered,clock = engine.timed(torch,session._private_hop_credit.gather)
                    engine.add(timings,'candidate_actual_complete_gradient_gather',clock)
                    routing,clock = engine.timed(torch,lambda:shared_factual_audit(session,gathered,incoming))
                    engine.add(timings,'candidate_extra_shared_factual_routing_audit',clock)
                    with hooks._admitted(hop,True):
                        training,clock = engine.timed(torch,lambda:session._private_hop_credit.commit(gathered))
                    engine.add(timings,'candidate_one_native_Adam',clock)
                    del gathered,incoming
                else:
                    training,clock = engine.timed(torch,lambda:session.train_step(audit=False))
                    engine.add(timings,'other_whole_TRAIN_updates',clock)
                (probabilities,logits),clock = engine.timed(torch,session.factual_probabilities)
                engine.add(timings,'post_update_fullgraph_factual',clock)
                signature = metrics.signatures(logits,probabilities,*roles['TRAIN'])
                counters = dict(session.counters)
                if counters != hooks.expected_work(session,1,evaluations=1):
                    raise ValueError('Exact one-update/full-factual counters required')
                state = {key:value.detach().cpu().clone() for key,value in session.bodies[0].state_dict().items()}
                replay,clock = engine.timed(torch,lambda:hooks.fresh(method,hop,condition,9101,tensor,numerical_admitted=True))
                engine.add(timings,'fresh_replay_construction_preprocessing_factor_Adam',clock)
                _,clock = engine.timed(torch,lambda:replay.bodies[0].load_state_dict(state,strict=True))
                engine.add(timings,'in_memory_post_update_weight_restore',clock)
                (again,bank),clock = engine.timed(torch,replay.factual_probabilities)
                engine.add(timings,'fresh_replay_fullgraph_factual',clock)
                if (metrics.signatures(bank,again,*roles['TRAIN']) != signature or
                        not torch.allclose(bank,logits,atol=ATOL,rtol=RTOL) or
                        not torch.allclose(again,probabilities,atol=ATOL,rtol=RTOL)):
                    raise ValueError('Complete factual replay/logits/TRAIN identities changed')
                if replay.counters != hooks.expected_work(replay,0,evaluations=1):
                    raise ValueError('Replay must have zero optimizer transitions')
                row = dict(condition=condition,seed=9101,updates=1,training=training,counters=counters,
                           replay_counters=dict(replay.counters),factual_replay_verified=True,
                           replay_max_abs_logits=float((bank-logits).abs().max()),TRAIN_prediction_signatures=signature,
                           routing_check=routing,VALID_access=False,TEST_access=False,quality_evidence=False)
                base.write(output/(condition+'.json'),row);rows.append(row)
                session = replay = None
                del state,probabilities,logits,again,bank
                gc.collect();torch.cuda.empty_cache()
            engine.check_bounds(spec,output,started)
            result = dict(schema='private-hop-all5-TRAIN-qualification-complete-v1',complete=True,
                          record_id=RECORD,conditions=list(CONDITIONS),seed=9101,rows=rows,
                          source_manifest_sha256=spec['source_manifest_sha256'],release_sha256=spec['_release_sha256'],
                          providers=providers,graph_shape=[19717,500],edge_shape=[2,88648],TRAIN_count=11829,
                          TRAIN_class_counts=[2461,4643,4725],original_tensors_and_targets_bound=True,
                          genuine_optimizer_updates=5,model_constructions=10,preprocessing_banks=10,
                          Adam_constructions=12,
                          training_route_forwards=32,training_gradient_VJPs=32,
                          extra_routing_audit_forwards=4,extra_routing_audit_VJPs=4,
                          factual_and_fresh_replay_forwards=34,total_route_forwards=70,total_gradient_VJPs=36,
                          shared_data_gradient_routing_verified=True,all_factual_replays_verified=True,
                          shared_data_gradient_routing_condition='private_missinghop',
                          timings=timings,inclusive_seconds=time.monotonic()-started,
                          torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                          torch_peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                          VALID_access=False,TEST_access=False,science_enabled=False,quality_evidence=False,
                          standalone_native_or_M1_learning_requalification=False,automatic_retry=False)
            base.write(output/'COMPLETE.json',result)
            return result
    except BaseException as error:
        base.write(output/'FAILURE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
                   completed_conditions=[row['condition'] for row in rows],timings=timings,
                   inclusive_seconds=time.monotonic()-started,VALID_access=False,TEST_access=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['engineering'],required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--release-sha256',required=True)
    args = parser.parse_args()
    base,spec,output = admit(args)
    result = execute(base,spec,output,time.monotonic())
    print(json.dumps(dict(complete=result['complete'],record_id=RECORD,VALID_access=False,TEST_access=False)))


if __name__ == '__main__':
    main()
