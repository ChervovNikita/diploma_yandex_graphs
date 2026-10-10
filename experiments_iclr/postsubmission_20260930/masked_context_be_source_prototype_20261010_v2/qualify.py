"""One finite complete-PubMed TRAIN-only engineering call. Science disabled."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from plan import CONDITIONS, SEEDS, condition, plan, resource_forecast
from native import HERE, PHASE, sha


def inside(path, existing=True):
    value = Path(path).resolve(strict=existing)
    if value == PHASE or not value.is_relative_to(PHASE):
        raise ValueError('Every caller artifact must remain inside this project phase')
    return value


def bind(row):
    path = inside(row['path'])
    if not path.is_file() or sha(path) != row['sha256']:
        raise ValueError('Exact caller artifact bytes required')
    return path


def array_fingerprint(array):
    """SHA256 of dtype/shape header and exact C-order bytes; no labels exposed."""
    header = json.dumps(dict(dtype=array.dtype.str, shape=list(array.shape)),
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(header+b'\n'+array.tobytes(order='C')).hexdigest()


def admit(path, digest):
    path = inside(path)
    if sha(path) != digest: raise ValueError('Exact engineering release digest required')
    spec = json.loads(path.read_text())
    if spec.get('schema') != 'masked-context-PubMed-TRAIN-only-engineering-release-v1':
        raise ValueError('Separate bounded engineering release required')
    required = ('enabled', 'root_engineering_authorized', 'source_review_approved',
                'complete_graph_data_custody_verified', 'external_hard_bound_confirmed',
                'fresh_resource_readiness_confirmed', 'ordinary_runtime_confirmed')
    if not all(spec.get(key) is True for key in required):
        raise ValueError('Engineering admission is inactive until root binds sources/data/runtime/resources')
    if any(spec.get(key) is not False for key in ('science_enabled', 'TEST_access', 'VALID_access', 'automatic_retry')):
        raise ValueError('No science, TEST, VALID or retries in engineering')
    if spec.get('condition') not in CONDITIONS or spec.get('seed') not in SEEDS or spec.get('steps') not in (1,2,3):
        raise ValueError('One declared condition/seed and1–3 complete TRAIN updates required')
    if spec.get('dataset') != 'PubMed' or spec.get('full_graph_shape') != [19717,500]:
        raise ValueError('Complete PubMed19717×500 required; no sampled/induced graph')
    splits = {'official_Planetoid':60, 'native_PolyFormer_60_20_20':11829}
    if spec.get('split_protocol') not in splits or spec.get('TRAIN_count') != splits[spec['split_protocol']]:
        raise ValueError('Explicit unaltered official/native TRAIN projection required')
    if spec['split_protocol'] == 'native_PolyFormer_60_20_20':
        if type(spec.get('split_seed')) is not int or not 0 <= spec['split_seed'] < 2**32:
            raise ValueError('Frozen native split seed required before numerical imports')
        split_identity = 'PolyFormer-class-balanced-60-20-20'
    else:
        if spec.get('split_seed') is not None:
            raise ValueError('Official fixed masks have no random split seed')
        split_identity = 'Planetoid-public-PubMed'
    if spec.get('split_identity') != split_identity:
        raise ValueError('Explicit official/native split identity required')
    custody = json.loads(bind(spec['split_custody']).read_text())
    if custody.get('schema') != 'masked-context-PubMed-TRAIN-role-custody-v1':
        raise ValueError('Exact root-bound TRAIN-only exporter custody required')
    for key in ('split_protocol', 'split_seed', 'split_identity', 'TRAIN_count'):
        if custody.get(key) != spec.get(key): raise ValueError('Split custody disagrees with release: '+key)
    if custody.get('train_bundle_sha256') != spec['train_bundle']['sha256']:
        raise ValueError('Split custody must identify the exact TRAIN bundle')
    if custody.get('feature_normalization') != 'PyG.NormalizeFeatures':
        raise ValueError('Native normalized feature target must be declared')
    bind(custody['exporter_source'])
    if set(custody.get('array_fingerprints', {})) != {'x', 'edge_index', 'train_ids', 'train_y'}:
        raise ValueError('All four TRAIN-only arrays require exact dtype/shape/byte fingerprints')
    if spec.get('maximum_active_seconds') is None or not 0 < spec['maximum_active_seconds'] <= 1800:
        raise ValueError('External finite engineering envelope at most1800 seconds required')
    manifest = HERE/'MANIFEST.json'
    if sha(manifest) != spec.get('source_manifest_sha256'): raise ValueError('Exact new source manifest required')
    for row in json.loads(manifest.read_text())['files']:
        file = (HERE/row['path']).resolve(strict=True)
        if not file.is_relative_to(HERE) or sha(file) != row['sha256'] or file.stat().st_size != row['bytes']:
            raise ValueError('New source payload changed')
    bind(spec['train_bundle'])
    # Bind the existing root-owned supervision/readiness evidence; this source
    # contains no owner or cleanup implementation and does not assert its facts.
    bind(spec['resource_readiness_evidence'])
    bind(spec['existing_owner_release'])
    if condition(spec['condition'])['auxiliary_rewire']:
        bind(spec['auxiliary_graph'])
        rewire = spec.get('frozen_rewire_recipe', {})
        if type(rewire.get('double_edge_swap_seed')) is not int or not 0 <= rewire['double_edge_swap_seed'] < 2**32:
            raise ValueError('Frozen legal double-edge-swap seed required')
        if type(rewire.get('swap_count')) is not int or rewire['swap_count'] <= 0:
            raise ValueError('Frozen positive swap count required')
        if type(rewire.get('rejection_cap')) is not int or rewire['rejection_cap'] < rewire['swap_count']:
            raise ValueError('Frozen rejection cap must cover the requested swaps')
        fingerprint = rewire.get('graph_fingerprint')
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise ValueError('Exact lowercase SHA256 graph array fingerprint required')
        bind(rewire['generator_source'])
        generated = json.loads(bind(rewire['generation_custody']).read_text())
        if generated.get('schema') != 'masked-context-degree-preserving-rewire-custody-v1' or generated.get('algorithm') != 'double_edge_swap':
            raise ValueError('Exact double-edge-swap generation custody required')
        for key in ('double_edge_swap_seed', 'swap_count', 'rejection_cap', 'graph_fingerprint'):
            if generated.get(key) != rewire[key]: raise ValueError('Rewire generation custody disagrees: '+key)
        if generated.get('generator_source_sha256') != rewire['generator_source']['sha256'] or generated.get('auxiliary_bundle_sha256') != spec['auxiliary_graph']['sha256']:
            raise ValueError('Rewire custody must identify generator and auxiliary bundle')
    elif spec.get('auxiliary_graph') is not None:
        raise ValueError('Auxiliary graph only admitted for rewired condition')
    output = inside(spec['output'], existing=False)
    if output.exists(): raise FileExistsError(output)
    spec['_engineering_release_sha256'] = digest
    spec['_split_custody'] = custody
    return spec, output


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def engineering(spec, output):
    started = time.monotonic()
    # Imports and numeric input loading only occur after complete admission.
    import numpy as np
    import torch
    from method import Session
    output.mkdir(parents=True, exist_ok=False)
    session = None
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        with np.load(bind(spec['train_bundle']), allow_pickle=False) as archive:
            if set(archive.files) != {'x', 'edge_index', 'train_ids', 'train_y'}:
                raise ValueError('Only public X/edge_index and projected TRAIN identities/labels allowed')
            arrays = {key:archive[key].copy() for key in archive.files}
        if arrays['x'].dtype != np.float32 or arrays['edge_index'].dtype != np.int64 or arrays['train_ids'].dtype != np.int64 or arrays['train_y'].dtype != np.int64:
            raise ValueError('Exact float32 features and int64 projected graph/roles required')
        if len(arrays['train_ids']) != spec['TRAIN_count'] or arrays['edge_index'].shape != tuple(spec['edge_shape']):
            raise ValueError('Complete frozen TRAIN and graph dimensions required')
        fingerprints = {key:array_fingerprint(value) for key,value in arrays.items()}
        if fingerprints != spec['_split_custody']['array_fingerprints']:
            raise ValueError('Loaded TRAIN-only arrays disagree with frozen exporter custody')
        tensors = {key:torch.from_numpy(value) for key,value in arrays.items()}
        auxiliary = None
        if spec.get('auxiliary_graph') is not None:
            with np.load(bind(spec['auxiliary_graph']), allow_pickle=False) as archive:
                if archive.files != ['edge_index']: raise ValueError('Only frozen auxiliary graph allowed')
                auxiliary_array = archive['edge_index'].copy()
                if auxiliary_array.dtype != np.int64 or array_fingerprint(auxiliary_array) != spec['frozen_rewire_recipe']['graph_fingerprint']:
                    raise ValueError('Loaded auxiliary graph disagrees with frozen dtype/shape/byte fingerprint')
                auxiliary = torch.from_numpy(auxiliary_array)
        device = torch.device(spec['device'])
        if device.type == 'cuda':
            index = 0 if device.index is None else device.index
            torch.cuda.set_device(index); torch.cuda.reset_peak_memory_stats(index)
        providers = {'torch':str(torch.__version__), 'numpy':str(np.__version__)}
        for name in ('torch-geometric', 'scipy', 'torch-scatter', 'torch-sparse'):
            try: providers[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: providers[name] = None
        if providers != spec['frozen_providers']: raise ValueError('Root-bound ordinary numerical providers required')
        session = Session(spec['condition'], spec['seed'], **dict(x=tensors['x'], edges=tensors['edge_index'],
            train_ids=tensors['train_ids'], train_y=tensors['train_y']), device=spec['device'], auxiliary_edges=auxiliary)
        trace = []
        for step in range(spec['steps']):
            if time.monotonic()-started >= spec['maximum_active_seconds']:
                raise TimeoutError('Finite engineering time exhausted; external supervisor must enforce hard termination')
            if device.type == 'cuda': torch.cuda.synchronize()
            tick = time.monotonic()
            value = session.train_step(audit=step == 0 and condition(spec['condition'])['core'])
            if device.type == 'cuda': torch.cuda.synchronize()
            trace.append(dict(step=step+1, update_seconds=time.monotonic()-tick, TRAIN=value))
            write(output/'PROGRESS.json', dict(engineering=True, science_enabled=False, steps=step+1,
                counters=session.counters, inclusive_seconds=time.monotonic()-started))
        modes_before = [module.training for body in session.bodies for module in body.modules()]
        tick = time.monotonic(); probabilities, logits = session.factual_probabilities()
        if modes_before != [module.training for body in session.bodies for module in body.modules()]:
            raise ValueError('Factual serving must preserve every prior module mode')
        if device.type == 'cuda': torch.cuda.synchronize()
        serving_seconds = time.monotonic()-tick
        if probabilities.shape != (19717,3) or not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(-1), torch.ones(19717, device=probabilities.device), atol=1e-5, rtol=1e-5):
            raise ValueError('Complete finite factual probability serving required')
        shapes = dict(probabilities=list(probabilities.shape), member_logits=list(logits.shape))
        del probabilities, logits
        expected = condition(spec['condition'])
        expected_factual = (spec['steps']+1)*expected['factual_views']
        expected_optimizer_steps = spec['steps']*(expected['bodies']+(1 if expected['core'] else 0))
        if (session.counters['updates'] != spec['steps']
            or session.counters['masked_forwards'] != spec['steps']*expected['masked_views']
            or session.counters['factual_forwards'] != expected_factual
            or session.counters['optimizer_steps'] != expected_optimizer_steps):
            raise ValueError('All complete declared TRAIN views must execute')
        peak = None if device.type != 'cuda' else dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        report = dict(schema='masked-context-PubMed-TRAIN-only-engineering-result-v1', complete=True,
            science_enabled=False, VALID_access=False, TEST_access=False, competence_established=False,
            runtime_qualification_scope='One condition and1–3 complete TRAIN updates only; no convergence or selected-checkpoint evidence',
            condition=spec['condition'], seed=spec['seed'], split_protocol=spec['split_protocol'],
            split_seed=spec['split_seed'], split_identity=spec['split_identity'],
            engineering_release_sha256=spec['_engineering_release_sha256'],
            split_custody=spec['split_custody'], array_fingerprints=fingerprints, serving_modes_preserved=True,
            source_manifest_sha256=spec['source_manifest_sha256'], data_SHA=spec['train_bundle']['sha256'],
            providers=providers, trace=trace, gradient_audit=getattr(session, 'audit', None),
            counters=session.counters, shapes=shapes, preprocessing_seconds=session.preparation_seconds,
            serving_seconds=serving_seconds, inclusive_seconds=time.monotonic()-started, timing_scope='includes numerical imports/input loading; outer owner charges admission and process startup separately', peak_CUDA_bytes=peak,
            predictor_parameters=sum(p.numel() for b in session.bodies for p in b.parameters()),
            decoder_parameters=0 if session.decoder is None else sum(p.numel() for p in session.decoder.parameters()),
            no_predicted_or_heldout_scores=True, no_checkpoints_saved=True)
        write(output/'COMPLETE.json', report)
        return report
    except BaseException as error:
        write(output/'FAILURE.json', dict(complete=False, engineering=True, science_enabled=False,
            error_type=type(error).__name__, error=str(error), inclusive_seconds=time.monotonic()-started,
            counters=None if session is None else session.counters, automatic_retry=False, VALID_access=False, TEST_access=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('describe','engineering','science'), default='describe')
    parser.add_argument('--release', type=Path); parser.add_argument('--release-sha256')
    args = parser.parse_args()
    if args.mode == 'science':
        parser.error('Science is unconditionally disabled in this source generation; root must freeze a separately reviewed scientific runner')
    if args.mode == 'describe':
        print(json.dumps(dict(proposal=plan(), forecast=resource_forecast()), indent=2, sort_keys=True)); return
    if args.release is None or args.release_sha256 is None: parser.error('Separate exact root engineering release required')
    spec, output = admit(args.release, args.release_sha256)
    report = engineering(spec, output)
    print(json.dumps(dict(complete=report['complete'], output=str(output), science_enabled=False)))


if __name__ == '__main__': main()
