"""Exactly two discarded full-native SupCon TRAIN updates; no scoring/snapshots.

Root supplies finite process ownership externally. This only checks engineering
eligibility of the sealed source's local/global paths, not method quality or
the complete schedule's checkpoint transition. No development/TEST reader.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_manifest(root, expected):
    require(sha(root/'MANIFEST.json') == expected, 'Frozen manifest bytes changed')
    for row in read(root/'MANIFEST.json')['files']:
        path = root/row['path']
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Frozen source payload changed: '+row['path'])


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    started = time.monotonic()
    pins = read(HERE/'SOURCE_BINDINGS.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--supcon-source', type=Path,
                        default=PHASE/'portable_wikics_supcon_loss_comparison_20261008_v1')
    parser.add_argument('--public-interface', type=Path,
                        default=PHASE/'portable_internal_be_public_interface_20261007_v2')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--polynormer', type=Path, required=True)
    parser.add_argument('--physical-gpu-uuid', choices=pins['normal_runtime']['physical_gpu_inventory'], required=True)
    parser.add_argument('--owned-gpu-memory-cap-bytes', type=int,
                        default=pins['maximum_owned_GPU_cap_bytes'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('supcon_source', 'public_interface', 'train', 'polynormer', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    require(not args.output.exists(), 'Fresh discarded-engineering output required')
    verify_manifest(HERE, args.source_manifest_sha256)
    verify_manifest(args.supcon_source, pins['supcon_manifest_sha256'])
    verify_manifest(args.public_interface, pins['public_manifest_sha256'])
    runtime = pins['normal_runtime']
    require(socket.gethostname() == runtime['hostname'] and Path.cwd() == Path(runtime['repository']),
            'Registered normal77 host/repository only')
    require(str(Path(sys.executable).absolute()) == runtime['python']
            and os.environ.get('PYTHONPATH', '') == runtime['PYTHONPATH'],
            'Existing normal77 executable/providers without allocation overlay')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == args.physical_gpu_uuid,
            'One explicit physical GPU mapped by the root owner')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                        text=True, timeout=10).splitlines()
    require(inventory == runtime['physical_gpu_inventory'], 'Normal77 physical inventory changed')
    cap = args.owned_gpu_memory_cap_bytes
    require(pins['minimum_owned_GPU_cap_bytes'] <= cap <= pins['maximum_owned_GPU_cap_bytes'],
            'Owned allocator cap outside the reviewed full-native range')
    free_mib = int(subprocess.check_output(['nvidia-smi', '--id='+args.physical_gpu_uuid,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    require(free_mib*1024**2 >= cap+pins['minimum_free_headroom_bytes'],
            'Fresh GPU free memory must cover the owned cap and headroom')
    require(sha(args.train) == pins['expected_TRAIN_npz_sha256']
            and sha(args.polynormer) == pins['expected_polynormer_sha256'],
            'Use the existing complete TRAIN and pinned native model bytes')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []; session = facade = None
    try:
        # Match provider preload before the sealed Session resets its trajectory.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        import torch
        providers = {'torch': str(torch.__version__)}
        providers.update({name: importlib.metadata.version(name) for name in
                          ('numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'ogb')})
        require(all(providers[name] == runtime[name] for name in providers), 'Registered normal77 numeric providers')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one mapped CUDA device')
        torch.cuda.set_device(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        require(cap < total_memory, 'Owned cap must fit physical memory')
        torch.cuda.set_per_process_memory_fraction(cap/total_memory, 0)
        torch.cuda.reset_peak_memory_stats(0)
        cli = load('_native_supcon_cli', args.supcon_source/'train.py')
        require(cli.public_dependency(args.public_interface) == args.public_interface, 'Same sealed public dependency')
        public = cli.load('_native_supcon_public', args.public_interface/'portable.py')
        replay = cli.load('_native_supcon_replay', args.supcon_source/'recompute.py')
        data = public._module('_native_supcon_TRAIN_data', 'data.py')
        train = data.load_npz(args.train, ('x', 'edge_index', 'ids', 'y'))
        data.check_projection('wikics', train, None, {'split_index': 0})
        expected = read(args.public_interface/'PUBLIC_DATA_PINS.json')['wikics']['projected_raw_tensor_fingerprints']
        fingerprints = {}
        for name, tensor in {'x':train['x'], 'edge_index':train['edge_index'],
                             'train_ids':train['ids'], 'train_y':train['y']}.items():
            digest = hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            require(list(tensor.shape) == expected[name]['shape'] and str(tensor.dtype) == expected[name]['dtype']
                    and digest == expected[name]['contiguous_raw_bytes_sha256'], 'Same ordered TRAIN pins: '+name)
            fingerprints[name] = dict(expected[name], verified=True)
        require(sha(args.train) == pins['expected_TRAIN_npz_sha256'], 'TRAIN changed during loading')
        specification = cli.identity('supcon_eq2', replay)
        seed = pins['engineering_seed']
        session, facade = cli.make_session(public, replay, specification, seed, 'cuda:0', args.polynormer)
        require(session.config['model'] == pins['full_model_geometry'], 'Original complete native widths/layers required')
        require(session.arm == session.model.arm == 'be_unit_contrastive'
                and session.model.members == 4 and len(session.optimizers) == 1 and session.model.contrastive
                and not session.model.independent, 'Original shared unit BE and one Adam')
        session.replay_prediction_diagnostics = True  # Existing replay diagnostic switch, no extra forward.
        # Audit the same original operator on both devices; neither result replaces the replay panel.
        cpu_panel = torch.linspace(0, 579, steps=512, device='cpu').long()
        device_panel = torch.linspace(0, 579, steps=512, device=session.device).long()
        require(torch.equal(cpu_panel, device_panel.cpu())
                and hashlib.sha256(cpu_panel.numpy().tobytes()).hexdigest() == pins['CPU_panel_raw_sha256'],
                'Original CPU/CUDA512 panel identity')
        expected_aux_labels = train['y'].to(session.device)[device_panel]
        observed_auxiliary_calls = []
        original_alignment = facade.alignment_loss
        def audit_alignment(a, b, labels, task, temperature=.2, identities=None):
            require(tuple(a.shape) == tuple(b.shape) == (4,512,512)
                    and a.requires_grad and b.requires_grad and torch.equal(labels, expected_aux_labels),
                    'Actual facade sees both live native512-feature views and original panel labels')
            observed_auxiliary_calls.append(dict(shape=list(a.shape), both_views_differentiated=True,
                                                 panel_labels_match_original_operator=True))
            return original_alignment(a, b, labels, task, temperature, identities)
        facade.alignment_loss = audit_alignment
        for stage in ('local', 'global'):
            global_mode = stage == 'global'
            if global_mode:
                session.model.set_global(True)  # Discarded path coverage; no checkpoint restore/selection.
            require(all(body.body._global == global_mode for body in session.model.models), 'Native stage flags')
            batches = data.batches('wikics', train, session.config['training'], 101 if global_mode else 1, seed, session.device)
            batch, labels = next(batches)
            require(next(batches, None) is None and len(labels) == 580, 'Exactly one complete original TRAIN batch')
            before = dict(session.execution_totals)
            torch.cuda.synchronize(0); update_started = time.monotonic()
            result = session.train_step(batch, labels)
            torch.cuda.synchronize(0)
            delta = {k:v-before[k] for k,v in session.execution_totals.items()}
            require(delta == dict(shadow_member_forwards=8, replay_member_forwards=8, output_cotangent_collections=1,
                member_reverse_collections=8, optimizer_bank_updates=1, exact_member_RNG_endpoint_checks=1),
                'One complete original16-forward/8-VJP/1-Adam update')
            gradients = [p.grad for p in session.model.parameters() if p.grad is not None]
            require(gradients and all(torch.isfinite(g).all() for g in gradients), 'Actual finite accumulated parameter gradients')
            max_grad = float(torch.stack([g.abs().max() for g in gradients]).max())
            require(max_grad > 0 and all(torch.isfinite(x).all() for x in result.values()), 'Nonzero finite gradient and loss outputs')
            session.core['selection'].finite_state(session.model, session.optimizers)
            require(session.last_replay_diagnostics['exact_member_RNG_endpoint']
                    and session.last_replay_diagnostics['parameter_versions_unchanged_before_Adam'], 'Actual replay RNG/update order checks')
            rows.append(dict(stage=stage, train_objects=580, full_graph_nodes=11701, full_graph_edges=442907,
                own_views=2, discarded_updates=1, losses={k:float(v) for k,v in result.items()},
                execution_delta=delta, parameter_tensors_with_grad=len(gradients), finite_parameter_gradients=True,
                maximum_absolute_gradient=max_grad, replay_diagnostics=session.last_replay_diagnostics,
                peak_CUDA_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                peak_CUDA_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
                update_seconds=time.monotonic()-update_started))
            write(args.output/'PROGRESS.json', dict(engineering_only=True, rows=rows, predictive_scoring=False))
            del gradients, result, batch, labels
        cost = dict(session.public_cost)
        require(session.steps == 2 and facade.alignment_source_calls == 2 and facade.residual_source_calls == 0
                and len(observed_auxiliary_calls) == 2, 'Exactly two SupCon-only discarded updates')
        require(cost['training_member_calls_attempted'] == cost['training_member_calls_returned'] == 32
                and cost['development_member_calls_attempted'] == cost['development_member_calls_returned'] == 0
                and cost['update_calls_attempted'] == cost['update_calls_returned'] == 2, 'All actual calls charged; no inference/evaluation calls')
        require(torch.cuda.max_memory_reserved(0) <= cap, 'Measured reserved CUDA memory respects owned cap')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        record = dict(schema='WikiCS-SupCon-two-update-native-qualification-v1', complete=True,
            engineering_only=True, condition='supcon_eq2', seed=seed, discarded_updates=2,
            source_manifest_sha256=args.source_manifest_sha256, supcon_manifest_sha256=pins['supcon_manifest_sha256'],
            public_manifest_sha256=pins['public_manifest_sha256'], TRAIN_npz_sha256=sha(args.train),
            polynormer_sha256=sha(args.polynormer), providers=providers, physical_gpu_uuid=args.physical_gpu_uuid,
            python=sys.executable, hostname=socket.gethostname(), full_model_geometry=session.config['model'],
            owned_GPU_memory_cap_bytes=cap, initial_free_GPU_bytes=free_mib*1024**2,
            ordered_TRAIN_fingerprints=fingerprints, original_CPU_CUDA_panel_identity=True,
            CPU_panel_raw_sha256=pins['CPU_panel_raw_sha256'], actual_auxiliary_calls=observed_auxiliary_calls,
            execution_totals=dict(session.execution_totals), public_API_counters=cost, rows=rows,
            no_development_or_TEST_load=True, predictive_scoring=False, snapshots_written=False,
            local_checkpoint_transition_checked=False, direct_global_flag_for_discarded_path_coverage=True,
            scientific_fit=False, method_quality_established=False, full_training_schedule_qualified=False,
            reuse_execution_or_whole_family_criteria_changed=False, automatic_retry=False,
            peak_CUDA_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
            peak_CUDA_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
            CPU_user_seconds=usage.ru_utime, CPU_system_seconds=usage.ru_stime,
            peak_RSS_bytes=int(usage.ru_maxrss*1024), inclusive_seconds=time.monotonic()-started)
        write(args.output/'QUALIFIED.json', record)
    except BaseException as error:
        write(args.output/'FAILURE.json', dict(complete=False, engineering_only=True, rows=rows,
            error_type=type(error).__name__, error=str(error), automatic_retry=False,
            source_manifest_sha256=args.source_manifest_sha256, discarded_steps=session.steps if session else 0,
            execution_totals=dict(session.execution_totals) if session else None,
            public_API_counters=dict(session.public_cost) if session else None,
            no_development_or_TEST_load=True, predictive_scoring=False, snapshots_written=False,
            inclusive_seconds=time.monotonic()-started))
        raise
    print(json.dumps(dict(engineering_complete=True, discarded_updates=2, output=str(args.output))))


if __name__ == '__main__':
    main()
