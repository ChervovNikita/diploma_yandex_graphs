"""TRAIN-only, full-shape engineering check; never a scientific fit or donor.

One fresh local/global component is checked per process. No VALIDATION or TEST
targets, scores, selection, checkpoint publication or fitted logits are used.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PHASE = ROOT.parent
REPO = PHASE.parent.parent
UUID = 'GPU-44039938-fd82-41d2-fefd-de71514e2fac'
BANK_MANIFEST = 'eb34fcb8bc406219007ee39ee9789f1de9bc56acf3cb16164aeab95b4493752f'
BANK_PROTOCOL = '55f4146b450f2e04c518e07df9a118bb21f7dc65e18dc659eb34e6dbe8edfb46'
REFERENCE_MANIFEST = '54695f4ed086ed843835a7b040e8ac82a7e211cf17e39ae06b226a23c60b5c05'
REFERENCE_PROTOCOL = '56c96715bc84eb23a9da8cb982f23a1147db18e16787df5690b7e7163b464708'


def verify(record):
    path = PHASE / record['path']
    assert path.resolve().is_relative_to(PHASE) and not path.is_symlink()
    assert path.stat().st_size == record['bytes']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256']
    return path


def tree_difference(torch, left, right):
    """Exact equality is reported separately from an FP32 tolerance diagnostic."""
    result = dict(exact=True, close=True, maximum_absolute_error=0.0, tensors=0)
    def visit(a, b):
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
            result['tensors'] += 1
            result['exact'] &= torch.equal(a, b)
            if a.is_floating_point() and a.numel():
                delta = float((a.to(torch.float64) - b.to(torch.float64)).abs().max())
                result['maximum_absolute_error'] = max(result['maximum_absolute_error'], delta)
                result['close'] &= torch.allclose(a, b, rtol=5e-5, atol=5e-6)
            else:
                result['close'] &= torch.equal(a, b)
        elif isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b)
            for key in a: visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b): visit(x, y)
        else:
            assert a == b
    visit(left, right)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('bank', 'reference'), required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--stage', choices=('local', 'global'), required=True)
    args = parser.parse_args()
    assert Path.cwd() == REPO
    assert subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.splitlines() == [UUID]
    source = PHASE / ('accuracy_first_graph_view_source_preparation_20261004_v2' if args.family == 'bank'
                     else 'accuracy_first_graph_view_reference_source_preparation_20261004_v1')
    sys.path.insert(0, str(source))
    from runtime import load_runtime
    from driver import train_step
    if args.family == 'bank':
        from bank import build_bank, set_stage, snapshot, restore, rng_state
        from driver import eval_logits as evaluate, edge_tensors
        rt = load_runtime(execute=True, manifest_sha256=BANK_MANIFEST, protocol_sha256=BANK_PROTOCOL)
        from views import TrainRole, construct_views, native_edges
    else:
        from native_reference import build, snapshot, restore, evaluate
        rt = load_runtime(execute=True, manifest_sha256=REFERENCE_MANIFEST, protocol_sha256=REFERENCE_PROTOCOL)
        TrainRole, construct_views, native_edges = rt.views.TrainRole, rt.views.construct_views, rt.views.native_edges
    torch, np = rt.torch, rt.numpy
    torch.set_num_threads(1)
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    bindings = json.loads((ROOT / 'INPUT_BINDINGS.json').read_text())
    data = json.loads(verify(bindings['manifest']).read_text())
    assert data['public_graph'] == bindings['public_graph'] and data['train_labels'] == bindings['train_labels']
    with np.load(verify(data['public_graph']), allow_pickle=False) as archive:
        assert set(archive.files) == {'features', 'edge_index', 'train_mask', 'val_mask', 'test_mask'}
        x_cpu = archive['features'].copy()
        raw = archive['edge_index'].copy()
        ids = tuple(int(i) for i in np.flatnonzero(archive['train_mask'][:, 0]))
        val_ids = tuple(int(i) for i in np.flatnonzero(archive['val_mask'][:, 0]))
        test_ids = tuple(int(i) for i in np.flatnonzero(archive['test_mask'][:, 0]))
    with np.load(verify(data['train_labels']['0']), allow_pickle=False) as archive:
        assert set(archive.files) == {'ids', 'labels'} and np.array_equal(archive['ids'], ids)
        labels = tuple(int(y) for y in archive['labels'])
    assert x_cpu.shape == (24492, 300) and x_cpu.dtype == np.float32 and np.isfinite(x_cpu).all()
    role = TrainRole(24492, 0, ids, labels, val_ids, test_ids)
    bundle = construct_views(role, native_edges(24492, ((int(a), int(b)) for a, b in raw.T)), 17)
    coverage_path = PHASE / 'graph_view_actual_TRAIN_mask_coverage_execution_root_20261004_v1/split0_COVERAGE.json'
    saved_coverage = json.loads(coverage_path.read_text())
    assert saved_coverage == dict(bundle['coverage'], coverage_origin='official_TRAIN',
        scientific_freeze_eligible=bundle['coverage']['coverage_eligible'])
    assert saved_coverage['scientific_freeze_eligible']
    x = torch.from_numpy(x_cpu).to(device)
    if args.family == 'bank':
        graphs = edge_tensors(rt, bundle, device)
        torch.cuda.reset_peak_memory_stats()
        model, optimizer, _ = build_bank(rt, args.condition, 17, device)
        set_stage(model, args.stage == 'global')
        def step(update):
            return train_step(rt, model, optimizer, x, graphs, ids, labels, args.condition, 17, update)
        passes = 8
        cpu_copy = __import__('bank').cpu_tree
    else:
        graphs = {name: torch.tensor(value, dtype=torch.long, device=device).t().contiguous()
                  for name, value in dict(bundle['views'], native=bundle['native']).items()}
        torch.cuda.reset_peak_memory_stats()
        member = 0 if args.condition == 'native_member' else None
        model, optimizer, _ = build(rt, args.condition, 0, member, device)
        model._global = args.stage == 'global'
        def step(update):
            return train_step(rt, model, optimizer, x, graphs, ids, labels, args.condition, update)
        passes = 1 if args.condition == 'native_member' else 3
        cpu_copy = rt.helpers.cpu_tree
        rng_state = rt.helpers.rng_state
    # This is a stage-specific fresh-initialization check, not a 200-update handoff.
    update = 1 if args.stage == 'local' else 201
    binding = dict(engineering_only=True, condition=args.condition, stage=args.stage)
    selection = dict(global_=args.stage == 'global', actual_update=update)
    selection['global'] = selection.pop('global_')
    torch.cuda.synchronize(); started = time.perf_counter()
    _ = step(update)
    torch.cuda.synchronize(); step_seconds = time.perf_counter() - started
    torch.cuda.synchronize(); started = time.perf_counter()
    expected_logits = evaluate(rt, model, x, graphs['native']).detach().cpu()
    torch.cuda.synchronize(); eval_seconds = time.perf_counter() - started
    started = time.perf_counter()
    image = snapshot(rt, model, optimizer, device, selection, binding)
    snapshot_seconds = time.perf_counter() - started
    started = time.perf_counter()
    restore(rt, model, optimizer, device, image, binding)
    actual_logits = evaluate(rt, model, x, graphs['native']).detach().cpu()
    torch.cuda.synchronize(); restore_and_eval_seconds = time.perf_counter() - started
    function_replay = tree_difference(torch, expected_logits, actual_logits)
    restore(rt, model, optimizer, device, image, binding)
    _ = step(update + 1)
    first = cpu_copy(rt, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            gradients={n: p.grad for n, p in model.named_parameters()}, rng=rng_state(rt, device)))
    restore(rt, model, optimizer, device, image, binding)
    _ = step(update + 1)
    second = cpu_copy(rt, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             gradients={n: p.grad for n, p in model.named_parameters()}, rng=rng_state(rt, device)))
    next_update_replay = tree_difference(torch, first, second)
    torch.cuda.synchronize()
    result = dict(UTC=datetime.now(timezone.utc).isoformat(), family=args.family, condition=args.condition,
        stage=args.stage, graph_shape=list(x.shape), source_manifest_sha256=rt.manifest_sha256,
        protocol_sha256=rt.protocol_sha256, selected_device='cuda:0', GPU_UUID=UUID,
        parameters=sum(p.numel() for p in model.parameters()), step_seconds=step_seconds,
        evaluation_seconds=eval_seconds, snapshot_seconds=snapshot_seconds,
        restore_and_evaluation_seconds=restore_and_eval_seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        training_updates_checked=3, training_forwards_and_backwards=passes * 3,
        native_evaluation_forwards=8 if args.family == 'bank' else 2,
        function_replay=function_replay, next_update_replay=next_update_replay,
        exact_function_requirement_met=function_replay['exact'],
        implementation_close=function_replay['close'] and next_update_replay['close'],
        independent_scientific_result=False, predictive_values_reported=False,
        VALIDATION_or_TEST_targets_read=False, scientific_training_updates=0,
        checkpoint_or_logits_saved=False, eligible_as_donor=False,
        component_stage_fresh_initialization=True, co_resident_with_original_Amazon_queue=True,
        exact_whole_schedule_or_competence_qualification=False)
    path = ROOT / (args.family + '_' + args.condition + '_' + args.stage + '.json')
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), file=sys.stderr)
        raise
