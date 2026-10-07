"""Disabled VJP engineering only:8 fullgraph updates/128 member forwards and reload."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
from run_cell import CONDITIONS, admit, bound, inside, load, make_session, require, sha
from recompute import MODE


def tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item, torch)


def cpu_rng(streams, torch):
    require(all(value.device.type == 'cpu' and value.dtype == torch.uint8
                for stream in streams for value in stream.values()), 'CPU byte member RNG states')


def qualify_mode(session, facade, condition, global_mode, data, train, valid, output):
    torch = session.torch
    session.model.set_global(global_mode)
    torch.cuda.reset_peak_memory_stats()
    before = session.steps
    costs_before = dict(session.execution_totals)
    batches = data.train_batches('wikics', train, 101 if global_mode else 1, session.seed, session.device)
    batch, labels = next(batches)
    require(len(labels) == 580 and tuple(batch['x'].shape) == (11701, 300)
            and tuple(batch['edge_index'].shape) == (2, 442907), 'Complete original fullgraph TRAIN update')
    result = session.train_step(batch, labels)
    require(next(batches, None) is None and session.steps == before + 1
            and all(torch.isfinite(value).all() for value in result.values()), 'One original finite two-view update')
    charged = {key: value - costs_before[key] for key, value in session.execution_totals.items()}
    require(charged == dict(shadow_member_forwards=8, replay_member_forwards=8,
        output_cotangent_collections=1, member_reverse_collections=8, optimizer_bank_updates=1,
        exact_member_RNG_endpoint_checks=1), 'Complete16forwards/8VJPs/oneAdam after all replays')
    path = output / (condition + ('_global.pt' if global_mode else '_local.pt'))
    session.save_training_state(path, epoch=0)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(all(value.device.type == 'cpu' for value in tensors(saved, torch)), 'Original CPU snapshot tensors')
    cpu_rng(saved['streams'], torch)
    require(saved['cpu_rng'].device.type == saved['cuda_rng'].device.type == 'cpu'
            and saved['cpu_rng'].dtype == saved['cuda_rng'].dtype == torch.uint8, 'CPU byte host/device RNG snapshot')
    require(session.restore_training_state(path) == 0 and session.steps == before + 1, 'Original restore primitive')
    cpu_rng(session.streams, torch)
    require(all(parameter.device == session.device for parameter in session.model.parameters()), 'Restored CUDA parameters')
    moments = [value for optimizer in session.optimizers for state in optimizer.state.values()
               for key, value in state.items() if key in ('exp_avg', 'exp_avg_sq')]
    require(moments and all(value.device == session.device for value in moments), 'Restored CUDA Adam moments; scalar step may remain CPU')
    require(all(body.body._global == global_mode for body in session.model.models), 'Restored local/global mode')
    session.model.eval(); count = batches_served = 0
    with torch.no_grad():
        for development_batch, development_y in data.validation_batches('wikics', train, valid, session.device):
            logits, representations = session.forward(development_batch); pooled = session.serving(logits)
            session.core['selection'].finite_predictions(logits, pooled)
            require(torch.isfinite(representations).all() and tuple(logits.shape) == (4, 5274, 10)
                    and tuple(pooled.shape) == (5274, 10), 'Complete finite four-member development serving')
            count += len(development_y); batches_served += 1
    require(count == 5274 and batches_served == 1, 'Complete original development population')
    torch.cuda.synchronize()
    return dict(condition=condition, global_mode=global_mode, updates=1, own_views=2, train_objects=580,
                development_objects=count, original_CPU_reload_passed=True, CUDA_Adam_moments_passed=True,
                CPU_byte_RNG_passed=True, snapshot_sha256=sha(path), completed_source_steps=session.steps,
                execution_mode=MODE, execution_accounting_this_update=charged,
                replay_diagnostics=session.last_replay_diagnostics,
                active_original_loss_calls=dict(alignment=facade.alignment_source_calls, residual=facade.residual_source_calls),
                peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(), peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True); parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args(); cfg, pins, public_root = admit(args.release, args.release_sha256, engineering=True)
    output = inside(cfg['output']); output.mkdir(parents=True, exist_ok=False); started = time.monotonic(); rows = []
    public = load('_unit_mechanism_qualifier_public', public_root / 'portable.py')
    sys.modules['portable'] = public
    data = load('_unit_mechanism_qualifier_data', public_root / 'data_interface.py')
    # Same provider preload order as the unchanged full driver, before Session seeding.
    from ogb.graphproppred import Evaluator as GraphEvaluator
    from ogb.linkproppred import Evaluator as LinkEvaluator
    train, valid, origin = data.load_train_valid('wikics', bound(cfg['train']), bound(cfg['development']))
    try:
        for condition, weights in CONDITIONS.items():
            session, facade = make_session(public, condition, cfg['seed'], 'cuda:0', bound(pins['polynormer']))
            session.replay_prediction_diagnostics = True  # Recorded differences have no rejection threshold.
            torch = session.torch; torch.cuda.reset_peak_memory_stats()
            for global_mode in (False, True):
                rows.append(qualify_mode(session, facade, condition, global_mode, data, train, valid, output))
                (output / 'PROGRESS.json').write_text(json.dumps(dict(rows=rows, engineering_only=True), indent=2) + '\n')
            require(session.steps == 2 and facade.alignment_source_calls == (2 if weights[0] else 0)
                    and facade.residual_source_calls == (2 if weights[1] else 0), 'Fixed active original component calls')
            del session, facade; gc.collect(); torch.cuda.empty_cache()
        require(len(rows) == 8 and {row['condition'] for row in rows} == set(CONDITIONS)
            and all(row['replay_diagnostics']['exact_member_RNG_endpoint'] for row in rows), 'All four conditions and both native stages qualified')
        require(max(row['peak_reserved_GPU_bytes'] for row in rows) < cfg['owned_GPU_memory_cap_bytes'], 'Actual reserved peak within released recomputation cap')
        result = dict(schema='WikiCS-unit-mechanism-representative-qualification-v2', complete=True,
            execution_mode=MODE, bitwise_author_parity_claimed=False, discarded_training_member_forwards=128,
            engineering_only=True, numerical_equivalence_claimed=False, discarded_training=True, scientific_fit=False,
            TEST_access=False, automatic_retry=False, rows=rows, source_manifest_sha256=cfg['source_manifest_sha256'],
            facade_sha256=sha(Path(__file__).with_name('run_cell.py')), qualifier_sha256=sha(__file__),
            release_sha256=args.release_sha256, physical_gpu_uuid=cfg['physical_gpu_uuid'], data=origin,
            seconds=time.monotonic() - started)
        (output / 'QUALIFIED.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    except BaseException as error:
        (output / 'FAILURE.json').write_text(json.dumps(dict(complete=False, engineering_only=True, rows=rows,
            error_type=type(error).__name__, error=str(error), automatic_retry=False, seconds=time.monotonic() - started), indent=2) + '\n')
        raise
    print(json.dumps(dict(engineering_complete=True, output=str(output), discarded_updates=8)))


if __name__ == '__main__':
    main()
