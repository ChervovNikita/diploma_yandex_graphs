"""One root-owned full-WikiCS FiLM shared4 update/restore check; never a fit.

Import is stdlib only. Root supplies actual runtime/data/code admission and an
external finite owner. No arrays or numerical libraries are used until run().
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time


def _load(path):
    spec = importlib.util.spec_from_file_location('_film_readiness_family', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(config_path, output):
    started = time.perf_counter()
    raw = Path(config_path).read_bytes(); cfg = json.loads(raw)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    source = _load(Path(__file__).with_name('run_family.py'))
    family = None
    try:
        family = source.Family(cfg, output, hashlib.sha256(raw).hexdigest(), training_only=True)
        t = family.torch
        def budget():
            family.synchronize()
            limits = cfg['family_limits']
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
            if time.perf_counter()-started > limits['wall_seconds'] or rss > limits['rss_bytes']:
                raise RuntimeError('Finite full-WikiCS readiness wall/RSS cap exceeded')
            if t.cuda.max_memory_allocated(family.device) > limits['cuda_bytes']:
                raise RuntimeError('Finite readiness CUDA cap exceeded')
        def ownership(model, optimizer):
            expected = {id(p) for p in model.parameters() if p.requires_grad}
            actual = [id(p) for group in optimizer.param_groups for p in group['params']]
            if set(actual) != expected or len(actual) != len(set(actual)):
                raise ValueError('Every registered native/factor parameter exactly once')
            model.check_optimizer_ownership(optimizer)
        t.cuda.reset_peak_memory_stats(family.device)
        model, optimizer, streams = family.make(7301, 0, 'baseline', True, 4, 'shared')
        adapter = family._film_adapter
        ownership(model, optimizer); budget()
        model.train(); optimizer.zero_grad()
        logits, hidden = family.training_outputs(model, streams, True)
        if logits.shape != (4, 580, 10) or hidden.shape != (4, 580, 128):
            raise ValueError('Four full-graph native FiLM TRAIN trajectories required')
        targets = family.data.y[family.train_ids]
        loss = family.F.cross_entropy(logits.flatten(0, 1), targets.repeat(4))
        if not t.isfinite(loss).item():
            raise ValueError('Nonfinite readiness own CE')
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not t.isfinite(parameter.grad).all().item():
                raise ValueError('Missing/nonfinite registered gradient: '+name)
        optimizer.step()
        del logits, hidden, loss
        model.eval()
        with t.no_grad():
            served = family.logits(model, streams, True, family.train_ids)
            if served.shape != (4, 580, 10) or not t.isfinite(served).all().item():
                raise ValueError('Finite native full-graph FiLM serving before save')
        del served; budget()
        checkpoint = output/'readiness_unselected.pt'
        t.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'streams': streams, 'step': 1, 'scientifically_selected': False}, checkpoint)
        old_counts = {'TRAIN': sum(c.film_counts_train.get('film_conv_calls', 0) for c in adapter.convs),
                      'eval': sum(c.film_counts_eval.get('film_conv_calls', 0) for c in adapter.convs)}
        del model, optimizer, streams, adapter
        family._fit_model = family._single_adapter = family._film_adapter = None
        state = t.load(checkpoint, map_location='cpu', weights_only=True)
        restored, opt, resumed = family.make(7301, 0, 'baseline', True, 4, 'shared')
        restored.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer']); resumed = state['streams']
        ownership(restored, opt)
        if any(not t.equal(value.detach().cpu(), state['model'][key]) for key, value in restored.state_dict().items()):
            raise ValueError('Strictly restored registered state differs')
        if any(not t.is_tensor(v['step']) or v['step'].item() != 1 for v in opt.state.values()):
            raise ValueError('Actual single AdamW update state required')
        restored.eval()
        with t.no_grad():
            served = family.logits(restored, resumed, True, family.train_ids)
            if served.shape != (4, 580, 10) or not t.isfinite(served).all().item():
                raise ValueError('Finite label-free native FiLM serving after strict restore')
        budget()
        new_calls = sum(c.film_counts_eval.get('film_conv_calls', 0) for c in family._film_adapter.convs)
        if old_counts != {'TRAIN': 8, 'eval': 8} or new_calls != 8:
            raise ValueError('Exact native calls: four update routes and two four-route evals')
        result = {'status': 'ACTUAL_FULL_WIKICS_FiLM_SHARED4_READINESS_COMPLETE',
            'complete': True, 'study_kind': 'native_PyG_FiLM_WikiCS',
            'training_updates': 1, 'scientific_fit_or_selection': False,
            'VALID_payload_read': False, 'TEST_access': False,
            'optimizer_ownership_complete': True, 'finite_registered_gradients': True,
            'strict_model_and_optimizer_restore': True, 'numerical_parity_claim': False,
            'predictive_utility_gate': False, 'parameters': sum(p.numel() for p in restored.parameters()),
            'native_film_conv_completed_calls': 24,
            'source_hashes': family.source_hashes, 'learning_lock': family.learning_lock,
            'smoke_checkpoint': {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                                 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
            'elapsed_seconds': time.perf_counter()-started,
            'cuda_peak_allocated_bytes': t.cuda.max_memory_allocated(family.device),
            'cuda_peak_reserved_bytes': t.cuda.max_memory_reserved(family.device),
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
        source.write_json(output/'READINESS_RESULT.json', result)
        return result
    except BaseException as error:
        source.write_json(output/'FAILURE.json', {'status': 'FAILED_RETAINED', 'complete': False,
            'type': type(error).__name__, 'message': str(error), 'scientific_fit_or_selection': False,
            'TEST_access': False, 'retry_or_rescue': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.config, args.output)


if __name__ == '__main__':
    main()
