"""Tiny finite native-head AMP skip fixture; every optimizer map is denied.

Root schedules this once. No graph, label authority, fit, checkpoint or score
is read. This qualifies the real GradScaler/basic-session accounting path,
not native geometry, predictive quality or resource admission.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if (socket.gethostname() != 'anogena-2-0'
            or subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).split()
            != ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']):
        raise ValueError('Exact authorized one-GPU route required')
    root = Path(args.project_root).resolve()
    stage = Path(__file__).resolve().parent
    out = root / args.output
    if out.exists() or not out.resolve().is_relative_to(root):
        raise ValueError('Fresh confined synthetic fixture output required')
    out.mkdir()
    sys.path.insert(0, str(stage / 'runtime'))
    import numpy as np
    import torch
    from source_utils import PINS, receipt, load
    from adapters.sehgnn_native import BasicSession
    state_path, state_hash = PINS['state_helpers']
    state_row = receipt(root, state_path)
    if state_row['sha256'] != state_hash:
        raise ValueError('Pinned native RNG helpers changed')
    states = load(root, state_row, 'amp_fixture_states')
    helper_path = root / 'private_source_zero_update_helpers_20261012_v2/runtime/qa/practical_zero_update_resources.py'
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != 'effa70e31e64881322ccfb5a6e7e3903ccdb6f887c694a97f6b63d0bd2e267e0':
        raise ValueError('Exact reviewed optimizer denial helper required')
    spec = importlib.util.spec_from_file_location('sehgnn_amp_fixture_denial', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.manual_seed(87101)
    torch.cuda.manual_seed_all(87101)
    rows, started = [], time.monotonic()
    try:
        class NativeHeadFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projector = torch.nn.Linear(1, 5, bias=False)
                torch.nn.init.zeros_(self.projector.weight)
                # This is the original native final LayerNorm operator/config.
                self.normalization = torch.nn.LayerNorm(5, elementwise_affine=False, eps=1e-5)

            def forward(self, batch, features, labels, label_embedding):
                return self.normalization(self.projector(features['M']))

        # Bypass only the full constructor: execute the exact prospective
        # BasicSession.train_epoch on synthetic finite native-head inputs.
        with helper.OptimizerDenial(torch) as denied:
            model = NativeHeadFixture().to(device)
            session = BasicSession.__new__(BasicSession)
            session.rt = {'torch': torch, 'numpy': np, 'device': device}
            session.engine = states
            session.model = torch.nn.ModuleList([model])
            session.models = tuple(session.model)
            session.optimizer = torch.optim.Adam(session.model.parameters(), lr=.001)
            session.scalar = torch.cuda.amp.GradScaler(init_scale=65536, enabled=True)
            session.member_rng = [states.capture_rng(np, torch)]
            session.counters = {'own_epochs': 0, 'own_member_forwards': 0, 'own_backwards': 0, 'actual_Adam_steps': 0}
            ids = torch.arange(4)
            truth = torch.zeros(4, 5, device=device)
            truth[:, 0] = 1
            session.ctx = SimpleNamespace(train_loader=[ids], train_count=4,
                feats={'M': torch.ones(4, 1)}, label_feats={}, targets_cuda=truth)
            initial = {k: v.detach().clone() for k, v in session.model.state_dict().items()}
            for epoch in range(1, 17):
                scale_before = float(session.scalar.get_scale())
                calls_before = denied.denied_step_calls
                counters_before = dict(session.counters)
                value = session.train_epoch()
                maps = denied.denied_step_calls - calls_before
                scale_after = float(session.scalar.get_scale())
                gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in session.model.parameters())
                if (session.counters['own_epochs'] != counters_before['own_epochs'] + 1
                        or session.counters['own_member_forwards'] != counters_before['own_member_forwards'] + 1
                        or session.counters['own_backwards'] != counters_before['own_backwards'] + 1
                        or maps not in (0, 1)
                        or session.counters['actual_Adam_steps'] != counters_before['actual_Adam_steps'] + maps):
                    raise ValueError('Complete cycle/observed dispatch accounting differs')
                if maps == 0 and (gradients_finite or not scale_after < scale_before):
                    raise ValueError('Fixture did not demonstrate genuine scaled-gradient overflow skip/backoff')
                if maps == 1 and (not gradients_finite or scale_after != scale_before):
                    raise ValueError('Finite GradScaler dispatch differs')
                if any(not torch.equal(initial[k], v) for k, v in session.model.state_dict().items()):
                    raise ValueError('Optimizer denial failed to preserve weights')
                rows.append({'epoch': epoch, 'finite_own_BCE': value['mean_native_own_BCE'],
                    'scale_before': scale_before, 'scale_after': scale_after,
                    'unscaled_gradients_finite': gradients_finite, 'optimizer_owner_attempts': 1,
                    'skipped_dispatches': 1 - maps, 'denied_finite_dispatches': maps,
                    'actual_optimizer_updates': 0})
            if (sum(r['skipped_dispatches'] for r in rows) < 1
                    or sum(r['denied_finite_dispatches'] for r in rows) < 1
                    or denied.denied_functional_calls != 0 or denied.actual_optimizer_kernels != 0
                    or session.optimizer.state):
                raise ValueError('Required skipped and finite-denied dispatch paths did not both occur')
        torch.cuda.synchronize(device)
        result = {'schema': 'native-SeHGNN-synthetic-AMP-cycle-fixture-v1', 'status': 'passed',
            'scope': 'Exact BasicSession.train_epoch plus pinned native RNG helpers; synthetic 1-to-5 projector and original final LayerNorm/BCE; constructor/geometry and resources not qualified',
            'source': {'fixture': receipt(root, Path(__file__)),
                'basic_session': receipt(root, stage / 'runtime/adapters/sehgnn_native.py'),
                'native_rng_helpers': state_row, 'optimizer_denial': receipt(root, helper_path)},
            'complete_cycles': len(rows), 'optimizer_owner_attempts': len(rows),
            'legal_AMP_overflow_skips': sum(r['skipped_dispatches'] for r in rows),
            'finite_dispatches_denied': denied.denied_step_calls, 'cycles': rows,
            'elapsed_seconds': time.monotonic() - started, 'scientific_fits': 0,
            'actual_optimizer_updates': 0, 'optimizer_kernels': 0, 'predictive_scores': 0,
            'real_graph_or_targets_read': False, 'CUDA': True, 'GO': False}
        (out / 'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps({k: result[k] for k in ('status', 'complete_cycles', 'legal_AMP_overflow_skips', 'finite_dispatches_denied', 'actual_optimizer_updates', 'scientific_fits', 'GO')}), flush=True)
    except BaseException as exc:
        (out / 'FAILURE.json').write_text(json.dumps({'status': 'failed', 'error': str(exc),
            'traceback': traceback.format_exc(), 'completed_cycles': rows, 'scientific_fits': 0,
            'actual_optimizer_updates': 0, 'GO': False}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
