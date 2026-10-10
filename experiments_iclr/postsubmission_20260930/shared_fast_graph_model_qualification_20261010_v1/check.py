"""One full-TRAIN integration check; no validation/test scoring or fit."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    repo = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
    phase = repo / 'experiments_iclr/postsubmission_20260930'
    assert socket.gethostname() == 'anogena-2-0' and Path.cwd() == repo
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac']
    output = args.output.resolve()
    assert output.is_relative_to(phase) and not output.exists()
    source = phase / 'shared_fast_graph_model_interface_20261010_v1/common_routes.py'
    factors_path = phase / 'portable_internal_be_public_interface_20261007_v2/core/factors.py'
    train = phase / 'masked_context_pubmed_allocation_preparation_20261010_v1/data_v4/TRAIN_ONLY.npz'
    assert digest(source) == '84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773'
    assert digest(train) == 'eeef8034e27a9973c784e1d44966df191af92b7f0b3fc860ac990e2de2e9087e'
    assert digest(repo / 'models.py') == '07a6c1c452486802713a1a040ab24f9e9f8504660d731eb5b6417e2357f0f303'
    assert digest(factors_path) == '9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3'
    output.mkdir()
    started = time.monotonic()
    identity = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    import os
    launch = dict(PID=os.getpid(), start_ticks=int(identity[19]), UTC=datetime.now(timezone.utc).isoformat(), source_sha256=digest(source), train_sha256=digest(train), VALID_access=False, TEST_access=False)
    (output / 'LAUNCH.json').write_text(json.dumps(launch, indent=2) + '\n')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    with np.load(train, allow_pickle=False) as archive:
        assert set(archive.files) == {'x', 'edge_index', 'train_ids', 'train_y'}
        arrays = {k: archive[k].copy() for k in archive.files}
    assert arrays['x'].shape == (19717, 500) and arrays['edge_index'].shape == (2, 88648)
    assert len(arrays['train_ids']) == len(arrays['train_y']) == 11829
    tensors = {k: torch.from_numpy(v).to('cuda:0') for k, v in arrays.items()}
    graph = SimpleNamespace(edge_index=tensors['edge_index'])
    batch = dict(graph=graph, x=tensors['x'], edge_index=tensors['edge_index'])
    native = module('_qualification_native_models', repo / 'models.py')
    common = module('_qualification_common_routes', source)
    factors = module('_qualification_existing_factors', factors_path)
    rows = []
    try:
        for backbone in ('GCN', 'GAT', 'SAGE'):
            for kind in ('baseline', 'separable', 'exchange'):
                tick = time.monotonic()
                torch.manual_seed(590101)
                body = native.Model(backbone, 2, 500, 512, 3, 1, 8, 'LayerNorm', .2)
                body.to('cuda:0').eval()
                with torch.no_grad():
                    original = body(graph, tensors['x']).detach()
                body.to('cpu')
                adapter = common.NativeModelAdapter(torch, body)
                model = common.wrap_shared(torch, factors, body, adapter, members=4, kind=kind, rank=16, block_seed=870590101).to('cuda:0').eval()
                with torch.no_grad():
                    initial, hidden = model(batch)
                    error = float((initial - original[None]).abs().max())
                    scale = max(1., float(original.abs().max()))
                assert initial.shape == (4, 19717, 3) and hidden.shape == (4, 19717, 512)
                assert error <= 1e-5 + 1e-4 * scale, 'Native initial function differs materially'
                del initial, hidden, original
                factors.initialize_first_factor(model.body.input_linear, 590101 + 135000)
                streams = []
                for member in range(4):
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(590101 + 300001 + 1009 * member)
                        streams.append(dict(cpu=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state(0).clone()))
                scope = lambda m: common.existing_member_rng_scope(torch, streams, m, 0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.)
                model.check_optimizer_ownership(optimizer)
                roles = model.parameter_roles()
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, hidden = model(batch, route_scope=scope)
                loss = torch.nn.functional.cross_entropy(logits[:, tensors['train_ids']].flatten(0, 1), tensors['train_y'].repeat(4))
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                changed = {role: sum(not torch.equal(before[n], dict(model.named_parameters())[n].detach()) for n in names) for role, names in roles.items()}
                assert changed['native_slow'] > 0 and changed['private_fast'] > 0
                assert kind == 'baseline' or changed['shared_block'] > 0
                del before, logits, hidden, loss
                model.eval()
                with torch.no_grad():
                    selected, representation = model(batch)
                checkpoint = output / (backbone + '__' + kind + '.pt')
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), streams=streams), checkpoint)
                with torch.no_grad():
                    next(model.parameters()).add_(.1)
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                model.load_state_dict(saved['model'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                streams[:] = saved['streams']
                with torch.no_grad():
                    restored, restored_representation = model(batch)
                restoration_error = float((restored - selected).abs().max())
                assert restoration_error <= 1e-5 + 1e-4 * max(1., float(selected.abs().max()))
                torch.cuda.synchronize()
                rows.append(dict(backbone=backbone, kind=kind, native_unit_factor_max_abs_error=error, restored_max_abs_error=restoration_error, parameter_counts={role: sum(dict(model.named_parameters())[n].numel() for n in names) for role, names in roles.items()}, updated_parameter_tensors=changed, train_updates=1, backwards=1, optimizer_steps=1, seconds=time.monotonic()-tick))
                (output / 'PROGRESS.json').write_text(json.dumps(dict(completed_cases=len(rows), rows=rows), indent=2) + '\n')
                del model, body, adapter, optimizer, saved, streams, selected, representation, restored, restored_representation, scope
                gc.collect()
                torch.cuda.empty_cache()
        report = dict(complete=True, rows=rows, train_updates=9, optimizer_steps=9, full_graph_nodes=19717, TRAIN_objects=11829, source_sha256=digest(source), train_sha256=digest(train), seconds=time.monotonic()-started, peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(0), VALID_access=False, TEST_access=False, scientific_fits=0, accuracy_result=False, integration_only=True, launch=launch)
        (output / 'COMPLETE.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ('complete', 'train_updates', 'seconds', 'peak_GPU_reserved_bytes', 'scientific_fits')}))
    except BaseException as error:
        (output / 'FAILURE.json').write_text(json.dumps(dict(error_type=type(error).__name__, error=str(error), completed_cases=len(rows), seconds=time.monotonic()-started, VALID_access=False, TEST_access=False), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
