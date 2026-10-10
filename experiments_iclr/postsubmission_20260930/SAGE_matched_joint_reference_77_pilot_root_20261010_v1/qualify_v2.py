"""Disposable full TRAIN check of the five fresh SAGE reference architectures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

def main():
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / 'CONFIG.json').read_text())
    repo = Path(cfg['native_repo'])
    assert socket.gethostname() == 'peptide' and Path.cwd() == repo
    admitted = json.loads((here / 'ACTUAL_READY_V1.json').read_text())
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines() == admitted['gpu_uuids']
    for row in json.loads((here / 'QUALIFICATION_FREEZE_V2.json').read_text())['bound_files']:
        q = repo / row['path']
        assert hashlib.sha256(q.read_bytes()).hexdigest() == row['sha256'], row['path']
    source = repo / 'experiments_iclr/postsubmission_20260930/SAGE_matched_joint_reference_77_source_20261010_v1/run_family.py'
    spec = importlib.util.spec_from_file_location('qualified_native_sage_F', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out = here / 'actual_qualification_v2'
    out.mkdir(exist_ok=False)
    family = module.Family(cfg, out, hashlib.sha256((here / 'CONFIG.json').read_bytes()).hexdigest())
    torch = family.torch
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1
    def forbidden(*args, **kwargs):
        raise RuntimeError('No VALID scoring during disposable TRAIN qualification')
    family.metrics = forbidden
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cases, keep, trajectories = [], [], 0
    for arm in module.ARM_SPECS:
        independent = []
        for m in range(arm['bundles']):
            members = 1 if arm['bundles'] == 4 else arm['members']
            model, optimizer, streams = family.make(7301, m, 'baseline', arm['factorized'], members, arm['architecture'])
            if arm['architecture'] == 'joint_untied':
                model.check_optimizer_ownership(optimizer)
                ptrs = [q.untyped_storage().data_ptr() for q in model.parameters()]
                assert len(set(ptrs)) == len(ptrs)
            if arm['bundles'] == 4:
                ids = {id(q) for q in model.parameters()}
                assert all(not ids.intersection(x) for x in independent)
                independent.append(ids)
                keep.append(model)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            bank = members > 1
            logits, hidden = family.training_outputs(model, streams, bank)
            targets = family.data.y[family.train_ids] if bank else family.data.y[family.data.train_mask]
            assert logits.shape == (members, 580, 10) and hidden.shape == (members, 580, 128)
            loss = family.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
            assert torch.isfinite(loss).item()
            loss.backward()
            gradients = [q.grad for q in model.parameters() if q.grad is not None]
            assert gradients and all(torch.isfinite(g).all().item() for g in gradients)
            assert sum(g.abs().sum().item() for g in gradients) > 0
            optimizer.step()
            trajectories += members
            cases.append(dict(arm=arm['stem']+'_F', fit_member=m, members=members, architecture=arm['architecture'], TRAIN_loss=float(loss.detach()), all_gradients_finite=True))
            del model, optimizer, streams, logits, hidden, targets, loss, gradients
    torch.cuda.synchronize()
    assert len(cases) == 8 and trajectories == 14
    result = dict(qualified=True, cases=cases, optimizer_updates_discarded=8, native_trajectories=14,
                  seconds=time.perf_counter()-start, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(), VALID_quality_scored=False,
                  TEST_access=False, scientific_fits=0, scientific_quality_evidence=False,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    assert result['cuda_peak_allocated_bytes'] < 32*1024**3
    (out/'QUALIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print('F_QUALIFIED_JSON='+json.dumps(result))

if __name__ == '__main__':
    main()
