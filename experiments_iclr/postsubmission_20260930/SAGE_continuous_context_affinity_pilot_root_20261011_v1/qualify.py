"""Nine disposable full-TRAIN units /18 native trajectories; no VALID scoring.

The sealed Family/helper and exact root config remain unchanged. One combined
own+auxiliary backward and Adam step per unit; no auxiliary VJP or replay.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import signal
import socket
import subprocess
import sys
import time
import traceback

R = Path('/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs')
P = R / 'experiments_iclr/postsubmission_20260930'
H = Path(__file__).resolve().parent
SOURCE = P / 'native_SAGE_continuous_context_affinity_source_20261011_v1/run_family.py'
HELPER = SOURCE.with_name('context_affinity_objective.py')
EXPECTED_ARMS = (
    'ordinaryM1_weighted_mixture', 'factorizedM1_weighted_mixture',
    'genuinefactorizedI4_weighted_mixture', 'shared4_weighted_private',
    'shared4_weighted_mixture', 'shared4_uniform_allclassmate',
)

TIMEOUT_SECONDS = 180
PEAK_LIMIT_BYTES = 32 * 1024**3


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def forbidden(*args, **kwargs):
    raise RuntimeError('Disposable TRAIN qualification forbids scoring, serving, fits and run')


def deadline(*args):
    raise TimeoutError('Prospective180-second qualification budget exceeded')


def tensor_hashes(banks):
    return {order: {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                          'C_bytes_sha256': hashlib.sha256(value.detach().cpu().numpy().tobytes(order='C')).hexdigest()}
                    for name, value in tensors.items()} for order, tensors in banks.items()}


def tree_equal(torch, left, right):
    """Exact saved state custody only; no forward or numerical parity check."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(torch, left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(torch, a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def main():
    started = time.perf_counter()
    require(socket.gethostname() == 'anogena-2-0' and Path.cwd() == R,
            'Exact authorized allocation/project required before reads/writes')
    require(subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
            == ['GPU-44039938-fd82-41d2-fefd-de71514e2fac'], 'Exact authorized sole physical GPU required')
    frozen_path = H / 'QUALIFICATION_FREEZE.json'
    frozen = json.loads(frozen_path.read_text())
    require(frozen.get('qualification_forward_units') == 9
            and frozen.get('qualification_native_trajectories') == 18
            and frozen.get('TRAIN_only') is True
            and frozen.get('VALID_quality_scored') is False
            and frozen.get('TEST_access') is False,
            'Prospective root9-unit/18-trajectory TRAIN-only freeze required')
    require(frozen.get('timeout_seconds', TIMEOUT_SECONDS) == TIMEOUT_SECONDS
            and frozen.get('max_cuda_peak_allocated_bytes', PEAK_LIMIT_BYTES) == PEAK_LIMIT_BYTES,
            'Exact180s/<32GiB bounded qualification budget')
    bound = {}
    for row in frozen['bound_files']:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Root-relative frozen project input required')
        path = R / rel
        require(str(rel) not in bound and digest(path) == row['sha256'], 'Frozen source/config/role bytes differ: ' + str(rel))
        bound[str(rel)] = row['sha256']
    config_path = H / 'CONFIG.json'
    require(all(str(path.relative_to(R)) in bound for path in (SOURCE, HELPER, config_path, Path(__file__).resolve())),
            'Root freeze must include exact Family/helper/config/qualifier bytes')
    config_bytes = config_path.read_bytes()
    cfg = json.loads(config_bytes)
    require(cfg.get('seeds') == [7301, 7403, 7507] and cfg.get('device') == 'cuda:0', 'Exact copied native config/seed roster required')
    output = H / 'actual_qualification_v1'
    output.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1, int(TIMEOUT_SECONDS - (time.perf_counter() - started))))
    cases, family, torch = [], None, None
    counts = {'native_forward_units': 0, 'native_member_trajectories': 0,
              'attempted_combined_backwards': 0, 'full_parameter_backwards': 0,
              'attempted_Adam_steps': 0, 'discarded_optimizer_updates': 0}
    paid_counts, failure, phase = {}, None, 'Family_target_preparation'
    try:
        spec = importlib.util.spec_from_file_location('continuous_context_actual_TRAIN_qualification', SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        require(module.ARMS == EXPECTED_ARMS, 'Exact six-arm sealed roster')
        family = module.Family(cfg, output, hashlib.sha256(config_bytes).hexdigest())
        torch = family.torch
        torch.set_num_threads(2)
        family.metrics = forbidden
        family.logits = forbidden
        family.fit = forbidden
        family.run = forbidden
        preparation = family.context_preparation
        require(preparation['TRAIN_rows'] == 580
                and preparation['target_policy'] == 'continuous_all_classmate_full_denominator_v1'
                and preparation['teacher_temperature'] == preparation['student_temperature'] == .2
                and len(preparation['nonuniform_target_rows_per_context_exact_diagnostic']) == 4
                and all(v > 0 for v in preparation['nonuniform_target_rows_per_context_exact_diagnostic'])
                and preparation['targets_use_VALID_or_predictions'] is False,
                'All580 TRAIN fixed continuous target/context law required')
        require(tensor_hashes(family.context_targets) == preparation['actual_loss_tensor_hashes'],
                'Transferred Q/E row-order custody differs')
        target_versions = {(order, name): value._version for order, tensors in family.context_targets.items() for name, value in tensors.items()}
        for order, tensors in family.context_targets.items():
            require(all(not value.requires_grad for value in tensors.values()), 'Targets/support must be stopped')
            labels = family.data.y[family.train_ids] if order == 'bank' else family.data.y[family.data.train_mask]
            allowed = ~torch.eye(580, device=family.device, dtype=torch.bool)
            positive = (labels[:, None] == labels[None, :]) & allowed
            require(torch.equal(tensors['full_allowed'], allowed)
                    and torch.equal(tensors['positive'], positive), 'Complete same-class support and all-nonself denominator in native order')
            for name in ('weights', 'uniform_weights'):
                weights = tensors[name]
                shape = (4, 580, 580) if name == 'weights' else (580, 580)
                require(tuple(weights.shape) == shape and torch.isfinite(weights).all().item()
                        and (weights >= 0).all().item()
                        and torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6, rtol=1e-6),
                        'Finite normalized full580 weighted/uniform target rows required')
                supports = weights > 0
                require(torch.equal(supports, positive[None].expand(4,-1,-1) if name == 'weights' else positive),
                        'Exactly all nonself classmates positive, weighted and uniform support equal')
        for arm in EXPECTED_ARMS:
            bank, genuine = arm.startswith('shared4_'), arm == 'genuinefactorizedI4_weighted_mixture'
            members, factorized = (4 if bank else 1), arm != 'ordinaryM1_weighted_mixture'
            mode = module.CONTEXT_MODES[arm]
            kept, identities, storages = [], [], []
            for member in range(4 if genuine else 1):
                phase = arm + '/member' + str(member) + '/factory'
                model, optimizer, streams = family.make(7301, member, 'baseline', factorized, members)
                params = dict(model.named_parameters())
                trainable = {id(value) for value in params.values() if value.requires_grad}
                actual = [id(value) for group in optimizer.param_groups for value in group['params']]
                require(len(actual) == len(set(actual)) and set(actual) == trainable,
                        'Every live original/private parameter must be owned exactly once')
                original_identity = {name: (id(value), value.untyped_storage().data_ptr()) for name, value in params.items()}
                if genuine:
                    ids = set(original_identity[name][0] for name in params)
                    pointers = set(original_identity[name][1] for name in params)
                    require(all(not ids.intersection(old) for old in identities)
                            and all(not pointers.intersection(old) for old in storages), 'Four genuine I4 bodies need disjoint parameters/storage')
                    identities.append(ids)
                    storages.append(pointers)
                    kept.append((model, optimizer, streams))
                body = model.body if bank else model
                head_ids = {id(value) for value in body.output_linear.parameters()}
                private_ids = {id(value) for part in body.modules() if isinstance(part, family.factors.FactorLinear) for value in (part.r, part.s)}
                require(bool(private_ids) == factorized and private_ids.issubset(trainable), 'Private factor Parameters/registration differ')
                model.train()
                optimizer.zero_grad(set_to_none=True)
                phase = arm + '/member' + str(member) + '/one_factual_TRAIN_forward'
                logits, hidden = family.training_outputs(model, streams, bank)
                counts['native_forward_units'] += 1
                counts['native_member_trajectories'] += members
                require(tuple(logits.shape) == (members, 580, 10) and tuple(hidden.shape) == (members, 580, 128)
                        and hidden.requires_grad and torch.isfinite(logits).all().item() and torch.isfinite(hidden).all().item(),
                        'One live finite full580 native H/logit tuple required')
                targets = family.data.y[family.train_ids] if bank else family.data.y[family.data.train_mask]
                target_bank = family.context_targets['bank' if bank else 'single']
                own = family.F.cross_entropy(logits.flatten(0, 1), targets.repeat(members))
                auxiliary, route_losses, pair_losses, paid = family.context_objective.context_auxiliary(torch, family.F, hidden, target_bank, mode)
                for key, value in paid.items():
                    paid_counts[key] = paid_counts.get(key, 0) + value
                require(auxiliary.requires_grad and auxiliary.grad_fn is not None
                        and torch.isfinite(own).item() and torch.isfinite(auxiliary).item()
                        and torch.isfinite(route_losses).all().item() and torch.isfinite(pair_losses).all().item(),
                        'Finite live own/auxiliary scalar paths required')
                hidden.retain_grad()
                auxiliary.retain_grad()
                loss = own + .05 * auxiliary
                phase = arm + '/member' + str(member) + '/one_combined_backward'
                counts['attempted_combined_backwards'] += 1
                loss.backward()
                counts['full_parameter_backwards'] += 1
                require(auxiliary.grad is not None and torch.isfinite(auxiliary.grad).all().item()
                        and auxiliary.grad.abs().sum().item() > 0
                        and hidden.grad is not None and torch.isfinite(hidden.grad).all().item()
                        and hidden.grad.abs().sum().item() > 0, 'Combined loss must reach the live auxiliary/H graph')
                gradients = {}
                for name, value in params.items():
                    require(value.grad is not None and torch.isfinite(value.grad).all().item(), 'Missing/nonfinite registered gradient: ' + name)
                    gradients[name] = float(value.grad.abs().sum().item())
                nonhead = sum(gradients[name] for name, value in params.items() if id(value) not in head_ids)
                private_nonhead = sum(gradients[name] for name, value in params.items() if id(value) in private_ids and id(value) not in head_ids)
                require(nonhead > 0 and (not factorized or private_nonhead > 0), 'Live finite nonhead/private-factor combined gradients required')
                phase = arm + '/member' + str(member) + '/one_Adam_step'
                counts['attempted_Adam_steps'] += 1
                optimizer.step()
                counts['discarded_optimizer_updates'] += 1
                require(all(torch.isfinite(value).all().item() for value in model.parameters()), 'Finite original Parameters after discarded Adam step')
                phase = arm + '/member' + str(member) + '/state_custody'
                checkpoint = output / (arm + '_member' + str(member) + '_discarded_state.pt')
                order_name = 'train_ids' if bank else 'native_boolean_mask'
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'streams': streams,
                            'step': 1, 'context_mode': mode,
                            'context_target_archive_sha256': preparation['archive_sha256'],
                            'TRAIN_H_order': order_name, 'context_contract': family.context_contract,
                            'context_actual_loss_tensor_hashes': preparation['actual_loss_tensor_hashes'],
                            'source_hashes': family.source_hashes}, checkpoint)
                with torch.no_grad():
                    next(iter(params.values())).view(-1)[0].add_(1)
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                require(state['step'] == 1 and state['context_mode'] == mode
                        and state['context_target_archive_sha256'] == preparation['archive_sha256']
                        and state['TRAIN_H_order'] == order_name
                        and state['context_contract'] == family.context_contract
                        and state['context_actual_loss_tensor_hashes'] == tensor_hashes(family.context_targets)
                        and state['source_hashes'] == family.source_hashes, 'Actual target/scalars/source/mode/TRAIN-order checkpoint binding')
                require(tree_equal(torch, streams, state['streams']), 'Saved stream bytes differ from the factual forward state')
                model.load_state_dict(state['model'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                streams = state['streams']
                require(tree_equal(torch, model.state_dict(), state['model'])
                        and tree_equal(torch, optimizer.state_dict(), state['optimizer'])
                        and tree_equal(torch, streams, state['streams']), 'Exact saved model/optimizer/stream restoration')
                require(original_identity == {name: (id(value), value.untyped_storage().data_ptr()) for name, value in model.named_parameters()},
                        'Original parameter objects/storage must persist across update/restoration')
                require(all(value._version == target_versions[order, name] for order, tensors in family.context_targets.items() for name, value in tensors.items()),
                        'A disposable update mutated frozen Q/E')
                torch.cuda.synchronize(family.device)
                require(torch.cuda.max_memory_allocated(family.device) < PEAK_LIMIT_BYTES
                        and time.perf_counter() - started < TIMEOUT_SECONDS, 'Prospective qualification resource bound')
                cases.append({'arm': arm, 'member': member, 'routes': members, 'mode': mode,
                              'seed': 7301, 'TRAIN_own_CE': float(own.detach().item()),
                              'TRAIN_auxiliary': float(auxiliary.detach().item()), 'TRAIN_total_loss': float(loss.detach().item()),
                              'nonhead_gradient_absolute_sum': nonhead,
                              'private_nonhead_gradient_absolute_sum': private_nonhead,
                              'registered_private_parameter_count': len(private_ids),
                              'combined_gradients_finite': True, 'live_H_and_auxiliary_graph': True,
                              'gradient_scope': 'One combined own+auxiliary backward; no isolated auxiliary VJP/gradient-effect attribution',
                              'original_parameter_identity_and_storage_retained': True,
                              'model_optimizer_streams_restored': True, 'targets_unchanged': True,
                              'checkpoint_target_mode_order_bound': True,
                              'discarded_checkpoint_bytes': checkpoint.stat().st_size,
                              'discarded_checkpoint_sha256': digest(checkpoint), 'operation_counts': paid})
                del logits, hidden, own, auxiliary, loss, route_losses, pair_losses, params, body, state
                if not genuine:
                    del model, optimizer, streams
            kept.clear()
            if genuine:
                del model, optimizer, streams
            torch.cuda.empty_cache()
        require(len(cases) == 9 and counts['native_forward_units'] == 9
                and counts['native_member_trajectories'] == 18
                and counts['full_parameter_backwards'] == counts['discarded_optimizer_updates'] == 9,
                'Exact9 disposable units/18 trajectories/9backwards/9updates')
        require(tensor_hashes(family.context_targets) == preparation['actual_loss_tensor_hashes']
                and digest(output / 'CONTEXT_TARGETS.npz') == preparation['archive_sha256'], 'Prepared targets persisted through every unit')
    except Exception as error:
        failure = {'phase': phase, 'error': repr(error), 'traceback': traceback.format_exc()}
    if torch is not None and family is not None:
        try:
            torch.cuda.synchronize(family.device)
        except Exception as error:
            if failure is None:
                failure = {'phase': 'final_synchronization', 'error': repr(error), 'traceback': traceback.format_exc()}
    signal.alarm(0)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(family.device) if torch is not None and family is not None else None
    qualified = failure is None and seconds < TIMEOUT_SECONDS and peak is not None and peak < PEAK_LIMIT_BYTES
    value = {'qualified': qualified, 'cases': cases, 'seconds': seconds, 'failure': failure,
             'counts': counts, 'context_operation_counts': paid_counts,
             'timeout_seconds': TIMEOUT_SECONDS, 'peak_limit_bytes_strict': PEAK_LIMIT_BYTES,
             'cuda_peak_allocated_bytes': peak,
             'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(family.device) if torch is not None and family is not None else None,
             'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'source_sha256': digest(SOURCE),
             'helper_sha256': digest(HELPER), 'qualifier_sha256': digest(__file__),
             'qualification_freeze_sha256': digest(frozen_path),
             'context_preparation': family.context_preparation if family is not None else None,
             'VALID_labels_loaded_only_by_existing_role_loader': family is not None,
             'VALID_quality_scored': False, 'TEST_access': False, 'scientific_fits': 0,
             'scientific_quality_evidence': False, 'auxiliary_only_VJPs': 0, 'extra_native_replay_forwards': 0}
    (output / 'QUALIFICATION.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'qualified': qualified, 'completed_cases': len(cases), 'seconds': seconds,
                      'TRAIN_trajectories': counts['native_member_trajectories'], 'peak_bytes': peak,
                      'quality_evidence': False, 'failure': failure['error'] if failure else None}))
    if not qualified:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
