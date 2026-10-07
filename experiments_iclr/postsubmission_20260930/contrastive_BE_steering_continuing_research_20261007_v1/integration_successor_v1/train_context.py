"""Disabled context-target full loop; explicit original public driver successor."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import sys
HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
PUBLIC_ROOT = PHASE / 'portable_internal_be_public_interface_20261007_v2'
sys.path.insert(0, str(PUBLIC_ROOT))
sys.path.insert(0, str(HERE.parent))
from portable import Session, recipe
import portable as public
from data_interface import load_train_valid, train_batches, validation_batches, _sha
from context_dispatch import (METHODS, configured_recipe, make_session,
    local_transition, restore_own_best_bank)
from context_recompute import install as install_replay, MODE


def admit(release_path, method, seed, inputs):
    release = json.loads(Path(release_path).read_text())
    required = ('root_adopted', 'source_review_approved',
        'fullgraph_targets_objective_and_selectors_qualified',
        'external_supervision_confirmed', 'complete_staged_protocol_frozen')
    if (release.get('schema') != 'context-target-cell-release-v1'
        or release.get('enabled') is not True
        or any(release.get(key) is not True for key in required)
        or release.get('TEST_access') is not False
        or release.get('automatic_retry') is not False
        or release.get('method') != method or release.get('seed') != seed):
        raise ValueError('Scientific release disabled pending complete explicit root admission')
    manifest_path = HERE / 'MANIFEST.json'
    if release.get('source_manifest_sha256') != _sha(manifest_path):
        raise ValueError('Exact reviewed successor source required')
    for row in json.loads(manifest_path.read_text())['files']:
        path = HERE / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or _sha(path) != row['sha256']:
            raise ValueError('Successor source changed')
    for row in json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())['files']:
        path = PHASE / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or _sha(path) != row['sha256']:
            raise ValueError('Original public/helper source binding changed')
    if set(release.get('inputs', {})) != set(inputs):
        raise ValueError('Exact frozen TRAIN/development/native input bindings required')
    for key, path in inputs.items():
        row = release['inputs'][key]
        expected = (PHASE / row['path']).resolve()
        if not expected.is_relative_to(PHASE) or expected == PHASE:
            raise ValueError('Frozen inputs must remain inside the project research repository')
        if path is None or str(path.resolve()) != str(expected):
            raise ValueError('Input identity differs from prospective release')
        if _sha(path) != row['sha256']:
            raise ValueError('Input bytes differ from prospective release')
    if method in ('single_native', 'independent4_native', 'single_common_context',
                  'independent4_route_context'):
        if release.get('whole_stage1_gate_passed') is not True:
            raise ValueError('Stage2 remains disabled before whole9 fixed Stage1 gate')
    return release


def json_write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def evaluate(session, train, valid):
    from ogb.graphproppred import Evaluator as GraphEvaluator
    from ogb.linkproppred import Evaluator as LinkEvaluator
    torch, np = session.torch, session.np
    session.model.eval(); chunks = []; member_chunks = []; truth = []
    with torch.no_grad():
        for batch, labels in validation_batches(session.task, train, valid, session.device):
            logits, _ = session.forward(batch); pooled = session.serving(logits)
            session.core['selection'].finite_predictions(logits, pooled)
            chunks.append(pooled.cpu()); member_chunks.append(logits.cpu()); truth.append(labels.cpu())
    prediction = torch.cat(chunks); members = torch.cat(member_chunks, 1)
    expected = {'wikics': 5274, 'collab': 160084, 'molhiv': 4113}[session.task]
    if len(prediction) != expected or members.shape[:2] != (session.model.members, expected):
        raise ValueError('Incomplete complete-VALID member/population work')
    if session.task == 'wikics':
        labels = torch.cat(truth)
        metric = float((prediction.argmax(-1) == labels).float().mean())
        per = [float((member.argmax(-1) == labels).float().mean()) for member in members]
    elif session.task == 'molhiv':
        labels = torch.cat(truth).reshape(-1, 1); evaluator = GraphEvaluator(name='ogbg-molhiv')
        metric = float(evaluator.eval({'y_true': labels.numpy(), 'y_pred': prediction.numpy()})['rocauc'])
        per = [float(evaluator.eval({'y_true': labels.numpy(), 'y_pred': member.numpy()})['rocauc']) for member in members]
    else:
        count = len(valid['positive']); evaluator = LinkEvaluator(name='ogbl-collab'); evaluator.K = 50
        def hits(values):
            return float(evaluator.eval({'y_pred_pos': values[:count].flatten(),
                                         'y_pred_neg': values[count:].flatten()})['hits@50'])
        metric = hits(prediction); per = [hits(member) for member in members]
    if not np.isfinite(metric) or not np.isfinite(per).all():
        raise ValueError('Complete selector metric nonfinite')
    return metric, per


def joint_snapshot(session, epoch, metric, per, run):
    torch = session.torch
    return session._cpu_tree({'model': session.model.state_dict(),
        'optimizers': [optimizer.state_dict() for optimizer in session.optimizers],
        'streams': session.streams, 'epoch': epoch,
        'global': session.task == 'wikics' and session.model.models[0].body._global,
        'selected_VALID': metric, 'member_VALID': per,
        'python_rng': random.getstate(), 'numpy_rng': session.np.random.get_state(),
        'cpu_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(session.cuda_index) if session.cuda_index is not None else None,
        'run': run, 'config': session.config,
        'checkpoint_kind': 'strict-first-maximum complete VALID joint snapshot',
        'exact_resume_supported': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('wikics',), required=True)
    parser.add_argument('--arm', choices=tuple(METHODS), required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=6101)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--train', type=Path, required=True, help='Complete numeric TRAIN NPZ')
    parser.add_argument('--valid', type=Path, required=True, help='Complete numeric VALID NPZ')
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    parser.add_argument('--polynormer', type=Path, required=True)
    parser.add_argument('--ncn-model', type=Path)
    parser.add_argument('--ncn-utils', type=Path)
    args = parser.parse_args(); release = admit(args.release, args.arm, args.seed,
        {'train': args.train, 'development': args.valid, 'polynormer': args.polynormer})
    if not args.output.resolve().is_relative_to(PHASE) or args.output.resolve() == PHASE:
        raise ValueError('Fresh output must remain inside this project research repository')
    started = time.monotonic(); trace = []; session = None
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        # Load evaluator providers before Session seeds its trajectory, as in
        # frozen run.py; evaluate() then accesses these cached modules.
        from ogb.graphproppred import Evaluator as GraphEvaluator
        from ogb.linkproppred import Evaluator as LinkEvaluator
        if args.task != 'wikics' or args.seed not in (8101, 8203, 8307):
            raise ValueError('Exact proposed context-target task/seed roster required')
        config = configured_recipe(public, args.arm)
        train, valid, origin = load_train_valid(args.task, args.train, args.valid)
        session, facade, policy, preparation = make_session(public, args.arm, args.seed,
            args.device, args.polynormer, train)
        install_replay(session)
        torch = session.torch
        if preparation is not None:
            metadata = {key: value for key, value in preparation.items()
                        if key not in ('masks', 'permutations')}
            session.np.savez_compressed(args.output / 'FROZEN_TRAIN_TARGETS.npz',
                masks=preparation['masks'], permutations=preparation['permutations'])
            metadata['frozen_targets_sha256'] = _sha(args.output / 'FROZEN_TRAIN_TARGETS.npz')
            json_write(args.output / 'TARGET_PREPARATION.json', metadata)
        run = {'task': args.task, 'arm': args.arm, 'seed': args.seed, 'device': str(session.device),
               'data': origin, 'core': session.core_provenance, 'native': session.native_provenance,
               'torch': str(torch.__version__), 'numpy': session.np.__version__,
               'author_execution_equivalence_claimed': False, 'TEST_scoring': False,
               'underlying_session_arm': session.arm, 'checkpoint_policy': policy.__dict__,
               'explicit_context_objective': config['context_target_method'],
               'source_release_sha256': _sha(args.release), 'execution_mode': MODE,
               'driver_sha256': _sha(__file__), 'exact_resume_supported': False}
        json_write(args.output / 'RUN.json', run)
        best = best_local = -float('inf')
        own_best = [-float('inf')] * session.model.members
        own_local = [-float('inf')] * session.model.members
        for epoch in range(1, config['training']['epochs'] + 1):
            if args.task == 'wikics' and epoch == config['training']['local_epochs'] + 1:
                local_transition(session,
                    lambda name: torch.load(args.output / name, map_location=session.device, weights_only=False),
                    policy, facade)
            last = None; epoch_updates = 0
            for batch, labels in train_batches(args.task, train, epoch, args.seed, session.device):
                result = session.train_step(batch, labels); epoch_updates += 1
                last = {'own_mean': float(result['own_mean']), 'aux': float(result['auxiliary'])}
            expected_updates = {'wikics': 1, 'collab': 18, 'molhiv': 258}[args.task]
            if epoch_updates != expected_updates:
                raise ValueError('Incomplete full TRAIN epoch')
            metric, per = evaluate(session, train, valid)
            if not policy.own_selected_four and metric > best:
                best = metric
                torch.save(joint_snapshot(session, epoch, metric, per, run), args.output / 'selected.pt')
            if args.task == 'wikics' and epoch <= config['training']['local_epochs'] and metric > best_local:
                best_local = metric
                # Same unused pooled-local artifact is retained for ordinary
                # independent4; its transition reads ONLY own_local_N.pt.
                torch.save(joint_snapshot(session, epoch, metric, per, run), args.output / 'selected_local.pt')
            if session.model.independent:
                for member, body in enumerate(session.model.models):
                    row = session._cpu_tree({'model': body.state_dict(),
                        'optimizer': session.optimizers[member].state_dict(), 'epoch': epoch,
                        'global': args.task == 'wikics' and body.body._global, 'metric': per[member]})
                    if per[member] > own_best[member]:
                        own_best[member] = per[member]; torch.save(row, args.output / ('own_best_' + str(member) + '.pt'))
                    if args.task == 'wikics' and epoch <= config['training']['local_epochs'] and per[member] > own_local[member]:
                        own_local[member] = per[member]; torch.save(row, args.output / ('own_local_' + str(member) + '.pt'))
            trace.append({'epoch': epoch, 'VALID_selector': metric, 'members': per, 'TRAIN': last,
                          'seconds': time.monotonic() - started})
            json_write(args.output / 'VALID_TRACE.json', trace)
            json_write(args.output / 'PROGRESS.json', {'epoch': epoch, 'complete_epochs': config['training']['epochs'],
                       'steps': session.steps, 'complete': False})
        if policy.own_selected_four:
            own_selected_epochs = restore_own_best_bank(session,
                lambda name: torch.load(args.output / name, map_location=session.device, weights_only=False),
                policy, facade)
            metric, per = evaluate(session, train, valid)
            json_write(args.output / 'OWN_BEST_BANK.json', {'VALID': metric, 'members': per,
                'own_selected_epochs': own_selected_epochs, 'checkpoint_policy': policy.__dict__})
            torch.save(session._cpu_tree({'model': session.model.state_dict(),
                'body_global': [getattr(getattr(body, 'body', None), '_global', False) for body in session.model.models],
                'evaluation_only': True, 'candidate': 'individual_best_bank_only', 'selected_VALID': metric,
                'run': run, 'config': config, 'own_selected_epochs': own_selected_epochs,
                'checkpoint_policy': policy.__dict__, 'exact_resume_supported': False}), args.output / 'selected.pt')
        if session.steps != config['training']['epochs'] * expected_updates:
            raise ValueError('Incomplete full training horizon')
        members = session.model.members
        expected_accounting = dict(shadow_member_forwards=2*members*session.steps,
            replay_member_forwards=2*members*session.steps,
            output_cotangent_collections=session.steps,
            member_reverse_collections=2*members*session.steps,
            optimizer_bank_updates=len(session.optimizers)*session.steps,
            exact_member_RNG_endpoint_checks=session.steps)
        if session.execution_totals != expected_accounting:
            raise ValueError('Complete shadow/replay/reverse/Adam work must be charged')
        if facade is not None and (facade.alignment_source_calls != session.steps
                                   or facade.residual_source_calls != 0):
            raise ValueError('Exact fixed-target alignment only, with no cross-member residual loss')
        json_write(args.output / 'COMPLETE.json', {'complete': True, 'task': args.task, 'arm': args.arm,
            'seed': args.seed, 'epochs': config['training']['epochs'], 'steps': session.steps,
            'selected_sha256': _sha(args.output / 'selected.pt'), 'TEST_scoring': False,
            'seconds': time.monotonic() - started, 'parameter_counts': session.core['factors'].factor_counts(session.model),
            'author_execution_equivalence_claimed': False, 'exact_resume_supported': False,
            'alignment_source_calls': facade.alignment_source_calls if facade else 0,
            'residual_source_calls': facade.residual_source_calls if facade else 0,
            'checkpoint_policy': policy.__dict__,
            'execution_accounting': dict(session.execution_totals)})
    except BaseException as error:
        json_write(args.output / 'FAILURE.json', {'complete': False, 'error_type': type(error).__name__,
            'error': str(error), 'epochs_completed': len(trace), 'steps': session.steps if session else 0,
            'seconds': time.monotonic() - started, 'automatic_retry': False})
        raise
    print(json.dumps({'complete': True, 'task': args.task, 'arm': args.arm, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
