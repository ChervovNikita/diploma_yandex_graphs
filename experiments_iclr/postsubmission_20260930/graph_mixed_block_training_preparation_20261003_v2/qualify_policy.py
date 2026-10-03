"""One same-state witness plus score-free full-graph resource/state replay.

Prepared only. A separate root qualification release is required before Torch,
data, label, or fitted-state access. No training driver main or selector called.
"""
import argparse
import copy
import inspect
import json
from pathlib import Path
import resource
import time
import traceback

import block_policy
import common as c


def assert_tree(torch, left, right, exact=False):
    if isinstance(left, torch.Tensor):
        c.require(isinstance(right, torch.Tensor), 'Replay tensor kind differs')
        if exact or not left.is_floating_point():
            c.require(torch.equal(left, right), 'Exact replay tensor differs')
        else:
            torch.testing.assert_close(left, right)
    elif type(left) in (list, tuple):
        c.require(type(left) is type(right) and len(left) == len(right), 'Replay sequence shape differs')
        for a, b in zip(left, right):
            assert_tree(torch, a, b, exact)
    elif isinstance(left, dict):
        c.require(type(right) is type(left) and left.keys() == right.keys(), 'Replay state keys differ')
        for key in left:
            assert_tree(torch, left[key], right[key], exact)
    else:
        c.require(left == right, 'Replay scalar differs')


def state(torch, model, optimizer, scheduler, streams, onecycle):
    return dict(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
        scheduler=onecycle.portable_state(scheduler), member_RNG=streams.state_dict(),
        CPU_RNG=torch.get_rng_state().clone(), parameter_names=[n for n, _ in model.named_parameters()])


def restore(torch, value, model, optimizer, scheduler, streams, onecycle):
    c.require(value['parameter_names'] == [n for n, _ in model.named_parameters()], 'Replay parameter order differs')
    model.load_state_dict(value['model']); optimizer.load_state_dict(value['optimizer'])
    onecycle.restore_state(scheduler, value['scheduler'])
    streams.load_state_dict(value['member_RNG']); torch.set_rng_state(value['CPU_RNG'])


def optimizer(torch, model):
    value = torch.optim.AdamW(model.parameters(), weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(value, total_steps=300, max_lr=1e-3, pct_start=.05)
    return value, schedule


def synthetic_witness(torch, mods):
    """Independent full-loss backward oracle, not another copy of block routing."""
    implementation, onecycle = mods['implementation'], mods['onecycle']
    graph = implementation.HeteroGraph({'0': 4, '1': 3}, [
        implementation.Relation(0, '0', '1', torch.tensor([0, 1, 2, 3, 0]), torch.tensor([0, 1, 2, 0, 2])),
        implementation.Relation(1, '1', '0', torch.tensor([0, 1, 2, 2, 0]), torch.tensor([0, 1, 2, 3, 2]))])
    features = {'0': torch.arange(20, dtype=torch.float32).reshape(4, 5)/19,
                '1': torch.arange(12, dtype=torch.float32).reshape(3, 4)/11}
    ids, labels = torch.tensor([0, 2, 3]), torch.tensor([0, 2, 1])
    torch.manual_seed(131)
    base = implementation.PrivateHGT(implementation.NativeHGT(graph, {'0': 5, '1': 4}, 64, 3, 3, 8, True),
        implementation.initialized_factors(64, 2, 3, 900132)['be'])
    rows = []
    for policy in c.POLICIES:
        production, oracle = copy.deepcopy(base), copy.deepcopy(base)
        blocks = block_policy.partition(implementation, production)
        oracle_blocks = block_policy.partition(implementation, oracle)
        opt, schedule = optimizer(torch, production)
        oracle_opt, oracle_schedule = optimizer(torch, oracle)
        seeds = [131+700001+1009*m for m in range(4)]
        streams, oracle_streams = implementation.MemberStreams(seeds), implementation.MemberStreams(seeds)
        initial_rng = torch.get_rng_state().clone()
        calls = [0]
        handle = production.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0]+1))
        production.train(); opt.zero_grad()
        logits = production(graph, features, '0', streams)
        own = implementation.mean_member_ce(logits, ids, labels)
        block_policy.backward(torch, policy, blocks, logits, ids, labels, own)
        actual_grad = {n: None if p.grad is None else p.grad.clone() for n, p in production.named_parameters()}
        after_backward_rng = torch.get_rng_state().clone()
        opt.step(); schedule.step(1); handle.remove()
        c.require(calls == [1] and torch.equal(initial_rng, after_backward_rng), 'Gradient routing advanced forward/RNG twice')
        production_state = state(torch, production, opt, schedule, streams, onecycle)

        torch.set_rng_state(initial_rng)
        oracle.train(); oracle_opt.zero_grad()
        reference_logits = oracle(graph, features, '0', oracle_streams)
        c.require(torch.equal(logits, reference_logits), 'Same-state forward logits differ')
        own_reference = implementation.mean_member_ce(reference_logits, ids, labels)
        pool_reference = torch.nn.functional.cross_entropy(reference_logits.mean(0)[ids], labels)
        if policy in ('own/own', 'pool/pool'):
            (own_reference if policy == 'own/own' else pool_reference).backward()
        else:
            own_reference.backward(retain_graph=True)
            own_gradient = {n: None if p.grad is None else p.grad.clone() for n, p in oracle.named_parameters()}
            oracle_opt.zero_grad(); pool_reference.backward()
            pool_gradient = {n: None if p.grad is None else p.grad.clone() for n, p in oracle.named_parameters()}
            oracle_opt.zero_grad()
            shared_names = {oracle_blocks['names'][id(p)] for p in oracle_blocks['shared']}
            for name, parameter in oracle.named_parameters():
                objective = policy.split('/')[0 if name in shared_names else 1]
                parameter.grad = (own_gradient if objective == 'own' else pool_gradient)[name]
        expected_grad = {n: None if p.grad is None else p.grad.clone() for n, p in oracle.named_parameters()}
        assert_tree(torch, actual_grad, expected_grad, exact=policy == 'own/own')
        oracle_opt.step(); oracle_schedule.step(1)
        oracle_state = state(torch, oracle, oracle_opt, oracle_schedule, oracle_streams, onecycle)
        assert_tree(torch, production_state, oracle_state, exact=policy == 'own/own')
        assert_tree(torch, streams.state_dict(), oracle_streams.state_dict(), exact=True)
        unused = [n for n, g in expected_grad.items() if g is None]
        c.require(unused and all(actual_grad[n] is None for n in unused), 'Witness must exercise genuine unused shared leaves')
        rows.append(dict(policy=policy, one_preupdate_forward=True, original_1_over_M_own_reduction=True,
            complete_gradient_field_matches_independent_backward_oracle=True,
            AdamW_OneCycle_step_matches=True, buffer_and_RNG_transition_matches=True,
            own_own_bitwise_exact=policy == 'own/own', unused_None_parameters=unused,
            numeric_comparison='Torch assert_close defaults for mixed floating tensors; exact own/own and all RNG/nonfloating state'))
        del production, oracle
    return dict(schema='mixed_block_single_same_state_witness_v1', status='passed', rows=rows,
                synthetic=True, real_graph_or_labels_opened=False, scientific_quality_claim=False)


def resource_case(torch, mods, dataset, policy, frozen, inputs, graph, features, schema, development, out, release):
    """One existing full graph, six steps, eval/save/restore and next-step replay.

    No validation/test scoring, early stop, epoch0 quality, or model selection.
    Full-state checkpoint uses the existing portable OneCycle implementation.
    """
    implementation, onecycle = mods['implementation'], mods['onecycle']
    split_record = next(r for r in frozen['splits'] if r['seed'] == 131)
    split, train_labels, _ = inputs.verify_split(split_record['descriptor'], development, 131)
    ids, labels = torch.tensor(split['train_ids'], dtype=torch.long), torch.tensor(train_labels, dtype=torch.long)
    model = c.build(mods, dataset, graph, schema, len(development['train_class_schema']), 131)
    blocks = block_policy.partition(implementation, model)
    opt, scheduler = optimizer(torch, model)
    torch.manual_seed(131+700001)
    streams = implementation.MemberStreams([131+700001+1009*m for m in range(4)])
    observed, started = [], time.perf_counter()
    out.mkdir()
    for epoch in range(6):
        before = c.memory()
        # Separate assignments match immutable fit's variable lifetimes: old
        # logits end after the new forward, old loss after the new own CE.
        model.train(); opt.zero_grad()
        logits = model(graph, features, '0', streams)
        own_loss = implementation.mean_member_ce(logits, ids, labels)
        c.require(bool(torch.isfinite(own_loss)), 'Nonfinite TRAIN mean-member CE')
        block_policy.backward(torch, policy, blocks, logits, ids, labels, own_loss)
        opt.step(); scheduler.step(epoch+1)
        after_train = c.memory()
        model.eval()
        with torch.no_grad():
            eval_logits = model(graph, features, '0')
        torch.save(state(torch, model, opt, scheduler, streams, onecycle), out/'complete_state.pt')
        torch.save(eval_logits, out/'member_logits.pt')
        # Retain eval logits until the next eval, as immutable fit retains its
        # member_logits variable across the next TRAIN transition.
        observed.append(dict(update=epoch+1, before_TRAIN=before, after_TRAIN=after_train,
                             after_eval_and_checkpoint=c.memory()))
        for values in (before, after_train, observed[-1]['after_eval_and_checkpoint']):
            c.require(values['VmPeak_bytes'] <= release['address_space_limit_bytes']
                      and values['VmRSS_bytes'] <= release['RSS_limit_bytes'], 'Declared resource envelope exceeded')
    saved = torch.load(out/'complete_state.pt', map_location='cpu', weights_only=True)
    for trial in range(2):
        if trial:
            restore(torch, saved, model, opt, scheduler, streams, onecycle)
        model.train(); opt.zero_grad()
        logits = model(graph, features, '0', streams)
        own_loss = implementation.mean_member_ce(logits, ids, labels)
        c.require(bool(torch.isfinite(own_loss)), 'Nonfinite TRAIN mean-member CE')
        block_policy.backward(torch, policy, blocks, logits, ids, labels, own_loss)
        opt.step(); scheduler.step(7)
        if not trial:
            first = state(torch, model, opt, scheduler, streams, onecycle)
        else:
            replay = state(torch, model, opt, scheduler, streams, onecycle)
    assert_tree(torch, first, replay, exact=True)
    replay_memory = c.memory()
    c.require(replay_memory['VmPeak_bytes'] <= release['address_space_limit_bytes']
              and replay_memory['VmRSS_bytes'] <= release['RSS_limit_bytes'], 'Replay resource envelope exceeded')
    row = dict(dataset=dataset, policy=policy, seed=131, status='qualified', consecutive_updates=6,
        next_step_complete_state_replay=True, full_state_rng_custody=True,
        module_semantic_partition=blocks['receipt'], per_update_memory=observed,
        replay_memory=replay_memory, wall_seconds=time.perf_counter()-started,
        validation_or_test_scored=False, model_selection=False,
        scientific_study_outcomes=False, qualification_only=True)
    c.write(out/'RECEIPT.json', row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True); parser.add_argument('--admission', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args(argv)
    c.require(Path(args.run_name).name == args.run_name and args.run_name not in ('', '.', '..'), 'Fresh qualification name required')
    provenance, frozen, sources, release, admission, original_bytes = c.admission(args.freeze, args.admission, 'qualification')
    c.require(release['run_name'] == args.run_name and release['resource_limits_installed_preimport'] is True,
              'Bound root qualification run and preimport resource envelope required')
    limit = resource.getrlimit(resource.RLIMIT_AS)[0]
    c.require(limit == release['address_space_limit_bytes'] and limit > 0, 'Actual inherited address-space limit differs')
    out = c.PACKET/'qualification'/args.run_name
    out.mkdir(parents=True, exist_ok=False)
    rows, errors, witness = [], [], None
    originals_preserved = False
    try:
        torch, runtime = c.runtime()
        c.write(out/'RUNTIME.json', dict(runtime, python_version=[3, 11, 14], address_space_limit_bytes=limit))
        mods = c.modules(provenance)
        _, custody = block_policy.fit_tree(inspect.getsource(mods['driver'].fit))
        c.write(out/'FIT_BODY_CUSTODY.json', custody)
        witness = synthetic_witness(torch, mods)
        c.write(out/'SAME_STATE_WITNESS.json', witness)
        for dataset in c.DATASETS:
            try:
                inputs, graph, features, schema, development = c.graph_inputs(mods, dataset, sources[dataset], mods['implementation'], torch)
                for policy in c.POLICIES:
                    case = out/dataset/policy.replace('/', '__'); case.parent.mkdir(exist_ok=True)
                    try:
                        rows.append(resource_case(torch, mods, dataset, policy, sources[dataset], inputs, graph, features, schema, development, case, release))
                    except Exception as error:
                        rows.append(dict(dataset=dataset, policy=policy, status='failed', error_type=type(error).__name__,
                                         error_message=str(error), traceback=traceback.format_exc()))
                del graph, features, schema, development
            except Exception as error:
                errors.append(dict(dataset=dataset, error_type=type(error).__name__, error_message=str(error)))
    except Exception as error:
        errors.append(dict(error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc()))
    finally:
        seen = {(r['dataset'], r['policy']): r for r in rows}
        rows = [seen.get((d, p), dict(dataset=d, policy=p, status='blocked', reason='qualification prerequisite failed'))
                for d in c.DATASETS for p in c.POLICIES]
        try:
            c.preserved(provenance, frozen, sources, args.freeze, args.admission, original_bytes)
            originals_preserved = True
        except Exception as error:
            errors.append(dict(error_type=type(error).__name__, error_message=str(error)))
        qualified = witness and witness['status'] == 'passed' and not errors and originals_preserved and all(r['status'] == 'qualified' for r in rows)
        result = dict(schema='mixed_block_focused_qualification_v1', status='qualified' if qualified else 'unqualified',
            **admission, rows=rows, errors=errors, synthetic_same_state_witness_passed=bool(witness and witness['status'] == 'passed'),
            originals_preserved=originals_preserved, validation_or_test_scored=False, training_driver_main_called=False,
            model_selection=False, scientific_study_outcomes=False, representative_resource_seed=131)
        c.write(out/'QUALIFICATION.json', result)
    print(json.dumps(dict(output=str(out), status=result['status'])))
    return 0 if result['status'] == 'qualified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
