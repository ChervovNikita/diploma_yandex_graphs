"""Read-only channels for native updates1/2; no replay or stopped epoch."""
from common import require
from state_observer import full_state


def parameter_mapping(unit, adam, torch):
    names = {id(p): prefix + '/' + name for prefix, root in zip(('encoder', 'predictor'), unit[:2])
             for name, p in root.named_parameters()}
    require(len(adam['param_groups']) == len(unit[2].param_groups) == 2, 'Original two Adam groups required')
    rows = []
    seen = set()
    for index, (live, saved) in enumerate(zip(unit[2].param_groups, adam['param_groups'])):
        expected = list(unit[index].named_parameters())
        require(len(live['params']) == len(saved['params']) == len(expected), 'Adam group length differs')
        for position, (parameter, key, (name, expected_parameter)) in enumerate(zip(live['params'], saved['params'], expected)):
            require(parameter is expected_parameter and id(parameter) not in seen, 'Adam name/order/ownership differs')
            seen.add(id(parameter))
            state = adam['state'].get(key, {})
            rows.append({'group': index, 'position': position, 'state_id': key, 'name': names[id(parameter)],
                         'shape': list(parameter.shape), 'gradient_present': parameter.grad is not None,
                         'state_fields': sorted(state),
                         'step': float(state['step'].item()) if 'step' in state else None})
    require(seen == set(names), 'Adam parameter coverage differs')
    return rows


def moment_transitions(unit, before, after, torch, differences, tolerance):
    """First two native Adam transitions from the actual unclipped pre-step grads.

    Formula comparisons are diagnostic scalar reports under the old rule. They
    neither replace Adam nor establish kernel causation. No parameter is written.
    """
    mapping = parameter_mapping(unit, before['Adam'], torch)
    reports = []
    for row in mapping:
        group, key = row['group'], row['state_id']
        name = row['name'].split('/', 1)[1]
        gradient = before['gradients'][group][name]
        old = before['Adam']['state'].get(key, {})
        new = after['Adam']['state'].get(key, {})
        settings = before['Adam']['param_groups'][group]
        require(settings['weight_decay'] == 0 and not settings.get('maximize', False)
                and not settings.get('amsgrad', False), 'Qualified native Adam family differs')
        if gradient is None:
            transition = differences(new, old, torch, 0, 0)
            reports.append({**row, 'expected_step_increment': 0, 'inactive_state': transition,
                            'transition_within_original_rule': transition['exact']})
            continue
        b1, b2 = settings['betas']
        previous_step = float(old['step'].item()) if old else 0.0
        first = old.get('exp_avg', torch.zeros_like(gradient))
        second = old.get('exp_avg_sq', torch.zeros_like(gradient))
        expected = {'exp_avg': first * b1 + gradient * (1 - b1),
                    'exp_avg_sq': second * b2 + gradient.square() * (1 - b2)}
        require(set(expected) <= set(new) and 'step' in new, 'Active Adam state missing after native step')
        actual_step = float(new['step'].item())
        moments = {field: differences(new[field], value, torch, tolerance['atol'], tolerance['rtol'])
                   for field, value in expected.items()}
        reports.append({**row, 'expected_step_increment': 1, 'before_step': previous_step,
                        'after_step': actual_step, 'step_increment_exact': actual_step == previous_step + 1,
                        'moment_formula_reports': moments,
                        'transition_within_original_rule': actual_step == previous_step + 1
                        and all(value['within_fixed_numeric_rule'] for value in moments.values())})
    return reports


class UpdateChannels:
    def __init__(self, native, observed, torch, np, exact, differences, compare, tolerance, limits):
        self.native, self.observed, self.torch, self.np = native, observed, torch, np
        self.exact, self.differences, self.compare = exact, differences, compare
        self.tolerance, self.limits = tolerance, limits
        self.frames, self.records, self.rows = {}, [], {'reference': [], 'candidate': []}
        self.bytes = self.peak = self.neutral_checks = 0
        self.backward_counts = {'reference': 0, 'candidate': 0}

    def install(self, unit, role):
        torch = self.torch
        active = {'step': 0, 'channels': {}, 'inputs': [], 'prestate': None, 'decoder_calls': 0}
        def read(callback):
            before = self.native.clone(self.native.rng(torch, self.np, True), torch)
            value = callback()
            self.exact(before, self.native.rng(torch, self.np, True), torch, 'update_channel_RNG_neutral')
            self.neutral_checks += 1
            return value
        def before_encoder(module, args):
            if active['step'] < 2:
                active['prestate'] = read(lambda: full_state(unit, self.native, torch, self.np))
                rowptr, col, value = args[1].csr()
                active['inputs'].append(read(lambda: {'x': self.observed.tensor_identity(args[0]),
                    'rowptr': self.observed.tensor_identity(rowptr), 'col': self.observed.tensor_identity(col),
                    'value': None if value is None else self.observed.tensor_identity(value)}))
        def after_encoder(module, args, value):
            if active['step'] < 2:
                active['channels']['encoder_output'] = read(lambda: self.native.clone(value, torch))
        def after_decoder(module, args, value):
            if active['step'] < 2:
                index = active['decoder_calls']
                require(index < 2, 'Expected two original root decoder calls per native update')
                active['inputs'].append(read(lambda: self.observed.tensor_identity(args[2])))
                active['channels']['positive_raw_logits' if index == 0 else 'negative_raw_logits'] = read(lambda: self.native.clone(value, torch))
                active['decoder_calls'] += 1
        original_backward = torch.Tensor.backward
        def backward(tensor, *args, **kwargs):
            require(not args and not kwargs and tensor.numel() == 1, 'Original scalar backward call differs')
            if active['step'] < 2:
                active['channels']['loss'] = read(lambda: self.native.clone(tensor, torch))
            original_backward(tensor, *args, **kwargs)
            self.backward_counts[role] += 1
        def pre(optimizer, args, kwargs):
            if active['step'] < 2:
                require(active['decoder_calls'] == 2 and active['prestate'] is not None, 'Selected update forward coverage differs')
                active['channels']['named_preAdam_gradients'] = read(lambda: self.native.capture(unit, torch, self.np)['gradients'])
        def post(optimizer, args, kwargs):
            before_rng = self.native.clone(self.native.rng(torch, self.np, True), torch)
            step = active['step']
            if step < 2:
                from step_observer import tensor_bytes
                frame = {'prestate': active['prestate'], 'inputs': active['inputs'], 'channels': active['channels']}
                size = tensor_bytes(frame, torch)
                require(self.bytes + size <= self.limits['selected_channel_tensor_payload_bytes'], 'Selected channel live tensor payload cap exceeded')
                self.peak = max(self.peak, self.bytes + size)
                self.rows[role].append({'step': step + 1, 'channel_identities': {k: self.observed.state_identity(v, torch) for k, v in frame['channels'].items()},
                    'prestate_sha256': self.observed.state_identity(frame['prestate'], torch), 'inputs': frame['inputs']})
                if role == 'reference':
                    self.frames[step] = frame
                    self.bytes += size
                    require(self.bytes <= self.limits['selected_channel_tensor_payload_bytes'], 'Selected reference channel cap exceeded')
                    self.peak = max(self.peak, self.bytes)
                else:
                    reference = self.frames.pop(step)
                    state = self.differences(frame['prestate'], reference['prestate'], torch, 0, 0)
                    inspected = all(f['prestate']['unregistered_module_state']['fully_inspected'] for f in (frame, reference))
                    inputs_exact = frame['inputs'] == reference['inputs']
                    reports, verdicts = [], []
                    for channel in frame['channels']:
                        verdict = {'channel': channel, 'passed_original_fixed_predicate': True}
                        try:
                            self.compare(frame['channels'][channel], reference['channels'][channel], torch,
                                         self.tolerance['atol'], self.tolerance['rtol'], 'update' + str(step + 1) + '/' + channel, reports)
                        except Exception as error:
                            verdict.update(passed_original_fixed_predicate=False, type=type(error).__name__, condition=str(error))
                        verdicts.append(verdict)
                    self.records.append({'step': step + 1, 'complete_preforward_state': state,
                        'state_fully_inspected': inspected, 'native_inputs_exact': inputs_exact,
                        'same_state_interpretation_allowed': state['exact'] and inspected and inputs_exact,
                        'interpretation': 'Original-rule trajectory comparison; unequal updated prestates forbid same-state causal interpretation.',
                        'original_fixed_comparator_reports': reports, 'original_fixed_comparator_verdicts': verdicts})
                    self.bytes -= tensor_bytes(reference, torch)
            active.update(step=step + 1, channels={}, inputs=[], prestate=None, decoder_calls=0)
            self.exact(before_rng, self.native.rng(torch, self.np, True), torch, 'update_channel_post_RNG_neutral')
            self.neutral_checks += 1
        handles = [unit[0].register_forward_pre_hook(before_encoder), unit[0].register_forward_hook(after_encoder),
                   unit[1].decoder.register_forward_hook(after_decoder), unit[2].register_step_pre_hook(pre), unit[2].register_step_post_hook(post)]
        require(torch.Tensor.backward is original_backward, 'Backward changed before observer install')
        torch.Tensor.backward = backward
        def remove():
            require(torch.Tensor.backward is backward, 'Backward observer unexpectedly replaced')
            torch.Tensor.backward = original_backward
            for handle in handles:
                handle.remove()
        return remove

    def finish(self):
        require(self.backward_counts == {'reference': 36, 'candidate': 36} and len(self.records) == 2
                and self.bytes == 0 and not self.frames, 'Complete selected update/epoch coverage required')
        return {'steps': [1, 2], 'roles': self.rows, 'comparisons': self.records,
                'backward_calls': self.backward_counts, 'RNG_neutral_observations': self.neutral_checks,
                'peak_selected_reference_tensor_payload_bytes': self.peak, 'retained_tensor_payload_bytes': self.bytes,
                'no_extra_replay': True, 'numerical_mismatch_truncates_native_epoch': False}
