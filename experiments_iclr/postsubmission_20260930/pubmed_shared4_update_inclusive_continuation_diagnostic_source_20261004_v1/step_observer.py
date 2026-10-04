"""Bounded read-only optimizer hooks for two complete native continuations."""
import time


def tensor_bytes(value, torch):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(child, torch) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(child, torch) for child in value)
    return 0


class StepObserver:
    """Hooks return None and never change args, tensors, optimizer or generators.

    First-copy CPU snapshots are retained only until the corresponding second
    pre/post hook compares them. Numerical mismatches are ledger entries, never
    raises or reasons to truncate either original 36-step native epoch.
    """
    def __init__(self, native, observed, torch, np, require, exact, differences,
                 write, output, progress, tolerance, limits):
        self.native, self.observed, self.torch, self.np = native, observed, torch, np
        self.require, self.exact, self.differences, self.write = require, exact, differences, write
        self.output, self.progress, self.tolerance, self.limits = output, progress, tolerance, limits
        self.frames = {}
        self.bytes = self.peak = self.neutral_checks = 0
        self.counts = {'reference_pre': 0, 'reference_post': 0, 'candidate_pre': 0, 'candidate_post': 0}
        self.ledger = {'reference': [], 'candidate': [], 'first_divergence': {},
                       'observer_wall_seconds': 0.0, 'numerical_mismatch_truncates_native_epoch': False,
                       'observer_is_causal_kernel_probe': False}

    def blocks(self, candidate, reference):
        return {key: self.differences(candidate[key], reference[key], self.torch,
                    0 if key in ('RNG', 'flags', 'invest') else self.tolerance['atol'],
                    0 if key in ('RNG', 'flags', 'invest') else self.tolerance['rtol'])
                for key in ('encoder', 'predictor', 'Adam', 'gradients', 'RNG', 'flags', 'invest')}

    def mark(self, name, step, rows):
        for rule in ('exact', 'within_fixed_numeric_rule'):
            channel = name + '/' + rule
            if channel not in self.ledger['first_divergence'] and any(not row[rule] for row in rows.values()):
                self.ledger['first_divergence'][channel] = {
                    'native_epoch': 6, 'step': step + 1,
                    'blocks': [key for key, row in rows.items() if not row[rule]],
                    'interpretation': 'First observed difference in this channel; no causal attribution.'}

    def install(self, unit, role):
        self.require(role in ('reference', 'candidate'), 'Known diagnostic copy required')
        pending = {'step': None, 'transition_pre': None}

        def observe(phase):
            started = time.monotonic()
            before_rng = self.native.clone(self.native.rng(self.torch, self.np, True), self.torch)
            step = self.counts[role + '_' + phase]
            self.require(step < 36, 'No extra native optimizer step admitted')
            if phase == 'pre':
                self.require(pending['step'] is None, 'Diagnostic native step already begun')
                pending['step'] = step
            else:
                self.require(pending['step'] == step, 'Diagnostic pre/post hook order differs')
            snapshot = self.native.capture(unit, self.torch, self.np)
            size = tensor_bytes(snapshot, self.torch)
            self.require(size <= self.limits['one_complete_snapshot_tensor_payload_bytes'], 'Diagnostic snapshot payload cap exceeded')
            digest = self.observed.state_identity(snapshot, self.torch)
            row = {'native_epoch': 6, 'step': step + 1, 'phase': phase,
                   'complete_state_sha256': digest, 'CPU_tensor_payload_bytes': size}
            if step < 2:
                from update_observer import parameter_mapping, moment_transitions
                row['optimizer_parameter_mapping'] = parameter_mapping(unit, snapshot['Adam'], self.torch)
                if phase == 'pre':
                    pending['transition_pre'] = snapshot
                else:
                    row['moment_step_transitions'] = moment_transitions(unit, pending['transition_pre'], snapshot,
                        self.torch, self.differences, self.tolerance)
                    pending['transition_pre'] = None
            if role == 'reference':
                self.require(self.bytes + size <= self.limits['reference_tensor_payload_bytes'], 'Diagnostic reference payload cap exceeded')
                frame = self.frames.setdefault(step, {})
                self.require(phase not in frame, 'Reference snapshot already present')
                frame[phase] = snapshot
                self.bytes += size
                self.peak = max(self.peak, self.bytes)
                self.require(len(self.frames) <= 36, 'Reference native step count cap exceeded')
            else:
                self.require(step in self.frames and phase in self.frames[step], 'Matching original native step snapshot absent')
                reference = self.frames[step].pop(phase)
                row['blocks'] = self.blocks(snapshot, reference)
                if phase == 'pre':
                    self.mark('pre_Adam_gradients', step, {'gradients': row['blocks']['gradients']})
                    self.mark('pre_Adam_model_or_Adam', step, {k: row['blocks'][k] for k in ('encoder', 'predictor', 'Adam')})
                else:
                    self.mark('post_Adam_model_or_Adam', step, {k: row['blocks'][k] for k in ('encoder', 'predictor', 'Adam')})
                self.bytes -= tensor_bytes(reference, self.torch)
                del reference
                if not self.frames[step]:
                    del self.frames[step]
            self.counts[role + '_' + phase] += 1
            if phase == 'post':
                pending['step'] = None
            self.ledger[role].append(row)
            self.progress.add(**{'diagnostic_optimizer_' + phase + '_observations': 1})
            self.progress.update(diagnostic_reference_payload_bytes=self.bytes,
                                 diagnostic_reference_peak_payload_bytes=self.peak)
            row['observer_wall_seconds'] = 0.0
            self.write(self.output / 'STEP_DIVERGENCE_PROGRESS.json', self.receipt())
            self.exact(before_rng, self.native.rng(self.torch, self.np, True), self.torch,
                       'diagnostic_optimizer_hook_complete_RNG_neutral')
            self.neutral_checks += 1
            elapsed = time.monotonic() - started
            row['observer_wall_seconds'] = elapsed
            self.ledger['observer_wall_seconds'] += elapsed
            # All snapshot/hash/compare/write work belongs to the diagnostic cap.

        def pre(optimizer, args, kwargs):
            observe('pre')

        def post(optimizer, args, kwargs):
            observe('post')

        handles = [unit[2].register_step_pre_hook(pre), unit[2].register_step_post_hook(post)]
        def remove():
            for handle in handles:
                handle.remove()
            if pending['step'] is not None:
                self.ledger['incomplete_native_step_observed'] = {'role': role, 'step': pending['step'] + 1}
        return remove

    def receipt(self):
        return {**self.ledger, 'hook_counts': dict(self.counts),
                'hook_RNG_neutral_checks': self.neutral_checks,
                'retained_reference_tensor_payload_bytes': self.bytes,
                'peak_reference_tensor_payload_bytes': self.peak,
                'retained_reference_steps': len(self.frames), 'limits': self.limits,
                'reference_snapshots_written_to_disk': False,
                'progress_receipt_is_lower_bound_until_next_hook_or_final_result': True,
                'memory_limit_scope': 'Tensor payload and frame count bounds; Python metadata and all transient/I/O costs remain under original supervisor RSS/wall and CUDA caps.'}

    def finish(self):
        self.require(all(count == 36 for count in self.counts.values()), 'Complete native diagnostic hook coverage required')
        self.require(self.neutral_checks == 144 and self.bytes == 0 and not self.frames, 'Diagnostic reference snapshots or hook checks incomplete')
        return self.receipt()
