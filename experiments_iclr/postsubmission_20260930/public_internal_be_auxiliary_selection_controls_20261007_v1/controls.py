"""Two full-training attribution controls; sealed public driver delegation.

Import/CLI parsing is stdlib only. No numerical source is imported until an
explicit run_complete call. This packet adopts no seeds, campaign or execution.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

PUBLIC_NAME = 'portable_internal_be_public_interface_20261007_v2'
PUBLIC_SHA = '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
TASKS = ('wikics', 'collab', 'molhiv')
CONTROLS = ('be_init_alignment_only', 'untied4_own_joint')
UPDATES = {'wikics': 1, 'collab': 18, 'molhiv': 258}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(control):
    require(control in CONTROLS, 'One of the two declared controls required')
    shared = control == 'be_init_alignment_only'
    return {'control_id': control, 'constructor_arm': 'be_init_contrastive' if shared else 'independent4',
        'architecture': 'shared initialized four-member BE' if shared else 'four original untied native bodies',
        'TRAIN_update_change': 'omit only residual_member_contrast' if shared else 'none: original independent4 own-only per-batch update',
        'local_restore_change_vs_constructor_arm': 'none: original joint package restoration' if shared else
            'WikiCS joint local restore replaces ordinary own-local restore and changes later trajectories',
        'whole_acquisition_trajectory_claimed_unchanged': False,
        'auxiliary_permission': 'All source-active trainable parameters; no block masking/detach' if shared else 'none',
        'own_reduction': 'mean across four two-view member losses' if shared else 'sum of four two-view member own losses; unscaled per-body gradients',
        'alignment_weight': .05 if shared else 0., 'residual_weight': 0.,
        'two_view_member_forward_contract': 8, 'native_Adam_calls_per_update': 1 if shared else 4,
        'reverse_API_contract': 'one source scalar backward per completed update; FLOPs/memory not equated by this count',
        'selector': 'strict first maximum complete VALID joint pool; joint WikiCS local model/Adam restore, live end-local RNG',
        'ordinary_own_selected_independent4': False, 'own_best_artifacts_used_for_selection_or_transition': False,
        'public_V2_manifest_sha256': PUBLIC_SHA, 'interface_program_sha256': sha(__file__),
        'source_preparation_adopts_execution': False, 'runtime_verified_at_preparation': False,
        'exact_resume_supported': False, 'TEST_scoring': False, 'campaign_or_seeds_adopted': False}


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


@contextmanager
def _aliases(values):
    missing = object(); old = {key: sys.modules.get(key, missing) for key in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for key, value in old.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def sources(public_root=None):
    root = Path(public_root or Path(__file__).resolve().parent.parent / PUBLIC_NAME).resolve(strict=True)
    require(sha(root / 'MANIFEST.json') == PUBLIC_SHA, 'Exact sealed public V2 required')
    for row in json.loads((root / 'MANIFEST.json').read_text())['files']:
        path = (root / row['path']).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and sha(path) == row['sha256']
                and path.stat().st_size == row['bytes'], 'Frozen public source changed: ' + row['path'])
    public = _module(root / 'portable.py', '_aux_selection_public_v2')
    with _aliases({'portable': public}):
        data = _module(root / 'data_interface.py', '_aux_selection_data')
        with _aliases({'data_interface': data}):
            driver = _module(root / 'train.py', '_aux_selection_original_driver')
    return public, driver


class ControlSession:
    """Preserve the original Session and expose labelled joint-control metadata."""
    def __init__(self, raw, control, config):
        self.raw, self.arm, self.config, self.control = raw, control, config, control
        self.definition = identity(control); self.last_train = None; self.in_train = False
        shared = control == 'be_init_alignment_only'
        require(raw.arm == self.definition['constructor_arm'] and raw.model.members == 4
                and raw.model.independent is not shared and len(raw.model.models) == (1 if shared else 4)
                and raw.model.contrastive is shared
                and len(raw.optimizers) == (1 if shared else 4), 'Exact source shared/untied architecture required')
        self.work = {'TRAIN_Session_forward_calls': 0, 'completed_TRAIN_member_view_forwards': 0,
            'native_Adam_calls': 0, 'completed_native_Adam_calls': 0, 'completed_updates': 0,
            'complete_VALID_evaluations': 0, 'complete_VALID_dispatch_seconds': 0.}
        original_forward = raw.forward
        def forward(*args, **kwargs):
            if self.in_train:
                self.work['TRAIN_Session_forward_calls'] += 1
            result = original_forward(*args, **kwargs)
            if self.in_train:
                self.work['completed_TRAIN_member_view_forwards'] += raw.model.members
            return result
        raw.forward = forward  # Exact source RNG/forward delegate; no extra view.
        for optimizer in raw.optimizers:
            original_step = optimizer.step
            def step(*args, _original=original_step, **kwargs):
                self.work['native_Adam_calls'] += 1
                result = _original(*args, **kwargs)
                self.work['completed_native_Adam_calls'] += 1
                return result
            optimizer.step = step

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def metadata(self):
        return {**self.definition, 'actual_work': dict(self.work), 'source_steps': self.raw.steps,
                'last_TRAIN_objectives': self.last_train,
                'reverse_API_count_at_completed_updates_source_contract': self.work['completed_updates'],
                'partial_failed_update_reverse_calls_not_instrumented': True}

    def _cpu_tree(self, value):
        result = self.raw._cpu_tree(value)
        if isinstance(result, dict) and 'model' in result:
            role = ('unused per-body diagnostic; never joint selection/transition input'
                    if 'optimizer' in result and 'optimizers' not in result else 'joint selected bank')
            result.update(auxiliary_selection_control=self.metadata(), method_identity=self.control,
                          control_checkpoint_role=role)
        return result

    def _alignment_step(self, batch, labels):
        """Original shared two-view step with only residual call/addend omitted."""
        torch = self.torch
        require(labels.device == self.device, 'Move batch/labels to source Session.device')
        self.model.train()
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        la, ha = self.raw.forward(batch); lb, hb = self.raw.forward(batch)
        selection, losses = self.core['selection'], self.core['objectives']
        selection.finite_predictions(la, self.serving(la)); selection.finite_predictions(lb, self.serving(lb))
        if not torch.isfinite(ha).all() or not torch.isfinite(hb).all():
            raise FloatingPointError('TRAIN representation')
        size = min(len(labels), self.config['contrastive']['max_objects'])
        index = torch.linspace(0, len(labels) - 1, steps=size, device=labels.device).long()
        own = .5 * (losses.own_supervision(la, labels, self.task) + losses.own_supervision(lb, labels, self.task))
        target_identity = batch['query'][index] if self.task == 'collab' else None
        alignment = losses.alignment_loss(ha[:, index], hb[:, index], labels[index], self.task, identities=target_identity)
        auxiliary = .05 * alignment
        total = own.mean() + auxiliary
        if not torch.isfinite(total):
            raise FloatingPointError('TRAIN loss')
        total.backward()  # Original unrestricted active-parameter permissions.
        active = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise FloatingPointError('TRAIN gradient')
        for optimizer in self.optimizers:
            optimizer.step()
        selection.finite_state(self.model, self.optimizers)
        self.raw.steps += 1
        return {'loss': total.detach(), 'own_mean': own.mean().detach(), 'auxiliary': auxiliary.detach(),
                'alignment': alignment.detach()}

    def train_step(self, batch, labels):
        self.in_train = True
        try:
            result = (self._alignment_step(batch, labels) if self.control == 'be_init_alignment_only'
                      else self.raw.train_step(batch, labels))
        finally:
            self.in_train = False
        self.work['completed_updates'] += 1
        self.last_train = {key: float(value) for key, value in result.items()}
        return result

    def save_training_state(self, *args, **kwargs):
        raise NotImplementedError('Use labelled joint selected snapshots; exact resume unsupported')

    def restore_training_state(self, *args, **kwargs):
        raise NotImplementedError('No unlabelled ordinary/base-arm resume path')


def run_complete(task, control, train, valid, output, seed, device='cpu',
                 polynormer=None, ncn_model=None, ncn_utils=None, public_root=None):
    """Explicit one-cell full source run, serial per process; no epoch-loop copy."""
    require(task in TASKS and type(seed) is int, 'Declared task and explicit integer seed required')
    definition = identity(control); output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic(); box = {}
    try:
        public, driver = sources(public_root)
        original_snapshot, original_write, original_evaluate = driver.joint_snapshot, driver.json_write, driver.evaluate

        def recipe(task):
            config = copy.deepcopy(public.recipe(task)); config['arms'] = [control]
            config['frozen_base_recipe_sha256'] = sha(public.ROOT / 'recipes' / (task + '.json'))
            config['frozen_base_contrastive_recipe'] = copy.deepcopy(config['contrastive'])
            config['contrastive']['alignment_weight'] = definition['alignment_weight']
            config['contrastive']['residual_weight'] = definition['residual_weight']
            config['effective_auxiliary_coefficients'] = {'A': definition['alignment_weight'], 'R': 0.}
            config['auxiliary_selection_control'] = definition
            return config

        def factory(task, arm, seed, device, polynormer, ncn_model, ncn_utils):
            require(arm == control, 'One labelled control per fresh run')
            raw = public.Session(task, definition['constructor_arm'], seed, device, polynormer, ncn_model, ncn_utils)
            session = ControlSession(raw, control, recipe(task)); box['session'] = session
            return session

        def annotate(value):
            session = box.get('session')
            return {**value, 'auxiliary_selection_control': session.metadata() if session else definition,
                    'method_identity': control, 'underlying_constructor_arm': definition['constructor_arm']}

        def snapshot(session, epoch, metric, per, run):
            return original_snapshot(session, epoch, metric, per, annotate(run))

        def evaluate(session, train, valid):
            began = time.monotonic(); result = original_evaluate(session, train, valid)
            session.work['complete_VALID_evaluations'] += 1
            session.work['complete_VALID_dispatch_seconds'] += time.monotonic() - began
            return result

        def write(path, value):
            name = Path(path).name; session = box.get('session')
            if isinstance(value, dict) and name in ('RUN.json', 'PROGRESS.json', 'COMPLETE.json', 'FAILURE.json'):
                value = annotate(value)
                if name == 'COMPLETE.json':
                    steps = session.config['training']['epochs'] * UPDATES[task]; work = session.work
                    require(session.steps == work['completed_updates'] == steps
                            and work['TRAIN_Session_forward_calls'] == 2 * steps
                            and work['completed_TRAIN_member_view_forwards'] == 8 * steps
                            and work['native_Adam_calls'] == work['completed_native_Adam_calls']
                                == definition['native_Adam_calls_per_update'] * steps
                            and work['complete_VALID_evaluations'] == session.config['training']['epochs'],
                            'Complete native TRAIN/VALID/views/update work required')
            elif name == 'VALID_TRACE.json' and session is not None:
                for row in value:
                    if 'auxiliary_selection_control_id' not in row:
                        row['auxiliary_selection_control_id'] = control
                        row['TRAIN']['control_objectives'] = dict(session.last_train)
            original_write(path, value)

        driver.Session, driver.recipe = factory, recipe
        driver.joint_snapshot, driver.json_write, driver.evaluate = snapshot, write, evaluate
        # Both control IDs differ from ordinary independent4, so the original
        # main uses joint selection/local transition and never own-bank assembly.
        arguments = ['auxiliary-selection-complete', '--task', task, '--arm', control, '--seed', str(seed),
            '--device', str(device), '--train', str(train), '--valid', str(valid), '--output', str(output)]
        for flag, value in (('--polynormer', polynormer), ('--ncn-model', ncn_model), ('--ncn-utils', ncn_utils)):
            if value is not None:
                arguments.extend([flag, str(value)])
        previous = sys.argv
        try:
            sys.argv = arguments
            driver.main()
        finally:
            sys.argv = previous
        return {'output': str(output), 'method_identity': control, 'metadata': box['session'].metadata(), 'TEST_scoring': False}
    except BaseException as error:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
            failure = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                'seconds': time.monotonic() - started, 'automatic_retry': False, 'TEST_scoring': False,
                'method_identity': control, 'auxiliary_selection_control': definition,
                'failure_stage': 'source setup before unchanged full driver'}
            (output / 'FAILURE.json').write_text(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--control', choices=CONTROLS, required=True)
    parser.add_argument('--seed', type=int, required=True, help='Caller-declared seed; no seed/campaign adopted here')
    parser.add_argument('--device', default='cpu')
    for name in ('train', 'valid', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('polynormer', 'ncn-model', 'ncn-utils', 'public-root'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(); result = run_complete(**vars(args))
    print(json.dumps({'complete': True, 'output': result['output'], 'method_identity': result['method_identity'], 'TEST_scoring': False}))


if __name__ == '__main__':
    main()
