"""Prospective eight-view controls through the unchanged public V2 interface.

Import is stdlib only. No CLI, download, process, resource qualification or
automatic launch path is provided. Numerical work requires a later explicit
call with later_execution_authorized=True. Source preparation is not adoption.
"""
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PUBLIC_NAME = 'portable_internal_be_public_interface_20261007_v2'
PUBLIC_MANIFEST_SHA = '190940ca9f8141ac45f739d064cb1ef76aaf1965ba8c91adb544ad8e38fef724'
CONTROLS = ('single8_own', 'single8_alignment', 'single8_virtual_contrast')
EXPECTED_UPDATES = {'wikics': 1, 'collab': 18, 'molhiv': 258}
TRAIN_OBJECTS_PER_EPOCH = {'wikics': 580, 'collab': 2 * 1179052, 'molhiv': 32901}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def control_spec(control):
    require(control in CONTROLS, 'Declared prospective single8 control required')
    return {'control': control, 'underlying_public_arm': 'single', 'parameter_bodies': 1,
        'Adam_optimizers': 1, 'Adam_steps_per_batch': 1, 'persistent_training_view_streams': 4,
        'views_per_stream_per_update': 2, 'full_own_supervised_views_per_update': 8,
        'own_reduction': 'mean of all eight full-object task losses',
        'alignment_weight': 0. if control == 'single8_own' else .05,
        'residual_weight': .05 if control == 'single8_virtual_contrast' else 0.,
        'temperature': .2, 'auxiliary_max_objects': 512,
        'stream_seed_rule': 'seed + 1009 * stream_index0 + 300001; same source BE stream rule',
        'virtual_axis': 'Four stochastic views of one parameter body, not four learned members',
        'serving': 'deterministic native single-body eval; no stochastic/view ensemble',
        'prior_attribution': 'Standard multi-view loss averaging and known supervised/view contrastive priors; suite CDLG-inspired residual proxy only in optional virtual extension',
        'checkpoint': 'Unchanged public complete-VALID strict first maximum and native local transition',
        'source_preparation_adopts_execution': False, 'exact_resume_supported': False}


def _verify_public(root):
    root = Path(root).resolve(strict=True)
    manifest = root / 'MANIFEST.json'
    require(sha(manifest) == PUBLIC_MANIFEST_SHA, 'Exact preserved public V2 seal required')
    for row in json.loads(manifest.read_text())['files']:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Public source path leaves root')
        path = (root / rel).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file()
                and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Frozen public V2 source changed: ' + row['path'])
    return root


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _aliases(values):
    missing = object(); previous = {name: sys.modules.get(name, missing) for name in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_public(public_root=None):
    root = _verify_public(public_root or Path(__file__).resolve().parent.parent / PUBLIC_NAME)
    public = _module(root / 'portable.py', '_single8_public_v2_portable')
    with _aliases({'portable': public}):
        data = _module(root / 'data_interface.py', '_single8_public_v2_data_interface')
        with _aliases({'data_interface': data}):
            driver = _module(root / 'train.py', '_single8_public_v2_complete_training')
    return public, driver


def _recipe(public, task):
    config = copy.deepcopy(public.recipe(task))
    require(config['contrastive'] == {'alignment_weight': .05, 'residual_weight': .05,
                                     'temperature': .2, 'max_objects': 512}, 'Fixed source auxiliary recipe differs')
    # This separate adapter family does not register arms in the frozen recipe.
    config['base_public_arms'] = config['arms']
    config['arms'] = list(CONTROLS)
    config['label_opportunity'] = 'eight_full_own_loss_views; four persistent streams twice; one body and one Adam step'
    config['serving_control'] = 'single-body deterministic eval; no virtual-view ensemble'
    config['prospective_control_family'] = True
    return config


class Single8Session:
    """One source single Session with a separate four-stream training view bank.

    Source Session remains the parameter/optimizer/serving implementation.
    Four training stream states are added to selected snapshot metadata; source
    serving streams and model.members remain single. Exact continuation is not
    offered by this adapter or the reused complete training driver.
    """
    def __init__(self, source_session, control, config):
        self.source_session = source_session
        self.control = self.arm = control
        self.config = config
        self.spec = control_spec(control)
        require(source_session.arm == 'single' and source_session.model.members == 1
                and len(source_session.model.models) == 1 and len(source_session.optimizers) == 1,
                'Exactly one unchanged ordinary body/Adam required')
        torch = self.torch
        # The first stream is a clone of the source's single stream. The other
        # three use the same persistent source member-stream seed schedule.
        self.training_view_streams = [{k: value.clone() for k, value in source_session.streams[0].items()}]
        for view in range(1, 4):
            with torch.random.fork_rng(devices=self.cuda_devices):
                seed = self.seed + 1009 * view + 300001
                torch.manual_seed(seed)
                stream = {'cpu': torch.get_rng_state()}
                if self.cuda_index is not None:
                    torch.cuda.manual_seed(seed)
                    stream['cuda'] = torch.cuda.get_rng_state(self.cuda_index)
                self.training_view_streams.append(stream)
        self.control_counts = {'training_body_forwards': 0, 'full_own_supervision_views': 0,
            'supervised_object_view_presentations': 0, 'Adam_step_calls': 0,
            'alignment_stream_object_pairs': 0, 'residual_selected_objects': 0}

    def __getattr__(self, name):
        return getattr(self.source_session, name)

    def forward(self, batch):
        require(not self.model.training, 'Use train_step for eight-view TRAIN; forward is deterministic serving only')
        return self.source_session.forward(batch)

    def serving(self, logits):
        require(logits.shape[0] == 1, 'No stochastic/virtual view bank may be served')
        return self.source_session.serving(logits)

    def _training_views(self, batch):
        torch = self.torch; rows = []
        for stream in self.training_view_streams:
            with torch.random.fork_rng(devices=self.cuda_devices):
                torch.set_rng_state(stream['cpu'])
                if self.cuda_index is not None:
                    torch.cuda.set_rng_state(stream['cuda'], self.cuda_index)
                # Always source member zero: no private routing or cloned body.
                rows.append(self.model.member_forward(batch, 0))
                self.control_counts['training_body_forwards'] += 1
                stream['cpu'] = torch.get_rng_state()
                if self.cuda_index is not None:
                    stream['cuda'] = torch.cuda.get_rng_state(self.cuda_index)
        return torch.stack([row[0] for row in rows]), torch.stack([row[1] for row in rows])

    def train_step(self, batch, labels):
        torch = self.torch
        require(labels.device == self.device and len(labels) > 0, 'Complete nonempty labels on Session.device required')
        self.model.train(); optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        la, ha = self._training_views(batch); lb, hb = self._training_views(batch)
        selection = self.core['selection']; losses = self.core['objectives']
        # Source finite-prediction check over all TRAIN views. These stochastic
        # means are only a finite guard, never the control's served prediction.
        selection.finite_predictions(la, self.source_session.serving(la))
        selection.finite_predictions(lb, self.source_session.serving(lb))
        if not torch.isfinite(ha).all() or not torch.isfinite(hb).all():
            raise FloatingPointError('TRAIN representation')
        own = .5 * (losses.own_supervision(la, labels, self.task)
                    + losses.own_supervision(lb, labels, self.task))
        self.control_counts['full_own_supervision_views'] += 8
        self.control_counts['supervised_object_view_presentations'] += 8 * len(labels)
        alignment = residual = own.sum() * 0
        if self.control != 'single8_own':
            size = min(len(labels), self.config['contrastive']['max_objects'])
            index = torch.linspace(0, len(labels) - 1, steps=size, device=labels.device).long()
            identity = batch['query'][index] if self.task == 'collab' else None
            alignment = losses.alignment_loss(ha[:, index], hb[:, index], labels[index],
                self.task, temperature=.2, identities=identity)
            self.control_counts['alignment_stream_object_pairs'] += 4 * size
            if self.control == 'single8_virtual_contrast':
                residual = losses.residual_member_contrast(ha[:, index], hb[:, index],
                                                           labels[index], temperature=.2)
                self.control_counts['residual_selected_objects'] += size
        auxiliary = self.spec['alignment_weight'] * alignment + self.spec['residual_weight'] * residual
        total = own.mean() + auxiliary
        if not torch.isfinite(total):
            raise FloatingPointError('TRAIN loss')
        total.backward()
        active = [parameter for parameter in self.model.parameters() if parameter.grad is not None]
        if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise FloatingPointError('TRAIN gradient')
        optimizer.step(); self.control_counts['Adam_step_calls'] += 1
        selection.finite_state(self.model, self.optimizers)
        self.source_session.steps += 1
        return {'loss': total.detach(), 'own_mean': own.mean().detach(), 'auxiliary': auxiliary.detach(),
                'alignment': alignment.detach(), 'residual': residual.detach()}

    def selected_snapshot_metadata(self):
        return self._cpu_tree({'single8_control': self.spec,
            'training_view_streams': self.training_view_streams,
            'control_counts': self.control_counts, 'steps': self.steps,
            'source_serving_members': self.model.members, 'exact_resume_supported': False})

    def save_training_state(self, *args, **kwargs):
        raise NotImplementedError('Use selected snapshots from run_complete; exact continuation is unsupported')

    def restore_training_state(self, *args, **kwargs):
        raise NotImplementedError('No incomplete view-RNG continuation or restart path is supplied')


def _make(public, task, control, seed, device, polynormer, ncn_model, ncn_utils):
    control_spec(control)
    source = public.Session(task, 'single', seed, device, polynormer, ncn_model, ncn_utils)
    return Single8Session(source, control, _recipe(public, task))


def make_session(task, control, seed=6101, device='cpu', polynormer=None,
                 ncn_model=None, ncn_utils=None, public_root=None, *, later_execution_authorized=False):
    """Later callable update/serving adapter; default refuses numerical work."""
    require(later_execution_authorized is True, 'Prospective source only; later independent adoption required')
    public, _ = _load_public(public_root)
    return _make(public, task, control, seed, device, polynormer, ncn_model, ncn_utils)


def run_complete(task, control, train, valid, output, seed=6101, device='cpu',
                 polynormer=None, ncn_model=None, ncn_utils=None, public_root=None,
                 *, later_execution_authorized=False):
    """Later full-population run through unchanged public train.main.

    Interface substitution is limited to this newly loaded private module.
    Original source modules/files/recipes/families remain unchanged. Calls
    must be serial within a process because the original argparse main reads
    sys.argv. No worker, queue, release, launch, retry or performance gate.
    """
    require(later_execution_authorized is True, 'Prospective source only; later independent adoption required')
    control_spec(control)
    public, driver = _load_public(public_root); box = {}
    original_snapshot, original_write = driver.joint_snapshot, driver.json_write

    def factory(task, arm, seed, device, polynormer, ncn_model, ncn_utils):
        require(arm == control, 'One fixed control per fresh full run')
        session = _make(public, task, arm, seed, device, polynormer, ncn_model, ncn_utils)
        box['session'] = session
        return session

    def snapshot(session, epoch, metric, per, run):
        value = original_snapshot(session, epoch, metric, per, run)
        value.update(session.selected_snapshot_metadata())
        return value

    def write(path, value):
        name = Path(path).name; session = box.get('session')
        if isinstance(value, dict) and name in ('RUN.json', 'COMPLETE.json', 'FAILURE.json', 'PROGRESS.json'):
            value = dict(value, single8_control=control_spec(control), adapter_sha256=sha(__file__),
                         public_V2_manifest_sha256=PUBLIC_MANIFEST_SHA)
            if session is not None:
                value['single8_counts'] = dict(session.control_counts)
            if name == 'COMPLETE.json':
                steps = session.config['training']['epochs'] * EXPECTED_UPDATES[task]
                counts = session.control_counts
                require(session.steps == steps and counts['Adam_step_calls'] == steps
                        and counts['training_body_forwards'] == counts['full_own_supervision_views'] == 8 * steps
                        and counts['supervised_object_view_presentations']
                            == 8 * session.config['training']['epochs'] * TRAIN_OBJECTS_PER_EPOCH[task],
                        'Complete source own population/eight-view/one-update counts required')
        original_write(path, value)

    driver.Session = factory
    driver.recipe = lambda task: _recipe(public, task)
    driver.joint_snapshot = snapshot
    driver.json_write = write
    arguments = ['single8-public-V2-callable', '--task', task, '--arm', control,
                 '--seed', str(seed), '--device', str(device), '--train', str(train),
                 '--valid', str(valid), '--output', str(output)]
    for flag, value in (('--polynormer', polynormer), ('--ncn-model', ncn_model), ('--ncn-utils', ncn_utils)):
        if value is not None:
            arguments.extend([flag, str(value)])
    previous = sys.argv
    try:
        sys.argv = arguments
        driver.main()  # Unchanged complete epochs/batches/evaluation/selection/failure handling.
    finally:
        sys.argv = previous
    return {'output': str(output), 'control': control, 'steps': box['session'].steps,
            'counts': dict(box['session'].control_counts), 'TEST_scoring': False}
