"""Prospective own-only single8 with one live TRAIN tape, public V2 full driver.

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
CONTROLS = ('single8_own',)
PREDECESSOR_SINGLE8_SHA = '2c73522de53a7ff6ffe940189191377cbb5c838bd7ad377ac89b0cc84d828a08'
EXECUTION_MODE = 'single8_own_sequential_CE_at_old_parameters_v1'
EXPECTED_UPDATES = {'wikics': 1}
TRAIN_OBJECTS_PER_EPOCH = {'wikics': 580}


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
        'alignment_weight': 0., 'residual_weight': 0.,
        'execution_mode': EXECUTION_MODE,
        'backward_calls_per_update': 8,
        'maximum_live_TRAIN_backbone_tapes': 1,
        'shadow_or_replay_forwards': 0,
        'temperature': .2, 'auxiliary_max_objects': 512,
        'stream_seed_rule': 'seed + 1009 * stream_index0 + 300001; same source BE stream rule',
        'virtual_axis': 'Four stochastic views of one parameter body, not four learned members',
        'serving': 'deterministic native single-body eval; no stochastic/view ensemble',
        'prior_attribution': 'Established multi-view mean-own-loss attribution control; no novel GNNM method',
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
    require(task == 'wikics', 'Only the unchanged full WikiCS single8_own task is supported')
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
        require(source_session.task == 'wikics' and source_session.steps == 0
                and not source_session.model.contrastive,
                'Fresh own-only full WikiCS native single required')
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
        self.control_counts = {'training_body_forward_call_attempts': 0,
            'training_body_forwards': 0, 'full_own_supervision_views': 0,
            'supervised_object_view_presentations': 0, 'backward_call_attempts': 0,
            'backward_calls': 0, 'Adam_step_calls': 0,
            'old_parameter_version_checks': 0, 'complete_accumulated_updates': 0,
            'maximum_live_TRAIN_backbone_tapes': 0, 'serving_body_forwards': 0,
            'alignment_stream_object_pairs': 0, 'residual_selected_objects': 0}

    def __getattr__(self, name):
        return getattr(self.source_session, name)

    def forward(self, batch):
        require(not self.model.training, 'Use train_step for eight-view TRAIN; forward is deterministic serving only')
        result = self.source_session.forward(batch)
        self.control_counts['serving_body_forwards'] += 1
        return result

    def serving(self, logits):
        require(logits.shape[0] == 1, 'No stochastic/virtual view bank may be served')
        return self.source_session.serving(logits)

    def train_step(self, batch, labels):
        """Accumulate eight CE/8 gradients at old parameters, then one native Adam.

        Ordering is A0,A1,A2,A3,B0,B1,B2,B3, exactly the predecessor's two
        four-stream calls. Only detached scalar diagnostics survive a view's
        backward. No TRAIN shadow/replay, pooled objective or auxiliary is used.
        """
        torch = self.torch
        require(self.task == 'wikics' and labels.device == self.device
                and len(labels) == TRAIN_OBJECTS_PER_EPOCH['wikics'],
                'Complete580 WikiCS TRAIN labels on Session.device required')
        require(self.control == 'single8_own' and not self.model.contrastive
                and self.model.members == 1 and len(self.optimizers) == 1
                and len(self.training_view_streams) == 4,
                'One own-only native function/Adam and four TRAIN streams required')
        self.model.train(); optimizer = self.optimizers[0]
        optimizer.zero_grad(set_to_none=True)
        selection = self.core['selection']; losses = self.core['objectives']
        parameters = tuple(self.model.parameters())
        old_versions = tuple(parameter._version for parameter in parameters)
        own_mean = torch.zeros((), dtype=torch.float32, device=self.device)
        for view in range(2):
            for stream in self.training_view_streams:
                # As in the predecessor, the stream advances only around its
                # native forward; backward runs outside this RNG context.
                with torch.random.fork_rng(devices=self.cuda_devices):
                    torch.set_rng_state(stream['cpu'])
                    if self.cuda_index is not None:
                        torch.cuda.set_rng_state(stream['cuda'], self.cuda_index)
                    self.control_counts['training_body_forward_call_attempts'] += 1
                    logits, representation = self.model.member_forward(batch, 0)
                    self.control_counts['training_body_forwards'] += 1
                    stream['cpu'] = torch.get_rng_state()
                    if self.cuda_index is not None:
                        stream['cuda'] = torch.cuda.get_rng_state(self.cuda_index)
                self.control_counts['maximum_live_TRAIN_backbone_tapes'] = 1
                # One-view probability computation is a finite guard only.
                # The sole scientific objective is the complete native own CE.
                selection.finite_predictions(logits.unsqueeze(0),
                    self.source_session.serving(logits.unsqueeze(0)))
                if not torch.isfinite(representation).all():
                    raise FloatingPointError('TRAIN representation')
                view_own = losses.own_supervision(logits.unsqueeze(0), labels, self.task).mean()
                scaled = view_own / 8
                if not torch.isfinite(view_own) or not torch.isfinite(scaled):
                    raise FloatingPointError('TRAIN own loss')
                own_mean = own_mean + scaled.detach()
                self.control_counts['full_own_supervision_views'] += 1
                self.control_counts['supervised_object_view_presentations'] += len(labels)
                self.control_counts['backward_call_attempts'] += 1
                scaled.backward()
                self.control_counts['backward_calls'] += 1
                # Default backward frees this view's graph; no attached output,
                # representation or loss is retained before the next forward.
                del logits, representation, view_own, scaled
        require(tuple(parameter._version for parameter in parameters) == old_versions,
                'Old parameters must remain unchanged across all eight CE backwards')
        self.control_counts['old_parameter_version_checks'] += 1
        active = [parameter for parameter in parameters if parameter.grad is not None]
        if not active or any(not torch.isfinite(parameter.grad).all() for parameter in active):
            raise FloatingPointError('TRAIN accumulated gradient')
        if not torch.isfinite(own_mean):
            raise FloatingPointError('TRAIN detached mean own loss')
        optimizer.step(); self.control_counts['Adam_step_calls'] += 1
        selection.finite_state(self.model, self.optimizers)
        self.source_session.steps += 1
        self.control_counts['complete_accumulated_updates'] += 1
        auxiliary = torch.zeros_like(own_mean)
        return {'loss': own_mean, 'own_mean': own_mean, 'auxiliary': auxiliary,
                'alignment': auxiliary, 'residual': auxiliary}

    def selected_snapshot_metadata(self):
        return self._cpu_tree({'single8_control': self.spec,
            'training_view_streams': self.training_view_streams,
            'control_counts': self.control_counts, 'steps': self.steps,
            'adapter_sha256': sha(__file__),
            'predecessor_single8_sha256': PREDECESSOR_SINGLE8_SHA,
            'public_V2_manifest_sha256': PUBLIC_MANIFEST_SHA,
            'execution_mode': EXECUTION_MODE,
            'source_serving_members': self.model.members, 'exact_resume_supported': False})

    def save_training_state(self, *args, **kwargs):
        raise NotImplementedError('Use selected snapshots from run_complete; exact continuation is unsupported')

    def restore_training_state(self, *args, **kwargs):
        raise NotImplementedError('No incomplete view-RNG continuation or restart path is supplied')


def _make(public, task, control, seed, device, polynormer, ncn_model, ncn_utils):
    control_spec(control)
    require(task == 'wikics' and ncn_model is None and ncn_utils is None,
            'Full WikiCS only; no NCN or additional task adapter')
    source = public.Session(task, 'single', seed, device, polynormer, ncn_model, ncn_utils)
    return Single8Session(source, control, _recipe(public, task))


def make_session(task, control, seed=6101, device='cpu', polynormer=None,
                 ncn_model=None, ncn_utils=None, public_root=None, *, later_execution_authorized=False):
    """Later callable update/serving adapter; default refuses numerical work."""
    require(later_execution_authorized is True, 'Prospective source only; later independent adoption required')
    control_spec(control)
    require(task == 'wikics', 'Full WikiCS single8_own only')
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
    require(task == 'wikics' and ncn_model is None and ncn_utils is None,
            'Full WikiCS single8_own only')
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
                         public_V2_manifest_sha256=PUBLIC_MANIFEST_SHA,
                         predecessor_single8_sha256=PREDECESSOR_SINGLE8_SHA,
                         execution_mode=EXECUTION_MODE)
            if session is not None:
                value['single8_counts'] = dict(session.control_counts)
            if name == 'COMPLETE.json':
                steps = session.config['training']['epochs'] * EXPECTED_UPDATES[task]
                counts = session.control_counts
                require(session.steps == steps and counts['Adam_step_calls']
                        == counts['complete_accumulated_updates']
                        == counts['old_parameter_version_checks'] == steps
                        and counts['training_body_forward_call_attempts']
                        == counts['training_body_forwards']
                        == counts['full_own_supervision_views']
                        == counts['backward_call_attempts']
                        == counts['backward_calls'] == 8 * steps
                        and counts['supervised_object_view_presentations']
                            == 8 * session.config['training']['epochs'] * TRAIN_OBJECTS_PER_EPOCH[task]
                        and counts['serving_body_forwards'] == session.config['training']['epochs']
                        and counts['maximum_live_TRAIN_backbone_tapes'] == 1
                        and counts['alignment_stream_object_pairs'] == counts['residual_selected_objects'] == 0,
                        'Complete single8 own population/sequential-backward/one-Adam/single-serving counts required')
        original_write(path, value)

    driver.Session = factory
    driver.recipe = lambda task: _recipe(public, task)
    driver.joint_snapshot = snapshot
    driver.json_write = write
    arguments = ['single8-own-sequential-public-V2-callable', '--task', task, '--arm', control,
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
            'counts': dict(box['session'].control_counts), 'execution_mode': EXECUTION_MODE,
            'adapter_sha256': sha(__file__), 'TEST_scoring': False}
