"""Pinned native-link policies and selectors; no model or source execution."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
BINDINGS_SHA256 = 'b959054804e0dc8e25939334ebc4dcd1c8d8b9d56dab68f26e41003e94468d30'
EVALUATION_MODE = 'VALID_only_disclosed_adaptation'


def _inside(path, root):
    root, path = Path(root).resolve(), Path(path)
    path = (path if path.is_absolute() else root/path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Policy/source path escapes the repository root: '+str(path))
    return path


def _receipt(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def source_bindings(root=None):
    """Authenticate metadata and read source bytes without executing source."""
    root = Path(root or PHASE.parent).resolve()
    path = _inside(HERE/'BINDINGS.json', root)
    if _receipt(path)['sha256'] != BINDINGS_SHA256:
        raise ValueError('Frozen link policy bindings changed')
    binding = json.loads(path.read_text())
    for row in binding['source_files']:
        actual = _receipt(_inside(PHASE/row['path_relative_to_phase'], root))
        if any(actual[key] != row[key] for key in ('bytes', 'sha256')):
            raise ValueError('Pinned roster/source changed: '+row['path_relative_to_phase'])
    roster = json.loads(_inside(PHASE/'reference_wrapper_redesign_20261013_v1/ROSTER.json', root).read_text())
    if len(roster['recipes']) != binding['roster_cells'] or binding['roster_cells'] != 27:
        raise ValueError('This policy requires the frozen 27-cell roster')
    recipes = {row['id']: row for row in roster['recipes']}
    for cell in binding['cells']:
        row = recipes[cell['cell_id']]
        if (row['architecture'] != cell['native_family'] or row['dataset'] != cell['dataset']
                or row['tracks'] != cell['tracks'] or row['native_fixed_config'] != cell['native_fixed_config']
                or row['source_bindings']['native_program'] != cell['native_program']):
            raise ValueError('Link policy differs from the frozen roster: '+cell['cell_id'])
    return binding


@dataclass(frozen=True)
class LinkPolicy:
    schema: str
    family: str
    native_family: str
    cell_id: str
    track: str
    roster_track: str
    maximum_epochs: int
    minimum_epochs: int
    patience: int | None
    metric: str
    direction: str
    ties: str
    evaluate_TRAIN_each_epoch: bool
    evaluation_mode: str
    validation_interval_epochs: int
    checkpoint0: bool
    initial_best_metric: float | None
    warmup_epochs: int
    native_schedule: dict
    required_gates: list
    source_phase_gates: dict
    source_role: str
    native_evaluation_roles: list
    runtime_evaluation_roles: list
    source_protocol_equivalent: bool
    source_RNG_trajectory_equivalent: bool
    numerical_qualification_required: bool
    fidelity_limitations: list
    source_binding: dict

    @classmethod
    def from_dict(cls, value, native_family, root=None):
        if not isinstance(value, dict):
            raise ValueError('An explicit link policy dictionary is required')
        native_family = str(native_family).upper()
        binding = source_bindings(root)
        cells = {row['native_family']: row for row in binding['cells']}
        if native_family not in cells:
            raise ValueError('Only frozen NCN/NCNC/BUDDY Collab cells have link policies')
        cell = cells[native_family]
        aliases = {'source_reference': 'native_replica', 'matched_extended': 'long_convergence'}
        track = value.get('track')
        if track not in aliases:
            raise ValueError('This package binds source_reference and matched_extended only; a pilot needs its own frozen policy')
        native_track = cell['tracks'][aliases[track]]
        source = track == 'source_reference'
        selection = native_track['selection']
        if selection != {'checkpoint0': False, 'components': [{'direction': 'max', 'metric': 'hits@50'}],
                         'native_callback_authority': cell['native_program']['path'],
                         'ties': 'first', 'validation_interval_epochs': 1}:
            raise ValueError('Unsupported source link selection contract')
        schedule = native_track['native_schedule']
        if schedule['warmup'] != 0 or schedule['type'] != 'constant native optimizer groups':
            raise ValueError('A new native phase/schedule requires explicit policy implementation')
        stopping = native_track.get('stopping', {})
        gates = [] if source else copy.deepcopy(stopping['required_gates'])
        if any(name != 'native_warmup_and_phase_transitions_complete' for name in gates):
            raise ValueError('An unknown native phase gate cannot be assumed complete')
        canonical = {
            'schema': 'frozen27-link-policy-v1', 'family': 'link', 'native_family': native_family,
            'cell_id': cell['cell_id'], 'track': track, 'roster_track': aliases[track],
            'maximum_epochs': native_track['maximum'],
            'minimum_epochs': native_track['maximum'] if source else stopping['minimum_units'],
            'patience': None if source else stopping['patience_validation_checks'],
            'metric': 'hits@50', 'direction': 'max', 'ties': 'first',
            'evaluate_TRAIN_each_epoch': False, 'evaluation_mode': EVALUATION_MODE,
            'validation_interval_epochs': 1, 'checkpoint0': False,
            'initial_best_metric': cell['source_selection_initial_metric'] if source else None,
            'warmup_epochs': 0, 'native_schedule': copy.deepcopy(schedule),
            'required_gates': gates, 'source_phase_gates': {name: True for name in gates},
            'source_role': 'native_M1_source_horizon_with_disclosed_evaluation_adaptation' if source else 'matched_extended_controller',
            'native_evaluation_roles': copy.deepcopy(cell['native_evaluation_roles']),
            'runtime_evaluation_roles': ['VALID'], 'source_protocol_equivalent': False,
            'source_RNG_trajectory_equivalent': False, 'numerical_qualification_required': True,
            'fidelity_limitations': copy.deepcopy(cell['fidelity_limitations']),
            'source_binding': {'bindings_sha256': BINDINGS_SHA256,
                'roster_cells': 27, 'native_program': copy.deepcopy(cell['native_program']),
                'source_files': copy.deepcopy(binding['source_files'])}}
        required = {'family', 'track', 'maximum_epochs', 'minimum_epochs', 'patience',
                    'metric', 'direction', 'ties', 'evaluate_TRAIN_each_epoch', 'evaluation_mode'}
        if not required <= set(value):
            raise ValueError('Declare link policy and evaluation adaptation explicitly; missing='+str(sorted(required-set(value))))
        for key, supplied in value.items():
            if key not in canonical or type(supplied) is not type(canonical[key]) or supplied != canonical[key]:
                raise ValueError('Policy field differs from the pinned link contract: '+key)
        return cls(**canonical)


def create_policy(job, root=None):
    if job.get('family') != 'link' or job.get('TEST_access') is not False:
        raise ValueError('Explicit family=link and TEST_access=false required')
    recipe = job['recipe']
    root = root or recipe.get('_repository_root')
    policy = LinkPolicy.from_dict(job['policy'], native_family=recipe['architecture'], root=root)
    cell = next(row for row in source_bindings(root)['cells'] if row['cell_id'] == policy.cell_id)
    if recipe.get('native_fixed_config') != cell['native_fixed_config']:
        raise ValueError('Job native_fixed_config must exactly match the frozen link cell')
    if recipe.get('dataset') != 'ogbl-collab' or job.get('split', 0) != 0:
        raise ValueError('The frozen link policy admits official ogbl-collab split0 only')
    for declared in (recipe.get('id'), job.get('recipe_id')):
        if declared is not None and declared != policy.cell_id:
            raise ValueError('The declared recipe ID differs from its frozen link cell')
    if policy.track == 'source_reference':
        if job.get('procedure') not in ('native_single', 'native') or job.get('method_settings') or any(
                job.get(key) for key in ('session_transform', 'extension_factory', 'session_extension')):
            raise ValueError('A source-reference role permits unextended native M1 only')
        if type(job.get('seed')) is not int or job['seed'] not in range(10) or job.get('body') not in (None, 0):
            raise ValueError('The frozen source role uses repetition seeds0..9 and native member zero')
    if job.get('procedure') == 'independent4':
        raise ValueError('I4 has four separately selected native fit jobs, never one pooled epoch selector')
    return policy


class LinkSelector:
    """One complete-epoch native body/boundary bank; strict first-tie selection."""
    def __init__(self, policy):
        if not isinstance(policy, LinkPolicy):
            raise ValueError('An authenticated LinkPolicy is required')
        self.policy = policy
        self.completed_epoch = 0
        self.best_metric = policy.initial_best_metric
        self.selected_epoch = None
        self.stale = 0

    def stopping_status(self):
        p = self.policy
        gates = all(p.source_phase_gates.get(name) is True for name in p.required_gates)
        plateau = p.patience is not None and gates and self.completed_epoch >= p.minimum_epochs and self.stale >= p.patience
        cap = self.completed_epoch >= p.maximum_epochs
        reason = 'patience' if plateau else ('fixed_source_horizon' if p.track == 'source_reference' else 'cap_without_patience_plateau') if cap else None
        return {'stop': plateau or cap, 'stop_reason': reason,
                'completed_epoch': self.completed_epoch, 'selected_epoch': self.selected_epoch,
                'best_metric': self.best_metric, 'stale_validation_checks': self.stale,
                'selected_checkpoint_available': self.selected_epoch is not None}

    def observe(self, epoch, metrics):
        if type(epoch) is not int or epoch != self.completed_epoch+1 or self.stopping_status()['stop']:
            raise ValueError('Selection requires the next complete native epoch before stopping')
        metric = float(metrics[self.policy.metric])
        if not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError('Full VALID Hits@50 must be finite in raw0..1 units')
        improved = self.best_metric is None or metric > self.best_metric
        gates = all(self.policy.source_phase_gates.get(name) is True for name in self.policy.required_gates)
        if improved:
            self.best_metric, self.selected_epoch, self.stale = metric, epoch, 0
        elif epoch >= self.policy.minimum_epochs and gates:
            self.stale += 1
        else:
            self.stale = 0
        self.completed_epoch = epoch
        return {**self.stopping_status(), 'save_selected': improved,
                'minimum_horizon_reached': epoch >= self.policy.minimum_epochs,
                'required_phase_gates_complete': gates}

    def observe_validation(self, epoch, result):
        if result.get('role') != 'VALID' or result.get('complete_source_VALID') is not True or (
                result.get('complete_VALID_positive_records'), result.get('complete_VALID_negative_records')) != (60084, 100000):
            raise ValueError('Only all official positive/negative VALID rows may select a link state')
        return self.observe(epoch, result['metrics'])

    def state(self):
        return {'schema': 'frozen27-link-selector-state-v1', 'completed_epoch': self.completed_epoch,
                'best_metric': self.best_metric, 'selected_epoch': self.selected_epoch, 'stale': self.stale,
                'policy': copy.deepcopy(self.policy.__dict__)}

    def load_state(self, state):
        if state.get('schema') != 'frozen27-link-selector-state-v1' or state['policy'] != self.policy.__dict__:
            raise ValueError('Resume cannot change the source/controller/evaluation policy')
        epoch, selected, stale, best = (state[key] for key in ('completed_epoch', 'selected_epoch', 'stale', 'best_metric'))
        if type(epoch) is not int or not 0 <= epoch <= self.policy.maximum_epochs or type(stale) is not int or not 0 <= stale <= epoch:
            raise ValueError('Invalid saved link epoch/staleness counters')
        if selected is not None and (type(selected) is not int or not 1 <= selected <= epoch):
            raise ValueError('Selected state must be a positive completed epoch')
        if best is not None and (type(best) not in (int, float) or not math.isfinite(best) or not 0 <= best <= 1):
            raise ValueError('Invalid saved link VALID metric')
        if selected is None:
            if best != self.policy.initial_best_metric or (epoch > 0 and self.policy.initial_best_metric is None):
                raise ValueError('Missing selected epoch is inconsistent with the source initialization rule')
        elif best is None or (self.policy.initial_best_metric is not None and best <= self.policy.initial_best_metric):
            raise ValueError('A saved selection must strictly improve its source sentinel')
        self.completed_epoch, self.selected_epoch, self.stale, self.best_metric = epoch, selected, stale, best


def create_selector(policy):
    return LinkSelector(policy)
