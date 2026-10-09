"""Disabled factual-control contracts; standard library only."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ('factual_native_single_P', 'factual_independent_native4_P')
POLICY = 'factual-only-native-body-BCE-single-live-FP32-BN-v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['runtime_disabled'] is True and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact disabled factual source seal')
    for root, rows in ((HERE, read(HERE / 'MANIFEST.json')['files']),
                       (HERE.parent, read(HERE / 'SOURCE_BINDINGS.json')['files'])):
        for row in rows:
            path = (root / row['path']).resolve(strict=True)
            require(path.is_relative_to(root) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed factual source/dependency bytes')
    return read(HERE / 'SOURCE_BINDINGS.json')


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    root_source_review_approved: bool = False
    intent: str = 'disabled'
    root_qualification_execution_approved: bool = False
    root_training_approved: bool = False
    source_seal_sha256: str | None = None
    joint_v2_source_review_binding: str | None = None
    native_qualification_binding: str | None = None
    new_training_qualification_binding: str | None = None
    prospective_quality_freeze_binding: str | None = None
    study_binding: str | None = None
    role_binding: str | None = None
    full_view_binding: str | None = None
    condition: str = 'factual_native_single_P'
    base_seed: int = 1
    body_seeds: tuple[int, ...] = ()
    adapter_seeds: tuple[int, ...] = ()
    member_rng_seeds: tuple[int, ...] = ()

    def require_enabled(self):
        require(self.enabled and self.root_source_review_approved, 'Inactive factual source: explicit root review/enablement required')
        require(self.condition in CONDITIONS and self.intent in ('qualification', 'scientific_training'), 'Exact factual condition/intent')
        for value in (self.source_seal_sha256, self.joint_v2_source_review_binding, self.native_qualification_binding,
                      self.study_binding, self.role_binding, self.full_view_binding):
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact prospective source/review/native/role/study binding')
        require(self.source_seal_sha256 == sha(HERE / 'SEAL.json'), 'Exact factual source seal')
        if self.intent == 'qualification':
            require(self.root_qualification_execution_approved and not self.root_training_approved, 'Explicit discarded-copy factual qualification only')
        else:
            require(self.root_training_approved, 'Factual scientific fits remain disabled')
            for value in (self.new_training_qualification_binding, self.prospective_quality_freeze_binding):
                require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Actual factual qualification and new combined pilot freeze required')
        count = 1 if self.condition == CONDITIONS[0] else 4
        require(type(self.base_seed) is int and 0 <= self.base_seed < 2**32, 'Explicit original outer role seed')
        require(not self.adapter_seeds and len(self.body_seeds) == count and len(self.member_rng_seeds) == count,
                'Only native body/stochastic seeds, no adapter dictionaries')
        require(len(set(self.body_seeds)) == count and len(set(self.member_rng_seeds)) == count, 'Independent bodies/streams use distinct frozen seeds')
        require(all(type(seed) is int and 0 <= seed < 2**32 for seed in (*self.body_seeds, *self.member_rng_seeds)), 'Explicit valid frozen paired seeds')


def dependencies(rt, modules):
    sources = source_gate()
    require(set(modules) == set(sources['injected_module_keys']), 'Exact factual/V2 injected dependencies')
    joint = modules['joint']
    require(sha(joint.__file__) == sources['joint_training_sha256']
            and Path(joint.source_gate.__code__.co_filename).resolve() == (Path(joint.__file__).parent / 'joint_contracts.py').resolve(), 'Actual V2 source and its own contract namespace')
    native_modules = {key: value for key, value in modules.items() if key != 'joint'}
    joint.validate_modules(rt, native_modules, joint.source_gate())
    return joint, native_modules
