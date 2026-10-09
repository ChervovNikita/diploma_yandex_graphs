"""Root-only disabled native source entry contracts; standard library only."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ('actor', 'director', 'keyword')
VIEWS = (None, *FAMILIES)
ASSIGNMENTS = ('actor', 'director', 'keyword', None)
NATIVE_SCALARS = 83659532
FAST_RANK1_PER_MEMBER = 97536
MODULE_KEYS = ('engine', 'adapter', 'metadata', 'helper', 'role_loader', 'source_views', 'diagnostics')
SPECS = {
    'BE_P': ('shared', 'be', 1, 'none', False),
    'BE_PS': ('shared', 'be', 1, 'assigned', False),
    'ADD_P': ('shared', 'add', 1, 'none', False),
    'ADD_PS': ('shared', 'add', 1, 'assigned', False),
    'ADD_P_copied_dictionary': ('shared', 'add', 1, 'none', True),
    'ADD_PS_neutral_all_source_average': ('shared', 'add', 1, 'neutral', False),
    'independent_ADD_P': ('independent', 'add', 1, 'none', False),
    'independent_ADD_PS': ('independent', 'add', 1, 'assigned', False),
    'native_single_P': ('single', 'native', 0, 'none', False),
    'rank4_single_P': ('single', 'add', 4, 'none', False),
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source_gate():
    seal = read(HERE / 'SEAL.json')
    require(seal['runtime_disabled'] is True and sha(HERE / 'MANIFEST.json') == seal['manifest_sha256'], 'Exact disabled source seal')
    for row in read(HERE / 'MANIFEST.json')['files']:
        path = (HERE / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed source payload')
    sources = read(HERE / 'SOURCE_BINDINGS.json')
    for row in sources['files']:
        path = (HERE.parent / row['path']).resolve(strict=True)
        require(path.is_relative_to(HERE.parent) and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed frozen native dependency')
    return sources


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    root_source_review_approved: bool = False
    intent: str = 'disabled'
    root_qualification_execution_approved: bool = False
    root_training_approved: bool = False
    source_seal_sha256: str | None = None
    native_qualification_binding: str | None = None
    new_training_qualification_binding: str | None = None
    prospective_quality_freeze_binding: str | None = None
    study_binding: str | None = None
    role_binding: str | None = None
    full_view_binding: str | None = None
    condition: str = 'ADD_PS'
    base_seed: int = 1
    body_seeds: tuple[int, ...] = ()
    adapter_seeds: tuple[int, ...] = ()
    member_rng_seeds: tuple[int, ...] = ()

    def require_enabled(self):
        require(self.enabled and self.root_source_review_approved, 'Inactive source: root source review and explicit enablement required')
        require(self.condition in SPECS and self.intent in ('qualification', 'scientific_training'), 'Exact declared condition/intent')
        for value in (self.source_seal_sha256, self.native_qualification_binding, self.study_binding, self.role_binding, self.full_view_binding):
            require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact prospective source/native/role/study binding')
        require(self.source_seal_sha256 == sha(HERE / 'SEAL.json'), 'Exact new source seal')
        if self.intent == 'qualification':
            require(self.root_qualification_execution_approved and not self.root_training_approved, 'Only an explicitly authorized discarded-copy qualifier')
        else:
            require(self.root_training_approved, 'Scientific training remains disabled')
            for value in (self.new_training_qualification_binding, self.prospective_quality_freeze_binding):
                require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Actual new full-input qualification and new freeze required')
        ownership = SPECS[self.condition][0]
        count = 1 if ownership == 'single' else 4
        require(type(self.base_seed) is int and 0 <= self.base_seed < 2**32, 'Explicit native base seed')
        require(len(self.member_rng_seeds) == count and len(set(self.member_rng_seeds)) == count, 'Every complete path has its own frozen stochastic seed')
        require(len(self.body_seeds) == (4 if ownership == 'independent' else 1), 'Fresh native body seeds, never teacher/donor reuse')
        require(len(self.adapter_seeds) == 4 and len(set(self.adapter_seeds)) == 4, 'Four declared dictionary seeds, including rank4 single capacity')
        for seed in (*self.body_seeds, *self.adapter_seeds, *self.member_rng_seeds):
            require(type(seed) is int and 0 <= seed < 2**32, 'Valid explicit prospective seed')
        if ownership == 'independent':
            require(len(set(self.body_seeds)) == 4, 'Four distinct independently initialized full bodies')


def validate_modules(rt, modules, sources):
    require(set(modules) == set(MODULE_KEYS), 'Exact seven injected callable dependencies')
    for key, module in modules.items():
        require(sha(module.__file__) == sources['modules'][key]['sha256'], 'Exact injected native module: ' + key)
    require(rt['model_class'] is rt['model_module'].SeHGNN and rt['model_module'].torch is rt['torch'], 'Pinned actual native class/runtime')
    require(sha(rt['model_module'].__file__) == sources['modules']['model']['sha256'] and sha(rt['native'].__file__) == sources['modules']['native_helpers']['sha256'], 'Exact native model and propagation/evaluator helpers')
    require(sha(modules['engine'].capture_rng.__code__.co_filename) == sources['modules']['state_helpers']['sha256'], 'Exact native RNG/CPU-state helpers')
