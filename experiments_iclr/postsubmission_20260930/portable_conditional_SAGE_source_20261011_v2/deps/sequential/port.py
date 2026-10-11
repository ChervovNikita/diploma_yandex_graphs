"""Small explicit patch adapter over the immutable completed E V2 owner.

Root supplies data custody, source/runtime admission and prospective protocol.
This module constructs no tensors and writes no original source. No CLI launch.
"""
import hashlib
import copy
from pathlib import Path
import types
import sys

E_HASHES = {
    "run_family.py": "dbe58a63e2976e865b475b3557ec8fad7568dcf31bfcd721de816772c51d55bd",
    "value_installer.py": "e92df73611dd2cf4c063dc7bee1c7182a1c07907a77777f10d2a748fd92bb23a",
    "value_transport.py": "cf97376779782482148ed90b57ea47d58fe18fd5554aab543d41cf3f0e9cf263",
    "contracts.py": "b50d8e24c2decf8c696f10bf694b724456011a0248fb47de99f47bceddc763d1",
}
ROUTES_SHA256 = "84ff13b28ef471ac0a4199e10d4284f24647d396e021646e2a21a06f96fb0773"
FACTORS_SHA256 = "9f185dfeb05a059f6c5d84062e1b6226ab29a288b4c7ab8fac08f8dfb523f9c3"
FIXED = {"depth": 2, "hidden": 128, "dropout": .2, "learning_rate": .001,
         "max_updates": 1000, "patience_updates": 300, "rank": 16,
         "member_seed_stride": 1000003, "factor_seed_offset": 2000003,
         "dropout_seed_offset": 3000007, "block_seed_offset": 4000037,
         "stalk_dim": 4, "baseline_policy": "fresh_required"}
PROFILES = {"ogbn-arxiv": (169343, 128, 40), "WikiCS": (11701, 300, 10), "GitHub": (37700, 128, 2), "CoauthorPhysics": (34493, 8415, 5)}
QUAL_ACTIONS = {"ROOT_ENGINE_EQUIVALENCE_ADMITTED": "engine_equivalence_WikiCS",
                "ROOT_ARXIV_RESOURCE_QUALIFICATION_ADMITTED": "arxiv_sequential_resource",
                "ROOT_GITHUB_RESOURCE_QUALIFICATION_ADMITTED": "github_sequential_resource",
                "ROOT_PHYSICS_RESOURCE_QUALIFICATION_ADMITTED": "physics_sequential_resource"}
GITHUB_SEEDS = (8101, 8219, 8363, 8501, 8663, 8819, 9001, 9199, 9403, 9623)
WIKI_TRAIN_SHA256 = "88c36e1983f17a9e53d3f13a473baaf9eff86f52bc4be56aba012321983fdc6d"


def checked_text(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Pinned source differs: " + str(path))
    return data.decode()


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError("Source amendment anchor differs: " + before[:70])
    return text.replace(before, after)


def check_profile(profile):
    if not isinstance(profile, dict) or profile.get('dataset') not in PROFILES:
        raise ValueError("Only declared arxiv/GitHub/Physics and engineering-only WikiCS profiles")
    if tuple(profile.get(k) for k in ('nodes', 'input_dim', 'classes')) != PROFILES[profile['dataset']]:
        raise ValueError("Pinned N/D/C dimensions differ")
    if profile['dataset'] == 'WikiCS' and (profile.get('train_rows') != 580 or profile.get('select_rows') != 5274):
        raise ValueError("Exact existing complete WikiCS role counts required")
    if profile['dataset'] == 'GitHub' and (profile.get('train_rows') != 22620 or profile.get('select_rows') != 3770 or profile.get('edges') != 578006):
        raise ValueError("Root must bind the declared complete GitHub support/custom roles; no fallback")
    if profile['dataset'] == 'CoauthorPhysics' and (profile.get('train_rows') != 100 or profile.get('select_rows') != 150 or profile.get('edges') != 530417 or profile.get('self_edges') != 34493):
        raise ValueError("Exact full Physics common model support and per-block roles required")
    for key in ("train_rows", "select_rows", "edges", "self_edges"):
        if type(profile.get(key)) is not int or profile[key] < (0 if key == "self_edges" else 1):
            raise ValueError("Root must freeze factual graph/role counts: " + key)
    if profile["train_rows"] + profile["select_rows"] > profile['nodes'] or profile["self_edges"] > profile["edges"]:
        raise ValueError("Factual count bounds differ")
    for key in ("provider_identity_sha256", "graph_support_sha256"):
        value = profile.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Root must bind actual provider/preprocessing identity")


def render_sources(e_directory, profile):
    """Source-only; returns complete derived source plus explicit change list."""
    check_profile(profile)
    directory = Path(e_directory)
    texts = {name: checked_text(directory / name, digest) for name, digest in E_HASHES.items()}
    owner = texts["run_family.py"]
    n, d, c = PROFILES[profile['dataset']]
    changes = [
        ('x.shape != (11701, 300)', f'x.shape != ({n}, {d})'),
        ('train_ids.shape != (580,)', f'train_ids.shape != ({profile["train_rows"]},)'),
        ('((0,) if training_only else (5274,))', f'((0,) if training_only else ({profile["select_rows"]},))'),
        ('y.max() >= 10', f'y.max() >= {c}'),
        ('edges.max() >= 11701', f'edges.max() >= {n}'),
        ('{"nodes": 11701,', f'{{"nodes": {n},'),
        ('TRAIN_rows=580,', f'TRAIN_rows={profile["train_rows"]},'),
        ('VALID_rows=0 if training_only else 5274', f'VALID_rows=0 if training_only else {profile["select_rows"]}'),
        ('c.get("depth", 2), 300, c.get("hidden", 128), 10,', f'c.get("depth", 2), {d}, c.get("hidden", 128), {c},'),
        ('(1, 300), generator=gen', f'(1, {d}), generator=gen'),
        ('adapter = self.routes.NativeModelAdapter(torch, body)\n        installs = []',
         'adapter = self.routes.NativeModelAdapter(torch, body)\n        if wrapper is not None:\n            wrapper.native_adapter = adapter\n        installs = []'),
        ('hidden.shape[-1]', 'self.cfg["hidden"]'),
        ('"backwards": step, "adam_steps": step', '"backwards": (4 if bank else 1) * step, "adam_steps": step'),
        ('"full_expected_groups": 15', '"full_expected_groups": 5 * len(seeds)'),
        ('"full_expected_optimizer_acquisition_bundles": 24', '"full_expected_optimizer_acquisition_bundles": 8 * len(seeds)'),
        ('"full_expected_native_body_fit_records": 24', '"full_expected_native_body_fit_records": 8 * len(seeds)'),
        ('"full_native_trajectories_per_roster_forward": 51', '"full_native_trajectories_per_roster_forward": 17 * len(seeds)'),
    ]
    for before, after in changes:
        owner = replace(owner, before, after, 3 if before == 'hidden.shape[-1]' else 1)
    if profile['dataset'] == 'CoauthorPhysics':
        owner = replace(owner, '        for seed in self.seed_block:\n            for spec, arm in zip(ARM_SPECS, arms):',
                        '        for seed in self.seed_block:\n            self.activate_role_block(seed)\n            for spec, arm in zip(ARM_SPECS, arms):')
    start = owner.index('                model.train()\n                optimizer.zero_grad()')
    end = owner.index('                model.eval()', start)
    old_update = owner[start:end]
    owner = replace(owner, old_update,
        '                train = own_ce_update(self, model, optimizer, streams, bank)\n'
        '                train_loss = train_own_ce = train["own_ce"]\n'
        '                train_pool_ce = train["pool_ce"]\n'
        '                train_supcon = train_supcon_routes = None\n')
    aux_start = owner.index('    def within_route_supcon(')
    aux_end = owner.index('    def probability_pool_ce(', aux_start)
    owner = replace(owner, owner[aux_start:aux_end],
        '    def within_route_supcon(self, *args, **kwargs):\n'
        '        raise ValueError("No auxiliary objective in the fixed own-CE port")\n\n')
    # Text only: no model/normalization/support/equation mutation.
    owner = owner.replace('WikiCS', profile['dataset']).replace('TRAIN580/VALID5274 required', 'Frozen complete TRAIN/SELECT rows required')
    owner = owner.replace('Ten-class role labels required', f'{c}-class role labels required').replace('three-seed', 'root-frozen seed')
    owner = owner.replace('580', str(profile['train_rows']))  # Dormant zero-valued legacy cost fields only.
    installer = replace(texts["value_installer.py"], '11701', str(n), 4).replace('WikiCS', profile['dataset'])
    return {"owner": owner, "installer": installer,
            "helper": texts["value_transport.py"], "contracts": texts["contracts.py"]}, changes


def bind(e_directory, cfg, root_validate, release):
    """Return the root-integrated Family class; metadata admission precedes tensors.

    root_validate(cfg, *, action, executing_source, port_receipt) must validate
    host/GPU, runtime, all source/role/provider hashes, fixed seeds/support,
    external watchdog/caps, and the new root protocol. Old E admission is invalid.
    """
    cfg, release = copy.deepcopy(cfg), copy.deepcopy(release)
    if release.get("status") not in (*QUAL_ACTIONS, "ROOT_CONFIRMATION_FIT_ADMITTED"):
        raise ValueError("Inactive packet: root release required")
    if not callable(root_validate) or any(cfg.get(k) != v for k, v in FIXED.items()):
        raise ValueError("Root admission and exact preserved numerical recipe required")
    if cfg.get("accumulation") != "sequential_own_ce" or not cfg.get("seeds"):
        raise ValueError("Fixed nonempty seed roster and sequential shared policy required")
    if release["status"] == "ROOT_CONFIRMATION_FIT_ADMITTED" and release.get("capable_reference_decision") != "ROOT_ADMITTED":
        raise ValueError("Root capable-reference decision must precede confirmation")
    profile = cfg["profile"]
    dataset = profile['dataset']
    if (release['status'] == 'ROOT_ENGINE_EQUIVALENCE_ADMITTED') != (dataset == 'WikiCS'):
        raise ValueError("WikiCS is only the existing-graph engine equivalence check")
    expected_dataset = {'ROOT_ARXIV_RESOURCE_QUALIFICATION_ADMITTED': 'ogbn-arxiv', 'ROOT_GITHUB_RESOURCE_QUALIFICATION_ADMITTED': 'GitHub', 'ROOT_PHYSICS_RESOURCE_QUALIFICATION_ADMITTED': 'CoauthorPhysics'}
    if release['status'] in expected_dataset and (dataset != expected_dataset[release['status']] or release.get('engine_equivalence') != 'ROOT_VERIFIED_PASS'):
        raise ValueError("Correct factual profile and existing verified engine required")
    if release['status'] == 'ROOT_CONFIRMATION_FIT_ADMITTED':
        resource_key = {'GitHub': 'actual_github_resource_qualification', 'ogbn-arxiv': 'actual_arxiv_resource_qualification', 'CoauthorPhysics': 'actual_physics_resource_qualification'}[dataset]
        if any(release.get(k) != 'ROOT_VERIFIED_PASS' for k in ('engine_equivalence', resource_key)):
            raise ValueError("Separate engine and chosen actual-graph resource receipts required before fit")
        if dataset == 'GitHub' and (cfg['seeds'] != list(GITHUB_SEEDS) or release.get('github_metric_and_competence') != 'ROOT_ADOPTED_MICRO_F1_EQUALS_ACCURACY'):
            raise ValueError("Root fixed GitHub ten-seed/metric/competence/source freeze required")
        if dataset == 'CoauthorPhysics' and cfg['seeds'] != list(GITHUB_SEEDS):
            raise ValueError("Root fixed paired ten-mask Physics roster required")
    if dataset == 'CoauthorPhysics':
        blocks = cfg.get('role_blocks')
        if not isinstance(blocks, dict) or set(blocks) != {str(s) for s in GITHUB_SEEDS} or any(set(v) != {'train_npz', 'select_npz'} for v in blocks.values()):
            raise ValueError("Root must bind every paired Physics safe role block")
        active = cfg.get('active_role_seed')
        if active not in GITHUB_SEEDS or cfg['train_npz'] != blocks[str(active)]['train_npz'] or cfg['valid_npz'] != blocks[str(active)]['select_npz']:
            raise ValueError("Initial safe loader must bind one declared active block")
    sources, _ = render_sources(e_directory, profile)
    checked_text(cfg["common_routes"], ROUTES_SHA256)
    checked_text(cfg["factors"], FACTORS_SHA256)
    receipt = {"profile": profile, "base_E_sha256": E_HASHES,
               "derived_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
               "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "own_ce_sha256": hashlib.sha256(Path(__file__).with_name("own_ce.py").read_bytes()).hexdigest()}
    module = types.ModuleType("root_integrated_sequential_sage_owner")
    module.__file__ = str(Path(e_directory).resolve() / "run_family.py")
    sys.modules[module.__name__] = module
    exec(compile(sources["owner"], module.__file__, "exec"), module.__dict__)
    original_load = module.load_source
    update_module = original_load("root_sequential_own_ce", Path(__file__).with_name("own_ce.py"))
    module.own_ce_update = update_module.update
    installer = types.ModuleType("root_arxiv_value_installer")
    installer.__file__ = str(Path(e_directory).resolve() / "value_installer.py")
    sys.modules[installer.__name__] = installer
    exec(compile(sources["installer"], str(Path(e_directory) / "value_installer.py"), "exec"), installer.__dict__)
    def load(name, path):
        if name == "endpoint_value_installer":
            return installer
        result = original_load(name, path)
        if name == "endpoint_value_contracts_counts":
            result.PROTOCOL = cfg["root_frozen_protocol"]
        return result
    module.load_source = load
    def validate(actual_cfg, qualification_only=False):
        if actual_cfg != cfg or (qualification_only != (release["status"] in QUAL_ACTIONS)):
            raise ValueError("Release action/config differs; no automatic fit continuation")
        return root_validate(actual_cfg, action=QUAL_ACTIONS[release['status']] if qualification_only else "fixed_family_fit",
                             executing_source=module.__file__, port_receipt=receipt)
    module.validate_config = validate
    original = module.Family
    class ProfiledFamily(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.edge_metadata != {"nodes": profile['nodes'], "edges": profile["edges"], "self_edges": profile["self_edges"]}:
                raise ValueError("Complete actual support differs from frozen root audit")
            if dataset == 'WikiCS' and self.learning_lock['safe_role_sha256'].get('train') != WIKI_TRAIN_SHA256:
                raise ValueError("Reuse the exact existing E WikiCS safe TRAIN materialization")
            self.port_receipt = receipt
            self.active_role_seed = cfg.get('active_role_seed')
        def activate_role_block(self, seed):
            if dataset != 'CoauthorPhysics' or self.training_only or self._fit_model is not None:
                raise ValueError("Physics role switch only before complete seed-block acquisition")
            block = cfg['role_blocks'][str(seed)]
            with self.np.load(block['train_npz'], allow_pickle=False) as t:
                if set(t.files) != {'x', 'edge_index', 'ids', 'y'}: raise ValueError("Exact Physics safe TRAIN block")
                train_ids, train_y = t['ids'].copy(), t['y'].copy()
            with self.np.load(block['select_npz'], allow_pickle=False) as t:
                if set(t.files) != {'ids', 'y'}: raise ValueError("Exact Physics SELECT-only block")
                select_ids, select_y = t['ids'].copy(), t['y'].copy()
            if any(a.dtype != self.np.int64 for a in (train_ids, train_y, select_ids, select_y)) or any(ids.shape != y.shape or ids.shape != (count,) for ids, y, count in ((train_ids, train_y, 100), (select_ids, select_y, 150))):
                raise ValueError("Exact per-block100TRAIN/150SELECT shapes")
            ids = self.np.concatenate((train_ids, select_ids))
            if len(self.np.unique(ids)) != 250 or ids.min() < 0 or ids.max() >= 34493 or any(y.min() < 0 or y.max() >= 5 for y in (train_y, select_y)):
                raise ValueError("Unique disjoint Physics TRAIN/SELECT IDs and labels")
            tensor = lambda a: self.torch.from_numpy(a).to(self.device)
            self.train_ids, self.valid_ids, self.valid_y = map(tensor, (train_ids, select_ids, select_y))
            self.data.y.zero_(); self.data.train_mask.zero_()
            self.data.y[self.train_ids] = tensor(train_y); self.data.train_mask[self.train_ids] = True
            self.active_role_seed = seed
            # Root custody binds identical full public X/support in all blocks;
            # only tiny safe role arrays are deserialized again here.
            module.write_json(self.output / 'ACTIVE_ROLE_BLOCK.json', {'seed': seed, 'TRAIN_rows': 100, 'SELECT_rows': 150,
                               'full_public_graph_reused': True, 'CAL_CONFIRM_truth_access': False})
        def within_route_supcon(self, *args, **kwargs):
            raise ValueError("No auxiliary or cross-route objective in this port")
    return ProfiledFamily, receipt


if __name__ == "__main__":
    raise SystemExit("Inactive source. Root integrates bind(); no data, fit or qualification launched.")
