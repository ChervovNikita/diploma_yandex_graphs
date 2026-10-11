"""Small explicit patch adapter over the immutable completed E V2 owner.

Root supplies data custody, source/runtime admission and prospective protocol.
This module constructs no tensors and writes no original source. No CLI launch.
"""
import hashlib
import copy
from pathlib import Path
import types

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
    if not isinstance(profile, dict) or any(profile.get(k) != v for k, v in {
            "dataset": "ogbn-arxiv", "nodes": 169343, "input_dim": 128, "classes": 40}.items()):
        raise ValueError("Explicit official ogbn-arxiv dimensions required")
    for key in ("train_rows", "select_rows", "edges", "self_edges"):
        if type(profile.get(key)) is not int or profile[key] < (0 if key == "self_edges" else 1):
            raise ValueError("Root must freeze factual graph/role counts: " + key)
    if profile["train_rows"] + profile["select_rows"] > 169343 or profile["self_edges"] > profile["edges"]:
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
    changes = [
        ('x.shape != (11701, 300)', 'x.shape != (169343, 128)'),
        ('train_ids.shape != (580,)', f'train_ids.shape != ({profile["train_rows"]},)'),
        ('((0,) if training_only else (5274,))', f'((0,) if training_only else ({profile["select_rows"]},))'),
        ('y.max() >= 10', 'y.max() >= 40'),
        ('edges.max() >= 11701', 'edges.max() >= 169343'),
        ('{"nodes": 11701,', '{"nodes": 169343,'),
        ('TRAIN_rows=580,', f'TRAIN_rows={profile["train_rows"]},'),
        ('VALID_rows=0 if training_only else 5274', f'VALID_rows=0 if training_only else {profile["select_rows"]}'),
        ('c.get("depth", 2), 300, c.get("hidden", 128), 10,', 'c.get("depth", 2), 128, c.get("hidden", 128), 40,'),
        ('(1, 300), generator=gen', '(1, 128), generator=gen'),
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
    owner = owner.replace('WikiCS', 'ogbn-arxiv').replace('TRAIN580/VALID5274 required', 'Frozen complete TRAIN/SELECT rows required')
    owner = owner.replace('Ten-class role labels required', 'Forty-class role labels required').replace('three-seed', 'root-frozen seed')
    owner = owner.replace('580', str(profile['train_rows']))  # Dormant zero-valued legacy cost fields only.
    installer = replace(texts["value_installer.py"], '11701', '169343', 4).replace('WikiCS', 'ogbn-arxiv')
    return {"owner": owner, "installer": installer,
            "helper": texts["value_transport.py"], "contracts": texts["contracts.py"]}, changes


def bind(e_directory, cfg, root_validate, release):
    """Return the root-integrated Family class; metadata admission precedes tensors.

    root_validate(cfg, *, action, executing_source, port_receipt) must validate
    host/GPU, runtime, all source/role/provider hashes, fixed seeds/support,
    external watchdog/caps, and the new root protocol. Old E admission is invalid.
    """
    cfg, release = copy.deepcopy(cfg), copy.deepcopy(release)
    if release.get("status") not in ("ROOT_FULL_TRAIN_QUALIFICATION_ADMITTED", "ROOT_CONFIRMATION_FIT_ADMITTED"):
        raise ValueError("Inactive packet: root release required")
    if not callable(root_validate) or any(cfg.get(k) != v for k, v in FIXED.items()):
        raise ValueError("Root admission and exact preserved numerical recipe required")
    if cfg.get("accumulation") != "sequential_own_ce" or not cfg.get("seeds"):
        raise ValueError("Fixed nonempty seed roster and sequential shared policy required")
    if release["status"] == "ROOT_CONFIRMATION_FIT_ADMITTED" and release.get("capable_reference_decision") != "ROOT_ADMITTED":
        raise ValueError("Root capable-reference decision must precede confirmation")
    profile = cfg["profile"]
    sources, _ = render_sources(e_directory, profile)
    checked_text(cfg["common_routes"], ROUTES_SHA256)
    checked_text(cfg["factors"], FACTORS_SHA256)
    receipt = {"profile": profile, "base_E_sha256": E_HASHES,
               "derived_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
               "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "own_ce_sha256": hashlib.sha256(Path(__file__).with_name("own_ce.py").read_bytes()).hexdigest()}
    module = types.ModuleType("root_integrated_sequential_sage_owner")
    module.__file__ = str(Path(e_directory).resolve() / "run_family.py")
    exec(compile(sources["owner"], module.__file__, "exec"), module.__dict__)
    original_load = module.load_source
    update_module = original_load("root_sequential_own_ce", Path(__file__).with_name("own_ce.py"))
    module.own_ce_update = update_module.update
    installer = types.ModuleType("root_arxiv_value_installer")
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
        if actual_cfg != cfg or (qualification_only != (release["status"] == "ROOT_FULL_TRAIN_QUALIFICATION_ADMITTED")):
            raise ValueError("Release action/config differs; no automatic fit continuation")
        return root_validate(actual_cfg, action="full_TRAIN_qualification" if qualification_only else "fixed_family_fit",
                             executing_source=module.__file__, port_receipt=receipt)
    module.validate_config = validate
    original = module.Family
    class ArxivFamily(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.edge_metadata != {"nodes": 169343, "edges": profile["edges"], "self_edges": profile["self_edges"]}:
                raise ValueError("Complete actual support differs from frozen root audit")
        def within_route_supcon(self, *args, **kwargs):
            raise ValueError("No auxiliary or cross-route objective in this port")
    return ArxivFamily, receipt


if __name__ == "__main__":
    raise SystemExit("Inactive source. Root integrates bind(); no data, fit or qualification launched.")
