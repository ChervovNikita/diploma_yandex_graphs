"""Root-supplied public features/edges and isolated TRAIN/VALID bundles; no loader."""
from dataclasses import dataclass
from types import SimpleNamespace
from .caps import CLOSED
from .native import require, preprocess


@dataclass(frozen=True)
class Role:
    ids: object
    targets: object
    custody_sha256: str


@dataclass(frozen=True)
class Inputs:
    factual: object
    train: Role
    validation: Role
    input_custody_sha256: str
    split_seed: int
    counts: dict


def prepare(torch, x, edges, train, validation, custody_sha256, device, costs, caps=CLOSED):
    caps.require('source_bound', 'data', 'runtime')
    require(x.shape == (19717, 500) and x.dtype == torch.float32 and not x.requires_grad
            and torch.isfinite(x).all().item(), 'Full normalized PubMed19717x500 feature matrix')
    require(edges.dtype == torch.long and edges.ndim == 2 and edges.shape[0] == 2
            and edges.shape[1] > 0 and int(edges.min()) >= 0 and int(edges.max()) < 19717,
            'Complete custody-bound original PubMed graph; no induced graph')
    for role in (train, validation):
        require(len(role.custody_sha256) == 64 and role.ids.dtype == torch.long
                and role.targets.dtype == torch.long and role.ids.ndim == 1
                and role.targets.shape == role.ids.shape and not role.targets.requires_grad
                and role.ids.numel() > 0 and role.ids.unique().numel() == role.ids.numel()
                and int(role.ids.min()) >= 0 and int(role.ids.max()) < 19717
                and int(role.targets.min()) >= 0 and int(role.targets.max()) < 3,
                'Complete separate in-range role identities and categorical targets')
        require(all((role.targets == c).any().item() for c in range(3)), 'All three classes represented in the fixed role')
    require(not set(train.ids.cpu().tolist()) & set(validation.ids.cpu().tolist()) and len(custody_sha256) == 64,
            'Disjoint TRAIN/VALID and explicit root input custody; no full-y interface')
    require(11828 <= len(train.ids) <= 11830 and 3941 <= len(validation.ids) <= 3943,
            'Full predeclared class-stratified floor60/20/20 representative; no shortened TRAIN task')
    with costs.measure('literal_native_full_graph_mono_preprocessing') as row:
        mats = preprocess(x.to(device), edges.to(device), caps)
        require(len(mats) == 3 and all(m.shape == (19717, 500) and m.dtype == torch.float32
                and m.device == device and torch.isfinite(m).all().item() for m in mats), 'All native K2 full-graph monomial channels')
        row.update(cached_monomial_bytes=sum(m.numel()*m.element_size() for m in mats),
                   graph_or_feature_or_population_shrink=False, labels_used_in_preprocessing=False)
    return Inputs(SimpleNamespace(list_mat=mats),
        Role(train.ids.to(device), train.targets.to(device), train.custody_sha256),
        Role(validation.ids.to(device), validation.targets.to(device), validation.custody_sha256),
        custody_sha256, 190111, dict(nodes=19717, features=500, classes=3,
                                    TRAIN=len(train.ids), VALID=len(validation.ids)))
