"""Complete native graph provider with separate TRAIN and selector truth custody."""
from dataclasses import dataclass
import hashlib
import json

from .caps import CLOSED
from .context import CLASS_DIM, LABEL_PATHS, TrainOnly, make_plan, prepare_views
from .runtime_gate import PROJECT, require, read


@dataclass(frozen=True)
class ValidationOnly:
    ids: tuple
    targets: object
    custody_sha256: str


@dataclass(frozen=True)
class NativeSeedInputs:
    seed: int
    feats: dict
    label_feats: dict  # Keys only for native constructor; values never used in prediction.
    data_size: dict
    train: TrainOnly
    validation: ValidationOnly
    paired: object
    train_loader: object
    role_binding: dict
    feature_shapes: dict


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_static(rt, engine, loader, release, sources, roles, costs, caps=CLOSED):
    caps.require("source_bound", "data", "runtime")
    with costs.measure("exact_once_loaded_complete_development_inputs"):
        spec = read(PROJECT / sources["runtime_sources"]["source_expectations"])
        data = loader.load(release["input_root"], release["roles"]["1"]["path"], spec)
        require(data.input_bindings == release["development_files"] and tuple(data.counts) == (4932, 2393, 6124, 7971)
                and data.class_dim == CLASS_DIM, "Exact complete source-defined native populations/classes")
        require(tuple(None if f is None else len(f)//n for f, n in zip(data.features, data.counts))
                == (3489, 3341, 3341, None), "Full native input widths; implicit keyword identity")
        for role in roles.values():
            data.with_roles(role)
    static = engine.prepare_static(rt, data, costs)  # Exact saved native implementation.
    with costs.measure("exact_full_native_CPU_label_operators") as row:
        meta = rt["native"].hg_propagate_sparse_pyg(static.adjs, "M", 4, 5, [], prop_feats=False,
                                                  echo=True, prop_device="cpu")
        require(set(meta) == set(LABEL_PATHS), "All twelve native label paths before diagonal removal")
        operators = {key: rt["remove_diag"](meta[key]) for key in LABEL_PATHS}
        inventory = {}
        for key, op in operators.items():
            r, c, value = op.coo()
            inventory[key] = {"nnz": int(op.nnz()), "COO_logical_bytes":
                r.numel()*r.element_size()+c.numel()*c.element_size()
                +(0 if value is None else value.numel()*value.element_size())}
        row.update(native_label_operator_inventory=inventory,
                   COO_logical_bytes_are_not_total_sparse_cache_RSS=True)
    return static, operators


def prepare_seed(rt, static, operators, role, role_binding, spec, costs, caps=CLOSED):
    caps.require("source_bound", "data", "runtime")
    torch, np, native = rt["torch"], rt["numpy"], rt["native"]
    data = static.data.with_roles(role)
    require(spec["role_seed"] == role["seed"] == spec["optimizer_seed"], "Declared literal native split/optimizer seed")
    native.set_random_seed(spec["optimizer_seed"])
    shuffled = np.asarray(sorted(data.train_ids + data.valid_ids), dtype=np.int64)
    np.random.shuffle(shuffled)  # Literal native split RNG consumption.
    boundary = int(shuffled.shape[0] * .2)
    require(np.sort(shuffled[boundary:]).tolist() == list(data.train_ids)
            and np.sort(shuffled[:boundary]).tolist() == list(data.valid_ids), "Frozen role matches exact native shuffle")
    require(0 < len(data.train_ids) <= 10000 and len(data.valid_ids) > 0, "Complete native one-minibatch TRAIN task")
    with costs.measure("native_per_seed_full_features_and_two_frozen_contexts") as row:
        feats = {key: value.detach().clone() for key, value in static.raw_feats.items()}
        train = TrainOnly(tuple(data.train_ids), torch.FloatTensor(data.train_labels),
                          digest({"role": role_binding, "TRAIN": data.train_ids, "input": data.input_bindings}))
        validation = ValidationOnly(tuple(data.valid_ids), torch.FloatTensor(data.valid_labels),
                                    digest({"role": role_binding, "VALID": data.valid_ids, "input": data.input_bindings}))
        # Constructed before native model initialization, generator=None, just as native.
        train_loader = torch.utils.data.DataLoader(np.asarray(train.ids, dtype=np.int64),
                            batch_size=10000, shuffle=True, drop_last=False)
        plan = make_plan(train, spec["context_seed"], caps)
        paired = prepare_views(torch, rt["SparseTensor"], train, plan, operators, caps)
        require(all(all(v > 0 for v in a["positive_bits"]) and all(v > 0 for v in a["negative_bits"])
                    for a in paired.support_audit), "Every declared context has both binary classes; fail without resampling")
        require(not set(train.ids) & set(validation.ids)
                and validation.targets.shape == (len(validation.ids), CLASS_DIM)
                and torch.isfinite(validation.targets).all().item(), "Separate complete finite selector targets")
        row.update(feature_bytes=sum(v.numel()*v.element_size() for v in feats.values()),
                   positive_label_sparse_applications=24, knownness_sparse_applications=6,
                   additional_signed_sparse_applications=0,
                   context_label_bytes=sum(v.numel()*v.element_size() for view in paired.views for v in view.label_features.values()),
                   context_field_bytes=sum(v.local_field.numel()*v.local_field.element_size()
                                           + v.global_field.numel()*v.global_field.element_size() for v in paired.views),
                   context_identity_sha256=plan.identity_sha256, support_audit=paired.support_audit,
                   TRAIN_targets_only_in_context_builder=True, full_y_exists=False,
                   graph_or_path_or_width_shrink=False)
    return NativeSeedInputs(spec["optimizer_seed"], feats, {key: None for key in LABEL_PATHS},
                            static.data_size, train, validation, paired, train_loader,
                            role_binding, static.feature_shapes)
