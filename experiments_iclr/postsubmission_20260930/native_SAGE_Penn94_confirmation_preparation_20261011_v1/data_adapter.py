"""Caller-supplied Penn94 preparation; no imports, files, models or scorer.

Custodian functions may decode public release metadata and generate safe roles.
Scientific model views accept TRAIN targets only. This source is unexecuted.
"""

EXPECTED_NODES = 41554
CLASSES = 2
ROLE_SEED = 11709
PUBLIC_POLICY = "PyG_LINKX_one_hot_without_gender_author_undirected_support_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def custodian_metadata(np, local_info):
    """Custodian only: separate target column before scientific public export.

    Matches the saved provider's int64 local_info convention and gender minus
    one labels. Returned private targets must not enter public/model bundles.
    """
    require(local_info.ndim == 2 and local_info.shape[0] == EXPECTED_NODES
            and local_info.shape[1] >= 3, "Complete Penn94 metadata required")
    require(np.isfinite(local_info).all(), "Finite categorical release metadata required")
    integer = local_info.astype(np.int64)
    require(np.equal(integer, local_info).all(), "No truncation of noninteger metadata")
    private_y = integer[:, 1] - 1
    require(np.isin(private_y, (-1, 0, 1)).all(), "Two known targets or missing -1 required")
    require(set(np.unique(private_y[private_y >= 0]).tolist()) == {0, 1},
            "Both eligible Penn94 classes required")
    public_metadata = np.concatenate((integer[:, :1], integer[:, 2:]), axis=1).copy()
    return public_metadata, private_y.copy()


def public_tensors(torch, one_hot, non_target_metadata, raw_edge_index):
    """PyG per-column unique/one_hot; author undirected/coalesced support.

    Caller supplies the pinned PyG one_hot function and custodian-approved
    non-target columns. Raw edge weights are not consumed by native SAGE.
    Original loops remain once if present; no new loops are inserted.
    """
    require(torch.get_default_dtype() == torch.float32, "Native float32 provider required")
    metadata = torch.as_tensor(non_target_metadata, device="cpu")
    edges = torch.as_tensor(raw_edge_index, device="cpu")
    require(metadata.ndim == 2 and metadata.shape[0] == EXPECTED_NODES
            and metadata.shape[1] >= 2 and metadata.dtype == torch.int64,
            "Custodian-approved complete int64 non-target metadata required")
    require(edges.ndim == 2 and edges.shape[0] == 2 and edges.dtype == torch.int64,
            "Int64 COO support required")
    require(edges.numel() > 0 and int(edges.min()) >= 0
            and int(edges.max()) < EXPECTED_NODES, "Full graph indices outside domain")
    columns, category_counts = [], []
    for column in range(metadata.shape[1]):
        _, inverse = metadata[:, column].unique(sorted=True, return_inverse=True)
        encoded = one_hot(inverse)
        require(encoded.dtype == torch.float32, "Pinned one_hot must yield float32")
        columns.append(encoded)
        category_counts.append(encoded.shape[1])
    x = torch.cat(columns, dim=-1).contiguous()
    src, dst = edges[0], edges[1]
    codes = torch.cat((src * EXPECTED_NODES + dst,
                       dst * EXPECTED_NODES + src)).unique(sorted=True)
    canonical_edges = torch.stack((codes // EXPECTED_NODES,
                                   codes % EXPECTED_NODES)).contiguous()
    metadata_record = {
        "public_policy": PUBLIC_POLICY, "nodes": EXPECTED_NODES,
        "features": x.shape[1], "classes": CLASSES,
        "encoded_category_counts": category_counts,
        "target_column_in_features": False,
        "raw_edge_entries": edges.shape[1],
        "undirected_coalesced_edge_entries": canonical_edges.shape[1],
        "original_loop_entries": int((src == dst).sum()),
        "canonical_loop_entries": int((canonical_edges[0] == canonical_edges[1]).sum()),
        "new_self_loops": 0, "edge_weights_consumed": False,
        "missing_target_nodes_removed_from_graph": False,
        "one_hot_dense_entries": x.numel(),
        "coalesce_input_entries": 2 * edges.shape[1],
    }
    return x, canonical_edges, metadata_record


def custodian_roles(np, decoded_split0, eligible_ids):
    """Custodian only: published split 0 plus one label-free VALID separation.

    No file or pickle reader is provided. A separately admitted custodian
    supplies the decoded, source-bound published split and eligible IDs.
    """
    require(set(decoded_split0) == {"train", "valid", "test"},
            "Exactly the first published split's three memberships required")
    eligible = np.asarray(eligible_ids)
    require(eligible.dtype == np.int64 and eligible.ndim == 1
            and len(np.unique(eligible)) == len(eligible), "Unique eligible int64 IDs required")
    require(len(eligible) > 0 and eligible.min() >= 0 and eligible.max() < EXPECTED_NODES,
            "Eligible IDs outside full graph")
    official = {}
    for name in ("train", "valid", "test"):
        ids = np.asarray(decoded_split0[name])
        require(ids.dtype == np.int64 and ids.ndim == 1 and len(ids) > 0
                and len(np.unique(ids)) == len(ids), "Unique nonempty published IDs required")
        require(ids.min() >= 0 and ids.max() < EXPECTED_NODES, "Published ID outside graph")
        official[name] = np.sort(ids).copy()
    combined = np.concatenate([official[k] for k in ("train", "valid", "test")])
    require(len(np.unique(combined)) == len(combined)
            and np.array_equal(np.sort(combined), np.sort(eligible)),
            "Complete disjoint published memberships must cover eligible nodes")
    shuffled = np.random.RandomState(ROLE_SEED).permutation(official["valid"])
    midpoint = len(shuffled) // 2
    require(midpoint > 0 and midpoint < len(shuffled), "Nonempty disjoint SELECT/CAL required")
    roles = {"train": official["train"], "select": np.sort(shuffled[midpoint:]),
             "cal": np.sort(shuffled[:midpoint]), "evaluation_ids": official["test"]}
    record = {
        "published_split_index": 0, "original_memberships_preserved": True,
        "SELECT_CAL_source": "published VALID only",
        "permutation": "NumPy RandomState MT19937, sorted official VALID IDs",
        "role_seed": ROLE_SEED, "CAL_first_floor_half": True,
        "counts": {key: len(ids) for key, ids in roles.items()},
        "eligible_nodes": len(eligible),
        "missing_target_nodes_retained_publicly": EXPECTED_NODES - len(eligible),
        "role_selection_uses_predictions_or_outcomes": False,
    }
    return roles, record


def native_train_view(torch, namespace, x, edge_index, train_ids, train_y,
                      representation, device):
    """Only TRAIN truth enters the model-facing graph; others are constant zero.

    SELECT/CAL targets and evaluation targets require separate owner/scorer
    bundles. Native Model, TABMModel and label-free wrapper interfaces are reused.
    """
    require(representation in ("two_logit_CE", "one_logit_BCE"), "Fixed binary representation required")
    require(x.ndim == 2 and x.shape[0] == EXPECTED_NODES and x.dtype == torch.float32,
            "Admitted complete float32 public feature matrix required")
    require(edge_index.ndim == 2 and edge_index.shape[0] == 2
            and edge_index.dtype == torch.int64, "Admitted public edge support required")
    ids = torch.as_tensor(train_ids, device=device)
    targets = torch.as_tensor(train_y, device=device)
    require(ids.dtype == torch.int64 and targets.dtype == torch.int64,
            "Custodian int64 TRAIN IDs and targets required; no numeric coercion")
    require(ids.ndim == 1 and targets.shape == ids.shape and ids.numel() > 0
            and ids.unique().numel() == ids.numel(), "Exact unique TRAIN IDs/y required")
    require(int(ids.min()) >= 0 and int(ids.max()) < EXPECTED_NODES
            and bool(((targets == 0) | (targets == 1)).all()), "TRAIN values outside fixed domain")
    require(targets.unique().numel() == CLASSES, "Both TRAIN classes required")
    features, edges = x.to(device), edge_index.to(device)
    label_dtype = torch.long if representation == "two_logit_CE" else torch.float32
    labels = torch.zeros(EXPECTED_NODES, device=device, dtype=label_dtype)
    labels[ids] = targets.to(label_dtype)
    mask = torch.zeros(EXPECTED_NODES, device=device, dtype=torch.bool)
    mask[ids] = True
    data = namespace(x=features, edge_index=edges, y=labels, train_mask=mask)
    batch = {"x": features, "edge_index": edges,
             "graph": namespace(edge_index=edges)}
    return data, batch, ids


def raw_binary_probability_pool(torch, logits, representation, pooling):
    """Label-free float32 serving only; no accuracy/NLL or target reader."""
    require(pooling in ("probability_mean", "logit_mean"), "Fixed pooling required")
    require(logits.dtype == torch.float32, "Native float32 raw authority required")
    require(bool(torch.isfinite(logits).all()), "Finite admitted route logits required")
    if representation == "two_logit_CE":
        require(logits.ndim == 3 and logits.shape[-1] == CLASSES
                and logits.shape[0] in (1, 4), "One/four two-class native routes required")
        return logits.softmax(-1).mean(0) if pooling == "probability_mean" else logits.mean(0).softmax(-1)
    require(representation == "one_logit_BCE" and logits.ndim == 2
            and logits.shape[0] in (1, 4), "One/four original binary route vectors required")
    p = logits.sigmoid().mean(0) if pooling == "probability_mean" else logits.mean(0).sigmoid()
    return torch.stack((1 - p, p), dim=-1)


if __name__ == "__main__":
    raise SystemExit("Source-only Penn94 adapter: root admits data/source/roles and qualifies before any execution")
