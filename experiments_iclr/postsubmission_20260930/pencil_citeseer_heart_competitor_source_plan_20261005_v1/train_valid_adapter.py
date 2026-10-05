"""Thin TRAIN/VALID adapter for pinned PENCIL Citeseer-HeaRT.

No module-load numerical imports, data reads, entry point or TEST interface.
Caller authenticates exact available files and root qualification before calling.
Native modules must come from retained PENCIL pin2d32e29; resource/fit release
must bind their bytes. This adapter does not run a model or create cached files.
"""


def build_train_valid(x, train, valid, pool, configs, *, dataset_class, unique_edge_function):
    import torch
    from torch_geometric.data import Data
    if x.device.type != "cpu" or x.dtype != torch.float32 or tuple(x.shape) != (3327, 3703) or not bool(torch.isfinite(x).all()):
        raise ValueError("Require root-qualified original finite float32 feature tensor")
    for rows, shape in ((train, (3870, 2)), (valid, (227, 2)), (pool, (227, 500, 2))):
        if rows.device.type != "cpu" or rows.dtype != torch.int64 or tuple(rows.shape) != shape:
            raise ValueError("Require exact authenticated TRAIN/VALID shapes/dtypes on CPU")
        if not bool(((rows >= 0) & (rows < len(x))).all()):
            raise ValueError("Endpoint population differs")
    if bool((train[:, 0] == train[:, 1]).any()) or bool((valid[:, 0] == valid[:, 1]).any()):
        raise ValueError("Positive self links must be removed by the native input loader")
    if not torch.equal(pool[:, :250, 0], valid[:, 0, None].expand(-1, 250)) or not torch.equal(pool[:, 250:, 1], valid[:, 1, None].expand(-1, 250)):
        raise ValueError("Fixed VALID pool row/anchor association differs")
    if configs.dataset != "heart-citeseer" or configs.use_features is not True or configs.feature_fusion != "early":
        raise ValueError("Only the predeclared same-input feature/early-fusion comparator is prepared")
    cfg = configs.sampling_config.edge_ego
    if cfg.depth_neighbors != [[2, 20]] or cfg.neg_ratio != 1 or cfg.percent != 100 or cfg.method.name != "global" or cfg.replace is not False:
        raise ValueError("Complete native Citeseer sampling recipe differs")
    edge = torch.cat((train.t(), train.t()[[1, 0]]), dim=1)
    graph = Data(num_nodes=3327, x=x, edge_index=edge, edge_weight=torch.ones(edge.size(1)),
                 id=torch.arange(3327, dtype=torch.int64))
    # Exactly native read_heart_split_edges' ordering and deduplication, without
    # pickle cache lookup/write and without opening any other split.
    flat_neg = pool.reshape(-1, 2)
    original_edges = torch.cat((valid, flat_neg), dim=0)
    labels = torch.cat((torch.ones(len(valid), dtype=torch.int64),
                        torch.zeros(len(flat_neg), dtype=torch.int64)))
    unique, original_to_unique, stats = unique_edge_function(original_edges)
    expected = torch.sort(original_edges, dim=1).values
    if not torch.equal(unique[original_to_unique], expected) or labels.shape != (113727,):
        raise ValueError("Native unordered query deduplication failed complete pool restoration")
    split = {"train": {"edge": train},
             "valid": {"edge": unique, "edge_neg": torch.empty((0, 2), dtype=torch.int64)}}
    train_raw = dataset_class(graph, configs.sampling_config, pretrain_mode=False,
                              split_edge=split, data_split="train")
    valid_raw = dataset_class(graph, configs.sampling_config, pretrain_mode=False,
                              split_edge=split, data_split="valid", labels=labels,
                              orig_to_unique=original_to_unique)
    if len(valid_raw) != len(unique) or not torch.equal(valid_raw.all_edges_with_y, unique):
        raise ValueError("Native VALID unique-query traversal differs")
    if len(train_raw) > 7740 or len(train_raw) < 3870:
        raise ValueError("Native ratio-one negative population is outside its valid underfill bounds")
    return train_raw, valid_raw, {"TRAIN_positives": 3870, "TRAIN_mixed_queries_constructor": len(train_raw),
                    "VALID_positive_queries": 227, "VALID_negative_occurrences": 113500,
                    "VALID_original_occurrences": 113727, "VALID_unique_queries": len(unique),
                    "deduplication_stats": stats, "TEST_interface": False}


def restore_valid_scores(unique_logits, valid_raw):
    """Complete native ranking input; no metric or selection in this helper."""
    import torch
    if tuple(unique_logits.shape) != (len(valid_raw),) or not bool(torch.isfinite(unique_logits).all()):
        raise ValueError("Require finite scores for every native unique VALID query")
    complete = valid_raw.scatter_preds(unique_logits)
    if tuple(complete.shape) != (113727,):
        raise ValueError("Complete fixed500-negative population was not restored")
    return complete[:227], complete[227:].reshape(227, 500)
