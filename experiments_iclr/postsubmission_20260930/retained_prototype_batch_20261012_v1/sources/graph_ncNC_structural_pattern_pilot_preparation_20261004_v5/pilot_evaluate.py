"""Complete VALID scoring and exact bound OGB shared-pool Hits50."""
from pathlib import Path
from time import perf_counter
from pilot_common import require, file_sha
from pilot_model import train_flag
from pilot_data import native_graph, tensor_sha


def evaluator(context):
    from ogb.linkproppred import Evaluator
    import inspect
    pin = context["authority"]["ogb_evaluator"]
    source = Path(inspect.getfile(Evaluator)).resolve()
    require(source == Path(pin["path"]).resolve() and file_sha(source) == pin["sha256"], "OGB evaluator source differs")
    # The sealed design fixes collab's metric. Bypass only the metadata-reading
    # constructor; use the exact source-bound OGB parsing/eval/_eval_hits methods.
    # No unbound master.csv or implicit dataset access is needed.
    metric = Evaluator.__new__(Evaluator)
    metric.name = "ogbl-collab"
    metric.eval_metric = "hits@50"
    metric.K = 50
    require(metric.eval_metric == "hits@50", "OGB official collab metric differs")
    return metric


def hits50(metric, positive, negative):
    require(len(positive) == 60084 and len(negative) == 100000, "Incomplete official VALID score pools")
    return strict_hits50(metric, positive, negative)


def strict_hits50(metric, positive, negative):
    """Pinned OGB arithmetic; tiny fabricated pools are used only by QA."""
    import torch
    require(positive.ndim == negative.ndim == 1 and len(positive) > 0 and len(negative) >= 50, "Incomplete official score pools")
    require(positive.dtype == negative.dtype == torch.float32 and bool(torch.isfinite(positive).all()) and bool(torch.isfinite(negative).all()), "Invalid served scores")
    actual = float(metric.eval({"y_pred_pos": positive, "y_pred_neg": negative})["hits@50"])
    threshold = torch.topk(negative, 50)[0][-1]
    explicit = float((positive > threshold).sum()) / len(positive)
    require(actual == explicit, "Bound OGB strict shared-pool Hits50 rule differs")
    return actual


def score_valid(model, data, mods, *, mode=None):
    import torch
    train_flag(model, False)
    torch.cuda.synchronize(0)
    started = perf_counter()
    graph = native_graph(data["pairs"], len(data["x"])) if isinstance(model, tuple) else mods["graph_ops"].Graph.from_pairs(data["pairs"], len(data["x"]))
    with torch.no_grad():
        encoder, decoder = model if isinstance(model, tuple) else (model.encoder, model.decoder)
        h = encoder(data["x"], graph)
        outputs = []
        batches = []
        for queries in (data["valid_positive"], data["valid_negative"]):
            scores, count = [], 0
            iterator = mods["native_utils"].PermIterator(queries.device, len(queries), 131072, False)
            for indices in iterator:
                if isinstance(model, tuple):
                    current = decoder(h, graph, queries[indices].T).flatten()
                else:
                    current = decoder(h, graph, queries[indices], mode).mean(dim=1)
                require(current.shape == (len(indices),) and bool(torch.isfinite(current).all()), "Complete served query batch differs")
                scores.append(current.detach().cpu())
                count += len(indices)
            require(count == len(queries), "VALID tail or query rows omitted")
            outputs.append(torch.cat(scores))
            batches.append(len(scores))
    torch.cuda.synchronize(0)
    receipt = {"positive_queries": len(outputs[0]), "negative_queries": len(outputs[1]), "query_batches": batches,
               "encoder_calls": 1, "wall_seconds": perf_counter() - started,
               "score_digests": {"positive": tensor_sha(outputs[0]), "negative": tensor_sha(outputs[1])},
               "graph": "complete_TRAIN_only", "serving_pool": "mean_raw_logits", "all_query_rows_complete": True}
    return outputs[0], outputs[1], receipt


def mean_native_scores(rows):
    import torch
    require(len(rows) == 4, "Complete independent bank required")
    positive = torch.stack([row[0] for row in rows], dim=1).mean(1)
    negative = torch.stack([row[1] for row in rows], dim=1).mean(1)
    return positive, negative
