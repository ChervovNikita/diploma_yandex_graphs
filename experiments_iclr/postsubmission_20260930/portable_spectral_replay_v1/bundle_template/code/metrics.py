"""Exact CPU-v3 saved-logit metric function; no author-evidence dependencies."""
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce

def metrics(logits,targets):
    if len(logits) != len(targets) or not targets: raise ValueError("Incomplete validation evidence")
    tp=fp=fn=entries=0; pooled_loss=member_loss=graph_member_loss=graph_pooled_loss=0.0
    for pred,y in zip(logits,targets):
        if pred.shape != (4,y.shape[0],121) or not bool(torch.isfinite(pred).all()): raise ValueError("Invalid validation member logits")
        pooled=pred.mean(0); truth=y.bool(); guessed=pooled>=0
        tp+=int((truth&guessed).sum()); fp+=int((~truth&guessed).sum()); fn+=int((truth&~guessed).sum())
        this_pooled=float(bce(pooled,y,reduction="sum")); pooled_loss+=this_pooled
        risk=float(bce(pred,y[None].expand_as(pred),reduction="sum"))/4
        graph_pooled_loss+=this_pooled/y.numel()
        member_loss+=risk; graph_member_loss+=risk/y.numel(); entries+=y.numel()
    gap=(graph_member_loss-graph_pooled_loss)/len(targets)
    if gap < -2e-7: raise ValueError("Convex-BCE member/pooled Jensen gap is negative")
    return {"micro_f1":2*tp/max(1,2*tp+fp+fn),"pooled_bce":pooled_loss/entries,
            "mean_member_bce":member_loss/entries,"graph_mean_member_bce":graph_member_loss/len(targets),
            "graph_mean_pooled_bce":graph_pooled_loss/len(targets),"graph_mean_bce_jensen_gap":gap}
