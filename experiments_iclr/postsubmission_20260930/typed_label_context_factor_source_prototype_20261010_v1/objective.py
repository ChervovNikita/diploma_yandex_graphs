"""Disabled direct-own BCE/serving helpers. No trainer, optimizer or launch entry."""
from .caps import CLOSED
from .context import half_inputs


def paired_epoch_loss(torch, bank, views, train_bundle, feature_dict, ledger, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime", "scientific")
    if tuple(train_bundle.ids) != views.plan.train_ids:
        raise ValueError("Use the exact custody-bound full TRAIN bundle")
    member_losses = []
    total_queries = len(views.plan.train_ids)
    for m in range(len(bank.members)):
        device = next(bank.members[m].parameters()).device
        loss = None
        for view in views.views:
            batch, features, labels, field = half_inputs(
                torch, view, view.query_ids, feature_dict,
                bank.condition == "shared_global_mul4", caps)
            features = {key: value.to(device) for key, value in features.items()}
            labels = {key: value.to(device) for key, value in labels.items()}
            batch = batch.to(device)
            logits = bank.forward_member(m, batch, features, labels, field)
            position = torch.tensor(view.query_positions, dtype=torch.long,
                                    device=train_bundle.targets.device)
            truth = train_bundle.targets[position].to(device=logits.device, dtype=torch.float32)
            term = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth)
            weighted = term*(len(view.query_ids)/total_queries)
            loss = weighted if loss is None else loss+weighted
            ledger.append({"scope": "TRAIN_context_forward", "member": m,
                           "context_nodes": len(view.context_ids), "query_nodes": len(view.query_ids),
                           "complete_native_channels": 37, "graph_or_path_shrink": False})
        member_losses.append(loss)
    # One external native optimizer step per complete paired epoch. Shared mean,
    # untied sum preserves unscaled own-body gradients. No pooled training credit.
    return (sum(member_losses)/len(member_losses) if bank.shared else sum(member_losses)), tuple(member_losses)


def serving_logits(torch, bank, views, ids, feature_dict, ledger, caps=CLOSED):
    caps.require("source_bound", "model", "data", "runtime")
    if bank.training:
        raise ValueError("Serving requires native eval mode and selected replay state")
    members = []
    with torch.no_grad():
        for m in range(len(bank.members)):
            device = next(bank.members[m].parameters()).device
            context_logits = []
            for view in views.views:
                batch, features, labels, field = half_inputs(
                    torch, view, ids, feature_dict,
                    bank.condition == "shared_global_mul4", caps)
                features = {key: value.to(device) for key, value in features.items()}
                labels = {key: value.to(device) for key, value in labels.items()}
                batch = batch.to(device)
                context_logits.append(bank.forward_member(m, batch, features, labels, field))
                ledger.append({"scope": "serving_context_forward", "member": m,
                               "context_nodes": len(view.context_ids), "query_nodes": len(ids),
                               "inference_context_density_matches_training": True})
            members.append(torch.stack(context_logits).mean(dim=0))
    member_bank = torch.stack(members)  # Context-averaged members, not selected best contexts.
    return member_bank.mean(dim=0), member_bank  # Native mean raw logits; sigmoid>0.5 downstream.
