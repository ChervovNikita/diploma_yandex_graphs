"""Actual serving and same-bank mean from the same complete native VALID call."""


def validate(session, family, gate):
    torch = session.torch
    session.model.eval()
    if family == "node":
        with torch.no_grad():
            ids = torch.nonzero(session.data.val_mask, as_tuple=False).flatten()
            raw = [route.raw_logits[ids] for route in session._routes(False)]
            served, mean = gate(raw)["served_raw_logits"], torch.stack(raw).mean(0)
            labels = session.data.y[ids]
            loss = torch.nn.functional.cross_entropy(served, labels)
            mean_loss = torch.nn.functional.cross_entropy(mean, labels)
        return {"outputs": served, "loss": float(loss), "metrics": {
                "accuracy": float(served.argmax(-1).eq(labels).float().mean())},
            "role": "VALID", "VALID_scope": "complete_source_VALID", "VALID_nodes": len(ids),
            "VALID_node_ids": ids.detach().cpu(), "VALID_ids": ids.detach().cpu(),
            "VALID_labels": labels.detach().cpu(), "test_scoring": False, "TEST_accessed": False,
            "same_bank_mean_outputs": mean, "same_bank_mean_loss": float(mean_loss),
            "same_bank_mean_metrics": {"accuracy": float(mean.argmax(-1).eq(labels).float().mean())},
            "selection_uses": "actual declared serving output/metric only"}
    if session.recipe.get("smoke_valid_batch_limit") is not None:
        raise ValueError("Complete native VALID required")
    outputs, means, labels, ids = [], [], [], []
    loss_sum = mean_loss_sum = 0.
    count = skipped = 0
    with torch.no_grad():
        for batch in session.valid_loader:
            batch = batch.to(session.device)
            if batch.x.shape[0] == 1:  # exact source main_pyg.eval skip condition
                skipped += 1
                continue
            raw = [session.model.run_member(member, batch)[1] for member in range(4)]
            served, mean = gate(raw)["served_raw_logits"], torch.stack(raw).mean(0)
            y = batch.y.view(served.shape)
            mask = y == y
            if mask.any().item():
                loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    served.float()[mask], y.float()[mask], reduction="sum"))
                mean_loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
                    mean.float()[mask], y.float()[mask], reduction="sum"))
                count += int(mask.sum())
            outputs.append(served.detach().cpu()); means.append(mean.detach().cpu())
            labels.append(y.detach().cpu()); ids.append(batch.smoke_graph_id.detach().cpu())
    if not outputs or not count:
        raise ValueError("Complete native VALID output and observed-label support required")
    outputs, means, labels, ids = torch.cat(outputs), torch.cat(means), torch.cat(labels), torch.cat(ids)
    observed = labels[:, 0] == labels[:, 0]
    if len(torch.unique(labels[observed, 0])) != 2:
        raise ValueError("Native VALID AUC support required")
    return {"outputs": outputs, "loss": loss_sum / count,
        "metrics": session.evaluator.eval({"y_true": labels.numpy(), "y_pred": outputs.numpy()}),
        "VALID_graph_ids": ids.tolist(), "VALID_graphs": len(outputs), "VALID_scope": "complete_source_VALID",
        "VALID_ids": ids, "VALID_labels": labels, "native_skipped_batches": skipped, "TEST_accessed": False,
        "same_bank_mean_outputs": means, "same_bank_mean_loss": mean_loss_sum / count,
        "same_bank_mean_metrics": session.evaluator.eval({"y_true": labels.numpy(), "y_pred": means.numpy()}),
        "selection_uses": "actual declared serving output/metric only"}
