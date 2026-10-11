"""Exact saved native training function; caller supplies F and TABM_INITS=4.
No stock evaluator, dataset loader or automatic TEST path is imported.
"""

def train_step_tabm(model, data, optimizer, is_binary):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for i in range(TABM_INITS):
        out = model(data, data.x, tabm_seed=i)
        if is_binary:
            loss = F.binary_cross_entropy_with_logits(out[data.train_mask].squeeze(-1), data.y[data.train_mask])
        else:
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].long())
        loss = loss / TABM_INITS
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss
