"""Inactive helper for a prospective experiment; no trainer or launch entrypoint.

Never imported or numerically exercised during preparation. The default factory
refuses construction. Existing sealed P0/core sources do not import this file.
"""

FEATURE_WIDTH = 512
LABEL_WIDTH = 64
CLASSES = 10


def _require(value, message):
    if not value:
        raise ValueError(message)


def make_query_value_gate(*, torch, later_execution_authorized=False):
    """Build one gate only after a separate release; no RNG reset or random draw.

    C4, joint single and one-path own one instance each. Fully untied U4 owns
    four independent instances, registered in their respective route modules.
    This helper does not construct, patch, optimize, load or launch any bank.
    """
    _require(later_execution_authorized is True,
             "Disabled value-gate helper; separate source integration/release required")

    class QueryValueGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.nn.Parameter(torch.zeros(
                (LABEL_WIDTH, FEATURE_WIDTH), dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.zeros(
                LABEL_WIDTH, dtype=torch.float32))

        def forward(self, message, query_H):
            _require(message.ndim == query_H.ndim == 2
                     and message.shape[1] == LABEL_WIDTH
                     and query_H.shape == (message.shape[0], FEATURE_WIDTH),
                     "Gate requires aggregated 64D message and matching target 512D H")
            _require(not query_H.requires_grad,
                     "The supplied target H must be detached from frozen native B")
            _require(message.dtype == query_H.dtype == self.A.dtype == torch.float32
                     and message.device == query_H.device == self.A.device,
                     "Gate inputs/parameters must share float32 dtype and device")
            gamma = 2 * torch.sigmoid(torch.nn.functional.linear(
                query_H.detach(), self.A, self.b))
            return message * gamma  # No additive shift; zero message stays zero.

    return QueryValueGate()


def lookup_permitted_label_embedding(embedding, *, visible_count,
                                     visible_labels, erase_class_identity=False):
    """Transform value inputs only, after the caller has excluded the entire Q.

    For the permanent C4-erased arm, every training and serving call supplies
    visible_labels=None and erase_class_identity=True. The mean embedding row
    equals feeding the fixed uniform 10-class code to the original matrix.
    The caller never changes train_labels, targets, visibility or native inputs.
    """
    _require(type(visible_count) is int and visible_count >= 0
             and tuple(embedding.weight.shape) == (CLASSES, LABEL_WIDTH)
             and type(erase_class_identity) is bool,
             "Original 10x64 embedding and explicit visible-anchor count/policy required")
    if erase_class_identity:
        _require(visible_labels is None,
                 "Identity-erased value lookup must not receive class labels")
        return embedding.weight.mean(dim=0, keepdim=True).expand(visible_count, -1)
    _require(visible_labels is not None and visible_labels.ndim == 1
             and visible_labels.shape[0] == visible_count,
             "Original visible TRAIN labels are required for the identity-aware arm")
    return embedding(visible_labels)
