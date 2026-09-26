"""Probability averaging of every archived dropout draw, with no selection."""
import numpy as np


def probability_pool_decisions(member_logits):
    """Return argmax of the arithmetic mean of per-draw softmax probabilities.

    Float64 log-sum-exp avoids overflow and underflow. No labels are inputs.
    The input must contain every one of the four fixed dropout draws.
    """
    logits = np.asarray(member_logits, dtype=np.float64)
    assert logits.ndim == 3 and logits.shape[0] == 4 and logits.shape[2] == 18
    assert np.isfinite(logits).all()
    shifted = logits - logits.max(axis=2, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
    peak = log_probability.max(axis=0)
    log_mean_probability = peak + np.log(np.exp(log_probability - peak).sum(axis=0)) - np.log(4)
    assert np.isfinite(log_mean_probability).all()
    return log_mean_probability.argmax(axis=1)
