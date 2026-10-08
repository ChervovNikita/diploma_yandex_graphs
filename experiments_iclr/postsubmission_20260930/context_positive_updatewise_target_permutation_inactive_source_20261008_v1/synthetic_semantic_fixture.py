"""Small stdlib-only synthetic fixture; no model/data/autograd qualification."""
import itertools
import json
import math
import random
from types import SimpleNamespace

from updatewise_target_permutation import install, permutation_for_update, PERMUTATIONS


def softmax(row):
    maximum = max(row)
    values = [math.exp(x - maximum) for x in row]
    return [x / sum(values) for x in values]


def loss_and_score_gradient(scores, weights):
    """Original symmetric full-denominator target CE at scaled score level."""
    members, size = len(scores), len(scores[0])
    loss = 0.
    gradients = []
    for m in range(members):
        forward = [softmax(row) for row in scores[m]]
        reverse = [softmax([scores[m][i][j] for i in range(size)]) for j in range(size)]
        loss -= sum(weights[m][i][j] * (math.log(forward[i][j]) + math.log(reverse[i][j]))
                    for i in range(size) for j in range(size)) / (2 * members * size)
        gradients.extend((forward[i][j] - weights[m][i][j]
                          + reverse[j][i] - weights[m][j][i]) / (2 * members * size)
                         for i in range(size) for j in range(size))
    return loss, gradients


class Cache:
    """Small structural stand-in for cache indexing, not an actual Torch tensor."""
    shape, requires_grad, _version = (4, 512, 512), False, 0
    def __init__(self, targets=("Q0", "Q1", "Q2", "Q3"), fail_copy=False):
        self.targets = tuple(targets)
        self.fail_copy = fail_copy
    def __getitem__(self, index):
        if self.fail_copy:
            raise MemoryError("synthetic target-cache copy failure")
        return Cache(tuple(self.targets[i] for i in index))


def lifecycle_fixture(fail=False, wrong_method=False, fail_copy=False):
    session = SimpleNamespace(task="wikics", seed=6101, steps=0, optimizers=[object()],
        model=SimpleNamespace(members=4, independent=False, contrastive=True),
        config={"contrastive": {"alignment_weight": .05, "residual_weight": 0.,
                                "temperature": .2, "max_objects": 512},
                "context_target_method": {"method": "shared_route", "mode": "route",
                                          "permuted": False, "own_selected_four": False}},
        streams=[{"cpu": "unchanged dropout stream " + str(m)} for m in range(4)],
        execution_totals={"exact_member_RNG_endpoint_checks": 0})
    own, residual = object(), object()
    facade = SimpleNamespace(mode="route", session=session, weight_cache=Cache(fail_copy=fail_copy),
                             own_supervision=own, residual_member_contrast=residual)
    received = []
    def alignment(a, b, labels, task, temperature=.2, identities=None):
        assert (a, b, labels, task, temperature, identities) == ("a", "b", "labels", "wikics", .2, None)
        received.append(facade.weight_cache)
        return facade.weight_cache.targets
    facade.alignment_loss = alignment
    session.core = {"objectives": facade}
    def update(batch, labels):
        assert batch == "batch" and labels == "labels"
        first = facade.alignment_loss("a", "b", labels, "wikics")
        # Simulate a repeated cotangent/loss evaluation during the SAME update.
        second = facade.alignment_loss("a", "b", labels, "wikics")
        assert first == second and received[-1] is received[-2]
        assert sorted(first) == ["Q0", "Q1", "Q2", "Q3"]
        if fail:
            raise RuntimeError("synthetic failure")
        session.steps += 1
        return first
    session.train_step = update
    original_cache, original_streams = facade.weight_cache, repr(session.streams)
    if wrong_method:
        session.config["context_target_method"].update(method="shared_route_permuted", permuted=True)
        try:
            install(session, 866102)
        except ValueError:
            pass
        else:
            raise AssertionError("node-incidence PERMUTED facade accepted as factual ROUTE")
        assert session.train_step is update and facade.alignment_loss is alignment
        return
    adapter = install(session, 6101 + 860001)
    assert adapter.metadata()["assignment_draws"] == 0
    assert facade.own_supervision is own and facade.residual_member_contrast is residual
    if fail or fail_copy:
        try:
            session.train_step("batch", "labels")
        except (MemoryError, RuntimeError):
            pass
        else:
            raise AssertionError("synthetic failure was not propagated")
        try:
            session.train_step("batch", "labels")
        except ValueError:
            pass
        else:
            raise AssertionError("failed step allowed retry")
        assert adapter.metadata()["last_binding"]["status"] == "failed"
        assert adapter.metadata()["assignment_draws"] == 1 and session.steps == 0
    else:
        for step in range(3):
            result = session.train_step("batch", "labels")
            expected = tuple("Q" + str(i) for i in permutation_for_update(866102, step))
            assert result == expected
            assert adapter.metadata()["last_binding"]["update"] == step
        assert adapter.metadata()["assignment_draws"] == adapter.metadata()["completed_updates"] == 3
        assert adapter.metadata()["delegated_loss_calls"] == 6
    assert facade.weight_cache is original_cache and repr(session.streams) == original_streams
    assert adapter.active is None
    try:
        facade.alignment_loss("a", "b", "labels", "wikics")
    except ValueError:
        pass
    else:
        raise AssertionError("unbound alignment accepted")


def main():
    assert len(PERMUTATIONS) == 24 and set(PERMUTATIONS) == set(itertools.permutations(range(4)))
    python_rng = random.getstate()
    draws = [permutation_for_update(866102, step) for step in range(12)]
    assert random.getstate() == python_rng
    assert draws == [permutation_for_update(866102, step) for step in range(12)]
    assert all(sorted(order) == [0, 1, 2, 3] for order in draws)
    lifecycle_fixture()
    lifecycle_fixture(fail=True)
    lifecycle_fixture(fail_copy=True)
    lifecycle_fixture(wrong_method=True)
    assert random.getstate() == python_rng

    # Four deliberately asymmetric, unequal-concentration directed targets.
    factual = [
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[.6, .4, 0.], [0., .7, .3], [.2, 0., .8]],
        [[.4, 0., .6], [.2, .8, 0.], [0., .5, .5]],
        [[.5, .2, .3], [.1, .5, .4], [.3, .2, .5]]]
    common = [[sum(q[i][j] for q in factual) / 4 for j in range(3)] for i in range(3)]
    scores = [[[.23 * (m + 1) * (i - j) + .11 * i * j + .07 * m * j
                for j in range(3)] for i in range(3)] for m in range(4)]
    reference_loss, reference_gradient = loss_and_score_gradient(scores, [common] * 4)
    losses, gradients = [], []
    for order in PERMUTATIONS:
        targets = [factual[index] for index in order]
        assert all(sum(targets[m][i][j] for m in range(4)) == sum(factual[m][i][j] for m in range(4))
                   or math.isclose(sum(targets[m][i][j] for m in range(4)),
                                   4 * common[i][j], abs_tol=1e-14)
                   for i in range(3) for j in range(3))
        value, gradient = loss_and_score_gradient(scores, targets)
        losses.append(value)
        gradients.append(gradient)
    loss_error = abs(sum(losses) / 24 - reference_loss)
    mean_gradient = [sum(g[k] for g in gradients) / 24 for k in range(36)]
    gradient_error = max(abs(x - y) for x, y in zip(mean_gradient, reference_gradient))

    # Arbitrary common score Jacobian with two shared and four private coordinates.
    # Route states are unequal. This checks raw-gradient pullback, not Adam.
    jacobian = [[math.sin(.17 * (score + 1) * (parameter + 2))
                 if parameter < 2 or parameter == score // 9 + 2 else 0.
                 for parameter in range(6)] for score in range(36)]
    def pullback(gradient):
        return [sum(jacobian[k][p] * gradient[k] for k in range(36)) for p in range(6)]
    raw_reference = pullback(reference_gradient)
    raw_mean = [sum(pullback(g)[p] for g in gradients) / 24 for p in range(6)]
    block_error = max(abs(x - y) for x, y in zip(raw_mean, raw_reference))

    deltas = [[q[i][j] - common[i][j] for i in range(3) for j in range(3)] for q in factual]
    covariance_error = 0.
    for i in range(9):
        for j in range(9):
            variance = sum(d[i] * d[j] for d in deltas) / 4
            between = sum(deltas[order[0]][i] * deltas[order[1]][j] for order in PERMUTATIONS) / 24
            covariance_error = max(covariance_error, abs(between + variance / 3))
    assert max(loss_error, gradient_error, block_error, covariance_error) < 1e-12
    assert max(abs(gradients[0][k] - reference_gradient[k]) for k in range(36)) > 1e-4
    print(json.dumps({"synthetic_only": True, "passed": True, "assignments_enumerated": 24,
        "loss_mean_error": loss_error, "score_gradient_mean_error": gradient_error,
        "shared_private_pullback_mean_error": block_error, "target_cross_route_covariance_error": covariance_error,
        "adapter_lifecycle": "one binding/update; repeated loss reuse; failure no retry; cache restored",
        "global_Python_rng_unchanged": True, "member_stream_standins_unchanged": True,
        "node_incidence_PERMUTED_input_rejected": True,
        "native_Torch_autograd_or_model_qualified": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
