"""Inactive updatewise bijection of the same four frozen factual ROUTE targets.

No driver, target preparation, model, score reader or execution release is added.
Install only after the unchanged context replay installer and before cost timing.
"""
import hashlib
import itertools
import json
import random
import time

ALGORITHM = "sha256_seed_step__private_MT19937_randrange24__lexicographic_v1"
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def permutation_for_update(assignment_seed, update):
    """Stateless private PRNG draw; neither global Python nor Torch RNG is used.

    randrange uses rejection sampling; all 24 bijections, including identity
    and repeats on adjacent updates, remain eligible. The uniform assignment
    law in the saved argument is the usual idealized PRNG draw law.
    """
    require(type(assignment_seed) is int and 0 <= assignment_seed < 2**64,
            "explicit assignment seed in [0,2**64) required")
    require(type(update) is int and update >= 0, "zero-based completed-step count required")
    payload = json.dumps([ALGORITHM, assignment_seed, update], separators=(",", ":")).encode("ascii")
    seed = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return PERMUTATIONS[random.Random(seed).randrange(len(PERMUTATIONS))]


class UpdatewiseTargetPermutation:
    """Decorate an already constructed factual ROUTE facade and replay update.

    The original facade remains the same class/object, preserving the replay
    installer's type contract. Its own CE, residual-zero method, panel checks
    and alignment operation are unchanged. Only its cached target order is
    replaced while an update is active.
    """
    def __init__(self, session, assignment_seed):
        facade = session.core["objectives"]
        require(session.task == "wikics" and session.model.members == 4
                and not session.model.independent and session.model.contrastive
                and len(session.optimizers) == 1, "shared four-member WikiCS only")
        require(getattr(facade, "mode", None) == "route"
                and getattr(facade, "session", None) is session,
                "existing factual ROUTE facade required; COMMON/cycle/PERMUTED are not inputs")
        method = session.config.get("context_target_method", {})
        require(method.get("method") == "shared_route" and method.get("mode") == "route"
                and method.get("permuted") is False and method.get("own_selected_four") is False,
                "factual shared_route dispatch identity required, not shared_route_permuted")
        specification = session.config["contrastive"]
        require(specification["alignment_weight"] == .05 and specification["residual_weight"] == 0.
                and specification["temperature"] == .2 and specification["max_objects"] == 512,
                "unchanged .05/.2/residual-free/512-panel contract required")
        weights = facade.weight_cache
        require(tuple(weights.shape) == (4, 512, 512) and not weights.requires_grad,
                "four frozen factual targets after original panel normalization required")
        require(session.steps == 0 and not hasattr(session, "updatewise_target_permutation"),
                "one fresh inactive-adapter installation only")
        require("exact_member_RNG_endpoint_checks" in getattr(session, "execution_totals", {}),
                "install unchanged context VJP replay before this decorator")
        require(type(assignment_seed) is int and 0 <= assignment_seed < 2**64,
                "explicit assignment seed in [0,2**64) required")
        self.session, self.facade = session, facade
        self.assignment_seed, self.factual_weights = assignment_seed, weights
        self.target_version = weights._version
        self.original_alignment, self.original_update = facade.alignment_loss, session.train_step
        self.active = self.last = None
        self.failed, self.next_update = False, 0
        self.assignment_draws = self.completed_updates = self.delegated_loss_calls = 0
        self.assignment_wall_seconds = 0.

    def alignment_loss(self, a, b, labels, task, temperature=.2, identities=None):
        require(self.active is not None and self.session.steps == self.active["update"],
                "alignment requires this update's already-bound permutation")
        require(self.facade.weight_cache is self.active["weights"],
                "bound target cache changed during loss/replay")
        self.delegated_loss_calls += 1
        # One original call computes BOTH directions using this same cache.
        return self.original_alignment(a, b, labels, task, temperature, identities)

    def train_step(self, batch, labels):
        require(not self.failed and self.active is None, "failed/nested update cannot be retried here")
        update = self.session.steps
        require(update == self.next_update, "original completed-step ordinal changed")
        require(self.factual_weights._version == self.target_version,
                "frozen factual target tensor was mutated")
        started = time.monotonic()
        binding_finished = False
        self.last = {"update": update, "permutation": None, "status": "binding"}
        try:
            permutation = permutation_for_update(self.assignment_seed, update)
            self.assignment_draws += 1
            self.last["permutation"] = list(permutation)
            assigned = self.factual_weights[list(permutation)]
            self.active = {"update": update, "permutation": permutation, "weights": assigned}
            self.facade.weight_cache = assigned
            self.assignment_wall_seconds += time.monotonic() - started
            binding_finished = True
            self.last["status"] = "bound"
            # Cache is held throughout shadow views, cotangents, and all VJPs.
            result = self.original_update(batch, labels)
            require(self.session.steps == update + 1, "original update must advance steps exactly once")
            self.completed_updates += 1
            self.next_update = update + 1
            self.last["status"] = "complete"
            return result
        except BaseException:
            self.failed = True
            self.last["status"] = "failed"
            raise
        finally:
            if not binding_finished:
                self.assignment_wall_seconds += time.monotonic() - started
            # Validation/serving never needs a draw or the training target cache.
            self.facade.weight_cache = self.factual_weights
            self.active = None

    def metadata(self):
        """JSON-safe source/step identity; stateless replay needs no RNG snapshot.

        Caller must include this in run/progress/selected/terminal custody.
        This source does not add exact-fit resume support to the original driver.
        """
        return {"condition": "shared_updatewise_factual_target_permutation",
                "execution_authorized": False, "algorithm": ALGORITHM,
                "assignment_seed": self.assignment_seed,
                "update_ordinal": "original session.steps before update; zero based; no epoch101 reset",
                "assignment_draws": self.assignment_draws,
                "completed_updates": self.completed_updates,
                "delegated_loss_calls": self.delegated_loss_calls,
                "assignment_wall_seconds": self.assignment_wall_seconds,
                "last_binding": dict(self.last) if self.last is not None else None,
                "failed": self.failed, "uses_original_member_or_global_Torch_RNG": False,
                "target_scope": "same normalized factual Q0,Q1,Q2,Q3; bijection every update",
                "held_through": "both original loss directions, cotangents, all replay VJPs",
                "exact_fit_resume_added": False}


def install(session, assignment_seed):
    """Inactive hook only: no new arm registration or training is dispatched."""
    adapter = UpdatewiseTargetPermutation(session, assignment_seed)
    adapter.facade.alignment_loss = adapter.alignment_loss
    session.train_step = adapter.train_step
    session.updatewise_target_permutation = adapter
    return adapter
