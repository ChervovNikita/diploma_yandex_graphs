"""Unexecuted GRU-style complete feasible-schedule laws with injected backend.

No Torch import, model/RNG creation, sampling, teacher forcing or numerical
qualification has occurred. All parameter tensors and uniforms are external.
This changes the source GNNM factor placement; the graph encoder has no BE.
"""
from math import isfinite

from .domain import GeneratedSchedule


def make_recurrent_decoder(*, torch_api, mode: str, n_branches: int,
                           embedding_dimension: int, hidden_dimension: int,
                           parameter_tensors, ledger):
    """Future module factory from externally initialized parameter tensors.

    factored: shared W_x,W_h,W_o; branch diagonals, gate/output biases and h0.
    single: one conventional decoder, no diagonal factors.
    untied: M conventional decoders with private W_x,W_h,W_o and biases.
    Parameter pairing, parameter/cost matching and initialization remain unset.
    """
    if mode not in ("factored", "single", "untied"):
        raise ValueError("Declared decoder arm required")
    dimensions = (n_branches, embedding_dimension, hidden_dimension)
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("Explicit positive branch/embedding/hidden dimensions required")
    if mode == "single" and n_branches != 1:
        raise ValueError("Single-decoder control has exactly one decoder law")
    M, d, h, p = n_branches, embedding_dimension, hidden_dimension, embedding_dimension + 2
    base_shapes = {"W_x": (3 * h, p), "W_h": (3 * h, h), "W_o": (2, h)}
    shapes = {name: ((M, *shape) if mode == "untied" else shape)
              for name, shape in base_shapes.items()}
    shapes.update({"b_x": (M, 3 * h), "b_h": (M, 3 * h),
                   "b_o": (M, 2), "h0": (M, h)})
    if mode == "factored":
        shapes.update({"r_x": (M, p), "s_x": (M, 3 * h),
                       "r_h": (M, h), "s_h": (M, 3 * h),
                       "r_o": (M, h), "s_o": (M, 2)})
    if set(parameter_tensors) != set(shapes):
        raise ValueError("Supply exactly the declared matrices, private biases/states and arm factors")
    for name, shape in shapes.items():
        if tuple(parameter_tensors[name].shape) != shape:
            raise ValueError("Parameter shape differs from the declared decoder architecture: " + name)
        if not torch_api.isfinite(parameter_tensors[name]).all().item():
            raise ValueError("Finite externally initialized parameter tensors required")

    class RecurrentScheduleDecoder(torch_api.nn.Module):
        def __init__(self):
            super().__init__()
            for name in shapes:
                self.register_parameter(name, torch_api.nn.Parameter(parameter_tensors[name]))
            self.mode = mode
            self.n_branches = M
            self.embedding_dimension = d
            self.hidden_dimension = h
            self.work_ledger = ledger

        def _check(self, encoded, branch):
            if type(branch) is not int or not 0 <= branch < M:
                raise ValueError("Decoder branch outside the declared law")
            if tuple(encoded.H.shape) != (encoded.domain.n_nodes, d):
                raise ValueError("Reuse the same complete static graph embedding H")

        def _weight(self, name, branch):
            weight = getattr(self, name)
            return weight[branch] if self.mode == "untied" else weight

        def _step(self, branch, state, node_embedding, previous_label, blocked):
            # A zero start token and blocked flag are deterministic prefix inputs.
            token = node_embedding.new_tensor((float(previous_label), float(blocked)))
            x = torch_api.cat((node_embedding, token), dim=-1)
            if self.mode == "factored":
                gx = torch_api.nn.functional.linear(x * self.r_x[branch], self.W_x)
                gh = torch_api.nn.functional.linear(state * self.r_h[branch], self.W_h)
                gx = gx * self.s_x[branch] + self.b_x[branch]
                gh = gh * self.s_h[branch] + self.b_h[branch]
            else:
                gx = torch_api.nn.functional.linear(x, self._weight("W_x", branch), self.b_x[branch])
                gh = torch_api.nn.functional.linear(state, self._weight("W_h", branch), self.b_h[branch])
            xr, xz, xn = gx.chunk(3, dim=-1)
            hr, hz, hn = gh.chunk(3, dim=-1)
            reset = torch_api.sigmoid(xr + hr)
            update = torch_api.sigmoid(xz + hz)
            candidate = torch_api.tanh(xn + reset * hn)
            state = (1 - update) * candidate + update * state
            if self.mode == "factored":
                logits = torch_api.nn.functional.linear(state * self.r_o[branch], self.W_o)
                logits = logits * self.s_o[branch] + self.b_o[branch]
            else:
                logits = torch_api.nn.functional.linear(state, self._weight("W_o", branch), self.b_o[branch])
            if not torch_api.isfinite(state).all().item() or not torch_api.isfinite(logits).all().item():
                raise ArithmeticError("Decoder state/logits lost numerical finiteness")
            if blocked:
                # Forced zero has probability 1 and log density 0, independent of theta.
                log_probabilities = torch_api.stack((logits.new_zeros(()), logits.new_full((), float("-inf"))))
            else:
                log_probabilities = torch_api.nn.functional.log_softmax(logits, dim=-1)
                if not torch_api.isfinite(log_probabilities).all().item():
                    raise ArithmeticError("Allowed schedule decision lost positive finite log probability")
            return state, log_probabilities

        def _charge(self, encoded, kind, retain_score_gradient):
            self.work_ledger.record_decoder(
                kind=kind, n_nodes=encoded.domain.n_nodes,
                n_edges=len(encoded.domain.edges), embedding_dimension=d,
                hidden_dimension=h, factored=self.mode == "factored",
                retain_score_gradient=retain_score_gradient,
            )

        def generate(self, encoded, *, branch: int, uniforms):
            """Unconditional branch draw from caller-supplied independent uniforms.

            Exactly N uniforms in [0,1), in the recorded order, including ignored
            draws for forced-zero steps. No future-lock clamping API is offered.
            Branch selection for the mixture is also supplied by the caller.
            """
            self._check(encoded, branch)
            uniforms = tuple(float(value) for value in uniforms)
            if len(uniforms) != encoded.domain.n_nodes or any(not isfinite(value) or not 0 <= value < 1 for value in uniforms):
                raise ValueError("Caller must supply N valid sampling uniforms")
            self._charge(encoded, "generation", False)
            prefix = [None] * encoded.domain.n_nodes
            previous = 0
            with torch_api.no_grad():
                state = self.h0[branch]
                for index, node in enumerate(encoded.domain.order):
                    blocked = encoded.domain.blocked(node, prefix)
                    state, logp = self._step(branch, state, encoded.H[node], previous, blocked)
                    if blocked:
                        label = 0
                    else:
                        probability = logp[1].exp().item()
                        if not 0 < probability < 1:
                            raise ArithmeticError("Free-step sampling probability reached a floating-point boundary")
                        label = int(uniforms[index] < probability)
                    prefix[node] = label
                    previous = label
            return GeneratedSchedule(tuple(prefix), branch, encoded.domain.graph_id, encoded.domain.order_id)

        def log_density(self, encoded, assignment, *, branch: int,
                        retain_score_gradient: bool):
            """Exact teacher-forced full branch log density, up to arithmetic.

            All complete feasible assignments are accepted, irrespective of the
            branch that generated them. Infeasible assignments have density zero.
            The gradient flag controls storage, never the law or masks.
            """
            self._check(encoded, branch)
            assignment = encoded.domain.assignment(assignment)
            if not encoded.domain.feasible(assignment):
                return encoded.H.new_full((), float("-inf"))
            self._charge(encoded, "density", retain_score_gradient)

            def replay():
                prefix = [None] * encoded.domain.n_nodes
                previous = 0
                state = self.h0[branch]
                # Include the empty-graph law's zero parameter derivative.
                score = encoded.H.sum() * 0 + state.sum() * 0
                for node in encoded.domain.order:
                    blocked = encoded.domain.blocked(node, prefix)
                    state, logp = self._step(branch, state, encoded.H[node], previous, blocked)
                    label = assignment[node]
                    score = score + logp[label]
                    prefix[node] = label
                    previous = label
                if not torch_api.isfinite(score).item():
                    raise ArithmeticError("Feasible full-schedule score lost numerical finiteness")
                return score

            if retain_score_gradient:
                if not torch_api.is_grad_enabled():
                    raise ValueError("Generating score requires an enabled gradient context")
                return replay()
            with torch_api.no_grad():
                return replay()

        def cross_member_log_densities(self, encoded, samples):
            """Full [S,M] matrix; only generating scores retain autograd.

            The self score from discrete generation is recomputed here; no
            generation/density reuse is claimed. Other M-1 densities are forward
            only, because the score-function reward is detached.
            """
            samples = tuple(samples)
            if not samples:
                raise ValueError("Nonempty externally designed sample batch required")
            rows = []
            for sample in samples:
                if sample.graph_id != encoded.domain.graph_id or sample.order_id != encoded.domain.order_id:
                    raise ValueError("All full densities must use the same graph and fixed order")
                self._check(encoded, sample.generating_branch)
                if not encoded.domain.feasible(sample.assignment):
                    raise ValueError("Generating records must be complete feasible schedules")
                rows.append(torch_api.stack(tuple(
                    self.log_density(encoded, sample.assignment, branch=branch,
                                     retain_score_gradient=branch == sample.generating_branch)
                    for branch in range(M)
                )))
            self.work_ledger.record_cross_matrix(len(samples), M)
            return torch_api.stack(rows)

    return RecurrentScheduleDecoder()
