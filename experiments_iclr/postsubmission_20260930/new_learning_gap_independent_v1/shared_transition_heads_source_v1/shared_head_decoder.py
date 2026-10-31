"""Unexecuted member-independent recurrence with private output heads.

All backend/parameters/graphs/orders/uniforms are externally supplied. Nothing
has been imported, constructed, sampled, compiled or numerically executed.
This is an attribution control, not a new methodology claim.
"""
from math import isfinite

from .contracts import GeneratedSchedule


def make_shared_transition_decoder(*, torch_api, head_mode: str,
                                   n_branches: int, embedding_dimension: int,
                                   hidden_dimension: int,
                                   parameter_tensors, ledger):
    """Future module factory; recurrence and h0 have NO member axis.

    ordinary_heads: private unconstrained dense output matrices and biases.
    factored_heads: one common output matrix, private diagonal factors/biases.
    No head, member identity, mixture weight or predicted probability enters the
    transition. Previous sampled labels are the ordinary autoregressive input.
    """
    if head_mode not in ("ordinary_heads", "factored_heads"):
        raise ValueError("Declared ordinary/factored private-head control required")
    dims = (n_branches, embedding_dimension, hidden_dimension)
    if any(type(value) is not int or value <= 0 for value in dims):
        raise ValueError("Explicit positive branch/embedding/hidden dimensions required")
    M, d, h, p = n_branches, embedding_dimension, hidden_dimension, embedding_dimension + 2
    shapes = {"W_x": (3 * h, p), "W_h": (3 * h, h),
              "b_x": (3 * h,), "b_h": (3 * h,), "h0": (h,),
              "b_o": (M, 2)}
    if head_mode == "ordinary_heads":
        shapes["W_o"] = (M, 2, h)
    else:
        shapes.update({"W_o": (2, h), "r_o": (M, h), "s_o": (M, 2)})
    if set(parameter_tensors) != set(shapes):
        raise ValueError("Supply exactly shared recurrence/h0 and declared private-head tensors")
    for name, shape in shapes.items():
        if tuple(parameter_tensors[name].shape) != shape:
            raise ValueError("Parameter shape differs from the control: " + name)
        if not torch_api.isfinite(parameter_tensors[name]).all().item():
            raise ValueError("Finite externally initialized parameter tensors required")

    class SharedTransitionHeadDecoder(torch_api.nn.Module):
        def __init__(self):
            super().__init__()
            for name in shapes:
                self.register_parameter(name, torch_api.nn.Parameter(parameter_tensors[name]))
            self.head_mode = head_mode
            self.n_branches = M
            self.embedding_dimension = d
            self.hidden_dimension = h
            self.work_ledger = ledger

        def _check(self, encoded, branch=None):
            if branch is not None and (type(branch) is not int or not 0 <= branch < M):
                raise ValueError("Head branch outside the declared mixture")
            if tuple(encoded.H.shape) != (encoded.domain.n_nodes, d):
                raise ValueError("Reuse one complete static node embedding H")

        def _transition(self, state, node_embedding, previous_label, blocked):
            # No branch argument and no member-dependent parameter/input here.
            token = node_embedding.new_tensor((float(previous_label), float(blocked)))
            x = torch_api.cat((node_embedding, token), dim=-1)
            gx = torch_api.nn.functional.linear(x, self.W_x, self.b_x)
            gh = torch_api.nn.functional.linear(state, self.W_h, self.b_h)
            xr, xz, xn = gx.chunk(3, dim=-1)
            hr, hz, hn = gh.chunk(3, dim=-1)
            reset = torch_api.sigmoid(xr + hr)
            update = torch_api.sigmoid(xz + hz)
            candidate = torch_api.tanh(xn + reset * hn)
            state = (1 - update) * candidate + update * state
            if not torch_api.isfinite(state).all().item():
                raise ArithmeticError("Shared recurrent state lost numerical finiteness")
            return state

        def _head_log_probabilities(self, branch, state, blocked):
            if self.head_mode == "ordinary_heads":
                logits = torch_api.nn.functional.linear(state, self.W_o[branch], self.b_o[branch])
            else:
                logits = torch_api.nn.functional.linear(state * self.r_o[branch], self.W_o)
                logits = logits * self.s_o[branch] + self.b_o[branch]
            if not torch_api.isfinite(logits).all().item():
                raise ArithmeticError("Private-head logits lost numerical finiteness")
            if blocked:
                return torch_api.stack((logits.new_zeros(()), logits.new_full((), float("-inf"))))
            logp = torch_api.nn.functional.log_softmax(logits, dim=-1)
            if not torch_api.isfinite(logp).all().item():
                raise ArithmeticError("Free-step head lost positive finite log probabilities")
            return logp

        def _charge(self, encoded, kind, heads_per_step, retain_score_graph):
            self.work_ledger.record_shared_trajectory(
                kind=kind, n_nodes=encoded.domain.n_nodes,
                n_edges=len(encoded.domain.edges), embedding_dimension=d,
                hidden_dimension=h, heads_per_step=heads_per_step,
                factored_heads=self.head_mode == "factored_heads",
                retain_generating_score_graph=retain_score_graph,
            )

        def generate(self, encoded, *, branch: int, uniforms):
            """Unconditional draw; every sampled history gets its own recurrence.

            N independent uniforms and mixture branch selection are supplied by
            the caller. Forced decisions consume an ignored uniform slot. No
            future-lock clamping or retry of numerical failures is implemented.
            """
            self._check(encoded, branch)
            if branch is None:
                raise ValueError("Generating head must be supplied")
            uniforms = tuple(float(value) for value in uniforms)
            if len(uniforms) != encoded.domain.n_nodes or any(not isfinite(value) or not 0 <= value < 1 for value in uniforms):
                raise ValueError("Exactly N valid caller-supplied uniforms required")
            self._charge(encoded, "generation", 1, False)
            prefix = [None] * encoded.domain.n_nodes
            previous = 0
            with torch_api.no_grad():
                state = self.h0
                for index, node in enumerate(encoded.domain.order):
                    blocked = encoded.domain.blocked(node, prefix)
                    state = self._transition(state, encoded.H[node], previous, blocked)
                    logp = self._head_log_probabilities(branch, state, blocked)
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

        def full_log_densities(self, encoded, assignment, *, generating_branch=None):
            """All M complete densities from ONE same-y recurrent trajectory.

            With a generating branch, shared states retain its score graph;
            nonself heads use detached states under no_grad. No cross-head graph
            or duplicate gradient replay of the recurrence is built. With None,
            the entire full-density vector is calculated under no_grad.
            """
            self._check(encoded, generating_branch)
            assignment = encoded.domain.assignment(assignment)
            if not encoded.domain.feasible(assignment):
                return encoded.H.new_full((M,), float("-inf"))
            retain = generating_branch is not None
            if retain and not torch_api.is_grad_enabled():
                raise ValueError("Generating score graph requires an enabled gradient context")
            self._charge(encoded, "density", M, retain)

            def replay():
                prefix = [None] * encoded.domain.n_nodes
                previous = 0
                state = self.h0
                # Zero anchors preserve the empty-graph score's zero derivative.
                anchor = encoded.H.reshape(-1)[:1].sum() * 0 + self.h0[:1].sum() * 0
                scores = [anchor if branch == generating_branch else encoded.H.new_zeros(())
                          for branch in range(M)]
                for node in encoded.domain.order:
                    blocked = encoded.domain.blocked(node, prefix)
                    state = self._transition(state, encoded.H[node], previous, blocked)
                    label = assignment[node]
                    for branch in range(M):
                        if branch == generating_branch:
                            logp = self._head_log_probabilities(branch, state, blocked)
                        else:
                            with torch_api.no_grad():
                                logp = self._head_log_probabilities(branch, state.detach(), blocked)
                        scores[branch] = scores[branch] + logp[label]
                    prefix[node] = label
                    previous = label
                result = torch_api.stack(scores)
                if not torch_api.isfinite(result).all().item():
                    raise ArithmeticError("Feasible complete head density lost numerical finiteness")
                return result

            if retain:
                return replay()
            with torch_api.no_grad():
                return replay()

        def log_density(self, encoded, assignment, *, branch: int,
                        retain_score_gradient: bool):
            """Compatibility scalar query; it evaluates/charges ALL M heads.

            Use full_log_densities/cross_member_log_densities for reuse. Calling
            this scalar method separately M times would repeat the recurrence.
            """
            self._check(encoded, branch)
            if branch is None or type(retain_score_gradient) is not bool:
                raise ValueError("Explicit branch and boolean gradient choice required")
            scores = self.full_log_densities(
                encoded, assignment,
                generating_branch=branch if retain_score_gradient else None,
            )
            return scores[branch]

        def cross_member_log_densities(self, encoded, samples):
            """Full [K,M] matrix with one recurrent replay per supplied y.

            Repeated assignments are not deduplicated across records. Each row
            retains only its generating score graph and reuses all prefix states
            and masks for every nonself head's exact density contribution.
            """
            samples = tuple(samples)
            if not samples:
                raise ValueError("Nonempty externally designed sample batch required")
            rows = []
            for sample in samples:
                if sample.graph_id != encoded.domain.graph_id or sample.order_id != encoded.domain.order_id:
                    raise ValueError("All densities must use the same full graph and recorded order")
                self._check(encoded, sample.generating_branch)
                if sample.generating_branch is None or not encoded.domain.feasible(sample.assignment):
                    raise ValueError("Generating records must be complete feasible schedules")
                rows.append(self.full_log_densities(
                    encoded, sample.assignment, generating_branch=sample.generating_branch,
                ))
            self.work_ledger.record_cross_matrix(len(samples), M)
            return torch_api.stack(rows)

    return SharedTransitionHeadDecoder()
