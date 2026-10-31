"""Unexecuted evaluation-only vectorization of the pinned scalar GRU laws.

Copied equations/masks are disclosed and require exhaustive tiny-state parity.
No sampling, score estimator, gradient or optimizer operation is replaced.
"""


def all_member_log_densities(*, torch_api, decoder, encoded, assignments,
                            family, counts):
    """Full [A,M] complete densities; one row per supplied feasible assignment.

    Private transition: A*M recurrent trajectories. Shared transition: A
    recurrent trajectories and A*M head contributions. All operations no-grad.
    Every row is retained; batching never changes the support or architecture.
    """
    assignments = tuple(encoded.domain.assignment(row) for row in assignments)
    if not assignments or any(not encoded.domain.feasible(row) for row in assignments):
        raise ValueError("Nonempty complete feasible assignment batch required")
    A, M, N = len(assignments), decoder.n_branches, encoded.domain.n_nodes
    h, d, p = decoder.hidden_dimension, decoder.embedding_dimension, decoder.embedding_dimension + 2
    if family not in ("private", "shared") or tuple(encoded.H.shape) != (N, d):
        raise ValueError("Pinned decoder family and complete static H required")
    def add(key, value):
        counts[key] = counts.get(key, 0) + value
    shared = family == "shared"
    factored = decoder.head_mode == "factored_heads" if shared else decoder.mode == "factored"
    recurrences = A if shared else A * M
    add("evaluation_chunks", 1)
    add("evaluation_complete_assignments", A)
    add("evaluation_complete_member_densities", A * M)
    add("evaluation_logical_recurrent_trajectories", recurrences)
    add("evaluation_logical_recurrent_steps", N * recurrences)
    add("evaluation_logical_head_steps", N * A * M)
    add("evaluation_recurrent_dense_multiply_terms", N * recurrences * (3 * h * p + 3 * h * h))
    add("evaluation_head_dense_multiply_terms", N * A * M * 2 * h)
    # Supplied teacher-forced y makes the mask common to all members even
    # when their recurrent states differ. Compute it once per assignment.
    add("evaluation_mask_neighbor_inspection_bound", A * 2 * len(encoded.domain.edges))
    add("evaluation_mask_setup_neighbor_visits", 2 * len(encoded.domain.edges))
    if factored:
        diagonal = h + 2 if shared else p + 8 * h + 2
        add("evaluation_diagonal_multiply_elements", N * A * M * diagonal)
    with torch_api.no_grad():
        labels = encoded.H.new_tensor(assignments).reshape(A, N)
        position = {node: index for index, node in enumerate(encoded.domain.order)}
        masks = []
        for index, node in enumerate(encoded.domain.order):
            earlier = [neighbor for neighbor in encoded.domain.neighbors[node] if position[neighbor] < index]
            mask = (labels[:, earlier] == 1).any(dim=-1) if earlier else torch_api.zeros(A, dtype=torch_api.bool, device="cpu")
            masks.append(mask)

        def transition(state, node, previous, blocked, branch=None):
            x = torch_api.cat((encoded.H[node].expand(A, d), previous[:, None], blocked.to(labels.dtype)[:, None]), dim=-1)
            if shared:
                gx = torch_api.nn.functional.linear(x, decoder.W_x, decoder.b_x)
                gh = torch_api.nn.functional.linear(state, decoder.W_h, decoder.b_h)
            elif decoder.mode == "factored":
                gx = torch_api.nn.functional.linear(x * decoder.r_x[branch], decoder.W_x) * decoder.s_x[branch] + decoder.b_x[branch]
                gh = torch_api.nn.functional.linear(state * decoder.r_h[branch], decoder.W_h) * decoder.s_h[branch] + decoder.b_h[branch]
            else:
                wx = decoder.W_x[branch] if decoder.mode == "untied" else decoder.W_x
                wh = decoder.W_h[branch] if decoder.mode == "untied" else decoder.W_h
                gx = torch_api.nn.functional.linear(x, wx, decoder.b_x[branch])
                gh = torch_api.nn.functional.linear(state, wh, decoder.b_h[branch])
            xr, xz, xn = gx.chunk(3, dim=-1)
            hr, hz, hn = gh.chunk(3, dim=-1)
            reset, update = torch_api.sigmoid(xr + hr), torch_api.sigmoid(xz + hz)
            candidate = torch_api.tanh(xn + reset * hn)
            state = (1 - update) * candidate + update * state
            if not torch_api.isfinite(state).all().item():
                raise ArithmeticError("Batched recurrent state lost numerical finiteness")
            return state

        def head(state, branch, blocked):
            if factored:
                logits = torch_api.nn.functional.linear(state * decoder.r_o[branch], decoder.W_o) * decoder.s_o[branch] + decoder.b_o[branch]
            else:
                wo = decoder.W_o[branch] if shared or decoder.mode == "untied" else decoder.W_o
                logits = torch_api.nn.functional.linear(state, wo, decoder.b_o[branch])
            if not torch_api.isfinite(logits).all().item():
                raise ArithmeticError("Batched head logits lost numerical finiteness")
            # Scalar sources skip softmax for forced-zero decisions. Apply
            # the same structural law and check only free-step log values.
            logp = encoded.H.new_zeros((A, 2))
            logp[:, 1] = float("-inf")
            free = ~blocked
            if free.any().item():
                free_logp = torch_api.nn.functional.log_softmax(logits[free], dim=-1)
                if not torch_api.isfinite(free_logp).all().item():
                    raise ArithmeticError("Batched free step lost positive finite log probability")
                logp[free] = free_logp
            return logp

        scores = encoded.H.new_zeros((A, M))
        states = [decoder.h0.expand(A, h)] if shared else [decoder.h0[branch].expand(A, h) for branch in range(M)]
        previous = encoded.H.new_zeros(A)
        for index, node in enumerate(encoded.domain.order):
            blocked = masks[index]
            if shared:
                states[0] = transition(states[0], node, previous, blocked)
            for branch in range(M):
                if not shared:
                    states[branch] = transition(states[branch], node, previous, blocked, branch)
                state = states[0] if shared else states[branch]
                logp = head(state, branch, blocked)
                selected = logp.gather(1, labels[:, node].to(torch_api.long)[:, None]).squeeze(-1)
                scores[:, branch] += selected
            previous = labels[:, node]
        if not torch_api.isfinite(scores).all().item():
            raise ArithmeticError("Batched complete density lost numerical finiteness")
        return scores
