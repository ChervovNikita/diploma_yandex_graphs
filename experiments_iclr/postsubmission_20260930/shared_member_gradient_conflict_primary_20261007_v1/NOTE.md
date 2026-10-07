# Shared-member optimization: saved primary-source conclusions

## Scope

This record covers CAGrad, arXiv 2110.14048v1, methods sections 3.1–3.3, and FAMO, arXiv 2306.03792v1, section 3. These are bounded method reads, not two full-paper reads. The retained HTML and extracted blocks are source evidence. No model or author code was executed.

The CAGrad author's README at commit `dc3d48152b6196945cfd56144879b9d42353b095` identifies the paper as NeurIPS 2021. The retrieved FAMO method provides a usable optimization prior. Its publication venue is not certified by this record.

## What can be reused

CAGrad calculates the gradient of each task loss. It chooses a direction that improves the worst first-order task change while staying in a ball around the average gradient. A small dual problem has one weight per task. Four GNNM members could be treated as four losses, with the shared parameters playing the role of the common task representation.

The theorem bounds the sum of squared average-loss gradient norms under smoothness, a suitable fixed gradient-descent step and a conflict radius below one. This is a stationarity result. It gives no global optimum, heldout-accuracy or Adam guarantee. Pairwise negative gradient cosines alone do not show that the actual optimizer harms a member. The relevant first-order quantity is that member's gradient dotted with the actual proposed displacement.

FAMO uses changes in task losses to adjust weights without computing every task gradient separately. The update uses weights proportional to `softmax(w)/(loss + epsilon)`, normalized to sum to one. The weight update uses changes in log losses. The underlying optimization relates to MGDA on log risks. It should not be described as preserving stationarity of the arithmetic mean loss. A second loss measurement also has a cost and requires a clearly defined stochastic-view policy.

Using either optimizer on ensemble-member losses is an adaptation of prior work. It is not sufficient evidence of a new method. A search also returned the unresolved lead *An Ensemble Strategy with Gradient Conflict for Multi-Domain Neural Machine Translation*, DOI 10.1145/3638248. Its method has not been read here, so ensemble-specific absence claims remain unsupported.

## A proposal requiring critique

One possible diagnostic separates shared and private parameter displacements. For member `m`, let `g_shared_m` and `g_private_m` be its supervised gradients. Let `delta_private_m` be its native private Adam displacement and `u0` the native shared Adam displacement. Define `b_m = -dot(g_private_m, delta_private_m)`.

A proposed shared displacement minimizes its distance from `u0` subject to `dot(g_shared_m, u) <= b_m` for every member. This accounts for the private part of the same first-order loss change. If every `b_m` is nonnegative, `u = 0` is feasible. Negative private progress can make the constraints infeasible. The proposal is unimplemented and its distinction from gradient projection and shared/private multi-task optimization is unverified.

Even a feasible exact projection protects only the linearized TRAIN losses. Finite steps, stochastic views, nonlinear graph operations and heldout quality require separate evidence. Native optimizer moments and the applied projected displacement must be recorded as different objects. The dedicated literature agent is checking the closest priors before this direction receives compute.

## Scientific decision

Continue the already frozen internal BatchEnsemble initialization/contrastive comparison. Do not add a known gradient optimizer to that running family. First use completed member competence and common-error analyses to determine whether shared-update interference is a plausible failure mechanism. A later comparison would need native Adam, a faithful prior optimizer and any proposed variant at the same data, horizon and serving rule, with measured gradient/update costs.
