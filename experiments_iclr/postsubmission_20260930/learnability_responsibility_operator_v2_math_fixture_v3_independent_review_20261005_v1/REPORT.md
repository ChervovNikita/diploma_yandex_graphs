# Independent operator V2 / math fixture V3 review

5 October 2026. **PASS for the bounded scalar-constant repair and interpretation of the saved toy fixture; no concrete blocking issue found.** Source remains disabled. No scientific/native admission follows. This reviewer did not author the repair or fixture and did not import or execute either.

## Repair and derivative ownership

The exact V1→V2 diff contains only `from math import isfinite, log` and replacement of `log_probs.new_tensor(float(M)).log()` with the Python scalar `log(M)` in `_query_objective`. The latter subtracts the same mathematical constant from log-sum-exp member log probabilities, preserving probability-mean CE and its parameter derivatives. Finite-dtype rounding is a qualification matter, not exact bit-identity. V2 removes the constant construction that the saved V2 attempt reports failing with `FuncTorchGradWrapper` under Torch 2.1.2. That failure artifact remains intact.

Independent AST comparison confirms all 12 other operator functions unchanged, including the finite solver, response construction, private partial, `stop_q`, validation and `episode_step`; no function is added or removed. The stored patch equals the exact byte delta. V2 source pin is `fba3ca3d4bb35da0438923941d97bc4833004dd9fd385735c2352f343693b3e3`; manifest is `f68ea05c8f256224144d762f77aa3025a475b259d4f548b9b712e7ae4be1a5f7`. All four packet members authenticate; the hard source guard and false scientific release flags remain.

Q remains a separate input to `torch.func.grad(_main_loss, argnums=1)`, so the private gradient excludes Q-to-private differentiation while its returned value retains Q dependence for the outer theta derivative. `stop_q` detaches only that supplied Q. The source still recomputes the complete response/Q/private step after updating theta, from original persistent private state, then commits detached states.

## What fixture V3 establishes

`check_math.py` is byte-identical to fixture V2 (`394356bae58a3d66e359c49ee7a1fea7f2350703a2571cf0c7c462fb8ecef056`). Its saved result, execution-release source/fixture/manifest hashes and transport stdout reconcile exactly. The result reports Torch `2.1.2+cu118`, CPU float64 and `PASS_CPU_MATH_FIXTURE_ONLY`.

| Check | Actual scope and saved evidence |
|---|---|
| Finite assignment | Four fixed random-cost/path-affinity cases: (items,members)=(1,4),(6,4),(9,4),(6,1). Final positivity, row/column balance and objective no greater than uniform are asserted. Maximum saved marginal residual is 2.22e-16. This does not check every iteration's descent, optimality after eight steps, all graphs/scales or native float32. |
| Private partial ownership | On one deterministic dense tanh/linear-head fixture, independently reconstructed CE-plus-weighted-margin loss uses detached Q and plain autograd. Returned private updates agree exactly (reported maximum error 0). Live and stopped-Q forward private updates also agree by exact tensor equality at that same state. |
| Live outer gradient | One normalized random direction through all shared weight/bias tensors, at one parameter state, is checked against central differences at three step sizes. Saved absolute errors are 1.04e-8, 9.35e-10 and 1.05e-10, below the fixture's 5e-8 gate. This checks one directional derivative, not the full Jacobian/Hessian or every higher-order path. |
| Stopped-Q ownership | The stopped outer gradient equals a genuinely fixed-Q reference function at that state (reported error 0). Live-minus-stopped gradient norm is observed as 1.443e-5. That norm is reported, not asserted positive by the fixture, and is not an efficacy result. |

The checks call internal math helpers, not public `episode_step`. They therefore do **not** numerically qualify the public release gate, post-core recomputation/commit values, state persistence across episodes, M=1 full outer identity, native parameter partition/context, sparse higher-order AD, optimizer/history behavior, acquisition/held-A exclusions, resource fairness, or predictive benefit. Those omissions are scope limits rather than defects in this deliberately bounded fixture. No optional test expansion is proposed here.

Exact whole-bank relabeling as one multibranch graph remains an equality/architecture boundary, not an impossible superiority gate. A same-capacity all-branch learner trained directly on served pool CE or standard own-plus-pool CE remains the useful learning-rule comparator. This fixture evaluates neither scientific learner; nonzero Q-chain credit supplies no ensemble-necessity claim.

## Access and disposition

Static source/JSON/hash/AST inspection and already saved CPU engineering results only. **0 source-module imports/executions, 0 SSH/GPU/server actions, 0 native/data/history/checkpoint/logit/scientific result reads, 0 compilation, 0 source/predecessor/canonical edits, 0 agents.** The saved launcher/transport text was read without invoking it or following paths. Only this fresh review folder was written. Native/scientific qualification, future releases and experiments remain unadmitted.
