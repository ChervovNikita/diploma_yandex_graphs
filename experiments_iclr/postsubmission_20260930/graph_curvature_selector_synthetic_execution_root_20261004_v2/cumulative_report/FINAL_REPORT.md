# Synthetic engineering qualification v2

All twelve exact corrected-harness cases passed on the authorized anogena-2 route. Supervisor PID 400448, suite PID 400450. PyTorch 2.1.2+cu118 / NumPy 1.26.4, CPU tensors, CUDA unavailable, OMP/MKL threads 1 and project-local TMPDIR. Verified sole GPU UUID: GPU-44039938-fd82-41d2-fefd-de71514e2fac.

Source v3 manifest: `d53d6963b3c66cb59061c4f5be6a81108adf8e9792a3f3cf93daea56fe35ec70`. V2 harness manifest: `d7d22ee87f4fb28ae925c3dc5c003575da57351e030b8d58340846b45d8642b2`. Constants SHA256: `633ea814bfc82c094ecb2d98699e77258f0f644d13cf17fa7f8d65691cf637dc`. Exact source, protocol, constants and staged bytes remained unchanged.

FP32 and FP64 full-control cases reached and completed all five returned-arm checks: rank3, nine candidate slots, seven trial maps. The fixed first graph pair returned a non-common pre-trial candidate, with complete model/optimizer/RNG custody checked. Selected graph/permuted arms abstained and the random arm was ineligible. This is no improvement claim. Both preserved independent-TRAIN counterexamples passed rank2, nine null candidate slots, one common trial and all five common-state returns.

The independent coupled Adam/AMSGrad equation oracle, private-gradient 1/4 normalization, live-factor gradients, duplicate trial independence, partial-trial failure discard, exact RNG, named alias/moment/step/bias transport, serialized state restoration, rank/D/cap/finite guards and Photo 512-coordinate head closure passed within the declared synthetic scope.

V1 remains preserved with seven passes and three harness oracle failures; its end-to-end five-arm assertions were not reached. Its FP64 restore factory had constructed FP32, and its independent TRAIN topology correctly produced rank at most two. V2 repaired only those fixture/oracle assumptions and retained the original topology as a regression. No constants, source selector/trials, candidate enumeration or predictive rules changed.

Costs retained: v1 supervisor wall 3.440172s, v1 failed-case wall sum 0.418682s; v2 supervisor wall 4.876723s. Both authorized attempts together: 8.316896s supervisor wall and 1.701796s summed case wall. Process CPU time, energy and GPU cost were not measured. One execution per version; no uncertain-outcome retry occurred.

This is synthetic engineering qualification. Actual native bodies, fresh warm-state/Adam/RNG custody and continuation correspondence, native trial memory/cost and predictive accuracy remain untested. No real data, warm states, scientific training, GPU work, other-route access or other-job operations occurred.
