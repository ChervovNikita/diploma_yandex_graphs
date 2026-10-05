# Same-state local derivative diagnosis

The original full-native synthetic qualification remains a failed engineering run. This separately frozen diagnostic retains its exact native architecture, model state, artificial graph, inputs, support/query targets and parameter direction. It performs no model fit or dataset scoring.

The base objective and autograd direction reproduce the preceding independent diagnosis exactly. At step 1e-5, the central finite difference still differs from the analytic direction by 0.089964. At steps 1e-6 and 1e-7, the differences are 1.004e-10 and 4.763e-9; both one-sided estimates also agree locally. The source did not introduce a qualification tolerance or declare a pass.

There are no captured ordinary activation sign transitions at either 1e-7 point. Other points do have transitions; the positive 1e-6 point has one transition despite its accurate numerical derivative. Sign changes alone therefore do not establish the cause of the larger-step discrepancy. All 248 transformed activation sites per point are explicitly skipped. The diagnostic establishes no global smoothness or full inner-gradient branch coverage.

The next minimal engineering check retains the original failure, fixes the local two-sided derivative scale prospectively, and completes the omitted stopped-responsibility/unused-coordinate, public episode, original-private-state recompute and native restoration checks. Actual full-graph memory/time qualification with the reviewed graph accessor remains separate. No predictive fit is released by this result.

The run took 16.897 worker seconds and peaked at 4,935,950,336 bytes RSS. Native state, inputs, RNG and temporary functional hooks were restored. All seven point records remain in RESULT.json. The complete 14,504,734-byte mask file stays in the authorized server repository, bound by SHA256 81212a4beccb32332c9361a72cca3cfa407b0058b8e6262ac2d558d00f4a8258.
