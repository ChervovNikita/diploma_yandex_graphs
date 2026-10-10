# Fixed GCN/GAT companion readout

Source only: root reviews and executes after **both** families close. The original SAGE analyzer remains unchanged. No training, model forward, checkpoint/trace access, owner, qualifier, retry or TEST scoring is added.

```text
analysis.py --gcn-output <GCN-familydir> --gcn-owner-end <GCN-root/OWNER_END.json> --gat-output <GAT-familydir> --gat-owner-end <GAT-root/OWNER_END.json> --report <new-report.json>
```

Both successful direct-wait/child-closure receipts are required first. Each complete 21-group/39-unit report must match its owner hash, exact frozen root CONFIG.json and DECISION.md, and explicit backbone. Configs must differ only by backbone. All 42 selected VALID archives must share exact ordered IDs/labels before scoring. Both outcomes are retained in one new JSON; neither outcome selects a backbone. Existing inputs are preserved; stdout is a bounded gate summary.

The proven member/class, stable float64 NLL, saved float32 decisions, paired three-seed descriptive uncertainty, coverage/loss/aggregation-only identity, repair/harm transitions, complementarity and factorial interaction calculations are retained. Exchange criteria are unchanged.

Added S−ordinary-I4 and S−each-single contrasts complete the prospective baseline comparison. Baseline accuracy transfer requires ≥0.2 mean accuracy points versus each independent bank, all seeds nonnegative and at least two positive, plus positive mean versus both singles. Proper-loss protection is reported separately: NLL deterioration ≤0.02 mean and ≤0.05 each seed versus each independent bank. A third boolean gives their conjunction. No baseline member floor or additional criterion is invented.

The script binds GCN config `aeee4950…`, decision `b6a6bcef…`; GAT config `3a5925a2…`, decision `75ad1b23…`; full hashes are literal constants in the source. These are development transfer studies on the encountered WikiCS graph, with unused confirmation still required. SAGE failure remains recorded.

Source SHA256: `e8885beb9ce424430822cd50820d844aa9684e80830fff74928c632535616d87`. Standard-library AST/text checks only; no numerical import, test, payload read or execution was performed during preparation. `CHANGE.patch` records the exact change from SAGE analyzer `6bcc5708…`.
