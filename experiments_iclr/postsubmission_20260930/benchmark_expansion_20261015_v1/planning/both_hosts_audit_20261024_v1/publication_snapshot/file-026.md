# Live metadata collection verification

The additive queue collector completed on both verified hosts at 22:29 UTC,
23 October 2026 (01:29 Moscow, 24 October). Eleven focused CPU metadata tests
also passed on each host. Exact test and collection receipts are retained beside
this note. No training, model forwarding, tensor deserialization or TEST access
was performed by this audit.

| Host | Terminal native RESULT records | I4 results | Observed predictor rows | Rows with bound evidence | Historical failed attempts preserved |
|---|---:|---:|---:|---:|---:|
| Allocation | 10 | 1 | 9 | 9 | 4 |
| Server51 | 33 | 6 | 22 | 19 | 3 |

Native ensemble bodies are not four separate reported ensemble predictors, so
the native-record and predictor-row counts differ. The three unbound server51
rows are retained failure rows. Server51 recovery was still being handled by the
parent agent during this snapshot; these are not its final post-recovery counts.

The allocation collection authenticated five preserved v1 witnesses and two v2
witnesses without rewriting them. The server51 snapshot contained no repaired
composed witnesses yet. All incomplete scientific contrasts remained withheld.

The report indices are 428,655 and 879,199 bytes respectively. Complete reports
remain on their respective servers, under this package's `REPORT-20261023T222906Z`
folder. `COLLECTOR_VERIFIED-allocation.json` and
`COLLECTOR_VERIFIED-server51.json` contain their exact paths and hashes.

This establishes actual queue-result discovery and historical witness
compatibility. It does not establish that every later training/finalization
branch will complete, or that the campaign has enough retained-storage space.
