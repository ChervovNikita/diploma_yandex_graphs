# Narrow v2 diagnostic source diff review — BLOCKED

Candidate manifest: `4766183c02176d6e0a68e83ba64415ec5fc7ab482cbe48a729026d941ed57c8e`. Static source review only; execution remains unauthorized.

**F01 is closed.** The parent observes elapsed/output after collection and after initial terminal/inventory/custody publication. Late exceedance or a second-check exception revokes success and returns failure. Last-check telemetry, unsampled post-child parent RSS and finite exceptional/status/fsync/stdio tails are explicit; no instantaneous OS or infinite publication guarantee is required.

**New blocking finding F02:** `supervise.py:339–342` inventories the initially written terminal. `366–373` then unconditionally rewrites that terminal with final-check telemetry and updates the top-level custody terminal hash, but leaves the terminal row inside `supervisor_output_files` unchanged. Even a normal PASS changes the checkpoint name, so final custody contains contradictory hashes for the same terminal path. Refresh that one row after the final terminal write, or explicitly define a historical snapshot and current custody separately. This needs no extra cap-check loop.

Nine bound files are byte-identical to v1. Twelve supervisor top-level functions and the whole module outside main are AST-identical; the retained diff is exact. The completed v1 review is reused for unchanged data, gradients, reductions, runtime and owned-supervisor scope. Hash/identity checks passed. No source modification, import/execution, numeric/state/score read, server access, staging, launch or experiment occurred.
