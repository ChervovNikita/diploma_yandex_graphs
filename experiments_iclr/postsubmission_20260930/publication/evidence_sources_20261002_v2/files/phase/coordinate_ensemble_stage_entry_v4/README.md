# Stage bridge v4 — timeout capacity only

Derived from independently reviewed stagev3. Request schema changes to_v4; the maximum finite per-operation timeout increases from1200 to1800 seconds so complete2000-update fits plus conservative worst-case checkpoint I/O can fit. No learning, path, source, dataset, mode, cell, GPU, idle-settling, label-isolation, horizon or selection guard changes. Exact complete-operation budgets still must fit the whole supervisor. No new request execution is admitted by this source file. Standard-library AST syntax checked only.
