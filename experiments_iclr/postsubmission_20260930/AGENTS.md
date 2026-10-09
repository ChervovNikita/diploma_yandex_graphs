# GNNM research execution scope

Use the literal scientific SSH destination
`anogena-2.ai0001053-01174@ssh-sr003-jupyter.ai.cloud.ru`, port `2222`,
with `/Users/alex/.ssh/mlspace__private_key_anogena.txt` and `-tt`.
Never infer a scientific destination from an SSH config alias. In particular,
`anogena` and the login without `-2` reach the restricted allocation.

Before project metadata reads, `cd`, writes or execution, verify hostname
`anogena-2-0` and the sole GPU UUID
`GPU-44039938-fd82-41d2-fefd-de71514e2fac`. A failed route check is not a job
failure. Preserve the attempt as excluded, correct the literal destination,
and never reconnect to the wrong allocation for monitoring or cleanup.
Seven-GPU access is authorized only for MacLink forwarding to the paired Mac.

Scientific deliberate operations stay within this local research workspace,
the allocation repository
`/home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs`,
or the authorized 18.77 repository
`/disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git`.
Use the reviewed MacLink wrapper for 18.77. Keep credentials private.
Ordinary incidental runtime caches are allowed; use normal host execution.
No sudo, filesystem namespaces, mount changes, unrelated-data access,
GENLINK work or PDF compilation.

Keep original paper scores unchanged. Preserve all declared outcomes,
failures, costs and decisions. Read frozen study criteria before interpreting
results. Numerical tolerances, surrogate diversity and setup checks are not
accuracy evidence or a manuscript acceptance verdict. Fresh manuscript
reviewers receive immutable paper/evidence without author history or a
requested verdict through the supplied paper-review guidance.
