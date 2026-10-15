# Portable metadata preparer v3

This is a narrow successor to published `portable_dispatcher_v2/prepare_projection.py`.
The first guarded allocation attempt is retained in `PORTABLE_PROJECTION_RUN_11.json`:
it failed after0.45 seconds before output generation or native work because the
metadata loop inherited the allocation configuration's environment hash while
reading the disabled77 configuration files.

The successor supplies each configuration's existing published manifest hash
explicitly. It does not change the allocation environment, any execution guard,
the host configurations, ownership logic, runtime or published v2 package.
The four path/projection/ownership functions are AST-identical. The actual
metadata-loop regression passes without hostname, GPU or remote access.

The allocation-only bootstrap retains the published allocation configuration
pin and ordinary guard. Only after that guard does the preparer read the
published manifest and all three configuration metadata files. Generated
source closure includes both the complete published v2 package and this
successor's files and manifest. Output is a new immutable
`portable_dispatcher_v2/generated_v3_config_pin` directory; all releases remain
unadmitted,77 remains disabled, and no GO is manufactured.

Root's single guarded metadata command uses the qualified allocation interpreter:

```sh
.venv/bin/python -B experiments_iclr/postsubmission_20260930/benchmark_expansion_20261015_v1/portable_projection_preparer_v3/prepare_projection.py
```

This command was not executed locally. The original30 source/metadata checks
and runtime comparisons are reused; only the actual metadata-loop regression
and unchanged-function comparison were run. No native jobs, tensor reads,
hardware probes, remote access or Git operations were performed.
