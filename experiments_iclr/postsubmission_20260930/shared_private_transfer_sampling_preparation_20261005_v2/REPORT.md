# Full-horizon TRAIN sampling transcript preparation v2

Root explicitly released one CPU-only transcript per host for seeds 20261005, 0, 1, 2, cycles 0 through 59. Soft bound 1800 seconds; owned-child external hard bound 2100 seconds. No models, optimizers, fits, VALID or TEST values. Each host uses its admitted original scientific source and exact installed core providers. The singleton retains only its two declared qualified overlays; peptide declares an empty PYTHONPATH. Host receipt/provider bytes may differ. Every component hash must match exactly.

The v1 harness is retained. The v2 diff changes only the PYTHONPATH assertion to a release-bound exact value and adds declared PYTHONPATH/sys.path to the result metadata. `native_episode_cycle.py`, `episode_geometry.py`, and the scientific `make_pair` and TRAIN loader bodies are unchanged.

CLI: `sampling_supervisor.py --release ROOT_RELEASE.json --release-sha256 SHA256`. Supervisor starts exactly one child with the release-bound interpreter, source and sampler; records PID/start ticks and authoritative Popen wait exit; hard-bound signals target its owned process group only. Root releases and launch receipts live in fresh host-specific execution folders. Scientific qualification requires complete transcript equality and a separate exact root release.
