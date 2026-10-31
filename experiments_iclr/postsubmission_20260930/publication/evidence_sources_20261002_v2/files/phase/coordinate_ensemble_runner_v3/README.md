# Runner v3 — prospective CoauthorCS / Amazon Photo source

Source-only derivative of immutable runner v2. The learning, label separation,
storage gates, optimizer strategies, metrics and profiling remain byte-for-byte
the v2 implementation apart from the module version text and dataset dispatch.
The v2 README documents their interface and measurement limits.

## Dataset mapping

- CLI/protocol dataset keys are exactly `CoauthorCS` and `AmazonPhoto`.
- Acquire uses standard PyG `Coauthor(root=dataset_root, name="CS")` or
  `Amazon(root=dataset_root, name="Photo")`.
- Acquire/qualification provider capture hashes the corresponding actual installed
  class source. Normal fit still reads only the acquired public bundles.
- Coauthor's nested `<dataset_root>/CS/raw` and `CS/processed` paths are discovered
  through the provider properties. Standard PyG `read_npz` processing is retained;
  there is no manual feature reconstruction, URL mutation, or substitute archive.

Root supplies the actual source/version bindings, dimensions, new storage gates
and prospectively frozen protocol. `PROTOCOL_TEMPLATE.json` contains the new
dataset keys but remains incomplete and `frozen: false`. The source manifest must
bind this new runner and all reviewed dependencies, including retained actual
Coauthor/NPZ reader source snapshots. All ordinary v2 invocation arguments apply.

The new stage bridge v2 permits acquire/qualify with this runner only and rejects
fit/test/preflight. Root independently owns any later fit admission, acquisition,
protocol freezing and GPU measurements. Existing Photo qualification stays on
its immutable v2 runner/request/source hashes.

No runner/bridge/model imports, compilation, tests, scripts, downloads, SSH,
dataset acquisition or model execution were performed preparing this packet.
