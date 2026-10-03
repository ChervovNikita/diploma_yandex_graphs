# V2 evaluator amendment: independent source review

This bounded amendment review covers only E1's one-line correction, unchanged sources/bindings, v2 custody and preservation of earlier reviews. It does not repeat the implementation audit, execute the evaluator or add fixtures.

**E1 / P2 is resolved in source.** At `evaluate.py:340`, v2 replaces:

```diff
-        graph_schema = descriptor(Path(frozen['study']['path']).parent/dataset/'GRAPH_SCHEMA.json'); original.append(graph_schema)
+        graph_schema = descriptor(verify(frozen['study']).parent/dataset/'GRAPH_SCHEMA.json'); original.append(graph_schema)
```

Graph-schema lookup now uses the verified study path, so an admitted PHASE-relative descriptor is resolved consistently with admission, independent of the launch working directory. The entire v2 evaluator equals v1 after exactly this substitution; the independently generated diff matches `V1_TO_V2.diff` exactly. No remaining source defect was found in this narrowly scoped correction.

`verify_source.py` is byte-identical. All existing required source records, module bindings, scientific/training/calibration authorities, graph authorities, optional native records and both evaluation templates are unchanged. The only added required dependency is the preserved v1 review JSON documenting E1. Thus the correction changes no model, gradient, selector, replay arithmetic, primary contrast, gate, calibration or optional-native isolation logic.

Final v2 manifest SHA256 `462c7bad84921d4fd7e136cc80af57c7854d57e1fc05634c79db4719ca25df41` matches its live seal and static receipt. All v2 payloads, required/optional source records and preserved v1 payloads match their hashes and lengths. V2 Python parses/compiles; the release template remains disabled. V1 retains manifest `1220da6e36b139a5826431d272dbd871e8e06b69428d2c74357c1f5927861a3e` and its matching seal. All six files across the three earlier review directories remain byte-for-byte unchanged.

This establishes source correction and custody only. No numerical runtime, evaluator, fixture, current run outcome, real data/label/tensor, fitted state, server access, training, source/canonical edit, Git operation, new agent or qualification/acceptance decision occurred. The v1 implementation-review limits carry forward. Actual complete40 closure, runtime replay, result correctness and scientific interpretation remain separate root responsibilities.
