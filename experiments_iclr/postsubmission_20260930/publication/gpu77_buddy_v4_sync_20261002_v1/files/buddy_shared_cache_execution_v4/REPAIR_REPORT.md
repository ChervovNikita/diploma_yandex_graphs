# BUDDY shared-cache source v4: installed OGB split metadata

Exact installed OGB1.3.6 sources fetched by root without dataset access establish that `PygLinkPropPredDataset.get_edge_split` uses `self.meta_info['split']`, and the class has no `dataset.split` attribute. Collab's pinned master.csv supplies `time`. V3's seven numerical CPU tests and independent source success are preserved; this newly discovered integration defect arose from source inspection before real-data execution, not a failed or completed data/cache run.

## Minimal delta from sealed v3

- `official_split_file` reads `meta_info.get('split')` instead of the absent `dataset.split`. It requires a nonempty canonical single basename, refusing dot/dot-dot, absolute/multiple-component, backslash and NUL values before any file read. Per-file reads and exact loader-source checks remain unchanged. No combined accessor is invoked and no compatibility monkeypatch is needed.
- One stdlib fixture verifies a loader with meta_info/time and no split attribute follows the official train path, then rejects unsafe metadata before Torch import. The original eleven fixtures remain.
- README/report and fresh evidence/manifest/seal record the source correction. All other executable files, CONFIG/pins/vendor/dependency extras and all seven CPU tests/tolerances are byte-identical to v3.

## Evidence and limits

Source evidence: `buddy_installed_ogb_source_preflight_v1/source/linkproppred/dataset_pyg.py` SHA `fa1f74362f671243d62b7c6d560433a87696b7ae8293c7da03233b2effce4e5d`; master.csv SHA `8c317b29892fa41c021446858a7fc6761c30d0f1af651f7942af8d85799a7054`. The separate root source receipt binds these and raw-reader sources. Local static checks and all twelve stdlib checks pass. No Torch/native model/hashing/OGB source, real data/model/checkpoint, GPU, remote action or network was executed/accessed by this repair.

Root must rerun all seven numerical CPU tests with this exact v4 identity, without skips/tolerance changes, and perform its source recheck before complete-data staging/cache work. V1/v2/v3 seals and every manifest entry were reverified unchanged. The new staging wrapper will bind v4 and its own frozen source/qualification receipt; it must stage only public/train/valid archive members and compare raw/processed graph order/features and official weighted train topology before cache hashing. Actual complete-data/resource/family gates remain pending.
