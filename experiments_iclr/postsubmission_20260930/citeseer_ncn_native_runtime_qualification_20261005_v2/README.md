# Native NCN runtime correction

The first zero-update probe failed on a missing torch_sparse import. The existing training environment is untouched. A private in-repository venv installs official PyG pt21/cu118 torch-sparse0.6.18 and torch-scatter2.1.2 wheels into an isolated dependency overlay. The successor uses that private interpreter with the overlay and existing pinned numerical libraries in PYTHONPATH. Probe scope is unchanged: one masked native TRAIN batch forward/backward with zero updates, complete VALID forward shapes with no ranking metric, and no TEST access.
