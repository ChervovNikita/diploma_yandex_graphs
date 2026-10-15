# Native OGB link split conversion and population completion

The first attempt completed six datasets but found the Collab TEST file absent
from the admitted cache. The next attempt recovered that exact file from the
retained official archive after matching its TRAIN/VALID bytes. It then exposed
a source mismatch: the official split contains NumPy arrays, while preparation
expected torch tensors directly. Both failures and costs are retained.

The native OGB `get_edge_split` uses `replace_numpy_with_torchtensor` on these
files. `native_link.py` adds that exact installed native conversion to the
original link population function and records its source. It performs no dtype
coercion, sorting, negative generation or cache rewrite. The original provider
sources remain unchanged. This driver reuses all six completed provider outputs
and the existing, authenticated original TEST file. It executes only Collab and
the original full contract builder in fresh `allocation_population_v3`.

The original CPU/RAM limits and 1,200-second supervisor bound apply. All
TEST access is population metadata; no predictions, metrics or training.
Normal incidental caches remain permitted. Run the qualified allocation
`.venv/bin/python -B complete.py` after immutable source transfer and route check.
This source is a declared successor; it does not overwrite previous attempts.
