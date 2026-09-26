# Fresh primary 432-cell reproduction

`prepare_primary_fresh.py` prepares a **new** study directory from the frozen
public `validation_tuning/` source. It refuses an existing target. It copies
only the five source/protocol files named by the original freeze and that
freeze itself. It downloads the exact public WikiCS, Actor, and filtered
Chameleon bytes to `target/data/`, asks PyG's Planetoid loader to download
Cora into `target/data/cora/`, and verifies every raw-file SHA-256. Finally it
runs the frozen runner's full source/data/tensor `check-freeze` gate. A failed
download leaves an incomplete **new** target for inspection; the script never
overwrites it on retry. Choose another absent target for a new attempt.

The runner's local Python dependency closure is complete: `tuning.py` imports
`models.py`, `verify_tuning.py` imports `tuning.py`, and neither imports other
local modules. The script checks this closure, all five frozen source hashes,
the exact non-Cora download list, and the eight expected Cora raw paths before
writing. Python/NumPy/PyTorch/PyG and their transitive packages must be
installed separately; the source's original dependency versions should be
used when available. The full `check-freeze` gate rejects changed raw bytes
or graph tensors under a different library environment. This preparation
script has not trained any model or verified a fresh model outcome.

Run these commands from the extracted anonymous archive root. In the public
repository, first run `cd experiments_iclr/iclr2027_compact`, then use the
same commands. The target `fresh_primary_432` must be absent and must be in
a writable copy of the archive. No administrator privileges are needed.

```sh
python experiments_iclr/fresh_primary/prepare_primary_fresh.py \
  --source experiments_iclr/validation_tuning --target fresh_primary_432 --check-only
python experiments_iclr/fresh_primary/prepare_primary_fresh.py \
  --source experiments_iclr/validation_tuning --target fresh_primary_432
cd fresh_primary_432
python tuning.py check-freeze
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python tuning.py preflight --device cpu
```

The above preparation and CPU preflight were actually verified with Python
3.11, PyTorch 2.1.2+cu118, and PyG 2.7.0. To run the full benchmark on an
available GPU, continue in the new target only after the input gate passes:

```sh
python tuning.py preflight --device cuda
python tuning.py run --dataset cora --device cuda
python tuning.py run --dataset wikics --device cuda
python tuning.py run --dataset actor --device cuda
python tuning.py run --dataset chameleon_filtered --device cuda
python verify_tuning.py audit-and-lock
python tuning.py score --device cuda
python verify_tuning.py audit-scores --device cuda
```

`run` reads no test indices or labels in its returned training bundle. The
fresh `audit-and-lock` command requires all 432 cells and refuses an existing
`VALIDATION_SELECTION_LOCK.json`. **Do not run `score` before it passes.**
`score` checks the complete 432-cell lock, and scores only each graph/arm's
validation-selected candidate and the prespecified `(0.001, 0)` default.
`audit-scores` independently replays the allowed original checkpoints and
requires the exact score set. Keep this new target separate from the supplied
compact records and any original author checkpoints; a later rerun cannot be
presented as a prospective original result. The original protocol's
2026-09-26 audit cutoff is historical metadata, not a fresh deadline.

For an inspection without downloads or writes, `--check-only` passed on the
supplied compact source on 2026-09-26. Passing it an existing target failed
with `FileExistsError` as intended. The exact source and public datasets were
then prepared in a new, isolated study directory in the original Python
environment (PyTorch 2.1.2 and PyG 2.7.0). The original `check-freeze` and
`preflight --device cpu` both passed with CUDA hidden and one CPU thread.
No 432-cell training, validation lock, test score, or independent fresh score
audit has been run. This preparation test establishes the input/dependency
gate, not reproduction of the reported experimental results.
