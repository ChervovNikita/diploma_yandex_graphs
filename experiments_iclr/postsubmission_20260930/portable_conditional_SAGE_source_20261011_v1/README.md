# Conditional SAGE: local Physics reproduction source

Four implemented commands reproduce the fixed Physics workflow from the pinned public archive. The engine preserves the E/V5 numerical functions and uses local receipts and permission, with no cluster credentials, fixed hostname/UUID or inherited PASS. This package is **source-only and has not been installed, imported with numerical packages, qualified or executed here**. A clean external installation/update/restoration run is still required before calling portability verified.

## Environment

Use Linux, Python 3.11 and a CUDA GPU exposed as cuda:0. Recorded core runtime is Torch 2.1.2+cu118, NumPy 1.26.4 and PyG 2.7.0. The native PyG source hashes are checked; alternate builds fail visibly. SciPy 1.11.4 is this installation recipe's pin; historical SciPy binary equivalence is not established.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install -r /ABS/PACKAGE/requirements.txt
```

All acquisition commands retain a 48 GiB CUDA/64 GiB RSS cap. Supply a finite external process watchdog, including pending native calls; 48 hours is a cap, not an ETA. The full graph has 34493 nodes, 8415 raw features and 530417 model-support entries, so this is not a laptop/toy example.

## Commands

Run each once, in order, with a fresh WORK directory outside the source bundle:

```bash
python /ABS/PACKAGE/cli.py prepare --work /ABS/WORK --download
# Or: prepare --work /ABS/WORK --raw /ABS/ms_academic_phy.npz
timeout --signal=TERM 48h python /ABS/PACKAGE/cli.py acquire --work /ABS/WORK
timeout --signal=TERM 48h python /ABS/PACKAGE/cli.py export --work /ABS/WORK
timeout --signal=TERM 2h python /ABS/PACKAGE/cli.py score --work /ABS/WORK
```

Preparation verifies the 16,374,005-byte archive, SHA256 e4d68468eba5fb8f6b280ffd406bd53e6e4295ba1dd2967d6fe99b978e4b3375, author commit 1e72912a0810cdf27ae54fd589a3b43358a2b161. It runs the preserved PyG CSR/dense-positive-binarization and loop removal/undirected coalescing, then inserts exactly one self edge per node for all models. It does not normalize features or add loops inside consumers.

The custodian necessarily reads all labels to make ten different custom stratified masks. Seeds are 8101,8219,8363,8501,8663,8819,9001,9199,9403,9623. For each seed use sequential per-class MT19937 RandomState(seed+5000011), sorted class IDs, 20 TRAIN/30 SELECT/30 CAL per class and restTEST; sort each union. Per block counts are 100/150/150/34093. These are not official masks or independent graphs. Model files expose only TRAIN and SELECT labels; CAL/TEST targets are separate.

## Fixed acquisition and closure

Core arms BASE/E4/N4/equipped_M1/genuine equipped_I4 use the original SAGE h128/depth2/LayerNorm/dropout.2, AdamW.001/zero decay, max1000/stale300, earliest strict SELECT accuracy. Shared routes make four CE/4 backwards then one AdamW step; singles/I4 use unscaled own CE and own selection. All original root/self/mean/FFN behavior remains. Core closes 50 banks/80 states.

The sixth arm is the published-author meanSAGE source-derived Torch port: h32/depth2/dropout.4; raw sparse input, stored-value dropout, shared dropped input for aggregate/skip, z/max(rowL2norm,1), ReLU only after layer1; first-layer .05sumW²; explicit TF1 Adam convention; latest tied minimum SELECT totalCE+L2, patience50/max100000. It adds 10 banks/10 states. This is not bitwise TensorFlow or reported-score replication.

Acquire writes exact local configs, runtime/package/file/role records, complete traces and selected model/optimizer/RNG states. Both full closures and all 90 state hashes are required before export. Export restores core states with no update and joins all 60 label-free CAL/TEST banks. Archives retain native float32 member/pool integer decisions plus member log probabilities. See INTERFACE.json.

Score checks all 60 banks/90 selected files before any CAL target is loaded, attempts all 60 fixed CPU float64 Adam temperatures (500 updates, LR.01, betas.9/.999, epsilon1e-8, zero decay, initial logT0, final endpoint), then requires all 60 finite endpoints before one complete TEST release. It reuses the original full raw/calibrated metrics, repair/coverage diagnostics and fixed two-primary-contrast inference. No survivor subset, retry, grid or extra GNN forward. Existing outputs/TEST marker prevent another attempt; failures are retained.

## Scope and remaining gaps

MANIFEST.json pins package bytes; ENGINE_NUMERICAL_AST_RECEIPT.json and CLI_SOURCE_CHECKS.json record source-only derivations. Donor E/port files remain as provenance carriers; their historical admission APIs are not invoked. Public local status describes actual permission/file checks, never past cluster verification. Run results and numerical equivalence have not been produced here.

The package currently exposes Physics only; historical WikiCS score replay is not added. Complete fresh execution, dependency installation, pending-call watchdog behavior and external runtime equivalence remain unverified. Upstream source/license provenance is listed in DEPENDENCY_BINDINGS.json and NOTICE.md; no new license grant or historical-outcome certificate is invented. Physics/Penn studies and original paper results remain untouched.
