One prospective portable Molhiv family: 8 frozen arms x 3 development seeds
=======================================================================

queue.py serially invokes unchanged public-V2 train.py for the complete family,
seed outer/arm inner order, 100 epochs and 25800 updates per cell. It does not
retry, resume, compare outcomes, score TEST, change recipes or automatically run
another task. This is source preparation; no family was launched by preparing it.
Future 18.77 execution remains pending actual live connection/runtime/device/data
verification. No contact with that machine occurred in this work.

Use a compatible selected CUDA runtime and explicit absolute paths, all inside
the chosen project root. No private hostname, predeclared GPU UUID, provider path,
author approval receipt or 24-cell resource-approval collection is required.
The selected physical GPU is observed and recorded at execution time; CUDA
visibility maps the logical chosen index to its physical nvidia-smi device.

  python queue.py --project-root /your/project --source /your/project/public-v2 \
    --train /your/project/roles/molhiv/train.npz --valid /your/project/roles/molhiv/valid.npz \
    --work-check /your/project/check/WORK_RECEIPT.json --device cuda:0 \
    --output /your/project/runs/molhiv_family

The standard representative work receipt has schema portable-representative-work-v1:
complete, exit_code0, reaped; source manifest, NPZ hashes, observed runtime and
physical GPU; actual single/be_init_contrastive/independent4 cases with complete
two-view TRAIN update, full VALID, snapshot/reload/serving; measured positive
peak_GPU_bytes. It records actual tests, not an approval. The queue uses its peak
plus fixed2GiB headroom and samples physical free/total memory before every fit.
Temporary memory shortage waits at most6h, then preserves an admission failure;
it does not stop another process or substitute a smaller population/batch.
Do not reuse a representative receipt across a changed source/data/runtime/GPU.
The initial representative scope uses the maximum-node batch from all frozen
seed/epoch orders; the separate maximum-edge batch is26 edges larger and remains
an explicit unmeasured scope. No full100-epoch resource/quality forecast is claimed.

Default absolute fit cap9h includes10s cleanup (active hard-10). Timeout stops only
the exact owned process group, with actual PID/start ticks, TERM/KILL/reap inside
the prospective bound. The cap is a ceiling, not a forecast. No shortened fit is
complete. Files record roster, owner/source/data/runtime/device, per-cell memory,
PID/start ticks, complete/caught failure/timeout, CPU/RSS/wall costs and closure.
RUSAGE_CHILDREN.ru_maxrss is recorded explicitly as a cumulative process maximum,
not an isolated per-cell peak. An unknown reap after owned cleanup aborts the
family immediately, retains cap/reap uncertainty and marks remaining cells
unlaunched; another fit is never admitted while that child remains unreaped.
No VALID quality value or checkpoint payload is opened by the queue. Binary
artifacts stay in each full train.py output. CLOSURE accounts all24 cells,
including unlaunched cells after a fatal queue interruption. Exit1 indicates
queue infrastructure failure; inspect closure for every missing/failed fit even
if the process otherwise exits normally. External SIGKILL may prevent closure;
missing CLOSURE/COMPLETE files mean incomplete, never success.

The wrapper does not add numerical-equivalence, novelty or scientific readiness
claims. A matched candidate-versus-control contrast measures the combined loss
package; internal-versus-boundary placement requires a separately matched GNNM
port. Three seeds are development evidence. Strong native/untied controls, later
8-view single control, unused confirmation, preserved failure accounting and
mechanism attribution remain necessary. BatchEnsemble/TabM/DICE/CDLG and public
backbones retain their prior credits in the frozen protocol. No author or active
Wiki24 files, jobs, scientific results or central ledger are modified.
