# Native resource qualification

12 October 2026. `profile_native_resources.py` prepares allocation resource evidence for the native packet. The root reviews its source and executes it. This child has performed source and pure accounting checks only; **there are no GPU measurements yet**.

## No-update boundary

The profiler verifies hostname `anogena-2-0` and the sole UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac` before project metadata reads or writes. It uses normal processes and the launcher's two Torch/OMP threads. It makes no remote connection, stops no other GPU process, and changes no host configuration. The existing authorized Physics training continues. Concurrent CUDA PIDs, process names, UUIDs and memory usages are recorded at startup and throughout cases; their presence does not reject memory qualification.

The latest root instruction is **graph-only qualification with zero optimizer updates and zero SELECT/CAL/test target access**. The driver opens `CUSTODY.json` metadata and verifies only the declared graph fingerprint. Workers open only that graph array and frozen source. They do not open or hash any target-bearing TRAIN/SELECT/CAL/evaluate file or endpoint ID file. Synthetic row IDs use the largest declared TRAIN and SELECT counts across the prepared roster; synthetic class targets are row index modulo class count. These are shape proxies, not prepared role membership or predictive evidence. Complete-TRAIN SupCon keeps the full declared TRAIN row count. Paired auxiliaries keep the first 512 synthetic rows and two complete synthetic supervised views.

Actual native model parameters receive gradients. Every worker replaces `torch.optim.Optimizer.__init__` with a rejecting guard. It never invokes the node adapter factory, creates an optimizer, calls an optimizer step, fits a model, computes accuracy or any target-based SELECT loss, selects a checkpoint, saves logits, calibrates, evaluates or creates a GO receipt. Parameter hashes and tensor version counters must remain identical before and after the probes. Objective values and gradients are not exported as research outcomes.

The frozen node bridge's `NativeBank` class is extracted from its AST and compiled alone, with the same frozen native `Model`/`TABMModel` and source loss functions. The surrounding factory, including AdamW construction, is excluded. A saved AST hash, source fingerprints and worker requests make this construction reviewable. No hand-maintained alternative bank is substituted.

## Required cases

Every case is a fresh CUDA process at **width 512, depth 5, FFN multiplier 1, heads 8, LayerNorm, dropout .2, four routes**. This is the widest/deepest frozen 100-point grid member. Dropout .4 does not change mask/tensor dimensions. Width/depth proxies bound the source tensor shapes, but a CUDA workspace algorithm can vary with shape; this is measured admission with explicit headroom, not proof that every kernel/workspace at all 100 points has been measured.

| Case | Covered native arms | Reason for coverage |
| --- | --- | --- |
| Original boundary4 own | Original GNNM | Original input `X*R` and output BE blocks have separate storage. |
| All-map shared4 own | All-map shared4, all-map single, native single, their separate I4 member jobs | The same body computations plus effective-weight autograd and three extra route graphs cover one-body shape/storage; four independently trained I4 members remain separate one-body jobs. |
| Coupled untied4 own | Coupled untied4 | Four complete parameter bodies are measured separately. Untied4 is not asserted to dominate all-map or original storage. |
| SAGE all-map4 GNCL + within-route SupCon | GNCL, within-route SupCon, combined GNCL+SupCon, ordinary/factorized SupCon singles and their separate I4 member jobs | Both nonzero loss graphs coexist, using the complete maximum TRAIN shape. Removing a loss or routes removes their extra tensor work. |
| SAGE all-map4 paired alignment + residual | Paired class-full alignment, paired alignment+residual | Both loss graphs coexist with two supervised forward banks. An additional four whole-panel hidden tensor shapes reserve for class-group gathering differences from synthetic labels. |
| SAGE all-map4 paired Eq2 | Paired SupCon Eq2 | Its doubled-row score matrix is measured separately; it is not replaced by the alignment case. |

Thus eight datasets × eight operators require **216 fresh case processes**: 168 cases for seven operators, plus 48 SAGE cases. Each defaults to one warmup and three measured forward/backward probes plus a label-free full-graph serving forward per iteration. `--backbone` and `--expected-datasets` allow a bounded subset; its manifest explicitly lists only that subset, and it cannot supply missing full-packet profiles. A fresh output directory is required for every attempt. The default case timeout is **300 seconds** for the root's initial bounded qualification.

## Measurements and admission

Receipts contain setup/whole-process wall time, per-iteration forward/objective/backward wall time, label-free serving-forward wall time, and CUDA event elapsed time for each. Timings with other CUDA PIDs are explicitly labeled **measured under concurrent CUDA workload**, not isolated GPU throughput. CUDA events bracket real CUDA work on the current stream and synchronize at the end. Their elapsed time can include host gaps caused by source `.item()` checks and GPU scheduling interference; it is not a kernel-busy-time sum. Warmups are excluded from the rate estimate and included in memory peaks and consumed resource cost.

Torch's peak allocated/reserved bytes are exact for its allocator in the fresh process. The parent samples that PID's NVIDIA process VRAM every .25 seconds; workers also sample at context, residency, backward and serving boundaries. Receipts preserve these samples and their resolution, including other CUDA processes' memory. The admission envelope takes the maximum sampled process usage and peak Torch reservation plus the largest observed process-minus-reserved overhead. Missing per-process memory evidence makes a case unqualified; concurrent workload alone does not.

Each worker sets only its own Torch allocator limit to the smaller of the driver's current free-memory budget and the worker's startup free memory, less **512 MiB context/nonallocator headroom**. The budget is rechecked before each case. This protects existing work without stopping or changing it. The measured peaks remain valid under this cap. A cap OOM is explicitly labeled infeasibility within the available concurrent headroom; it does not prove that the configuration would fail on an idle physical GPU. Sudden growth by another process still requires the root's normal resource supervision.

Two parameter-sized moment pools, two parameter-sized temporary pools and scalar step storage remain live throughout the measured probes: **4P + 8 bytes per parameter tensor**, in addition to the actual parameters, gradients, graph and activations. They are ordinary tensors; no optimizer is constructed or measured. Holding temporary-sized storage during the live backward graph is conservative for the declared AdamW allowance. It does not certify arbitrary optimizer implementations, precision changes or source changes. SELECT loss/gather shapes are reserved without computing those losses.

The default reservation is `ceil((measured_envelope × 1.15 + 512 MiB)/MiB)`. Both factors are explicit arguments. Allocation capacity is the smaller of the declared planning capacity and initially measured free VRAM under the existing workload. Each cell reports one-slot admission, `2 × reservation` for its worst same-cell pair, two-slot admission, and whether it permits only one slot. `RESOURCES.json` also reports the worst pair across all completed cells. This is reservation arithmetic; concurrent execution of **two new profiler/native jobs** and their throughput are unmeasured. The launcher must admit each launch against aggregate active reservations and current free VRAM, and use one slot whenever two reservations do not fit.

OOM receipts retain the phase, error, partial peaks, parent samples, exit status and log. OOM/missing cases are not omitted as outcomes, silently retried at smaller shapes, mapped to another architecture, or used to prune the frozen grid. A cell enters compiler-compatible `profiles` only after every required case completes. `missing_or_failed_cells`, `unadmitted_cells` and `one_slot_only_cells` are explicit. Any failed cell keeps the full packet unqualified. Any protocol or hardware revision belongs to the root before GO.

## Output and invocation

The profiler requires frozen native and runtime copies outside the live source directories. All inputs/output stay inside the allocation repository. Repeat `--custody` for each actual dataset receipt. A representative command, to run only on the verified allocation after source review, is:

```sh
python -B training_pipeline_v2_20261012/profile_native_resources.py \
  --project-root /home/jovyan/shares/SR003.nfs2/GENATATOR_PIPELINE/diploma_yandex_graphs/postsubmission_research_20260930 \
  --runtime-package FROZEN_NATIVE_PACKET/runtime \
  --native-model-source FROZEN_NATIVE_PACKET/native/models.py \
  --custody PREPARED_DATA/cora/CUSTODY.json \
  --expected-datasets 1 \
  --case-timeout-seconds 300 \
  --output RESOURCE_QUALIFICATION_cora_run01
```

For the full roster provide all eight custodies and omit `--expected-datasets 1`. `--worker-capacities` accepts a planning JSON with `worker_capacities_cuda_bytes`; any supplied server77 values remain explicitly unverified and do not authorize remote access or launch.

Outputs are `MANIFEST.json`, immutable request and final case JSONs, per-case logs, `RESOURCES.json`, and `COST_PROJECTION.json`. The manifest binds source/graph inputs and final result files. Resource entries match the native compiler's `dataset_id/backbone` key and positive reservation/wall/rate/evidence contract. There is no wildcard fallback. Resource entries retain detailed case rates, one/two-slot flags and `full_training_runtime_qualified:false`.

The compiler currently multiplies `estimated_seconds_per_update` by members. The supplied common rate deliberately retains the complete M4 measured-case ceiling, rather than dividing by four without an amortization proof. Compiler group time therefore adds conservatism for M4 jobs. `COST_PROJECTION.json` uses the relevant complete-case rate directly for each arm, with the default ×1.5 timing allowance plus 60 seconds setup per fit. Both documents label these as scheduling assumptions. A future compiler change to use the per-case map must preserve that distinction and cannot call these actual training times.

## Compute and disk report before GO

The profiler reports exact prospective starts and update ceilings even when rates remain unknown. For the default eight-dataset roster (three Planetoid split1 datasets, WikiCS split20, and four Amazon/Coauthor split10 datasets), the total split count is 63:

| Registered work | Default coefficient neighbors | Fixed native scalar coefficients |
| --- | ---: | ---: |
| Search optimizer starts | 227,200 | 76,800 |
| Comparative optimizer starts | 18,711 | 18,711 |
| Total fresh optimizer starts | **245,911** | **95,511** |
| Model optimizer update ceiling | **1,229,555,000** | **477,555,000** |
| Comparative banks / exports / scalar CAL fits / evaluation jobs, each | 12,285 | 12,285 |
| Registered scalar CAL updates | 6,142,500 | 6,142,500 |
| Reused I4 member0 fits | 3,213 | 3,213 |

Changing the coefficient mode is a prospectively frozen alternative. Resource failure is not permission to reduce it after outcomes. Actual custodies determine split totals; the script computes counts rather than assuming this table for another roster.

Measured case rates feed a maximum-shape planning projection over all fits. Lower width/depth points and one-body jobs use the larger measured proxy, so this can considerably exceed eventual time. Actual AdamW kernel time, target-based metric/selector CPU work, filesystem serialization, failed attempts/retries, concurrency and calibration/evaluation compute remain unmeasured. An unknown arm yields `estimates:null` and a null full-packet total, rather than treating that cost as zero. Measured-subset totals are explicitly separate. Root must present these limitations and measured receipts before requesting GO; this script grants no execution approval.

Disk projections retain **all search and comparative fit artifacts**. Per proxy fit, the checkpoint planning payload is `3P + buffers + scalar steps + 8 MiB padding` (model plus two Adam moments). Selected and final files plus recovery containing an embedded selected file give four such payloads; another 16 MiB covers a planning trace/metadata allowance. With recovery interval100, 51 recovery writes each use a two-payload allowance. Selected checkpoint overwrites vary from one to 5,000; gross writes are therefore projected at 104–5,103 payloads per fit. These are transparent padding assumptions, not serialized-size measurements or strict bounds. Atomic temporary writes require additional free-space headroom, and recovery trace/state growth can exceed padding. No checkpoint or optimizer is serialized by the profiler.

Label-free bank tensor bytes are calculated from metadata CAL/evaluate row counts, class count, registered banks and route views, with no endpoint file access. The default 100 MiB/s checkpoint throughput is an explicit unmeasured I/O scenario, not a host benchmark. Planned retained bytes and gross write traffic must be reported separately: replacing a file lowers retention, but still consumes I/O. Root must resolve storage capacity and I/O policy before GO; automatic deletion or reduced custody is not part of this profiler.

## Local preparation checks

Source parsing/help, AST bank extraction, exact arm coverage, coefficient-neighbor counts, eight-dataset 245,911-fit accounting, fixed-coefficient accounting, endpoint counts, unknown-rate propagation, and memory-envelope arithmetic can be checked without importing Torch, reading graph arrays or executing a model. These local checks do not qualify a GPU, native task, dataset provider or scientific outcome. Actual allocation qualification is still pending.
