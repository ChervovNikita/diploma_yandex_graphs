# Finite resource feasibility report

Status: **not admitted for scientific execution**. No GO, new fit, heldout score,
checkpoint deletion or automatic resource queue is created by this report.
The frozen catalogue is v21, hash
`1088d651a64f660bff122b6c8c03b92932e5c750da87de42e405ba279748a755`.
It contains 77,469 fits across 80 studies; original GNNM remains a new-data
comparator and all five original-paper datasets are excluded for every method.
NCNC2, BUDDY and HGT additions below are separate prospective deltas, not silently
included in that count. Source/file receipts are in `SOURCE_BINDINGS.json`.

## Known host storage, with the unpriced remainder explicit

| Host | v21 fits | Fits with no current-family checkpoint proxy | Covered single-full-checkpoint sum |
|---|---:|---:|---:|
| Allocation | 25,342 | 20,816 | 93.602 GB |
| 18.77, both GPUs | 52,127 | 42,725 | 89.557 GB |

These sums price only the covered subset using prior random, zero-update
same-study/method/factory shape receipts. They do not include trained buffers,
all adapter state, prefixes, optimizer maturity, atomic writes, input caches,
leases, banks, logs, failures or retries. They are incomplete point proxies,
not host upper bounds. The allocation observation reports 1.163 TB available;
quota and future external use are unknown. Current 18.77 disk/quota and runtime
have not been observed under the allocation-only scheduling override.

SeHGNN is assigned to 18.77/gpu1. For its 1,230-fit roster, the tensor-only
old duplicated state proxy is 12.224 TB; one selected full state per fit is
1.528 TB. Exact selected primary model tensors are 0.509 TB after full-state
leases can close. Its 360-fit HPO selected full cohort is 0.471 TB; temporary
compact HPO exports can coexist with those full authorities. The largest
complete grouped state is 4.020 GB. Existing atomic recovery writes can need
an additional 16.081 GB per active slot, or 32.163 GB for two slots, beyond
ordinary resident live checkpoint/prefix authorities. Prepared input files
add 8.253 GB, and declared logits/IDs add 21.309 MB. These numbers exclude the
listed buffers, adapter state, traces and retries. They are **not an upper bound**.
Retention improves residence time; it does not yet reduce repeated write volume.

Major unpriced studies include PolyFormer Coauthor CS/Physics (3,438 fits each,
allocation), NSD legacy/geometry (1,578+1,308, 18.77/gpu0), native QK WikiCS/Photo
(1,680+930, 18.77/gpu1), paid label stages (932 WikiCS on gpu0, 512 Photo on gpu1)
and growth (147 on gpu0). NSD nonpersistent `L` and paid-parent state must be
priced inside serving authorities, not assumed to be model-parameter bytes.

## Integer replay tapes are a separate retained output

The source LinkPrototype adapter writes an immutable `PERMUTATION.npy`,
`NEGATIVE_PAIRS.npy` and full `EPOCH.json` before each epoch's updates. Its
checkpoint adapter state contains RNG, epoch, selector and provenance; it does
not embed those arrays. The provider uses fixed source master 2026100401 and
per-fit directories. Sharing a seed does not currently deduplicate files.
The compact exporter therefore cannot remove this separate tape cost.

The complete TRAIN has 1,179,052 records. For a full equally sized negative draw,
raw int64 arrays cost `8*N + 16*N = 28,297,248` bytes per epoch, or 2.829725 GB per
100-epoch fit. Exact NPY headers, full negative row counts and EPOCH JSON layouts
still need one authenticated actual receipt; retries and filesystem overhead are
additional. v21 contains 177 tape-producing endpoint/count fits and 126 boundary
fits that do not use this provider. Conditional raw-tape totals are 0.500861 TB
for the 177-fit cohort, or 0.704601 TB when NCNC2's 72 additional provider fits
are included (249 in total), all currently assigned to 18.77/gpu0.

Keep raw arrays and receipts until a fit cannot retry and every paid replay
consumer has closed. Current code has no qualified lossless tape archive or
exact generator reconstruction release law. No tape removal or regeneration is
assumed for resource admission. Closed-tape lossless archiving can be a separate
finite engineering proof; it cannot be credited in the current bound.

## Safe execution with two slots

The launcher admits **at most two** simultaneous jobs per GPU. It must compare
the sum of qualified reservations against fresh VRAM headroom, avoid subtracting
owned process usage twice, and include unidentified/external usage. Two-slot
arithmetic is a planning capacity, not measured concurrent throughput.
Coauthor PolyFormer currently reserves 65,536 MiB per fit; two such fits cannot
fit an approximately 80 GB GPU. Those jobs must run alone unless an actual
family resource envelope supports a lower reservation. A small admitted job
can overlap a large one only when their sum fits.

Both 18.77 GPUs share one host: their four possible live slots require a joint
CPU RAM/disk/transient bound. Starting a job must reserve persistent banks and
serving states, open HPO cohorts, all transitive paid-source leases, raw tapes,
live recovery/prefix files, atomic writes, inputs, failed attempts and a declared
free-space margin. Dynamic VRAM scheduling alone does not establish disk safety.
Unknown resource cases stay disabled rather than receiving invented reservations
or an ETA. The 339 existing timings are random TRAIN gradient callbacks; they
do not measure optimizer/SELECT/convergence or whole study completion time.

## Finite remaining engineering set

Propose only the following necessary family gates for root supervision. Reuse
identical proved source/geometry receipts rather than rerunning a catalogue.

1. SeHGNN actual-input four-body BASIC/SUPPLY/JOINT and largest private-correction
   law; exact family serving state restore and prospective synthetic release.
2. Largest true nested PolyFormer Coauthor geometry and native QK local-to-global
   transition. The generic 631-case metadata proposal uses top-level geometry
   defaults and cannot certify nested PolyFormer maxima.
3. Exact paid-parent reconstruction and persistent state for the registered
   NSD/label/growth/GRU/landmark branches whose dependency laws remain unqualified.
4. BUDDY's five structural whole-TRAIN cases: baseline 1024, capable M1 256,
   genuine I4, all-map BE and original matched multistream adaptation, according
   to the exact source packet. All five are unpriced. Its prospective roster is
   114 fits, 18 banks and 26,254,200 optimizer maps; prepared cache and synthetic
   gradients do not certify actual-input execution.
5. NCNC2 completion and endpoint/count full native minibatch recursion/FP64-DP,
   persistent tape layouts, exact selected restore and endpoint construction.
   NCNC2 adds 72 fits and 122,400 maps; no new actual-input resource case is admitted.
6. HGT's explicit 13 structural cases with two proposed callbacks each (26);
   native unused gradients stay `None`. Full selected model/optimizer/OneCycle
   and private RNG reconstruction plus SELECT forward replay/constructor costs
   remain required. Its prospective roster is 570 fits, 450 banks and ten
   canonical conditions. HGT is not compact-export allowlisted, so retain full
   states until a separate audit and family restore proof.

The retention amendment is available as an immutable controller patch and v16
composition manifest. Six pure tests and a tiny synthetic CPU lifecycle passed;
full selected authorities remain present. The first failed fixture is preserved.
No release hook has executed; source-family actual-input serving and whole-host
storage admission remain open. A final controller/master successor must freeze
those choices, host resource receipts, complete counts and exact launch command
before the user verifies the scientific run.
