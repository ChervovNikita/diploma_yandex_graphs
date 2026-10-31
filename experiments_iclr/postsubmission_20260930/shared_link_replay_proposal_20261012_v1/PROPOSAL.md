# Prospective shared integer replay store

This is a source-equivalent storage and preparation-work proposal, not a new
scientific method, training result or live pipeline. No source epoch is generated
by its synthetic tests, no optimizer map runs, and no existing packet changes.

## Why one store can preserve the registered source trajectories

Both retained LinkPrototype adapters call the same source `draw_epoch` and
`write_epoch` using master2026100401 and the complete registered TRAIN geometry.
The sampler draws depend on the raw TRAIN graph, native source sampler/runtime,
node count and the epoch seed; the native permutation depends on TRAIN length,
batch size and that same isolated epoch seed. They do not read a fitted model,
route arm, learning rate, dropout rate or initialization seed. The provider
saves and restores the caller's complete source/global RNG state. All complete
permutation, negative arrays, masked graphs and context hashes are authenticated
before that epoch's optimizer work.

For equal authenticated data/source/runtime/device and epoch, deterministic
provider generation therefore produces equal integer arrays and contexts in
every fit. Replacing duplicate per-fit authorities by a shared authority plus
immutable per-fit reference leaves each query, tail policy and objective input
unchanged. It leaves neural initialization, private dropout streams, optimizer,
SELECT checkpoints and all 1,700 maps per fit unchanged. This is conditional on
source/runtime equality and RNG proof, not implied by matching only seed values.
The 126 native boundary fits with their different sampler law are excluded.
The proposal does not expand sharing to other samplers or datasets.

## Exact prospective authority and consumer protocol

1. A root-approved source successor declares one stream key over every GRAPH/
   TRAIN/source file receipt, full TRAIN geometry, master seed, batch/tail law,
   little-endian int64 schema, provider/native dependency byte hashes, numerical
   runtime and the actual selected device environment. Preserve the relevant
   source provider bindings and device/sparse operation identity. No scientific
   target file is decoded to build this key. Fit method/hyperparameters are
   deliberately absent because the provider never reads them.
2. Generate each epoch once using the unchanged authenticated source provider.
   This may be a separate label-free preparation authority or a lazy operation
   under an exclusive epoch lock. The producer needs a separately admitted
   source-memory/CPU/time envelope; lazy generation must fit alongside any live
   models on that host. The common store changes preparation work and costs,
   which must be logged explicitly.
3. Generation writes to a fresh attempt directory. Authenticate it through the
   unchanged source EpochReplay reader and complete source/data/RNG checks,
   atomically publish its immutable file/epoch manifest, and bind that receipt
   into the controller registry. A crash between directory publication and
   authority publication leaves an orphan requiring explicit root recovery.
   Failed partial attempts and error receipts remain preserved. Corruption
   blocks admission; the implementation must never silently regenerate a
   published authority or overwrite its consumer receipt.
4. Each fit at epoch e restores its own model/private RNG scope, reads the bound
   common arrays using the source EpochReplay constructor and preserves the
   complete source context checks. Save an immutable per-fit pointer identifying
   store key, e, epoch receipt, actual byte/work costs and the registry authority.
   Existing selected/recovery checkpoint epoch and RNG laws stay unchanged.
5. Lease the store through the complete prospective fit/retry/paid-consumer
   roster. A finished job cannot discard a tape needed by an unfinished source
   or downstream continuation. The prototype records open leases only and has
   no release/deletion function. Every run retains its result, failure, curves,
   optimizer counts and pointer provenance. Historical per-fit outputs are not
   migrated, relabelled, deduplicated or deleted.

## What is proved and what is pending

`shared_replay_store.py` is a stdlib lifecycle prototype with exclusive locks,
immutable manifests, exact per-file checks, preserved failed attempts and
per-consumer open leases. Five tests pass on fabricated text files: two
concurrent readers cause one producer call, corruption is rejected, a failed
attempt remains, input/runtime/master key differences separate stores, and
missing runtime or out-of-range epochs are denied. These tests qualify the
local lifecycle mechanics only; their `.npy` files are deliberately fabricated
text, not numerical source evidence. No Torch/graph/model is constructed.

The guarded metadata observation verifies the prospective NCNC2 exclusive
successor, confirms 249 provider fit records and 126 excluded native-boundary
records, verifies exact input/source file bytes, compares their declared source/
data/geometry keys and the literal replay_for_epoch AST. It hashes metadata and
raw files without decoding TRAIN, SELECT or heldout target values. Its runtime
key is unqualified, so source-key equality is not real CUDA equivalence.

Before integration, the root must admit or correctly inherit an actual full
source replay proof under the frozen current runtime/device and bound data.
Verify complete native permutation and returned negative rows, dropped tail,
all 17 native batch graph/query/support contexts, source forbidden/self-edge
checks and exact caller model/global/private RNG preservation. Verify source
reader and shared reader return equal arrays/context and preserve the checkpoint
next-epoch/retry law. No neural optimizer or scientific score is needed for that
narrow source proof. Actual constructor/CPU-RSS/CUDA/time and IO costs still need
resource admission. The root then reviews a distinct adapter/controller source
successor and includes this changed storage/preparation law in the user's GO
materials. This prototype does not authorise those actual-input operations.

## Storage arithmetic

The source budget records P=1,179,052 TRAIN records, raw directed entries2P and
native default negative request N=2P, with actual returned N unmeasured. Raw
payload per epoch is8P+16N=47,162,080 bytes at that request proxy. One100-epoch
roster is4,716,208,000 bytes, plus NPY headers/EPOCH JSON/manifests. Duplicating
it249times costs1,174,335,792,000 raw bytes; an admitted shared store could retain
one roster, plus small per-fit receipts/leases and failed generation attempts.
No measured compression factor is assumed, no source coordinates are stored,
and no tape deletion is credited. Resident and cumulative preparation costs
must both be reported. The claimed savings remain **prospective and unadmitted**.
