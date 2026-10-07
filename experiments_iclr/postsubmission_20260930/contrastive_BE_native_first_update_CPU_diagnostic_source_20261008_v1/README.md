# First native update diagnostic

Source only; execution disabled. Read exactly the existing discarded
shared_common_LOCAL and shared_route_LOCAL native qualifier checkpoints and
their whole qualification receipt. No new fit, forward, logits, graph or label
reader. No old paper/research result is recalculated.

The source verifies both states came from the first discarded LOCAL update at
seed901337, with the original CUDA backend, unit shared M4 architecture, same
core/native/model/optimizer recipe and identical member RNG endpoints. It then
constructs one original CPU model from the same native source/seed to recover
initial weights and parameter alias identities. It creates no optimizer and
executes no model forward. Constructor randomness is isolated by the original
seeded helper; pre-constructor target validation contains no RNG calls.

Checkpoints differ only in the fixed common/route target assignment. That does
not imply equal shared gradients under independently sampled member dropout.
Native floating execution is not certified bitwise identical. Source equality
and matched RNG opportunity support this bounded first-update comparison; the
observed parameter differences do not prove a causal gradient or quality claim.

Count unique live Parameter objects, validate all duplicate state slots, and
report exact shared storage between distinct Parameter objects without merging
distinct optimizer recipients. Private r/s factors are identified by their live
FactorLinear objects; all other native parameters remain shared. Initial private
factors must be exactly unit. Inactive local-stage global modules must remain
exactly equal to the reconstructed initial state in both snapshots.

Report per-parameter/per-member/aggregate common and route updates, and their
difference, with unique entry counts, exact nonzero counts, L2/RMS and maximum
absolute magnitude. All aliases are counted once by Parameter identity. Exact
nonzero counts include floating arithmetic variation; no threshold is used to
rescue a hypothesis or gate scientific activation.

Interpret only whether this initialization's discarded first updates differ.
Nontrivial positive-target TV can still yield zero cosine derivatives in a
stationary parallel-embedding configuration. Member competence, useful diversity,
accuracy, novelty and acceptance remain untested by this diagnostic.

Root fills a separate copy of RELEASE_TEMPLATE_DISABLED.json with the two LOCAL
checkpoint bindings from QUALIFIED.json and that receipt's own binding. Run the
existing qualified Python with -B and `compare.py --release <path>
--release-sha256 <SHA>` on the authorized allocation with CUDA_VISIBLE_DEVICES
empty. A finite300-active+10-cleanup310-second CPU supervisor and4GiB RSS cap
are proposed. Raw checkpoints remain server-only; output contains compact
parameter/update statistics and provenance, not tensors.
