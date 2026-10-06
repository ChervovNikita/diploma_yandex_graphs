# TRAIN-only internal-witness candidate census

The actual structural qualifier passed all six controls and cleanly closed.
Its first frozen cycle had internal-edge witnesses for168/3870 outer positives
and1093/62464 inner positives, but zero in all inner/outer sampled negatives and
all7740 native bank entries across61 current union-masked supports. This establishes
that this sampled cycle supplied no witnessed-negative contrast. It does not
establish that such TRAIN-absent pairs never exist, that the method failed, or
that weight sharing caused the recorded strict common errors.

## Exact exhaustive probe

For every undirected supported TRAIN edge(w,z), form the triangle co-neighbor
group S=N(w)∩N(z). Enumerate every unordered pair(u,v) in S; reject self and all
known TRAIN facts. Each surviving pair has internal common-neighbor edge(w,z).
Conversely, any internal-CN-edge witness for a TRAIN nonedge places u,v in this
group. The inverse edge enumeration is therefore exhaustive and its occurrence
count per pair is exactly LCL. Every discovered candidate is then independently
checked using its own C(u,v) and direct internal-edge count.

Only authenticated `train_pos.txt` is opened, plus acquisition/source/gate
metadata. No feature, checkpoint, learned model, VALID/TEST value, or held negative
pool is loaded. Full TRAIN opportunity is an upper bound for union-masked support
opportunity. Candidate pairs remain TRAIN-absent **unlabeled** pairs and may be
future/held positives. Eligibility does not certify representative negatives.

The result retains counts, LCL/CN/group-size histograms, endpoint concentration,
a canonical candidate-set digest, and at most16 example pairs. No full candidate
list or negative bank is written. The uniform-draw expected count/probability of
zero is explicitly hypothetical, not a verified PyG sampling-law claim.

Proposed bounds: CPU-only (`CUDA_VISIBLE_DEVICES=''`), one core, worker soft40s,
CPU soft45/hard50s, external wall60s, address space/RSS1GiB, result64KiB,
combined logs256KiB, total output1MiB, pre-child resource window60s. Work cap10m
pair occurrences and storage cap1m unique candidates abort with incomplete status;
there is no truncation, sampling fallback, automatic retry, or negative-zero
conclusion from an incomplete census. The existing reviewed `run_fit` loop and
ownership helper supply identity, terminal wait and telemetry. No new guard loop.

Six constructed graph cases compare exact candidates/counts against a separate
all-pairs oracle, including K4 rejection, K4-minus-edge, overlapping witnesses,
empty/triangle/cycle nulls. The work cap is checked to fail explicitly. These are
source checks; the actual graph probe awaits root activation. Existing v2 fixture
failure, v3 gate, fit/cost sources and E/J4/initialization queues are preserved.

## Saved targeted-negative literature assessment

No new paper read or novel sampler is claimed. Saved source scopes already cover:

- HeaRT's endpoint-personalized RA/PPR/feature-cosine hard evaluation candidates.
  Its official pools are evaluation data, not TRAIN inputs. Reusing a heuristic
  generation principle prospectively differs from importing those pool values.
- DNS/DENS and MixGCF: selection or graph-layer synthesis of hard candidates;
  AHNS: positive-aware adaptive hardness; DivNS: coverage/diversity beyond the
  hardest candidate. Topology-targeted negatives are an established type of
  sampling intervention, not a new ensemble principle.
- SRNS and Bayesian Negative Sampling: difficulty is not a true-negative label;
  reliability/posterior assumptions matter. A K4-minus-edge candidate can be an
  unobserved positive, especially in an edge-withheld benchmark.
- Importance weighting: conditioning on a topology-filtered candidate set changes
  the training target unless the proposal has full support and acquisition/
  sampling probabilities are handled. No weighting is implemented in this census.

If a new targeted sampler is considered, freeze it prospectively from TRAIN only,
use the same acquisition/targets for every architectural control, charge its full
work, and retain uniform sampling as a control. Historical E/J4 records would
then be historical references rather than matched new-sampler controls. Do not
use held positive identities to reject candidates or change official evaluation
pools. No sampler, cost cycle or full fit is launched by this probe.

If opportunity is sparse/absent, root may select a separately frozen denser-graph
successor before fitting. Current dynamic model dimension does not admit Collab's
loader, graph protocol or resource cost automatically. A census does not prove
that another graph will supply useful supervision or predictive gains.

Reused scopes: `gine_closest_control_v1/link_recommendation_gap_v1/REPORT.md`,
`small_representative_link_benchmark_scout_20261005_v1/REPORT.md`, and
`ranking_aligned_member_competence_control_20261006_v1/REPORT.md`. Canonical paper
ledger and original benchmark scores are unchanged.
