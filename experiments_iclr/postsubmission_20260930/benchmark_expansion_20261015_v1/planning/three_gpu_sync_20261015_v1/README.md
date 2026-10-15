# Three-GPU synchronization — 15 October 2026

The latest user instruction replaces occupied18.77 with18.51. The connection to
18.51 succeeded through MacLink and authenticated SSH. The seven-GPU allocation
is used only as the forwarding relay. No passwords are stored here.

- Allocation: one A10080GB, existing GRU worker preserved.
- Server51: hostolig, two A10080GB, existing other-user jobs preserved.
- Dedicated server51 checkout: /disk/10tb/home/shmelev/gnnm_iclr_validation_tuning/postsubmission_git.
- Both source checkouts initially match published0dd66efe12be9a851e5e8e56ad504677ed4a1b4c.
- Exact native source closure and PubMed input bytes transferred, with hashes
  checked on both ends; neither Mac retained dataset copies.

The pilot plan assigns one full eight-fit seed cohort to each physical GPU.
Each contains four native independent models plus originalGNNM and the uniform,
structured and shuffled residual variants. This retains all paired controls
on the same host. Allocation-based estimates are3.4–4.1hours of TRAIN+VALID
compute per cohort at2000epochs, or16.8–20.2hours at10000epochs. Setup, checkpoint
I/O and existing workloads add time. These are projections, not measured51 ETAs.

Queue construction, target qualification and actual launch status will be
recorded separately. The full6460-fit campaign remains unapproved and unlaunched.

## Verified checks and prepared full-campaign assignment

The server51 native and structured checks passed on GPU0 and GPU1 respectively.
Each performed two complete TRAIN updates and three whole VALID evaluations;
fresh checkpoint restore was bit exact. Elapsed child times were8.06s and13.61s.
Peak CUDA reservation was2.49GB and3.40GB; peak RSS1.38GB and1.43GB. These are
engineering checks, not prediction-quality measurements or convergence runs.

The full campaign dormant lists assign2,154/2,153/2,153fits and90ensemble
exports per GPU. Each has90complete paired cohorts. All6,730tasks are present
once and dependencies remain on the same device. A stratified scheduling proxy
(native=1,shared=4,export=.05) differs by0.06% between largest and smallest
assignment. Actual training durations remain uncertain. Source settings,
canonical identities and scientific hashes are unchanged. The lists are saved
on both servers; no full-campaign fit is authorized by these lists.

The pilot couples each optimizer seed with its assigned GPU/host. Compare
methods within each seed and disclose that hardware and seed are coupled.
