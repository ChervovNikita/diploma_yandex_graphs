# Independent repeated resource qualification source review

**GO for bounded v3 CPU resource qualification** against manifest `85503623773198a67f2f0add172bcd0b73b3718a0947f5f137659521dbbb0570`.

All seven payloads and the seal match their hashes/bytes. Ten bound original source records and the exact freeze were independently verified. The main fit remains byte-identical at SHA-256 `642c3478c2772d9b146fdea7e9b0cfbd97e2d319b38e428281d0441b1b8f77e8`. Python and embedded REMOTE source parse cleanly; no fixture suite or model work was repeated.

The narrow qualifier delta performs six successive TRAIN updates per frozen arm with the same all-seven resident initialized family construction. Previous eval/checkpoint references remain through the next TRAIN forward; TRAIN output/loss references remain through eval. Per-update complete-state/logit writes and VmSize/VmPeak/current/peak RSS observations cover the repeated-forward lifetime gap in the original probe. Arm closure deletes state/output/optimizer references. Only frozen first-split TRAIN targets enter the loss; no validation/test score, model selection or reusable study fit is produced.

The corrected wrapper sends every manifest payload plus manifest/seal. It sets 16 GiB RLIMIT_AS before Python/Torch import, one CPU thread with CUDA hidden, CPU 1200/1205 seconds and wall 1200 seconds. It monitors a 14 GiB sampled RSS stop, requires 24 GiB current host/cgroup headroom and acceptable CPU load, and terminates then kills after 5 seconds if needed. These are single-probe bounds, not five-worker study admission.

The probe loads/restores complete state but does not execute a post-load replay forward. Its result can support bounded repeated TRAIN-forward admission; it is not a guarantee for 300 epochs or the selected-state replay peak. Root should check the inner process peak RSS as well as the wrapper's sampled peak before study admission. The prior incomplete 35 cohort stays preserved, and a fresh complete 35 execution still requires explicit root supersession/release after successful qualification.

`CHECKS.json` records exact source custody and limits. No real labels, checkpoints, logits or quality outputs were opened, and no training, GPU or remote scientific execution was performed by this reviewer.
