# NCN / Collab factual context affinity — bounded CPU readiness

Common and split both passed their exact native CPU lifecycle checks.

- Four total original source optimizer transitions; four extra passive AUX gradient collections.
- Six full official VALID calls (60,084 positives + 100,000 negatives each), four native constructions.
- Both fresh checkpoint restores reproduced the served logits bit exactly.
- Complete native 65,536-positive + 65,536-negative task loss preserved in every update.
- Shared/input-bank AUX gradients were active; direct output-bank AUX gradients were zero.
- Actual semantic duplicate/conflict/self omissions, zero contexts, donor support and costs are in the JSON and fetched results.
- Checkpoints stay on the allocation; only JSON/log evidence was fetched.

The combined child wall time was 102.16 seconds; peak process RSS was 9.73 GiB.

**v1 is CPU-only.** Static audit identified a CUDA checkpoint reconstruction blocker:
saved sampler negatives are CPU tensors while the reconstruction moves the index to
native CUDA before using it. Preserve this v1 evidence. A new v2 must explicitly gather
on each source tensor's device and authenticate endpoints on CPU before CUDA admission.

No TEST access, GPU update, completed scientific fit, accuracy/convergence result or
novelty claim. NCNC/BUDDY, full-worker dispatch, CUDA pair admission and final TEST
reconstruction remain unqualified. The comprehensive campaign still needs user GO.
