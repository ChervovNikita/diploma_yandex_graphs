# Four NCN interaction recipes — bounded CPU readiness

All four explicit recipes passed two actual full-source NCN updates, whole
VALID and fresh checkpoint restoration. Source manifest: `8f6cc6ca60214b47653ba4c305f2b98dfae503c08ae52ee32b30de67ed132450`.

- Eight actual optimizer transitions, eight native constructions, twelve whole VALID calls.
- Twelve actual reverse collections and zero duplicate gradient/probe collections.
- I1/I2 actual autograd objective/block ownership and exact assignment of the returned tensors were verified before original Adam.
- I3/I4 use one audited J scalar; actual source/component gradients were active and finite.
- Every fresh restore reproduced original model/Adam/sampler/private/global RNG/modes/epoch state exactly and served/same-bank mean VALID logits bit exactly.
- Original65,536positive+65,536negative source queries, graph mask, native operators and optimizer options remain intact.
- Checkpoints remain on the allocation; JSON/log receipts were fetched.

The four qualified child cases took 216.27s in total; peak RSS was 9.65GiB.
The earlier repeat-backward equality guard failure is preserved separately:
0updates,1construction,4reverse collections,47.13s. Cumulative native work is
8updates/9constructions/12VALID/16reverse collections and263.41s child wall time.
No failure or cost was erased and no numerical tolerance or scientific source changed.

This is CPU engineering readiness only: no TEST access, GPU update, scientific
fit, convergence/accuracy result or novelty claim. CUDA/resources/full-worker,
selected-checkpoint mean export, other link families and final TEST reconstruction
remain separate gates. Full campaign launch still requires user GO.
