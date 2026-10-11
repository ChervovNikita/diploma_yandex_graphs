# One E4 runtime check for the portable binding

Run one E4 update and checkpoint round trip in a fresh Linux Python 3.11 environment using the README installation pins and only the portable bundle for project imports. This is the smallest actual numerical check that exercises the new binding, the removed unused run_base dependency, the exact PyG import receipts, conditional endpoint transport, sequential route accumulation and checkpoint restoration together.

## Procedure

1. Use a fresh work directory and the canonical prepared TRAIN and SELECT files. Construct the unchanged full ten-block config and source receipts through the portable workflow. Preserve the fixed 1000-update recipe in the config; call neither acquire nor Family.fit. Use a finite external watchdog and wall, CUDA and RSS caps. Do not reuse a current scientific checkpoint or outcome.
2. Call `engine.construct(cfg, fresh_output, common.local_validate)`. This validates all source and TRAIN/SELECT receipts and runs the portable constructor. Activate seed 8101, set the operation to `endpoint`, and call `family.make(8101, 0, 'baseline', True, 4, 'shared')`.
3. Invoke the bundled `own_ce.update(family, model, optimizer, streams, True, mode='sequential', qualification=False, capture_gradients=False)` once. Require the hook to report four backwards and exactly one Adam step, finite TRAIN loss and finite registered gradients. Do not publish the loss value as predictive evidence.
4. Save the post-update model, optimizer and streams to a fresh smoke checkpoint. Record its size and hash. Recreate the same E4 model, load with `torch.load(..., map_location='cpu', weights_only=True)`, load the model with `strict=True`, restore the optimizer and route streams, and set evaluation mode.
5. Perform one label-free forward through `family.logits` using SELECT IDs. Require finite output of shape `[4, 150, 5]`. Record pass or failure, runtime versions, source receipts and actual resource use; retain any failure. Do not compute SELECT accuracy or NLL, read CAL or TEST truth, run calibration, choose a retry configuration, or run a fit roster.

The smoke checkpoint is a runtime artifact after one update. It is not a scientifically selected state. The public exporter continues to require the entire actual 90-state closure; this procedure must not forge or bypass that closure.

## What the result establishes

A pass establishes that this fresh environment can construct the full Physics E4 portable path, perform one actual update, serialize and restore that path, and produce label-free logits. It does not establish numerical equivalence with the author environment, full scientific fitting, completed source portability for every arm, author reference integration, or the complete prepare/acquire/export/score workflow. Those limits belong with any portability claim.

Root owns execution. This memo contains a procedure and source review, with no new dependencies, numerical imports or execution by its author.
