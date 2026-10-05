# Launch metadata count clarification

The original launch packet is preserved unchanged: `shared_private_transfer_gpu77_block_launch_execution_root_20261005_v1/MANIFEST.json`, SHA256 `bae2f01af735dcba56ffd4281d39d4d6ec56eb8170cce38cefe8c30b12c22de0`.

In its `COMPACT_BINDINGS.json` (SHA256 `76d5998defb06f456cee6f4a4b08cadf1898f79ad6eef3220319e2f25e75f58d`), `blocks[*].scientific_children_started=0` was inherited from the authenticated **prefit freeze at06:44:42UTC**. It is the count at that freeze, and should have been labeled with that time. It does not describe the subsequent initial observation.

`INITIAL_OBSERVATION.json` (SHA256 `e185aba691d02bb0ea4a4a63cf3ad35299ad07fe577626e8477d568d79f41a29`) authenticated **one live first fit child per block at06:48:08UTC**: b1 child3333588/start1733275970, E_random_detached, cycle3/episode60; b2 child3333666/start1733276069, S_end_live, cycle4/episode14. Their exact identities and assigned CUDA placement matched. Both queues were live, with no observed failure.

Thus there were zero scientific children at freeze and two observed active first-fit children at the later snapshot. Twenty fits were authorized; no whole block or twenty-fit completion is asserted. No running source, receipt, score access, original sealed packet or release changed.
