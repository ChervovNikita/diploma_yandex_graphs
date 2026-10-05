# Owned Pubmed CPU input supervision v2

5 October 2026. Separately preserved repair of the complete independent v1 workflow findings. No execution, imports/tests, data/prediction/heldout payloads or server contact occurred.

All post-Popen monitoring/start-metadata/I/O is now inside try/except BaseException/finally. An ordinary monitoring exception or first KeyboardInterrupt enters cleanup before propagating; cleanup terminates and, if necessary, kills and reaps the same direct child. The nested finally covers interruption in the graceful phase. This does not promise recoverable receipts if the supervisor is externally killed or storage I/O cannot persist evidence; such transport remains unknown/failure, never success.

Raw os.kill delivers SIGTERM/SIGKILL only to the kernel-owned, still-unreaped direct child. No Popen poll/wait/terminate/kill method runs. A single os.wait4 path consumes terminal status and authoritative peak RSS and sets child.returncode, preventing the prior Popen signaling/reaper race. No unrelated PID/process is searched or signaled. The standard process environment and host configuration remain unchanged.

Before any staging, unique paths must match exactly the three known source programs plus SOURCE_MANIFEST.json and the seven root metadata/source files. Exact ordered executable manifest names, source hashes and sizes are checked again. Arbitrary source files and stdlib shadows cannot enter this staged packet. The independently prepared client successor must stage the same complete whitelist.

The fresh execution identity is v2. The same two input-only stages, CPU-only process, source/runtime/root review admissions,900-second per-stage hard bounds,2 GiB RSS and128 MiB owned output limits remain. Exact original data semantics and all scientific gates remain unchanged. Actual root releases, client/source review closure and execution remain absent. Focused independent re-review must cover the whole repaired data/client/supervision workflow.
