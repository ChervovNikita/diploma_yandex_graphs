# Minimal direct-worker ownership repair

The root-relayed fresh v1 review identifies likely escape of the elastic numerical rank into a different session than the supervisor-held launcher. That could exclude the rank from owned RSS accounting and group cleanup. V1 is preserved; no request to change its verdict is made.

V2's Popen argv is the admitted Python interpreter, `-B`, the exact worker and release arguments. `start_new_session=True` is unchanged, and the supervisor now holds/observes the numerical rank's PID itself. It passes an explicit RANK=0, LOCAL_RANK=0, WORLD_SIZE=1 environment. No intermediate elastic/launcher process remains. The existing held-child birth/session/group guards, WNOWAIT, sampled session RSS, owned killpg, physical terminal and custody logic are unchanged.

The worker still creates native NCCL/DDP rank0/world1, now using `file://` for `run01/RANK0_RENDEZVOUS` inside its newly created owned directory. No shared port or env rendezvous is needed. After native TRAIN and full VALID, it destroys the process group and removes its private rendezvous file before final success-envelope custody; a failed attempt retains its incomplete artifacts. The same setup and teardown timing scopes include this small rendezvous change.

All native/scientific/data work, selective noTEST adapter, caps and distribution versions are unchanged. Only the execution destination is fresh v2. Root's existing completed 16-package overlay metadata/inventory is bound without new package or runtime inspection. Root still needs a concrete external dependency admission in the unchanged schema plus a fresh independent source PASS and root release.

Author AST/byte checks establish the source topology and unchanged recipe/data/caps. No numerical or OS process launch was performed, so actual process behavior remains to be observed under the eventual authorized supervisor. This remains the inherited sampled ownership mechanism, not a process-escape sandbox.
