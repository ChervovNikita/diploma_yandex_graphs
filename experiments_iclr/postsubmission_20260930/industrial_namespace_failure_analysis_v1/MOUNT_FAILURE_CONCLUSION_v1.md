# Mount capability conclusion and execution provenance

## Conclusion

The current v2 mount/chroot route is unavailable in the observed unprivileged environment. The bounded root diagnostic created distinct user, mount, PID, network, IPC and UTS namespaces, with mapped UID 0 and PID 1. Each attempted direct mount failed with errno 13 (`EACCES`): minimal tmpfs, the runtime-image bind, and fresh proc. The runtime-image source and all destination hosting mounts were writable (`rw,noatime`) NFS mounts with no `shared:` or `master:` tags. The parent mount namespace, other namespace identities, and mountinfo remained unchanged.

Private-subtree propagation and readonly remount were skipped because their prerequisite bind failed. This closes the earlier conditional repair recommendation: changing `unshare` propagation flags, or avoiding the blanket propagation operation on `/`, does not supply the required mounts. The evidence identifies the rejected operations and errno; it does not identify the responsible kernel/LSM/container policy. Elevated mount permission or a separately specified isolation mechanism would be required. No unisolated fallback is accepted.

## Exact executed source

Root executed the reviewed draft, preserved as `root_namespace_capability_diagnostic_executed_draft.py`:

`806aa0d42ae47fb19e4e16dd76f74c3d7c52a74e890cfbf2bdf62e4c07d53dd8`

The later file `root_namespace_capability_diagnostic.py` was not executed in this run:

`2d8f6edcfde60e1c4920874e69f2a6a96a19f391abf3b306a1b15bd819aad39a`

The preserved draft was recovered by reversing the two known subsequent refinements and checking its complete byte hash against root's executed-source identification. The refinements added `-I -S -B` to the child interpreter and changed the remote outer interpreter from the repo venv to `/usr/bin/python3 -I -S -B`. They do not change the mount worker body, whose exact text matches the completed receipt's `argv` and has SHA-256:

`c9dd2a3136b99a937bf092e8ed4dec67093b5d8e3ccd02a43c7f8ef9090ea828`

The completed run must therefore remain attributed to `806aa0...`; interpreter isolation from the later `2d8f...` refinement was not observed. The draft, later refinement, initial failure, original conditional report, and root receipts are retained separately.

## Evidence and scope

The decisive receipt is `root_capability_runs/capability_v2/NAMESPACE_CAPABILITY.json`, with its root launch and stdout/stderr retained in the same directory. The earlier public probe failed before bootstrap/worker/source-label/model work. The bounded diagnostic reports no scientific, data, label or GPU action. This addendum was produced using local source/text/AST/scalar metadata analysis only; the author did not execute either diagnostic or any scientific code.

Sealed industrial v2 remains pinned to manifest `b73aea00210293667792dbe5124ed8780d35ba11dd5932ed132501ad972d54d8` and seal `e946a071bc0fdb5e8b35d4ca3835599155828ad59c61a1432ca309ea8b7234be`. The live precision study and root-owned run evidence were not modified.
