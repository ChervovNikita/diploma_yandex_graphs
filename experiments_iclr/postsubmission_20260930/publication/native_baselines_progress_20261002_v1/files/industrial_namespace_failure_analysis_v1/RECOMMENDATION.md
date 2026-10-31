# Namespace failure: root action

The failed public-only probe reached util-linux `unshare`'s automatic recursive private-propagation operation on `/`. It did not reach the bootstrap, worker, source-label parsing or model. The new capability receipt proves that all requested namespaces can be created with `--propagation unchanged`; it does not prove that mounts are usable.

Adding that flag alone cannot repair v2: `namespace_bootstrap.py:61` independently runs `mount --make-rprivate /`. The manual mount utility message about read-only mounting does not identify the kernel errno or establish that the inherited root's read-only flag caused the failure. Propagation changes act on mount metadata; a read-only filesystem alone is not evidence that every mount operation is prohibited.

## Immediate action

Review and run the new `root_namespace_capability_diagnostic.py` once with a fresh run identity. It uses the fixed authorized SSH route and isolated system Python (`-I -S -B`). Its remote worker verifies distinct user/mount/PID/net/IPC/UTS namespaces and mapped-root PID1, records destination/source hosting mount shared/master flags before mutation, and refuses shared hosting mounts. Direct standard-library `ctypes` mount calls report exact errno for tmpfs, runtime-image bind, subtree privacy, readonly remount and fresh proc. Mounts are limited to fresh diagnostic targets and disappear with the child namespace. Runtime files are never written or traversed. The parent records unchanged host mount namespace/mountinfo. Remote wall cap is 30 seconds. No scientific, label or GPU work is supplied.

Example root invocation, after review:

```sh
python3 root_namespace_capability_diagnostic.py --run-id capability_v2
```

## Conditional launcher repair

If the required mounts succeed and a private readonly runtime bind is observed, a new explicit source version can preserve the intended isolation:

1. Add `--propagation unchanged` to unshare. Pass the parent's mount/user namespace identities to the bootstrap and verify that both changed before any mount.
2. Replace the blanket `--make-rprivate /` with a check that the fresh namespace-root destination's hosting mount is not shared. Bind the reviewed runtime image there; ensure the new root bind has no shared/master propagation, using `--make-rprivate` on that new bind if needed and permitted. Check before making child mounts.
3. Keep readonly runtime/source/input binds, dedicated writable output, fresh tmp/dev/proc, explicit GPU device allowlist, chroot, descriptor/environment clearing, capability drop and the existing worker observations. Reuse the existing public-only probe as the operational check; no new numerical gate is proposed.

A `master:`/slave ancestor has no outward propagation to its master. A `shared:` destination can propagate new mount events and must be refused before the first bind. Once the worker's root bind is private, child mounts stay in that subtree. If it is already private, a redundant privacy syscall need not succeed; the diagnostic reports this case separately.

If bind, readonly remount, tmpfs or fresh proc are denied, a flag-only no-sudo repair cannot preserve this mount-based contract. Keep labels/models closed and preserve the failure. A writable copy/chmod substitute, inherited host proc, or dropping readonly/capability checks would change the intended boundary and is not recommended.

The author performed local text/AST/scalar metadata analysis only. The diagnostic has not been run. Sealed v2 and the live precision study were not changed.
