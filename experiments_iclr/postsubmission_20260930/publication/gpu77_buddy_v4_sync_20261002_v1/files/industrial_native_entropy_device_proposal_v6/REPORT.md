# Dormant v6 native entropy device proposal

This is a local source proposal. It is **unapproved, unactivated, and unexecuted**. It contains no launcher. The copied worker refuses before opening a device or installing policy unless its closed source flag is changed in a newly reviewed and sealed packet. The existing v5 package and all predecessor receipts are preserved.

## Exact observed blocker

The v5 root receipt completed `packaging`, `numpy`, `scipy`, `pandas`, and `yaml` after Landlock and seccomp, then failed while importing `delu`. Its complete exception graph has one `RuntimeError`, no cause or context, and no errno. The inner Python frames are `delu/__init__.py:5` → `delu/cuda.py:5` → `torch/__init__.py:290`; the message is `Unable to open /dev/urandom`. Thus the dependency frontier is the pinned Torch import reached transitively from Delu, rather than a failure of the Python entropy adapter.

The v5 adapter audit records one request for 16 bytes, one native `getrandom(318)` syscall with flags zero, and zero hard errors. Its activation bound both `os.urandom` and the already loaded stdlib `random._urandom`. Repeating those bindings cannot adapt a Torch C/C++ consumer that opens a pathname directly.

The pinned runtime manifest binds Delu 0.0.26, Torch 2.4.0+cu118, their import source files, the Torch `_C` extension, `libtorch_cpu.so`, and `libc10.so`. Only this manifest is available in the local runtime-image directory; the corresponding package source and binaries are absent locally. The receipt explicitly says `native_syscall_cause_verified=false` and `native_C_syscalls_fully_traced=false`. It therefore establishes the named native entropy dependency and Python import frontier, but does not prove the exact C++ function, native open syscall variant, or errno. CPU generator entropy bootstrap is a plausible interpretation, not a newly verified native stack. No remote lookup or runtime execution was used to strengthen that claim.

The current read grants exclude `/dev/urandom`; their file branch requires actual regular files. The seccomp source allows ordinary native `open/openat` without forbidden truncation flags, while Landlock denies ungranted reads. No source-only modification within the current identical read grants can make a pathname-dependent native random-device open succeed. Changing the numerical RNG, fabricating entropy, editing the pinned Torch binaries, or repeating the Python adapter would change the wrong interface or violate the current contract.

## Minimal prospective policy difference

`V5_WORKER_DELTA.patch` changes a separate dormant v5-worker derivative. It adds one hardcoded rule for `/dev/urandom`, never a directory or a caller-selected path. Before adding that rule, it opens an `O_PATH | O_CLOEXEC | O_NOFOLLOW` descriptor and validates the actual descriptor with `fstat`: the object must be a character device with Linux major 1 and minor 9. A symlink opened with `O_NOFOLLOW`, regular file, `/dev/random` major1/minor8, or other device fails closed. The rule is attached to that verified descriptor's object, so later pathname replacement cannot expand the granted object. No entropy bytes are read or hashed during validation.

The only added right is `LANDLOCK_ACCESS_FS_READ_FILE` (`1 << 2`). It grants no write, directory listing, creation, removal, or rename right. The descriptor is closed before restriction and is not inherited by scientific imports. Regular file rules keep their existing `S_ISREG` validation. The returned receipt identifies the validated device and explicit difference from v5. A new root-approved execution packet would need fresh source/contract pins and run identity; this proposal is not such a packet.

All original data/library/directory/output grants, handled filesystem mask, capability drop, descriptor cleanup, seccomp policy, content denial probes, Python adapter, import order, exception diagnostics, and CUDA checks are retained. The unchanged core policy continues to reject every unpinned ioctl, including random-device ioctls; the new rule supplies read access only. The v5 policy file SHA256 is `36f26ad757b88c3ff86bf39fac5ee5020cc3765b269ec7d8e2f7c8aeefef63a0`. The original outer and import-contract hashes are recorded in `SOURCE_CHECKS.json`; neither was edited.

This is explicitly an additional public-device read grant. Current repository/runtime read grants do not authorize its activation. No activation, root invocation, device open, entropy read, remote call, host write, GPU operation, data load, or scientific import occurred during this preparation. Standard OS entropy is an import execution dependency; the later frozen `delu.random.seed(seed)` scientific seed policy is retained.

## Why moving imports earlier changes qualification

The existing worker requires that scientific modules are absent before restriction and imports each module after both Landlock and seccomp. That measures native loader and package initialization under the content and syscall boundary. The v5 success of the first five imports is evidence about that exact ordering.

Moving Torch or Delu into unrestricted trusted startup would violate this qualification condition. Importing a pinned third-party package can execute Python and native initializers, open descriptors, or start threads before the sensitive-content and syscall controls apply. A later restriction does not undo those actions. Such a receipt could qualify later operations under restriction, but could not honestly claim scientific imports occurred after restriction.

A separately reviewed, tightly constrained bootstrap alternative could import a fixed pinned module closure under an early content boundary, then install the final restrictions and verify descriptor/thread state. It would require a new contract and trust analysis, and may still require a temporary real random-device read. It is not an authorized workaround here and has not been implemented. The one-object read grant is the smaller proposed change if the goal remains the existing post-restriction import test.

## Source validation

Ten pure stdlib mock cases execute the exact extracted proposed Landlock function against synthetic `os` and policy objects. They cover dormant refusal before any open/syscall, exact character1/9 READ_FILE, regular and symlink rejection, incorrect major/minor rejection, retained regular-library checks, open/rule errors, and missing ABI1. They check descriptor closure and that failed validation never reaches `restrict_self`. These checks call no real device, entropy provider, kernel policy, launcher, or scientific module.

The reversible delta reproduces the entire original v5 worker byte for byte. All 87 current predecessor source and receipt files from import-qualification v1–v5 are bound and rechecked. These are source checks, not a claim that the proposed device grant has passed a Linux runtime test. A future explicitly authorized run must still prove read-only device access, unchanged sensitive-content denials, native import completion or the next exact failure, and unchanged host state.
