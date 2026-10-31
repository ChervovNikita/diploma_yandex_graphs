# Passed CPU boundary: actual scope

Root's `cpu_content_root02_readonly_stdin` receipt is `PASSED_CPU_DIAGNOSTIC_ONLY`, with worker `PASSED_CPU_FIXTURE_ONLY`, ABI1 and 78 classic-BPF instructions. All 19 prepared fixture groups passed. The outer measured readonly major1/minor3 stdin; worker source and host namespaces/mountinfo remained unchanged.

The receipt records `EACCES` for excluded content/write/create, readonly writes, own-proc content and new `/dev/null` opens; `EPERM` for sampled truncation/open, metadata/xattr, memory/FD, io_uring, mount/namespace/exec, UNIX/foreign-FD and static-pin mutations; ordinary output writes/ftruncate; visible metadata; ioctl denial on regular/duplicated-pin FDs; the retained `/dev/null` pin returning device `ENOTTY`; and x32 child termination with `SIGSYS`.

This establishes the sampled CPU fixture boundary in the observed environment. It explicitly records no imports after restriction, no proc-content grant and no GPU compatibility scope. It does not establish every metadata syscall variant by runtime test, every alternate architecture by execution, native library compatibility, data/model/numerical parity or v2 chroot/mount absence. Exact syscall/source policy remains bound by the sealed worker, rather than inferred only from samples.

Passed receipt SHA-256: `7449b81ebe66571e58cb0f59d75e148b4e04bdd19bc293ca1c627af78fc46e7e`. Worker SHA-256: `2d0c1c18ae17432c20ded3cc509a90db5c5d3de9471b43f6a766e16e6b1973c4`.

The earlier readonly-stdin refusal and all immutable v1/v2 preparation files remain preserved. This interpretation only reads the root receipt JSON; it does not run the diagnostic or broaden its acceptance.
