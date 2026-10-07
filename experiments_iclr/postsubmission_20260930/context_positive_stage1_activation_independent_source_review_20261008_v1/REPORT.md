# Narrow Stage1 activation source review

8 October 2026. Reviewed current owner **`fb0033a1be86236e765d693e7e724218a08d5ed822f4991e29e65d9498bc78d6`**, schedule **`049c9d84388406cb9ddaf6d5a12f297fef5192a7efcd876c16d912a3780fa404`**, and run-cell **`7ddf8bf3f0ca51d15aa1c32171968ad0a942b6eeff9390bb13acb555e5b7890e`**.

**The material lifecycle defect found in the original preparation is resolved in the reviewed error/termination paths. No further material source defect identified within this narrow scope.** Original owner `b7cf754e…` could leave a spawned cell active when metadata writes or waits failed. The corrected owner retains active child/identity/cell before metadata writes, routes SIGTERM/SIGINT into failure cleanup, ignores repeated interrupts while cleaning up, and stops/reaps using owned PID/start-tick checks. Active state remains until the EXIT receipt is written. Cleanup outcome is retained in OWNER_FAILURE. Root preserved the original owner/schedule; this review changed neither.

The source/metadata checks confirm:

- Exactly nine unique common/route/permuted ×8101/8203/8307 cells, matching the frozen protocol. All release hashes/identities match; all six input bindings agree across cells, including frozen target archive/receipt/roles. Stage2 is unadmitted and retries/TEST are disabled.
- `run_cell.py` puts the v2 directory on `sys.path` before `runpy`; the driver then supplies its public/helper import paths. Every required v2 CLI input is forwarded, including targets, preprocessing receipt and role manifest. Run-cell imposes the measured Torch settings, explicit physical GPU and 20GiB allocator cap.
- The owner waits for 24GiB fresh GPU memory, runs one child at a time, applies the 32,390-second active wait plus bounded cleanup, and scopes TERM/KILL to the captured owned group. Failed/timed-out cells stop the family without retry. Success requires exit zero, full1100 epochs/steps, exact arm/seed, TEST closed and selected-checkpoint digest before family closure.
- The bound exact qualification copy matches the releases: complete, ten discarded native updates, about57.1seconds, peak reserved11.3867GiB, and no development/TEST labels or selected checkpoint scoring. This establishes finite engineering eligibility rather than scientific utility.
- The owner reads completion/custody metadata and checkpoint hashes; it does not deserialize checkpoints, inspect prediction tensors or open partial quality traces/logs.

Stdlib source/metadata inspection only: no framework invocation, server access, scientific role/target/checkpoint/fit-score access, new tests or publication. Full-family success and outcome opening remain future root-controlled events.
