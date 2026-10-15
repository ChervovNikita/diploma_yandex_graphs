# Separate NCNC native expansion port

This generation adds NCNC compatibility without editing the finished NCN
package, native source, frozen link bridge, runtime or pilot. It reuses the
same six C6/responsibility IDs and the unchanged portable input v2 callback's
two methods. BUDDY remains a separate pending task.

The C6/responsibility observer uses the existing source TaskCallScope depth,
and checks it against its own nesting clock. Only calls entering at source
depth0 are authenticated top-level task outputs. Native recursive completion
still executes unchanged; recursive outputs never become task loss rows or
gate inputs. Per-update receipts compare eight observed task calls and ignored
recursive calls with the source's own dispatch counts. Native output factors
remain restricted to task depth1, including no-grad evaluation.

NCNC keeps the source65,536-positive plus65,536-negative minibatch and positive
mean plus negative mean reduction. C6 uses the frozen detached prediction
features and J=.5F+.5L_actual_served. Responsibility uses the corrected v2
sampler and shared-F/private-W credit. No sampler, adjacency, native recursion,
source optimizer options or BN/dropout policy changes. Fresh checkpoints bind
the complete source task scope, sampler, global/private RNG and method/source.

The finite CPU checker performs two source-sized updates and three full VALID
calls per method, plus fresh-source restoration. It retains computed VALID
diagnostics and checkpoints after both transitions. Eight fixed methods run
sequentially with two CPU threads, a64GiB address-space cap per child and an
1,800-second timeout. The supervisor stops on the first failed case and never
retries it automatically. No old NCN check is duplicated. Actual qualification
is recorded separately; source preparation alone is not a passing result.

No full fit, completed TRAIN epoch, GPU training, TEST access, server77 access,
convergence/quality claim or campaign admission is provided by this package.
