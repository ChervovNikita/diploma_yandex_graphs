# Native DBLP full-graph CPU resource preparation

This source-only packet prepares one bounded resource qualification of the root-adopted native GAT, Simple-HGN and SeHGNN study. It does not authorize remote qualification or study training. QUALIFICATION_RELEASE_TEMPLATE.json has qualification_authorized false and execution_authorized false; the root must issue a separate exact qualification release.

## Existing evidence and exact protocol

The native v2 source manifest is 5956ed69b1c9646aedda0ce67d950081b23ca6ca77796ac7f98c4b900544af43. The native root freeze is 274b293cefa8dc8257ede17a0e5e6b1fb2fa1b53392ce0a34227bf87fc46b87a. The actually fetched original native CPU receipt is bound at SHA-256 641e83f82e44cd77eb13b8b184bf992eae1f81bffba47adaa43b29bb92be1a91.

That receipt records PASS for the earlier synthetic actual-author correspondence and 24 synthetic optimizer updates. No numerical fixtures are repeated here. Those synthetic checks do not qualify the memory or preprocessing cost of the full released DBLP graph.

qualify_native.py imports the byte-bound v2 input preprocessing, models and driver. It calls the original train_epoch directly, once for each arm at seed 131. Model construction, reseeding, Adam settings, SeHGNN shuffled TRAIN loader, CPU-disabled AMP and global RNG checkpoint representation match v2. No fit, predict_validation, validation_metrics or training main is called.

The graph includes the exact released node/link members and all admitted A/P/T attributes. GAT and Simple-HGN retain the original implicit identity feature projection and the original homogeneous edge order, relation labels, loops and message chunks. SeHGNN retains the exact normalized destination-first SciPy adjacency convention, five feature channels and four complete normalized label-product channels.

The existing explicit source pool physically contains TRAIN and VAL labels. Only the 974 frozen seed-131 TRAIN labels are retained as optimization targets or propagated one-hot inputs. The 243 VAL memberships are used as IDs only; no validation target tensor, validation loss/F1, test label member or test diagnostic is constructed. The only ZIP members opened are node/link.

SeHGNN inference uses sorted TRAIN, sorted VAL, then the sorted target complement derived from topology as one 4,057-node batch. Its class BatchNorm has track_running_stats false, so this full-batch composition is required even for unscored qualification. The other BatchNorm modules keep their original TRAIN update and inference behavior. All native inference is FP32 without autocast. Whole-target logits are checked for shape/finite values and saved as disposable unscored artifacts; they are not selected predictors or study fits.

## Bounds and measurements

The provisional qualification ceiling is one CPU process/thread, 16 GiB address space, 12 GiB observed RSS, 600 CPU seconds (605 hard limit) and 600 wall seconds. The remote launcher applies RLIMIT_AS and RLIMIT_CPU in preexec_fn before Python/Torch import; the child verifies those exact limits. The parent monitors RSS/wall time and reaps a killed child. A post-run process peak RSS check also closes qualification if the 12 GiB resident budget was exceeded.

Qualification requires at least 24 GiB current host/cgroup headroom, an available CPU affinity and a one-minute load no higher than 75% of affinity count. This separate screen accounts for current HGT workers through actual cgroup usage; it is not a claim about simultaneous native study capacity. These caps have not yet been measured on the full native graph and are not implicit permission to relax them.

Timing records separate source attribute streaming, homogeneous graph preparation, normalized adjacencies, feature propagation, full sparse metapath products, per-seed TRAIN label propagation, each exact native TRAIN epoch, unscored whole-target inference and disposable state/logit writes. Only all three qualified arms with original preservation produce a resource forecast. Failures/deferments and all three terminal slots remain in the transport receipt; no successful subset is used to decide competence or scientific merit.

The exact native v2 serial study has 15 cases and at most 4,000 epochs: five paired seeds × (300 GAT + 300 Simple-HGN + 200 SeHGNN). After measured qualification, its forecast includes preprocessing once, each seed's TRAIN label channels, model construction, maximum-cap epoch costs and one selected-state inference replay. A factor-two planning estimate is reported separately; it is not a guaranteed bound. The single-update probe does not establish later optimizer/checkpoint/allocator residency throughout a complete fit. Source-validation metric scalar overhead and later CPU/filesystem variation are unmeasured.

## Complete study plan and custody

EXECUTION_PLAN.json recommends the unchanged serial native v2 train_native.py initially. It shares the original full-graph preprocessing once and requires all 15 frozen seed/arm terminals, the original selections/replay and preserved sources. It does not reduce controls, select successful cases, retune the freeze, open TEST or fit calibration.

The resource result also estimates a prospective five-worker seed-block option and projects five times the measured process peak RSS. A parallel native schedule would need a separate minimal scheduler, source review, canonical all-15 closure, selected artifact hash/byte bindings and fresh root execution release. No parallel scheduler or study execution release is prepared in this packet.

BINDINGS.json binds 47 original source/provenance/CPU receipt records plus both root freezes. All own payloads, those sources, admitted data descriptors and the qualification release are verified before/after qualification. Disposable states include model/Adam/global RNG/parameter names but no label payload, validation selection or continuation admission.

run_cpu_remote.py with --admission pointing to the root qualification release and --receipt to a fresh local path uploads every own sealed payload plus manifest/seal and exact root metadata, preserving any already existing bytes. It uses only the authorized SSH account and canonical repository, checks the exact single GPU UUID with a read-only inventory query, records current Git HEAD and runs CPU with CUDA hidden. It makes no package/environment changes, GPU computation, tensor download or study launch. Its local result is metadata/logs/resource evidence only.

Local preparation checks are stdlib syntax, source fingerprints, frozen metadata equality, actual CPU receipt binding and root admission acceptance/rejection. No Torch, real dataset labels, numerical fixture or remote action is executed in preparation.
