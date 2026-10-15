# Native BUDDY private BatchNorm running state

This standalone port uses unchanged `private_bn_state_v3/banks.py` with four
route-private running means, variances and counters. The wrapped session owns
the banks and includes them with the complete native state in its checkpoint.
The original BUDDY model owns the same learned parameters, BN affine objects,
canonical buffer objects and Adam optimizer groups. The molecular attachment
is not used. The original shared-running-BN native baseline remains separate.

Native BUDDY TRAIN and VALID enter the original source route and then the
private bank context. Only bn_labels (16) and bn_feats (1024) execute; bn_RA (1)
is registered but inactive, and no embedding BN exists. The four bank sets add
33,408 buffer bytes and zero learned parameters. Original mixed 1024-query BCE,
query hydration, graph/hash inputs, route RNG, optimizer and full VALID
evaluator/diagnostic loss remain intact. Direct model/BN/sampler calls and
model-only state export/load are rejected outside the owned session lifecycle.

The seed0/year2007 materialized context authenticates the first two original
TRAIN batches and complete VALID inputs. The 1.230 GB context stays remote.
The first two TRAIN transitions are the proposed bounded qualification scope;
source support beyond that is not a full epoch or scientific qualification.

SMOKE_PLAN.json specifies a zero-step original baseline oracle followed by two
candidate transitions, three complete VALID calls and fresh restore. It counts
three constructions, 36 native forwards and three reverse collections. The
one first-old-state output/gradient comparison uses the predeclared 1e-6
tolerance; it does not authorize repeated backward or tolerance searches.
The supervisor requires root admission of the exact plan SHA256 and verifies
the source manifest. No native execution has been admitted for this port.

No CUDA/pair-resource/TEST/full-campaign readiness follows from source checks.
Post-attachment device migration, activation recomputation, SyncBN and
concurrent route contexts require separate support.
