# Native NCN expansion ports and finite CPU checks

New explicit callback for the three frozen C6 readouts and corrected
endpoint/query responsibility v2. The existing native NCN session still owns
negative sampling, the native shuffled 65,536-positive minibatch plus 65,536
negatives, adjacency masking, native four-route source forwards, dropout/RNG
and the original Adam groups. No native source file, runtime, frozen bridge or
pilot is changed. Additional task-call observers authenticate actual ordered
query endpoints and logits against the native sampler.

C6 methods keep the frozen detached confidence/JS scorer and exact uniform
start. Mean/global/conditional add 0/4/20 parameters to the existing predictor
head group. Every block receives one J=.5F+.5L_actual_served backward and one
native optimizer transition. F and L use the source positive mean plus negative
mean reduction. Whole VALID returns the actual served readout for selection
and the same-bank mean as a diagnostic from the same route forwards.

Corrected responsibility methods retain the source risk F for shared native
parameters and the weighted risk W for existing private BE factors. Unit,
query_iid and endpoint use the frozen v2 sampler; the nonself query control has
two independent Exp halves and self-loops one draw. Weights normalize within
the original positive/negative source groups. The two reverse collections use
one old state and one original optimizer transition. Raw marginal matching
does not establish matched normalized marginals or unbiased original risk.

`callback.py:create_adapter` pins the actual new manifest, callback, original
native bridge and frozen helper source receipts. `METHODS.json` lists exact
method IDs. The source session, sampler/private RNG, module modes, readout
parameters and native optimizer state remain in the complete wrapped
checkpoint. Method/source changes are rejected during restoration.

Portable input v2 already has its own native link callback. Its two methods
are checked through that unchanged callback rather than duplicated here.

## Scope

The new C6/responsibility port accepts NCN/Collab only. NCNC recursive task
scope and BUDDY joint source reduction/hashing ports remain pending. Existing
portable mappings describe those backbones, but this package does not qualify
them. No all-link readiness, predictive benefit or convergence claim follows.

`qualify_NCN_cpu.py` performs exactly two native-size CPU updates, three whole
VALID calls (60,084 positive plus100,000 negative queries each), a serialized
checkpoint and fresh-source restoration per method. `supervise_cpu.py` runs
the eight fixed methods sequentially with two CPU threads, a32GiB address-space
cap per child and1,200-second timeout. No full TRAIN epoch, scientific fit,
CUDA training, TEST scoring or final TEST reconstruction is authorized here.
The initial result and all failures/costs live under separate qualification
outputs. Source preparation alone does not establish a successful check.
