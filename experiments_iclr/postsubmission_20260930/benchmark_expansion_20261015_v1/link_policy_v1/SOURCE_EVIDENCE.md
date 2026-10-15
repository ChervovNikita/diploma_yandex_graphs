# Saved source evidence used for link policies

All paths below are relative to the phase directory. Files were read as text or
JSON; no native program was imported or executed. Their exact byte/SHA receipts
are in `BINDINGS.json` and are rechecked before constructing a policy.

## Frozen roster

`reference_wrapper_redesign_20261013_v1/ROSTER.json` has 27 recipes. Its three
link recipes are `collab-ncn`, `collab-ncnc` and `collab-buddy`, each on official
`ogbl-collab`. All three have:

- `tracks.native_replica.maximum=100`, no source patience, constant native
  optimizer groups and warmup0.
- `tracks.long_convergence.maximum=1000`, minimum100 and patience50 VALID
  checks, gated at `unit >= minimum_units`.
- Maximum raw `hits@50`, first ties, no checkpoint0 and interval1.
- Native source settings, repetition cohorts0..9/split0 and separate
  independent seed offsets `[0,100003,200003,300007]`.

These are metadata contracts; their original runtime-status fields remained
pending. `source_bindings.native_program` pins the actual saved native main.

## NCN/NCNC

`reference_wrapper_redesign_20261013_v1/native_sources/ncn/ncn__NeighborOverlap.py`:

- Lines20–23: source seeds Torch CPU/CUDA and NumPy.
- Lines84–155: source `test` predicts TRAIN, positive/negative VALID, then
  positive/negative TEST; it optionally computes embeddings on `full_adj_t`
  for TEST and reports Hits20/50/100.
- Lines256–262: repetition index seeds and a new `bestscore=None` per run.
- Lines280–291: constant Adam encoder/predictor groups and training epochs
  `range(1,1+args.epochs)`, with unconditional evaluation each epoch.
- Lines294–306: first result initializes bestscore; later updates require
  `valid_hits > bestscore[key][1]`.
- Lines323–325: Collab reports the validation-selected Hits50 pair.

`graph_link_competing_family_source_20261003_v1/excerpts/ncn__README.md.txt`
retains source README label0052: the Collab author command specifies
`--epochs 100 --runs 10`, original batch65536, testbs131072 and
`--use_valedges_as_input`. Generic parser defaults are not used as the Collab
horizon.

`reference_wrapper_redesign_20261013_v1/native_sources/ncn/ncn__utils.py`,
lines8–17: TRAIN PermIterator uses randperm and eval uses arange. This narrow
fact does not establish that all omitted source calls or source construction
have no RNG effect.

## BUDDY

`reference_wrapper_redesign_20261013_v1/native_sources/buddy/buddy_src_runners_run.py`:

- Lines37–48: source repetition seed function and cuDNN settings.
- Lines56–66: source seed before data/loaders/model construction and
  `val_res=test_res=best_epoch=0`.
- Lines70–80: full source epoch loop, evaluation every `eval_steps`, the
  TRAIN/VALID/TEST callback and strict `tmp_val_res > val_res` update. No
  early-stopping break appears in the saved main.
- Line204: default epochs100. Line225: default eval_steps1.

`link_covariance_idea_v1/sources/buddy_readme.md`, Experiments section, gives
the Collab BUDDY command with year2007/K50/lr.02 and cache flag. It has no epoch
or eval_steps override. Its repetition guidance adds `--reps 10`.

`reference_wrapper_redesign_20261013_v1/native_sources/buddy/buddy_src_data.py`,
lines44–64: BUDDY TRAIN shuffles; Collab VALID and TEST loaders shuffle; for
Collab `train_eval_loader=train_loader`. This establishes a concrete loader/RNG
difference from the adapted ordered-VALID-only integration. The policy adds no
TEST loader, fake evaluation pass or guessed RNG draw replacement.

## Existing adapted integration

`benchmark_expansion_20261015_v1/link_native_integration_v1/MANIFEST.json` is
pinned without modification. Its source-only freeze explicitly leaves actual
native numerical/full-epoch/resource qualification and link policy/dispatch
pending. This new package supplies the explicit policy component; it does not
amend that historical manifest or certify the remaining numerical work.
