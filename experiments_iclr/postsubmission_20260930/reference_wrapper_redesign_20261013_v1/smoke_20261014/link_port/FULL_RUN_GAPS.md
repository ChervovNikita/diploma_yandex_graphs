# Remaining link full-run requirements

The 15 initial link smoke cases passed before the preserved bias/task-scope
corrections. The parent is rerunning all 15 corrected cases. Short TRAIN maps,
complete VALID forwards and checkpoint replay do not establish complete epochs,
source horizons, convergence, selected states, or final benchmark serving.

1. **Bind the campaign worker/factories.** Campaign contracts declare
   `runtime.source_native_adapter:ncn/ncnc/buddy`, `native_config`, and
   `initialization_seed`; the smoke frontend accepts `architecture`,
   `native_fixed_config`, and `_seed`. No full campaign source-native worker is
   present. Implement an explicit contract-to-session binding that preserves
   fixed settings, source seed policy, data receipts and the distinct tracks.
   Locations: [campaign_compiler.py](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/engineering/campaign_compiler.py:72>),
   [create_session](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:530>).

2. **Expose completed native epochs.** Current `epoch` increments at epoch
   start, and `_next_batch` starts the following epoch while returning another
   update. Define explicit epoch length/completion and resume progress before
   placing validation or counting the horizon: NCN/NCNC use
   `floor(TRAIN_positive_count / 65536)` optimizer maps and drop the tail;
   BUDDY uses `ceil((positive_count + negative_count) / 1024)` maps including
   its source partial tail. Keep negative/permutation redraws at their native
   epoch boundary and retain native epoch-loss reporting reductions.
   Locations: [NCN sampler](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:326>),
   [BUDDY sampler](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:443>).

3. **Connect selected checkpoints and stopping.** Run the source track for
   100 complete epochs without early stopping. Run the longer track with cap
   1000, minimum 100 and patience 50 whole-VALID checks. Select maximum
   `hits@50`, keeping the first tie, after each completed epoch. Persist best
   complete model/buffer state, selector/stale counters, completed/selected
   epochs, curves, stop reason and costs together with session/global RNG.
   The selector primitive exists but is not connected to the smoke sessions.
   Locations: [training contract](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/README.md:107>),
   [LongerTrackSelector](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/engineering/convergence_policy.py:7>),
   [session state](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:247>).

4. **Give I4 members independent selectors/stopping.** The smoke bank advances
   all four members and returns empty aggregate `metrics`; member validation
   evidence is retained, but independent selected checkpoints/stopping states
   are absent. Select and stop each member from its own complete VALID metric,
   serve the mean of those individually selected raw logits, and reuse exactly
   member zero for the matched M1. Do not select one pooled bank epoch.
   Locations: [independent updates](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/independent_bank.py:77>),
   [independent validation](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/independent_bank.py:83>),
   [selection policy](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/README.md:113>).

5. **Add final selected-state evaluation.** The smoke loader intentionally
   admits GRAPH/TRAIN/VALID only; its VALID forwards use TRAIN topology. A full
   runner needs a separate final evaluator with all official TEST queries and
   negatives, the selected checkpoint, and native TEST graph construction:
   unweighted TRAIN+VALID input for NCN/NCNC; source weighted year-filtered
   TRAIN+VALID topology, propagated features and hashes for BUDDY. The current
   VALID path cannot serve as that evaluator. No TEST operation is authorized
   or performed by this document.
   Locations: [source topology contract](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/README.md:132>),
   [NCN VALID](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:367>),
   [BUDDY VALID](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:487>).

6. **Restore native cache custody and qualify durable resume.** BUDDY's fixed
   declaration has `cache_subgraph_features=True`, while smoke hydration
   recomputes native query features on every TRAIN/VALID call. Values remain
   source-native, but storage/timing is a disclosed smoke adaptation. Materialize
   and authenticate full-run native caches and their exact graph/query/config
   bindings, or explicitly retain that adaptation in the full-run contract and
   cost accounting. Complete-epoch rollover, partial-tail behavior, fresh-process
   reconstruction, selector restoration and stopping need qualification beyond
   the short replay checks.
   Locations: [streamed hydration](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/smoke_20261014/link_port/callbacks.py:461>),
   [native cache preprocessing](</Users/alex/Documents/ChatGPT/anogena allocation/postsubmission_research_20260930/reference_wrapper_redesign_20261013_v1/native_sources/buddy/buddy_src_datasets_elph.py:175>).

The bias formula and NCNC task-only output-hook discrepancies identified during
this review are corrected separately in `FIXES_20261014.md`. This list records
the remaining full-run work; it does not amend source settings or authorize fits.
