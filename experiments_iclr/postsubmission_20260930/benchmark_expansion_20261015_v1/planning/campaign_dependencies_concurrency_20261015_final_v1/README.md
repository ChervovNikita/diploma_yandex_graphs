# Campaign dependencies, storage and concurrency

Planning package for the unchanged 6,460-fit / 270-export campaign. The dormant queues assign 2,154 fits to allocation and 2,153 to each server51 GPU. This package starts no children, transfers no bytes, changes no scientific source, and confers no admission. Frozen PubMed pilots must drain before a successor concurrency test.

## Transfer inputs

| Inventory | Unique files | Bytes |
|---|---:|---:|
| Core graph, TRAIN/VALID roles and polynomial bases | 21 | 16,346,414,909 |
| Core files outside saved server51 receipt coverage | 19 | 16,029,302,220 |
| Native author snapshot | 19 | 213,253 |
| Local partial source closure, including native snapshot | 187 | 2,395,845 |

The two covered core files are PubMed graph and Cheb basis. Historical receipts do not establish present destination state. The two `CORE_DATA_*_TRANSFER_PLAN.json` files use the parent's stream format and canonical repository-relative `experiments_iclr/...` paths. Provider raw/preprocessing/release files and final role indices are additional dependencies.

`dependency_plan.py` deduplicates exact identities and refuses conflicts. With `--release`, it verifies the pinned normalized `RELEASE_UNADMITTED.json` and extracts its complete **declared** source closure. Without this argument it labels the local view partial. Optional `--queue` accepts the exact pinned server51 dormant queues and extracts source descriptors. `--inventory-providers` inventories existing provider `raw/**`, preprocessing markers and release markers by byte hashes. No dataset payload is decoded.

```
python3 dependency_plan.py --phase-root "$PHASE_PATH" --output "$NEW_OUTPUT" \
  --release "$EXACT_RELEASE" --inventory-providers \
  --queue "$EXACT_SERVER51_GPU0_QUEUE" --queue "$EXACT_SERVER51_GPU1_QUEUE"
```

The parent staged an earlier byte version of this script while metadata inventory proceeded. Keep that staged copy pinned; the independent `inventory_role_indices.py` companion inventories `split/**` and `split_dict.pt`, including TEST indices and preferred PT shadows, without replacing the running planner:

```
python3 inventory_role_indices.py --phase-root "$PHASE_PATH" \
  --provider-request "$PROVIDER_REQUEST" --output "$NEW_ROLE_PLAN"
```

The immutable molecular training provider reads only TRAIN/VALID roles and does not call `get_idx_split`; final selected serving separately requires original `test.csv.gz`. Link training requires exactly graph/TRAIN/VALID, while final serving needs original `test.pt`. Byte hashing/transferring those original index files does not perform held-out inference. `RUNTIME_AND_PROVIDER_REQUIREMENTS.json` records the source evidence. BUDDY additionally requires **datasketch 1.5.9**, absent from the saved server51 package inventory; fresh verification is needed. Unused original entrypoint imports are not automatically installation requirements.

## Conditional workload slices

Use `resource_scenarios_v2/TIME_SLICES.json`; v1 is retained as earlier arithmetic. These are explicit scenarios, not a universal campaign ETA.

| Representative setting | Measured evidence | Conditional burden |
|---|---|---|
| MolHIV GIN-VN conditional shared bank | 582.446 s TRAIN, 1,029 updates, one epoch | 16.18 TRAIN h per 100-epoch fit; 190 shared fits = 3,074 h |
| Eight MolHIV presets using that same shared-fit cost | Architecture/method substitution | 1,520 shared fits × 100 epochs = 24,592 TRAIN h; ideal equal three-GPU division = 341.6 days |
| MolBACE old GIN/drop0.5 pilot | Native 4.448 s/epoch; two shared classes 10.037–11.537 s/epoch | One 240-fit preset at 100 epochs = 59.1–67.1 worker h if those rates cover every included method |
| BUDDY current two-epoch lifecycle | 1,066.69 s, 3,170 updates, extra constructors/VALID/restores | Repeating that cadence: 14.82 h/100 epochs; 190 shared fits = 2,815 worker h |
| NCNC v6 two-epoch lifecycle | 180.20 s, 34 updates, lifecycle overhead | Repeating that cadence: 2.50 h/100 epochs; 180 shared fits = 450.5 worker h |
| Current PubMed-Cheb native pilot | 02→03 saved observation interval | 2,099 epochs: allocation 10.19 min; server51 GPUs 4.33 / 4.32 min |

MolHIV TRAIN scenarios exclude VALID, source TRAIN diagnostics, finalization, failures and retries; their native-reference/body fits are also outside the shared-fit totals. BUDDY/NCNC repeat-cadence scenarios include lifecycle overhead and are not mature epoch measurements. PubMed rates do not predict structured methods or other datasets. Dividing by three assumes equal speed, occupancy and no contention, which the observed host rates do not establish. JSON also includes 300/1,000-epoch sensitivity. A one-cohort, 27-preset / 646-fit tranche remains an optional proposal; no pruning or task selection is enacted.

## Disk admission

Both `selected.pt` and `recovery.pt` include optimizer state. Atomic replacement can require an extra full checkpoint. Measured MolHIV checkpoint bytes are 41,041,118 each, plus 331,564 for selected VALID: **82,413,800 retained payload bytes per observed fit**. Applying that one profile to all 4,306 server51 fits gives **354.9 GB before logs and other profiles**; this is a sensitivity scenario. PubMed native currently has 7,182,766-byte checkpoints and a 2.68 MB growing curve; no final export had been observed in that sample.

The prior free-space snapshot was 202.44 GB. The parent later reported about 700.89 GB after the user freed space. The latter improves feasibility but is not a current reservation. `resource_scenarios_v2/STORAGE_BUDGET.json` records observed sizes, all 27 legacy checkpoint priors, dense-Adam formulas, and their limits.

`admission.py disk INPUT.json` checks **incremental** remaining growth of active jobs and the next candidate, unfinished transfer bytes, temporary overlap, peer growth, a 20 GiB suggested floor, and an explicit retained-byte cap. Already retained/transferred bytes are not reserved twice. An unknown profile is held for measurement; insufficient headroom pauses new claims. No automatic deletion or change to current pilots is implemented. `DISK_INPUT_EXAMPLE.json` is illustrative, with provisional log allowance and a 500 GiB cap; its bytes must be replaced by current measurements and an accepted host budget.

## Two jobs per GPU

`CONCURRENCY_PLAN.json` specifies finite PubMed native+structured, molecular, and BUDDY+NCNC candidate tests after drain. It reuses valid lifecycle evidence and measures missing solo-versus-pair throughput, process memory, RAM/cache, CPU/IO, checkpoint and export growth. The BUDDY observed process peak is 4.92 GB; the historical 40 GiB repair reservation is not treated as demand. NCNC's 26.70 GB allocator reserve still requires an applicable total-process envelope.

`admission.py pair INPUT.json` computes summed solo-equivalent progress in a common interval while both children are active. Proposed acceptance requires at least **1.15× aggregate throughput**, each child's progress at least 0.40× solo, and all source, correctness, peer, memory and disk checks. Full-phase makespan must confirm gains survive constructor/VALID/save/export work. One optional repeat resolves ambiguity. Only the tested source/profile/host/GPU/thread tuple becomes an admission candidate; other pairs stay at one child.

Eight metadata tests passed. No new native run or remote access occurred in this package. Source transfer completion, full provider/source/descriptor closure, fresh host capacity, and execution admission remain the parent's recorded actions.
