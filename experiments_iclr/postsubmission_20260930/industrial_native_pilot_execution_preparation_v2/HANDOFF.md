# Root execution handoff

All commands below are future execution instructions for the separately authorized root. This source author has not run them. Keep sources sealed and put runtime/input/output artifacts in separate new repo-owned directories. The archive remains with the custodian. No extension or final-label phase is provided.

## 1. Verify and assemble the allowlist runtime

Read-only source verification can be run with an ordinary local Python; it imports no scientific module:

```sh
python3 "$packet/check_sources.py"
```

Use the separately provisioned Python 3.12.9 prefix and candidate environment bound in `CANDIDATE_RUNTIME.json`. Root already observed Torch/DGL 2.4.0+cu118, torchdata .8.0 and the NumPy/CUDA capability in that environment. Choose and review only necessary loader/system/NVIDIA driver library files or dedicated library trees. The assembler accepts explicit `--library-file`/`--library-tree` repetitions under `/lib*`, `/usr/lib*` or `/usr/local/nvidia/lib*`. If a loader symlink is selected, copy its actual target as another explicit library input when needed; no missing-link substitution is automatic.

```sh
python3 "$packet/build_runtime_image.py" \
  --python-prefix "$dedicated_python_prefix" \
  --environment "$candidate_environment" \
  --image "$new_runtime_image" \
  --library-file "$reviewed_loader_or_driver_library"
```

All prefix contents become part of the library manifest, so root must verify prefix custody first. The image is not a venv bind or a host-root snapshot. Interpreter and pyvenv launch metadata are relocated to `/opt/python` and `/opt/native`; scientific package bytes are preserved. Source preparation makes no claim that the complete image is yet operational. Missing relocation/shared-library dependencies stop the launch and remain a recorded blocker.

## 2. Freeze public roles, then observe worker isolation

Public extraction opens only `info.yaml`, `features.csv`, `edgelist.csv` and `split_masks_RL.csv` in the archive. It preserves released RL train/val/test roles, zero-membership rows and raw node-ID ordering; overlap stops the phase.

```sh
"$custodian_python" "$packet/custodian_export.py" public \
  --archive "$custodian_archive" --output "$new_closed_input_bundle"
```

Create a new empty dedicated probe output directory. The namespace-root argument must be a new path under the packet's project parent, outside packet/input/output directories. For CUDA, supply exactly the physical `/dev/nvidiaN` approved for UUID `GPU-44039938-fd82-41d2-fefd-de71514e2fac`; root determines the actual minor from its authorized device route. Do not assume `/dev/nvidia0`.

```sh
python3 "$packet/launch_isolated.py" isolation-probe \
  --runtime-image "$new_runtime_image" --inputs "$new_closed_input_bundle" \
  --outputs "$new_probe_output" --namespace-root "$new_probe_namespace_root" \
  --device cuda:0 --gpu-device "$approved_physical_gpu_node" \
  --gpu-uuid GPU-44039938-fd82-41d2-fefd-de71514e2fac
```

This mode imports no scientific package and performs zero model updates. Inspect `isolation_probe/receipt.json`, the retained phase result and root launcher stdout/stderr. Only actual observed excluded paths/mounts/capability state certify the worker boundary. The probe binds current source seal, exact public roles/members, runtime image, device node minor and UUID. Only a user namespace was tested before this v2 preparation; the full mount/network/PID/chroot/capability route is still a runtime gate.

CPU probe/qualification uses `--device cpu` without GPU arguments and certifies only CPU scope. A CPU certificate cannot authorize CUDA fits.

## 3. Separate archive-side source-label export

After the real public-only worker probe passes, use the separately controlled archive-side custodian process and the same candidate pandas/NumPy versions. It must have no model or test-report task. The preserved public-role bytes precede target open; CSV tokenization skips all non-source rows before numeric parsing. Only official train/val source buffers receive the native pandas float32 parser. No test missingness/class counts/labels/statistics are emitted.

```sh
"$custodian_python" "$packet/custodian_export.py" labels \
  --archive "$custodian_archive" --output "$new_closed_input_bundle" \
  --isolation-probe "$new_probe_output/isolation_probe/receipt.json"
```

The bundle must contain exactly frozen role metadata, four public files, two compact train/val packs and two bound receipts. The worker never receives the archive or raw targets. The archive-side process is root's custody obligation; its receipt flags are not a security substitute. No new custodian namespace launcher is requested or supplied.

## 4. Disposable full-graph qualification

Create a separate new empty science output volume and a new namespace-root path. Use the same reviewed image/input/device scope as the probe.

```sh
python3 "$packet/launch_isolated.py" qualify \
  --runtime-image "$new_runtime_image" --inputs "$new_closed_input_bundle" \
  --outputs "$new_science_output" --namespace-root "$new_qualify_namespace_root" \
  --device cuda:0 --gpu-device "$approved_physical_gpu_node" \
  --gpu-uuid GPU-44039938-fd82-41d2-fefd-de71514e2fac
```

The worker verifies actual OS observations before package/model imports, then exact source/input/image/installed dependency closure. All three configurations use cold seed 17 and the complete admitted graph. Qualification compares public preprocessing/oracle assembly, source masks/topology/eid order, native/Torch full outputs and gradients, populated-history AdamW state/options/parameters/logits, K1/K4 identity/private-bias transport and K1 Torch JVP/VJP/central-FD. It schedules seven disposable optimizer steps per config. Progress/check JSONL is written incrementally; failures retain paid-step counters, errors, elapsed/RSS evidence and forbid dependent native fits. No useful qualification checkpoint is retained.

The fixed tolerances are logits/parameters/moments atol 1e-6/rtol 1e-5; gradients atol 1e-6/rtol 1e-4; dual rtol 2e-4; FD rtol .05 under the unchanged borrowed rule requiring at least one of eps 1e-3 or 3e-4 per fixed direction. There is no fallback on OOM, unsupported deterministic primitives or failure. Preserve root launch failures even if the interpreter cannot reach its phase receipt.

## 5. Exactly nine native fits

Only a complete qualification `certificate.json` on the identical source/input/runtime/device may enter the grid. The science output directory must contain only `qualification` before this phase. Use a new namespace-root path.

```sh
python3 "$packet/launch_isolated.py" run-nine \
  --runtime-image "$new_runtime_image" --inputs "$new_closed_input_bundle" \
  --outputs "$new_science_output" --namespace-root "$new_grid_namespace_root" \
  --device cuda:0 --gpu-device "$approved_physical_gpu_node" \
  --gpu-uuid GPU-44039938-fd82-41d2-fefd-de71514e2fac
```

All nine cells are registered before the first fit. A failure marks later cells not started due to dependency failure, retains paid updates/checkpoints/traces and forbids selection. A completed matrix writes strict-best selection, competence, reused three-model validation probabilities and total useful-update/cost identities. A failed competence screen is a complete first-stage result and forbids later initializer extensions. Test remains CLOSED in every phase.

The first-stage packet has no resume/retry CLI that can erase a failed attempt. A changed implementation/environment or necessary correction requires a separate source version and fresh qualified identity before dependent scientific work.
