#!/usr/bin/env bash
# Source preparation only. This file is intentionally not an active launcher.
printf '%s\n' 'DISABLED: root source review, reader v3 export and owner/children terminal release required.' >&2
exit 64
# After independent root release, invoke the sealed collect.py directly using
# the existing native_ncn_runtime_20261005_v1 interpreter, repository cwd,
# CUDA_VISIBLE_DEVICES=GPU-44039938-fd82-41d2-fefd-de71514e2fac and OMP/MKL=2.
# Supply a separate root-authored release file and its exact SHA256.
# No training driver, live-training supervisor, automatic retry or resume.
