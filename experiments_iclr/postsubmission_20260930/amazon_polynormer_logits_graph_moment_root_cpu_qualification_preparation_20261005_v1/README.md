# Standalone CPU numerical qualification preparation

**Source only. Nothing has been executed.** This synthetic engineering qualifier accepts an explicit sealed source directory and its root-bound manifest/seal hashes. It does not hardbind source V1, load scientific data, import original models/loaders, initialize a GPU, or mutate the selected source. Python bytecode writing is disabled before source imports. Source V2 is the currently recommended corrected target.

The fixed checks cover:

1. Five seeded SPD four-variable QPs versus independent SciPy SLSQP solutions.
2. Known inverse-diagonal solutions at scales 1e-8, 1e-12, **1e-14**, 1e-16 and 1e-20, also compared with an independently scale-normalized SciPy optimizer.
3. Semidefinite identical/duplicate-member hull projection, including known minimum-norm weights and a density-bound solution.
4. Direct member-error Gram reconstruction.
5. Extreme stable log-mixtures, exact zero-residual skip and revival of a class whose probability underflows.
6. A tiny FP32 single-logit gap demonstrating separate graph probability class and native raw-logit scoring class. M4 retains probability-mean scoring.
7. All three declared head parameter counts.
8. One tiny discarded synthetic residual fit: exactly 150 updates, initial no-op included, retained minimum permitted training objective independently recomputed. This is charged as engineering work, separate from the scientific 90/36 budget.
9. Seven-node sparse restart diffusion against an independent finite-power expression; independent C&S Y−q correction and label reset; transported Grams/zero-mass fallback; forbidden anchor overlap rejection. `numerical.N=7` is temporary in the qualification process and restored in `finally`.

A fresh root-authorized process can invoke:

```sh
python qualify_numerical.py \
  --source /absolute/phase/amazon_polynormer_logits_graph_moment_source_preparation_20261005_v2 \
  --manifest-sha256 af15ac11349ac7d9dbc8608b4362d461195409ca6348fdfc67e7ac7b4a273b21 \
  --seal-sha256 2ed8fef0d08e420e140ddcddfa2929e00849797715d16e1e0be626e9cb2ef900 \
  --output /absolute/new/root/qualification/output
```

No numerical package or sealed source is imported during this preparation. The runner authenticates the selected source before imports, sets BLAS/OpenMP/PyTorch CPU threads to one, records Python/NumPy/SciPy/Torch versions and BLAS configuration, and checks that CUDA was not initialized. All numerical arrays/labels are generated explicitly or from fixed synthetic RNG seeds. No scientific custody/data loader is called.

Actual execution creates per-check records, `REPORT.json`, and `QUALIFICATION_RESULT.json`. Only complete success emits `PASS_SYNTHETIC_NUMERICAL_QUALIFICATION`; any failure emits `FAIL_SYNTHETIC_NUMERICAL_QUALIFICATION`, preserves its traceback and costs, and exits nonzero. Both results bind exact source hashes and the qualifier code hash. A passed qualifier grants no scientific execution or final-label authority. The root must admit the development run separately.

Only stdlib AST and source integrity checks have been performed for this preparation packet. The selected source V2 and preserved V1 have not been numerically run by this agent.
