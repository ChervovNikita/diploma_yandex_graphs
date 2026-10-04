# Native update and exposure schedule

The unchanged `native/run_lp.py::build_loaders` calls `reset_samples(epoch=epoch, seed=configs.seed)` for each epoch. Native sample reset seeds Python, NumPy, Torch and CUDA with `seed * 100 + epoch`. TRAIN generator/worker seeding uses that same value; its `DistributedSampler` shuffles and receives `set_epoch(epoch)`. VALID generator/worker seeds use the fit seed and its sampler is sequential.

The native 50% positive cycle uses generator seed `seed * 100 + 50 * epoch // 100`. For 811,405 filtered positive records, `int(round(811405 * 0.5))` is 405,702. Two consecutive cyclic slices therefore cover 811,404 records from that permutation and leave one out of the cycle. The harness retains this native rounding/exposure. Native global PyG negative sampling requests one negative per selected positive; theoretical underfill is recorded and retained without redraw. It does not add a future-positive rejection rule.

The unchanged `train_loop` zeroes gradients at epoch start, casts loss to float32 and divides by 8, then updates AdamW on each eighth batch and the final partial accumulation. The final partial retains divisor 8. There is no learning-rate scheduler; AdamW uses learning rate 1e-4 and weight decay 0.01.

The completed seed 0 epoch 0 resource run had 793 batches, hence `ceil(793 / 8) = 100` optimizer updates. A full fit has **2,000 updates only if every actual epoch has 793 batches**. The scientific worker records each actual loader length, mixed query population, optimizer-step count, sample/order digests and native counters, and verifies `ceil(actual_batches / 8)` every epoch. It reports whether the 793-batch / 2,000-update condition was observed rather than assuming it.

Native dataset construction initially uses its default seed before `utils.set_seed(fit_seed)` and the real native `get_feature_dim` sample. That ordering is retained. Every fit constructs a fresh scratch model and optimizer. Saving selected state does not feed state into the next epoch or the next fit.
