# Fixed CPU molecular bridge qualification

Authorization: parent follow-up after C6 source preparation requested a bounded
actual MolBACE/GIN allocation CPU check for all three fixed controls. Frozen
method manifest: `edef4b1f552a495e09e64855afbd96c7eb14ce5ef02d26cc13ec530d80e9805f`.
Seed offset1301009 was fixed and approved before any native execution.

Use only literal guarded allocation anogena-2-0 and GPU UUID ending
fefd-de71514e2fac; the model runs on CPU with CUDA_VISIBLE_DEVICES empty and
two Torch/OMP/MKL threads. Use the original native callback/data/config/seed
from `results/molbace_pilot_v1/fits/004_69902ee9d5d7427d/JOB.json`.

For mean/global/conditional separately, construct fresh native source GIN,
check exact initial mean and isolated gate RNG, then perform one32-TRAIN-graph
optimizer update. Evaluate complete official VALID151 using actual served
readout and same-bank mean. Save full wrapped source/optimizer/gate/mode/RNG
state. Construct a fresh matching session, restore, require bit-exact VALID
for both readouts and unchanged native Parameter identity. Restore RNG and
perform the second optimizer update on the same32 real TRAIN graphs, followed
by full VALID151. Two updates per method, maximum six total; six native
constructions and nine complete VALID calls if all pass.

The fixed32-graph batch is reused for engineering and explicitly bypasses
normal epoch sampling. No convergence, controller/pairing, GPU-memory,
predictive benefit, scientific fit or selection claim. No TEST access,
original paper rerun, worker interference, capacity fit, automatic retry or
node/link qualification. Preserve every attempt, failure, optimizer count,
wall/CPU time, peak RSS and checkpoint receipt in this separate package.
Frozen C6 method files stay byte-identical.
