# Synthetic CPU attempt v1

One authorized run completed on the verified anogena-2 route with CUDA hidden and OMP/MKL threads 1. Supervisor PID 400083, suite PID 400085. PyTorch 2.1.2+cu118 / NumPy 1.26.4 reported CUDA unavailable. Seven cases passed and three failed. All staged/source/constant bytes remained unchanged.

Elapsed supervisor wall time: 3.440172s; sum of case wall durations: 0.578519s; failed-case wall durations: 0.418682s. These are elapsed measurements, not process CPU or energy measurements.

The checkpoint failure is a harness factory dtype mismatch: an FP64 fixture is restored through a default-FP32 ToyTeacher factory. Native state serialization and transported Adam assertions before the failing strict dtype assertion had passed; optimizer/RNG checks after that assertion were not reached.

Both full-control tests demanded rank3 from a topology whose TRAIN nodes form an independent set. TRAIN injection/remasking makes the S term zero, so four cubic VJP columns lie in span(g, JᵀP_T S²r, JᵀP_T S³r); removing g leaves rank at most two. The source returned valid rank2/null receipts. The end-to-end selector and five-arm returned-state assertions were never reached.

These failures establish harness oracle errors, without establishing a selector defect. The exact v1 packet and failure results remain preserved. Root has separately authorized a minimal v2 fixture correction and one distinct CPU run. No actual-warm or predictive qualification is claimed.
