# Real training-only spectral collection

same PPI task; not six independent tasks or predictive evaluation.

| Cell | Layer ranks | Added trainable scalars | Frozen basis scalars |
| --- | --- | ---: | ---: |
| seed0_label10 | 0, 1, 1 | 768 | 256 |
| seed0_label100 | 3, 2, 2 | 2688 | 896 |
| seed1_label10 | 0, 0, 0 | 0 | 0 |
| seed1_label100 | 2, 1, 1 | 1536 | 512 |
| seed2_label10 | 1, 1, 0 | 768 | 256 |
| seed2_label100 | 1, 3, 7 | 4224 | 1408 |

Each adapter also stores12 common contrast scalars. The candidate keeps full-width member states; these counts do not imply faster inference.

All100%label cells select4-11total directions;10%cells select0-2. One10%seed selects none. No F2/validation/test outcome has been read.

All six records passed exact replay under the frozen rule (n0=1, cap16, positive discovery and screening). This tests code and local training-gradient evidence only. It supplies no accuracy result, final optimizer guarantee, or established novelty. F2 is still unopened and its forthcoming audit remains a training-only local proxy on two graphs.
