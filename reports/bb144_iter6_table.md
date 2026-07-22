| p | Cascade P_L/cycle [95% CI] (fail/tot) | BP+OSD P_L/cycle [95% CI] (fail/tot) | ratio (Cascade× better) |
|---|---|---|---|
| 0.002 | 3.641e-03 [3.16e-03, 4.18e-03] (206/512) | 2.319e-05 [1.17e-05, 4.57e-05] (8/2400) | 0.0064x |
| 0.003 | 3.641e-03 [3.16e-03, 4.18e-03] (206/512) | 2.557e-04 [2.15e-04, 3.04e-04] (130/3600) | 0.07x |
| 0.004 | 3.784e-03 [3.29e-03, 4.34e-03] (212/512) | 1.260e-03 [1.06e-03, 1.49e-03] (132/800) | 0.33x |
| 0.005 | 4.684e-03 [4.10e-03, 5.32e-03] (247/512) | 3.534e-03 [3.09e-03, 4.02e-03] (236/600) | 0.75x |
| 0.0055 | 5.110e-03 [4.49e-03, 5.79e-03] (262/512) | 5.876e-03 [5.18e-03, 6.64e-03] (280/500) | 1.1x |

\* = point flagged insufficient (n_failures < min_errors). ratio = BP+OSD P_L / Cascade P_L (>1 means Cascade wins). Cascade = 40k eval, best.pt @ step 38000, EMA weights (results/bb144_mw_v4.json, job 185232).
