| p | v7 Cascade p_block (fail/tot) | v7 Cascade P_L/cycle [95% CI] (fail/tot) | BP+OSD p_block (fail/tot) | BP+OSD P_L/cycle [95% CI] (fail/tot) | v6 Cascade p_block (fail/tot, ref) | ratio (Cascade× better, p_block) |
|---|---|---|---|---|---|---|
| 0.002 | 0.0003793 (200/527232) | 2.635e-06 [2.29e-06, 3.03e-06] (200/527232) | 0.003333 (8/2400) | 2.319e-05 [1.17e-05, 4.57e-05] (8/2400) | 0.4023 (206/512) | 8.8x |
| 0.003 | 0.005766 (200/34688) | 4.016e-05 [3.50e-05, 4.61e-05] (200/34688) | 0.03611 (130/3600) | 2.557e-04 [2.15e-04, 3.04e-04] (130/3600) | 0.4023 (206/512) | 6.3x |
| 0.004 | 0.04223 (200/4736) | 3.001e-04 [2.61e-04, 3.45e-04] (200/4736) | 0.165 (132/800) | 1.260e-03 [1.06e-03, 1.49e-03] (132/800) | 0.4141 (212/512) | 3.9x |
| 0.005 | 0.2061 (211/1024) | 1.615e-03 [1.41e-03, 1.85e-03] (211/1024) | 0.3933 (236/600) | 3.534e-03 [3.09e-03, 4.02e-03] (236/600) | 0.4824 (247/512) | 1.9x |
| 0.0055 | 0.35 (224/640) | 3.038e-03 [2.65e-03, 3.47e-03] (224/640) | 0.56 (280/500) | 5.876e-03 [5.18e-03, 6.64e-03] (280/500) | 0.5117 (262/512) | 1.6x |

\* = point flagged insufficient (n_failures < min_errors). ratio = BP+OSD p_block / v7 Cascade p_block (>1 means Cascade wins; this is the block-level win margin, distinct from the iter-6 table's P_L/cycle-based ratio). v7 Cascade = mixed-p 40k eval, EMA weights (results/bb144_v7_mixp.json). v6 Cascade column is reference-only (results/bb144_mw_v4.json, the flat ~0.40 floor iter-7 fixes) and is not part of the ratio. BP+OSD = results/bb144_mw_bposd_p{P}.json (p=0.003 uses the *_merged.json top-up); only decoder="BP+OSD" entries are used, never the live-probe "Cascade" entry embedded in those files.
