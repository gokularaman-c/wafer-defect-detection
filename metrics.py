# training_plots.py
# Usage:
# 1) Paste your full console log into the log_text variable below and run.
# 2) PNGs will be saved next to this script: loss_curve.png, acc_curve.png, f1_curve.png
# 3) metrics.csv will contain the parsed epoch-level metrics.

import re
import math
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 1) Paste your full training log here ──────────────────────────────────────
log_text = r"""
Epoch 01/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.8986 acc 0.668
  [train] batch 200/874 (22.9%) | loss 0.7233 acc 0.718
  [train] batch 300/874 (34.3%) | loss 0.6581 acc 0.735
  [train] batch 400/874 (45.8%) | loss 0.6015 acc 0.756
  [train] batch 500/874 (57.2%) | loss 0.5527 acc 0.773
  [train] batch 600/874 (68.6%) | loss 0.5184 acc 0.784
  [train] batch 700/874 (80.1%) | loss 0.4894 acc 0.794
  [train] batch 800/874 (91.5%) | loss 0.4683 acc 0.801
Epoch 01 | train loss 0.4530 acc 0.806 || val loss 0.1845 acc 0.922 macro-F1 0.769
  ↳ saved new BEST (macro-F1 0.769)
Epoch 02/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.2480 acc 0.876
  [train] batch 200/874 (22.9%) | loss 0.2434 acc 0.881
  [train] batch 300/874 (34.3%) | loss 0.2381 acc 0.885
  [train] batch 400/874 (45.8%) | loss 0.2387 acc 0.886
  [train] batch 500/874 (57.2%) | loss 0.2383 acc 0.886
  [train] batch 600/874 (68.6%) | loss 0.2346 acc 0.888
  [train] batch 700/874 (80.1%) | loss 0.2332 acc 0.888
  [train] batch 800/874 (91.5%) | loss 0.2301 acc 0.889
Epoch 02 | train loss 0.2277 acc 0.890 || val loss 0.2601 acc 0.954 macro-F1 0.897
  ↳ saved new BEST (macro-F1 0.897)
Epoch 03/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.1812 acc 0.912
  [train] batch 200/874 (22.9%) | loss 0.1857 acc 0.909
  [train] batch 300/874 (34.3%) | loss 0.1868 acc 0.909
  [train] batch 400/874 (45.8%) | loss 0.1853 acc 0.910
  [train] batch 500/874 (57.2%) | loss 0.1850 acc 0.910
  [train] batch 600/874 (68.6%) | loss 0.1834 acc 0.910
  [train] batch 700/874 (80.1%) | loss 0.1824 acc 0.910
  [train] batch 800/874 (91.5%) | loss 0.1806 acc 0.911
Epoch 03 | train loss 0.1808 acc 0.911 || val loss 0.1486 acc 0.970 macro-F1 0.911
  ↳ saved new BEST (macro-F1 0.911)
Epoch 04/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.1610 acc 0.921
  [train] batch 200/874 (22.9%) | loss 0.1647 acc 0.918
  [train] batch 300/874 (34.3%) | loss 0.1601 acc 0.920
  [train] batch 400/874 (45.8%) | loss 0.1609 acc 0.919
  [train] batch 500/874 (57.2%) | loss 0.1600 acc 0.920
  [train] batch 600/874 (68.6%) | loss 0.1596 acc 0.919
  [train] batch 700/874 (80.1%) | loss 0.1591 acc 0.920
  [train] batch 800/874 (91.5%) | loss 0.1576 acc 0.921
Epoch 04 | train loss 0.1577 acc 0.921 || val loss 0.0805 acc 0.965 macro-F1 0.893
Epoch 05/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.1362 acc 0.929
  [train] batch 200/874 (22.9%) | loss 0.1352 acc 0.930
  [train] batch 300/874 (34.3%) | loss 0.1349 acc 0.931
  [train] batch 400/874 (45.8%) | loss 0.1364 acc 0.930
  [train] batch 500/874 (57.2%) | loss 0.1365 acc 0.930
  [train] batch 600/874 (68.6%) | loss 0.1367 acc 0.930
  [train] batch 700/874 (80.1%) | loss 0.1361 acc 0.931
  [train] batch 800/874 (91.5%) | loss 0.1371 acc 0.931
Epoch 05 | train loss 0.1365 acc 0.931 || val loss 0.0679 acc 0.972 macro-F1 0.938
  ↳ saved new BEST (macro-F1 0.938)
Epoch 06/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.1169 acc 0.939
  [train] batch 200/874 (22.9%) | loss 0.1200 acc 0.939
  [train] batch 300/874 (34.3%) | loss 0.1155 acc 0.941
  [train] batch 400/874 (45.8%) | loss 0.1177 acc 0.940
  [train] batch 500/874 (57.2%) | loss 0.1201 acc 0.939
  [train] batch 600/874 (68.6%) | loss 0.1211 acc 0.939
  [train] batch 700/874 (80.1%) | loss 0.1216 acc 0.939
  [train] batch 800/874 (91.5%) | loss 0.1208 acc 0.939
Epoch 06 | train loss 0.1198 acc 0.940 || val loss 0.0839 acc 0.981 macro-F1 0.948
  ↳ saved new BEST (macro-F1 0.948)
Epoch 07/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0880 acc 0.950
  [train] batch 200/874 (22.9%) | loss 0.0981 acc 0.949
  [train] batch 300/874 (34.3%) | loss 0.1003 acc 0.948
  [train] batch 400/874 (45.8%) | loss 0.1004 acc 0.947
  [train] batch 500/874 (57.2%) | loss 0.1035 acc 0.946
  [train] batch 600/874 (68.6%) | loss 0.1042 acc 0.945
  [train] batch 700/874 (80.1%) | loss 0.1047 acc 0.945
  [train] batch 800/874 (91.5%) | loss 0.1040 acc 0.946
Epoch 07 | train loss 0.1060 acc 0.945 || val loss 0.0555 acc 0.981 macro-F1 0.941
Epoch 08/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0884 acc 0.955
  [train] batch 200/874 (22.9%) | loss 0.0860 acc 0.955
  [train] batch 300/874 (34.3%) | loss 0.0894 acc 0.953
  [train] batch 400/874 (45.8%) | loss 0.0918 acc 0.952
  [train] batch 500/874 (57.2%) | loss 0.0902 acc 0.953
  [train] batch 600/874 (68.6%) | loss 0.0912 acc 0.953
  [train] batch 700/874 (80.1%) | loss 0.0923 acc 0.952
  [train] batch 800/874 (91.5%) | loss 0.0934 acc 0.952
Epoch 08 | train loss 0.0931 acc 0.952 || val loss 0.0449 acc 0.984 macro-F1 0.959
  ↳ saved new BEST (macro-F1 0.959)
Epoch 09/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0712 acc 0.965
  [train] batch 200/874 (22.9%) | loss 0.0705 acc 0.963
  [train] batch 300/874 (34.3%) | loss 0.0710 acc 0.963
  [train] batch 400/874 (45.8%) | loss 0.0747 acc 0.960
  [train] batch 500/874 (57.2%) | loss 0.0782 acc 0.959
  [train] batch 600/874 (68.6%) | loss 0.0800 acc 0.958
  [train] batch 700/874 (80.1%) | loss 0.0820 acc 0.958
  [train] batch 800/874 (91.5%) | loss 0.0827 acc 0.958
Epoch 09 | train loss 0.0834 acc 0.958 || val loss 0.0521 acc 0.978 macro-F1 0.929
Epoch 10/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0614 acc 0.963
  [train] batch 200/874 (22.9%) | loss 0.0742 acc 0.959
  [train] batch 300/874 (34.3%) | loss 0.0720 acc 0.960
  [train] batch 400/874 (45.8%) | loss 0.0706 acc 0.962
  [train] batch 500/874 (57.2%) | loss 0.0737 acc 0.960
  [train] batch 600/874 (68.6%) | loss 0.0742 acc 0.960
  [train] batch 700/874 (80.1%) | loss 0.0768 acc 0.959
  [train] batch 800/874 (91.5%) | loss 0.0763 acc 0.959
Epoch 10 | train loss 0.0751 acc 0.960 || val loss 0.0810 acc 0.984 macro-F1 0.929
Epoch 11/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0514 acc 0.971
  [train] batch 200/874 (22.9%) | loss 0.0563 acc 0.971
  [train] batch 300/874 (34.3%) | loss 0.0577 acc 0.970
  [train] batch 400/874 (45.8%) | loss 0.0602 acc 0.968
  [train] batch 500/874 (57.2%) | loss 0.0609 acc 0.968
  [train] batch 600/874 (68.6%) | loss 0.0627 acc 0.967
  [train] batch 700/874 (80.1%) | loss 0.0636 acc 0.967
  [train] batch 800/874 (91.5%) | loss 0.0654 acc 0.966
Epoch 11 | train loss 0.0660 acc 0.966 || val loss 0.0590 acc 0.979 macro-F1 0.955
Epoch 12/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0475 acc 0.971
  [train] batch 200/874 (22.9%) | loss 0.0565 acc 0.969
  [train] batch 300/874 (34.3%) | loss 0.0593 acc 0.968
  [train] batch 400/874 (45.8%) | loss 0.0607 acc 0.968
  [train] batch 500/874 (57.2%) | loss 0.0602 acc 0.968
  [train] batch 600/874 (68.6%) | loss 0.0591 acc 0.968
  [train] batch 700/874 (80.1%) | loss 0.0601 acc 0.968
  [train] batch 800/874 (91.5%) | loss 0.0604 acc 0.968
Epoch 12 | train loss 0.0610 acc 0.968 || val loss 0.0560 acc 0.988 macro-F1 0.969
  ↳ saved new BEST (macro-F1 0.969)
Epoch 13/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0516 acc 0.972
  [train] batch 200/874 (22.9%) | loss 0.0533 acc 0.971
  [train] batch 300/874 (34.3%) | loss 0.0545 acc 0.971
  [train] batch 400/874 (45.8%) | loss 0.0549 acc 0.971
  [train] batch 500/874 (57.2%) | loss 0.0543 acc 0.971
  [train] batch 600/874 (68.6%) | loss 0.0552 acc 0.971
  [train] batch 700/874 (80.1%) | loss 0.0550 acc 0.971
  [train] batch 800/874 (91.5%) | loss 0.0547 acc 0.971
Epoch 13 | train loss 0.0551 acc 0.971 || val loss 0.0376 acc 0.986 macro-F1 0.959
Epoch 14/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0416 acc 0.979
  [train] batch 200/874 (22.9%) | loss 0.0413 acc 0.978
  [train] batch 300/874 (34.3%) | loss 0.0408 acc 0.978
  [train] batch 400/874 (45.8%) | loss 0.0463 acc 0.975
  [train] batch 500/874 (57.2%) | loss 0.0467 acc 0.975
  [train] batch 600/874 (68.6%) | loss 0.0459 acc 0.975
  [train] batch 700/874 (80.1%) | loss 0.0458 acc 0.975
  [train] batch 800/874 (91.5%) | loss 0.0466 acc 0.975
Epoch 14 | train loss 0.0475 acc 0.974 || val loss 0.0478 acc 0.986 macro-F1 0.957
Epoch 15/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0347 acc 0.980
  [train] batch 200/874 (22.9%) | loss 0.0411 acc 0.977
  [train] batch 300/874 (34.3%) | loss 0.0437 acc 0.976
  [train] batch 400/874 (45.8%) | loss 0.0432 acc 0.977
  [train] batch 500/874 (57.2%) | loss 0.0441 acc 0.976
  [train] batch 600/874 (68.6%) | loss 0.0430 acc 0.977
  [train] batch 700/874 (80.1%) | loss 0.0423 acc 0.977
  [train] batch 800/874 (91.5%) | loss 0.0441 acc 0.976
Epoch 15 | train loss 0.0448 acc 0.976 || val loss 0.0671 acc 0.988 macro-F1 0.967
Epoch 16/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0345 acc 0.979
  [train] batch 200/874 (22.9%) | loss 0.0372 acc 0.979
  [train] batch 300/874 (34.3%) | loss 0.0356 acc 0.979
  [train] batch 400/874 (45.8%) | loss 0.0372 acc 0.978
  [train] batch 500/874 (57.2%) | loss 0.0355 acc 0.979
  [train] batch 600/874 (68.6%) | loss 0.0365 acc 0.979
  [train] batch 700/874 (80.1%) | loss 0.0393 acc 0.978
  [train] batch 800/874 (91.5%) | loss 0.0401 acc 0.978
Epoch 16 | train loss 0.0402 acc 0.978 || val loss 0.0700 acc 0.989 macro-F1 0.969
  ↳ saved new BEST (macro-F1 0.969)
Epoch 17/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0307 acc 0.983
  [train] batch 200/874 (22.9%) | loss 0.0320 acc 0.982
  [train] batch 300/874 (34.3%) | loss 0.0340 acc 0.982
  [train] batch 400/874 (45.8%) | loss 0.0358 acc 0.981
  [train] batch 500/874 (57.2%) | loss 0.0379 acc 0.980
  [train] batch 600/874 (68.6%) | loss 0.0371 acc 0.981
  [train] batch 700/874 (80.1%) | loss 0.0383 acc 0.980
  [train] batch 800/874 (91.5%) | loss 0.0392 acc 0.980
Epoch 17 | train loss 0.0395 acc 0.980 || val loss 0.0335 acc 0.988 macro-F1 0.967
Epoch 18/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0218 acc 0.988
  [train] batch 200/874 (22.9%) | loss 0.0202 acc 0.989
  [train] batch 300/874 (34.3%) | loss 0.0248 acc 0.987
  [train] batch 400/874 (45.8%) | loss 0.0292 acc 0.984
  [train] batch 500/874 (57.2%) | loss 0.0314 acc 0.983
  [train] batch 600/874 (68.6%) | loss 0.0332 acc 0.982
  [train] batch 700/874 (80.1%) | loss 0.0357 acc 0.981
  [train] batch 800/874 (91.5%) | loss 0.0362 acc 0.981
Epoch 18 | train loss 0.0353 acc 0.981 || val loss 0.0901 acc 0.985 macro-F1 0.971
  ↳ saved new BEST (macro-F1 0.971)
Epoch 19/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0234 acc 0.986
  [train] batch 200/874 (22.9%) | loss 0.0223 acc 0.986
  [train] batch 300/874 (34.3%) | loss 0.0224 acc 0.986
  [train] batch 400/874 (45.8%) | loss 0.0273 acc 0.985
  [train] batch 500/874 (57.2%) | loss 0.0297 acc 0.984
  [train] batch 600/874 (68.6%) | loss 0.0317 acc 0.983
  [train] batch 700/874 (80.1%) | loss 0.0318 acc 0.982
  [train] batch 800/874 (91.5%) | loss 0.0306 acc 0.983
Epoch 19 | train loss 0.0299 acc 0.983 || val loss 0.0358 acc 0.990 macro-F1 0.975
  ↳ saved new BEST (macro-F1 0.975)
Epoch 20/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0276 acc 0.986
  [train] batch 200/874 (22.9%) | loss 0.0346 acc 0.982
  [train] batch 300/874 (34.3%) | loss 0.0329 acc 0.983
  [train] batch 400/874 (45.8%) | loss 0.0316 acc 0.984
  [train] batch 500/874 (57.2%) | loss 0.0327 acc 0.984
  [train] batch 600/874 (68.6%) | loss 0.0320 acc 0.984
  [train] batch 700/874 (80.1%) | loss 0.0333 acc 0.983
  [train] batch 800/874 (91.5%) | loss 0.0340 acc 0.983
Epoch 20 | train loss 0.0342 acc 0.983 || val loss 0.0362 acc 0.991 macro-F1 0.968
Epoch 21/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0299 acc 0.984
  [train] batch 200/874 (22.9%) | loss 0.0284 acc 0.985
  [train] batch 300/874 (34.3%) | loss 0.0254 acc 0.987
  [train] batch 400/874 (45.8%) | loss 0.0243 acc 0.987
  [train] batch 500/874 (57.2%) | loss 0.0247 acc 0.987
  [train] batch 600/874 (68.6%) | loss 0.0251 acc 0.987
  [train] batch 700/874 (80.1%) | loss 0.0258 acc 0.987
  [train] batch 800/874 (91.5%) | loss 0.0258 acc 0.987
Epoch 21 | train loss 0.0269 acc 0.986 || val loss 0.0699 acc 0.982 macro-F1 0.967
Epoch 22/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0307 acc 0.981
  [train] batch 200/874 (22.9%) | loss 0.0260 acc 0.986
  [train] batch 300/874 (34.3%) | loss 0.0280 acc 0.985
  [train] batch 400/874 (45.8%) | loss 0.0291 acc 0.985
  [train] batch 500/874 (57.2%) | loss 0.0277 acc 0.985
  [train] batch 600/874 (68.6%) | loss 0.0277 acc 0.985
  [train] batch 700/874 (80.1%) | loss 0.0285 acc 0.985
  [train] batch 800/874 (91.5%) | loss 0.0285 acc 0.985
Epoch 22 | train loss 0.0296 acc 0.984 || val loss 0.0368 acc 0.990 macro-F1 0.971
Epoch 23/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0206 acc 0.989
  [train] batch 200/874 (22.9%) | loss 0.0203 acc 0.988
  [train] batch 300/874 (34.3%) | loss 0.0205 acc 0.988
  [train] batch 400/874 (45.8%) | loss 0.0192 acc 0.989
  [train] batch 500/874 (57.2%) | loss 0.0207 acc 0.988
  [train] batch 600/874 (68.6%) | loss 0.0232 acc 0.987
  [train] batch 700/874 (80.1%) | loss 0.0235 acc 0.987
  [train] batch 800/874 (91.5%) | loss 0.0242 acc 0.987
Epoch 23 | train loss 0.0263 acc 0.987 || val loss 0.0430 acc 0.990 macro-F1 0.970
Epoch 24/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0240 acc 0.988
  [train] batch 200/874 (22.9%) | loss 0.0219 acc 0.988
  [train] batch 300/874 (34.3%) | loss 0.0184 acc 0.990
  [train] batch 400/874 (45.8%) | loss 0.0183 acc 0.990
  [train] batch 500/874 (57.2%) | loss 0.0193 acc 0.989
  [train] batch 600/874 (68.6%) | loss 0.0212 acc 0.988
  [train] batch 700/874 (80.1%) | loss 0.0230 acc 0.987
  [train] batch 800/874 (91.5%) | loss 0.0232 acc 0.987
Epoch 24 | train loss 0.0236 acc 0.987 || val loss 0.0567 acc 0.989 macro-F1 0.976
  ↳ saved new BEST (macro-F1 0.976)
Epoch 25/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0233 acc 0.988
  [train] batch 200/874 (22.9%) | loss 0.0252 acc 0.988
  [train] batch 300/874 (34.3%) | loss 0.0227 acc 0.989
  [train] batch 400/874 (45.8%) | loss 0.0224 acc 0.989
  [train] batch 500/874 (57.2%) | loss 0.0229 acc 0.989
  [train] batch 600/874 (68.6%) | loss 0.0234 acc 0.988
  [train] batch 700/874 (80.1%) | loss 0.0252 acc 0.987
  [train] batch 800/874 (91.5%) | loss 0.0256 acc 0.987
Epoch 25 | train loss 0.0261 acc 0.987 || val loss 0.0893 acc 0.988 macro-F1 0.970
Epoch 26/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0148 acc 0.992
  [train] batch 200/874 (22.9%) | loss 0.0131 acc 0.993
  [train] batch 300/874 (34.3%) | loss 0.0165 acc 0.991
  [train] batch 400/874 (45.8%) | loss 0.0190 acc 0.990
  [train] batch 500/874 (57.2%) | loss 0.0204 acc 0.989
  [train] batch 600/874 (68.6%) | loss 0.0207 acc 0.989
  [train] batch 700/874 (80.1%) | loss 0.0205 acc 0.989
  [train] batch 800/874 (91.5%) | loss 0.0201 acc 0.989
Epoch 26 | train loss 0.0194 acc 0.989 || val loss 0.0394 acc 0.994 macro-F1 0.986
  ↳ saved new BEST (macro-F1 0.986)
Epoch 27/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0209 acc 0.992
  [train] batch 200/874 (22.9%) | loss 0.0195 acc 0.991
  [train] batch 300/874 (34.3%) | loss 0.0187 acc 0.991
  [train] batch 400/874 (45.8%) | loss 0.0179 acc 0.991
  [train] batch 500/874 (57.2%) | loss 0.0174 acc 0.991
  [train] batch 600/874 (68.6%) | loss 0.0184 acc 0.991
  [train] batch 700/874 (80.1%) | loss 0.0175 acc 0.991
  [train] batch 800/874 (91.5%) | loss 0.0190 acc 0.990
Epoch 27 | train loss 0.0201 acc 0.990 || val loss 0.0601 acc 0.989 macro-F1 0.980
Epoch 28/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0255 acc 0.989
  [train] batch 200/874 (22.9%) | loss 0.0268 acc 0.987
  [train] batch 300/874 (34.3%) | loss 0.0311 acc 0.986
  [train] batch 400/874 (45.8%) | loss 0.0282 acc 0.986
  [train] batch 500/874 (57.2%) | loss 0.0251 acc 0.988
  [train] batch 600/874 (68.6%) | loss 0.0235 acc 0.988
  [train] batch 700/874 (80.1%) | loss 0.0243 acc 0.988
  [train] batch 800/874 (91.5%) | loss 0.0236 acc 0.988
Epoch 28 | train loss 0.0239 acc 0.988 || val loss 0.0425 acc 0.993 macro-F1 0.985
Epoch 29/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0159 acc 0.990
  [train] batch 200/874 (22.9%) | loss 0.0135 acc 0.992
  [train] batch 300/874 (34.3%) | loss 0.0147 acc 0.991
  [train] batch 400/874 (45.8%) | loss 0.0186 acc 0.990
  [train] batch 500/874 (57.2%) | loss 0.0204 acc 0.989
  [train] batch 600/874 (68.6%) | loss 0.0204 acc 0.989
  [train] batch 700/874 (80.1%) | loss 0.0193 acc 0.990
  [train] batch 800/874 (91.5%) | loss 0.0188 acc 0.990
Epoch 29 | train loss 0.0189 acc 0.990 || val loss 0.1314 acc 0.984 macro-F1 0.959
Epoch 30/30 — batches: 874, batch_size: 64
  [train] batch 100/874 (11.4%) | loss 0.0173 acc 0.990
  [train] batch 200/874 (22.9%) | loss 0.0179 acc 0.990
  [train] batch 300/874 (34.3%) | loss 0.0171 acc 0.991
  [train] batch 400/874 (45.8%) | loss 0.0158 acc 0.991
  [train] batch 500/874 (57.2%) | loss 0.0182 acc 0.990
  [train] batch 600/874 (68.6%) | loss 0.0190 acc 0.990
  [train] batch 700/874 (80.1%) | loss 0.0196 acc 0.990
  [train] batch 800/874 (91.5%) | loss 0.0195 acc 0.990
Epoch 30 | train loss 0.0189 acc 0.990 || val loss 0.0231 acc 0.995 macro-F1 0.988
  ↳ saved new BEST (macro-F1 0.988)

"""

# ── 2) Parser: extract epoch-level metrics from lines like:
# "Epoch 02 | train loss 0.2277 acc 0.890 || val loss 0.2601 acc 0.954 macro-F1 0.897"
EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+)(?:/\d+)?\s*\|\s*train\s+loss\s+([0-9.]+)\s+acc\s+([0-9.]+)\s*\|\|\s*val\s+loss\s+([0-9.]+)\s+acc\s+([0-9.]+)\s+macro-F1\s+([0-9.]+)",
    flags=re.IGNORECASE
)

def parse_log(log: str) -> pd.DataFrame:
    data = []
    for m in EPOCH_LINE.finditer(log):
        epoch = int(m.group(1))
        train_loss = float(m.group(2))
        train_acc = float(m.group(3))
        val_loss = float(m.group(4))
        val_acc = float(m.group(5))
        macro_f1 = float(m.group(6))
        data.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "macro_f1": macro_f1,
        })
    if not data:
        raise ValueError("No epoch lines found. Make sure your log lines match the expected format.")
    df = pd.DataFrame(data).sort_values("epoch").reset_index(drop=True)
    return df

# Optional: simple moving average smoothing (for prettier curves)
def smooth(x: List[float], window: int = 1) -> np.ndarray:
    if window <= 1: return np.asarray(x)
    pad = window - 1
    arr = np.pad(np.asarray(x), (pad, 0), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")

def mark_best(ax, x, y, mode="max"):
    if mode == "max":
        idx = int(np.argmax(y))
    else:
        idx = int(np.argmin(y))
    ax.scatter([x[idx]], [y[idx]], s=80, marker="o")
    ax.annotate(f"best @ epoch {x[idx]}: {y[idx]:.4f}",
                (x[idx], y[idx]),
                textcoords="offset points", xytext=(6, 8))
    return idx

def plot_line(x, ys, labels, title, ylabel, outfile, smooth_window=1):
    plt.figure(figsize=(8,5))
    for y, label in zip(ys, labels):
        y_plot = smooth(y, smooth_window)
        plt.plot(x, y_plot, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

def main():
    df = parse_log(log_text)
    df.to_csv("metrics.csv", index=False)

    epochs = df["epoch"].tolist()

    # ── Loss curves ────────────────────────────────────────────────────────────
    plot_line(
        epochs,
        [df["train_loss"].tolist(), df["val_loss"].tolist()],
        ["train loss", "val loss"],
        "Training vs Validation Loss",
        "Loss",
        "loss_curve.png",
        smooth_window=1,  # change to 3 for gentle smoothing
    )

    # ── Accuracy curves ────────────────────────────────────────────────────────
    plot_line(
        epochs,
        [df["train_acc"].tolist(), df["val_acc"].tolist()],
        ["train acc", "val acc"],
        "Training vs Validation Accuracy",
        "Accuracy",
        "acc_curve.png",
        smooth_window=1,
    )

    # ── Macro-F1 curve with best marker ───────────────────────────────────────
    plt.figure(figsize=(8,5))
    f1 = df["macro_f1"].to_numpy()
    plt.plot(epochs, f1, label="macro-F1")
    best_idx = mark_best(plt.gca(), np.array(epochs), f1, mode="max")
    plt.title("Validation Macro-F1 over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("f1_curve.png", dpi=160)
    plt.close()

    print("Saved: metrics.csv, loss_curve.png, acc_curve.png, f1_curve.png")
    print(f"Best macro-F1 at epoch {df.loc[best_idx, 'epoch']}: {df.loc[best_idx, 'macro_f1']:.4f}")

if __name__ == "__main__":
    main()


# Confusion matrix (needs: y_true, y_pred, class_names)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(y_true, y_pred, class_names, normalize=False, outfile="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(7,6))
    disp.plot(values_format=".2f" if normalize else "d", cmap=None, colorbar=False)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()