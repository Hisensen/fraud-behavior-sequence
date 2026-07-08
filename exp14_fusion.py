# -*- coding: utf-8 -*-
"""思想14·Multi-view Fusion: 各思想的分数做秩融合, 量化"多视角增益"。"""
import os

import numpy as np
from scipy.stats import rankdata

import bp_common as C

d = C.load_art()
print("== 思想14 Multi-view Fusion ==", flush=True)
views = {"MEM惊讶度": d["mem_topk"]}
for n, f in (("密度GMM", "e1_gmm"), ("原型距离", "e2_proto"),
             ("对比kNN", "e4_coles"), ("二元语法", "e5_bigram"),
             ("motif罕见", "e6_rare"), ("DTW轨迹", "e8_dtw"),
             ("嵌入kNN", "e9_knn"), ("VAE重构", "e12_vae")):
    p = f"blueprint_scores/{f}.npy"
    if os.path.exists(p):
        views[n] = np.load(p)

te = d["test"]
rk = {n: rankdata(s[te]) for n, s in views.items()}
for n in views:
    C.row(f"单视角 {n}", C.det_auc(d, views[n]))

from sklearn.metrics import roc_auc_score
y = d["y"][te]
base = roc_auc_score(y, rk["MEM惊讶度"])
print("  --- MEM + 单视角 秩融合 ---", flush=True)
best = ("MEM惊讶度", base, rk["MEM惊讶度"])
for n in views:
    if n == "MEM惊讶度":
        continue
    fu = rk["MEM惊讶度"] + rk[n]
    a = roc_auc_score(y, fu)
    print(f"  MEM+{n:<8} AUC={a:.4f} ({a-base:+.4f})", flush=True)
    if a > best[1]:
        best = (f"MEM+{n}", a, fu)
allf = sum(rk.values())
print(f"  全视角平均秩融合 AUC={roc_auc_score(y, allf):.4f}", flush=True)
print(f"  → 最优组合: {best[0]} AUC={best[1]:.4f} (单MEM {base:.4f})", flush=True)
