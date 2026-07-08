# -*- coding: utf-8 -*-
"""词频对齐基准(寄生型欺诈)可视化 —— 2×2: 同一批账户在四个空间里的黑白分布。
黑白事件配比完全相同, 预期: 统计/人空间全瞎, 只有偏移空间现形。输出 temporal_viz.png"""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import mem_rich as M
from cluster_experiment import embed, surprise_profile
from webapp import apply_znorm, encode_rec, load_bundle, stats_feats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                   "Heiti TC", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

b = load_bundle("temporal_v1")
raw = [json.loads(l) for l in open("data_temporal.jsonl", encoding="utf-8")]
users = [encode_rec(r, b["gq"]) for r in raw]
y = np.array([r["label"] for r in raw])

print("四种表征…", flush=True)
E_tr = embed(b["model"], users, "cpu")
torch.manual_seed(999)
model_rnd = M.MEMRich(b["fields"]); model_rnd.eval()
E_rnd = embed(model_rnd, users, "cpu")
S = np.array([stats_feats(u) for u in users])
pcs = M.per_position_ce(b["model"], users, "cpu")
zs = apply_znorm(pcs, b["zn"])
FP = surprise_profile(users, zs)


def knn_auc(X):
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    _, ii = nn.kneighbors(Xs)
    return roc_auc_score(y, y[ii[:, 1:]].mean(1))


def tsne(X):
    return TSNE(2, random_state=SEED, perplexity=30, init="pca").fit_transform(
        StandardScaler().fit_transform(X))


panels = [("① 统计特征空间（配比/节奏汇总）", S),
          ("② 人空间 · 未训练随机嵌入", E_rnd),
          ("③ 人空间 · 训练后 MEM 嵌入", E_tr),
          ("④ 偏移空间 · 惊讶指纹", FP)]
fig, axes = plt.subplots(2, 2, figsize=(14, 12.5))
for ax, (title, X) in zip(axes.flat, panels):
    print("t-SNE", title, flush=True)
    a = knn_auc(X)
    P = tsne(X)
    m = y == 0
    ax.scatter(P[m, 0], P[m, 1], s=9, c="#8aa5ad", alpha=0.55,
               linewidths=0, label="正常(500)")
    ax.scatter(P[~m, 0], P[~m, 1], s=9, c="#C24A32", alpha=0.7,
               linewidths=0, label="欺诈(500)")
    ax.set_title(f"{title}\n5近邻可分性 AUC = {a:.3f}", fontsize=14)
    ax.legend(fontsize=11, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("词频对齐基准（寄生型欺诈: 黑白事件配比完全相同, 只差时序结构）\n"
             "同一批 1000 个账户在四个空间里的黑白分布",
             fontsize=15, y=0.99)
fig.tight_layout()
fig.savefig("temporal_viz.png", dpi=140, bbox_inches="tight")
print("已保存 temporal_viz.png", flush=True)
