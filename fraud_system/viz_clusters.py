# -*- coding: utf-8 -*-
"""聚类质量可视化 —— 六格对照图, 隔离 MEM 训练效果 + 展示双表征定律。
输出 cluster_viz.png"""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import mem_rich as M
from cluster_experiment import embed, surprise_profile
from webapp import apply_znorm, encode_rec, load_bundle, stats_feats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                   "Heiti TC", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("载入模型与数据…", flush=True)
b = load_bundle("cluster_v1")
raw = [json.loads(l) for l in open("data_cluster.jsonl", encoding="utf-8")]
users = [encode_rec(r, b["gq"]) for r in raw]
wtype = np.array([r["wtype"] for r in raw])
btype = np.array([r["btype"] or "" for r in raw])
y = np.array([r["label"] for r in raw])
wi, bi = np.where(y == 0)[0], np.where(y == 1)[0]
uw = [users[i] for i in wi]
ub = [users[i] for i in bi]

print("训练后嵌入…", flush=True)
E_tr = embed(b["model"], users, "cpu")
print("未训练(随机权重)嵌入…", flush=True)
torch.manual_seed(999)
model_rnd = M.MEMRich(b["fields"])
model_rnd.eval()
E_rnd = embed(model_rnd, users, "cpu")
print("统计特征…", flush=True)
S = np.array([stats_feats(u) for u in users])
print("黑样本指纹(逐位置惊讶度)…", flush=True)
pcs_b = M.per_position_ce(b["model"], ub, "cpu")
zs_b = apply_znorm(pcs_b, b["zn"])
FP = surprise_profile(ub, zs_b)


def ari(X, labels, k=4):
    Xs = StandardScaler().fit_transform(X)
    return adjusted_rand_score(labels,
                               KMeans(k, n_init=10, random_state=SEED)
                               .fit_predict(Xs))


def tsne(X):
    Xs = StandardScaler().fit_transform(X)
    return TSNE(2, random_state=SEED, perplexity=30,
                init="pca").fit_transform(Xs)


PALW = {"工薪族": "#00997C", "学生党": "#2d5f9a",
        "个体商户": "#C24A32", "理财大户": "#9a7b2d"}
PALB = {"盗号爆发": "#C24A32", "慢速抽干": "#9a7b2d",
        "养卡套现": "#2d5f9a", "赌博出款": "#00997C"}

panels = [
    ("① 训练后 MEM 嵌入 · 白样本", E_tr[wi], wtype[wi], PALW),
    ("② 未训练(随机权重)嵌入 · 白样本", E_rnd[wi], wtype[wi], PALW),
    ("③ 统计特征 · 白样本", S[wi], wtype[wi], PALW),
    ("④ 训练后嵌入 · 黑样本按手法着色", E_tr[bi], btype[bi], PALB),
    ("⑤ 训练后嵌入 · 黑样本按底座人群着色", E_tr[bi], wtype[bi], PALW),
    ("⑥ 惊讶指纹 · 黑样本按手法着色", FP, btype[bi], PALB),
]

fig, axes = plt.subplots(2, 3, figsize=(19, 12))
for ax, (title, X, lab, pal) in zip(axes.flat, panels):
    print(f"t-SNE: {title}", flush=True)
    P = tsne(X)
    a = ari(X, lab)
    for name, color in pal.items():
        m = lab == name
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=8 if len(lab) > 500 else 22,
                       c=color, label=name, alpha=0.65, linewidths=0)
    ax.set_title(f"{title}\nKMeans ARI = {a:.3f}", fontsize=14)
    ax.legend(fontsize=10, markerscale=2, framealpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("聚类质量眼见为实：上排=训练效果隔离（同数据同降维，只差训没训练）"
             "  下排=双表征定律（嵌入认人·指纹认手法）",
             fontsize=15, y=0.995)
fig.tight_layout()
fig.savefig("cluster_viz.png", dpi=140, bbox_inches="tight")
print("已保存 cluster_viz.png", flush=True)
