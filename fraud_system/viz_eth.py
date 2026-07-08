# -*- coding: utf-8 -*-
"""真实以太坊数据的聚类可视化 —— 六格: 训练效果隔离(轮廓系数) + 钓鱼在
人空间/偏移空间的分布 + LLM 定名的手法簇。输出 eth_viz.png"""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import mem_rich as M
from cluster_experiment import embed, surprise_profile
from webapp import apply_znorm, encode_rec, load_bundle, pct_among, stats_feats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                   "Heiti TC", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

b = load_bundle("eth_v2")
raw = [json.loads(l) for l in open("eth_real.jsonl", encoding="utf-8")]
users = [encode_rec(r, b["gq"]) for r in raw]
y = np.array([r["label"] for r in raw])
wi, bi = np.where(y == 0)[0], np.where(y == 1)[0]
rng = np.random.RandomState(SEED)
ws = rng.choice(wi, 2000, replace=False)          # 白样本抽样(可视化用)

print("训练后嵌入(全部账户)…", flush=True)
E_tr = embed(b["model"], users, "cpu")
print("未训练随机权重嵌入…", flush=True)
torch.manual_seed(999)
model_rnd = M.MEMRich(b["fields"]); model_rnd.eval()
E_rnd = embed(model_rnd, users, "cpu")
S = np.array([stats_feats(u) for u in users])

print("全量惊讶度(最耗时, 用于指纹与报警池)…", flush=True)
pcs = M.per_position_ce(b["model"], users, "cpu")
zs = apply_znorm(pcs, b["zn"])
FP = surprise_profile(users, zs)
zsum = [sum(zs[k][i] for k in zs) for i in range(len(users))]
c1m = np.array([float(a.mean()) for a in zsum])
Ss = b["st_sc"].transform(S)
c2 = -b["ifo"].score_samples(Ss)
nn = NearestNeighbors(n_neighbors=5).fit(b["nn_ref"])
c4 = nn.kneighbors(Ss)[0][:, -1]
fused = np.array([np.mean([pct_among(c1m[i], b["c1m_w"]),
                           pct_among(c2[i], b["c2_w"]),
                           pct_among(c4[i], b["c4_w"])])
                  for i in range(len(users))])
alarm = np.where(fused > b["thr"])[0]
print(f"报警池 {len(alarm)}", flush=True)


def sil2(X, idx):
    Xs = StandardScaler().fit_transform(X[idx])
    lab = KMeans(2, n_init=10, random_state=SEED).fit_predict(Xs)
    return silhouette_score(Xs, lab), lab


def knn_auc(X, idx):
    """该空间里 5近邻黑占比 对标签的 AUC = 黑白可分性。"""
    Xs = StandardScaler().fit_transform(X[idx])
    yy = y[idx]
    nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    _, ii = nn.kneighbors(Xs)
    frac = yy[ii[:, 1:]].mean(1)
    return roc_auc_score(yy, frac)


def tsne(X):
    return TSNE(2, random_state=SEED, perplexity=30, init="pca").fit_transform(
        StandardScaler().fit_transform(X))


POP = {0: "#00997C", 1: "#9a7b2d"}
pop_names = {k: v["name"] for k, v in b.get("pop_names", {}).items()}
fp_names = {k: v["name"] for k, v in b["fp_km"]["names"].items()}
FPC = {0: "#C24A32", 1: "#9a7b2d", 2: "#2d5f9a"}

fig, axes = plt.subplots(2, 3, figsize=(19, 12))

# 上排: 白样本(抽样2000) 三个空间, KMeans(k=2) 轮廓系数
for ax, (title, X) in zip(axes[0], [
        ("① 训练后 MEM 嵌入 · 真实白样本", E_tr),
        ("② 未训练(随机权重)嵌入", E_rnd),
        ("③ 统计特征", S)]):
    print("t-SNE", title, flush=True)
    s, lab = sil2(X, ws)
    P = tsne(X[ws])
    for k in (0, 1):
        m = lab == k
        ax.scatter(P[m, 0], P[m, 1], s=7, c=POP[k], alpha=0.6, linewidths=0,
                   label=(pop_names.get(k, f"簇{k}") if "①" in title else f"簇{k}"))
    ax.set_title(f"{title}\nKMeans(k=2) 轮廓系数 = {s:.3f}", fontsize=14)
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

# ④ 人空间: 抽样白+全部钓鱼, 按标签着色
idx4 = np.concatenate([ws, bi])
print("t-SNE ④", flush=True)
P = tsne(E_tr[idx4])
a4 = knn_auc(E_tr, idx4)
m = y[idx4] == 0
axes[1][0].scatter(P[m, 0], P[m, 1], s=7, c="#8aa5ad", alpha=0.5,
                   linewidths=0, label="正常")
axes[1][0].scatter(P[~m, 0], P[~m, 1], s=11, c="#C24A32", alpha=0.8,
                   linewidths=0, label="真实钓鱼")
axes[1][0].set_title(f"④ 行为嵌入(人空间) · 钓鱼 vs 正常\n5近邻可分性 AUC = {a4:.3f}",
                     fontsize=14)

# ⑤ 偏移空间: 同一批点按标签着色
print("t-SNE ⑤", flush=True)
P = tsne(FP[idx4])
a5 = knn_auc(FP, idx4)
axes[1][1].scatter(P[m, 0], P[m, 1], s=7, c="#8aa5ad", alpha=0.5,
                   linewidths=0, label="正常")
axes[1][1].scatter(P[~m, 0], P[~m, 1], s=11, c="#C24A32", alpha=0.8,
                   linewidths=0, label="真实钓鱼")
axes[1][1].set_title(f"⑤ 惊讶指纹(偏移空间) · 同一批账户\n5近邻可分性 AUC = {a5:.3f}",
                     fontsize=14)

# ⑥ 报警池指纹, 按 LLM 定名的手法簇着色
print("t-SNE ⑥", flush=True)
Xa = b["fp_km"]["scaler"].transform(FP[alarm])
lab_a = np.linalg.norm(Xa[:, None] - b["fp_km"]["centers"][None],
                       axis=2).argmin(1)
P = tsne(FP[alarm]) if len(alarm) > 40 else None
for k in sorted(set(lab_a)):
    m2 = lab_a == k
    axes[1][2].scatter(P[m2, 0], P[m2, 1], s=26, c=FPC.get(k, "#555"),
                       alpha=0.85, linewidths=0,
                       label=fp_names.get(k, f"簇{k}"))
axes[1][2].set_title(f"⑥ 报警池({len(alarm)}个) · LLM 定名的手法簇", fontsize=14)

for ax in axes[1]:
    ax.legend(fontsize=10, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("真实以太坊数据(5680账户/680真实钓鱼) · 上排=训练效果 · "
             "下排=钓鱼在人空间混入人群、在偏移空间浮出 + LLM手法簇",
             fontsize=15, y=0.995)
fig.tight_layout()
fig.savefig("eth_viz.png", dpi=140, bbox_inches="tight")
print("已保存 eth_viz.png", flush=True)
