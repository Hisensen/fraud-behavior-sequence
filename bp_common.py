# -*- coding: utf-8 -*-
"""实验矩阵·公共评估: 统一三项指标 = 检测AUC / 白人群ARI / 黑手法ARI。"""
import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_art():
    a = np.load("blueprint_art.npz")
    m = json.load(open("blueprint_meta.json"))
    d = {k: a[k] for k in a.files}
    d["wtype"] = np.array(m["wtype"])
    d["btype"] = np.array([b if b else "" for b in m["btype"]])
    d["test"] = ~d["is_train"]
    return d


def det_auc(d, score):
    """检测 AUC: 测试集(400白+240黑)。score 为全体 N 维, 高=可疑。"""
    return roc_auc_score(d["y"][d["test"]], score[d["test"]])


def ari_kmeans(X, y_true, k):
    Xs = StandardScaler().fit_transform(X)
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs)
    return adjusted_rand_score(y_true, lab)


def white_ari(d, X_all):
    m = d["y"] == 0
    return ari_kmeans(X_all[m], d["wtype"][m], 4)


def black_ari(d, X_all):
    m = d["y"] == 1
    return ari_kmeans(X_all[m], d["btype"][m], 4)


def save_score(name, score):
    os.makedirs("blueprint_scores", exist_ok=True)
    np.save(f"blueprint_scores/{name}.npy", score)


def row(name, auc=None, wa=None, ba=None, note=""):
    f = lambda v: f"{v:.4f}" if v is not None else "  -   "
    print(f"  {name:<34} AUC={f(auc)}  白ARI={f(wa)}  黑ARI={f(ba)}  {note}",
          flush=True)
