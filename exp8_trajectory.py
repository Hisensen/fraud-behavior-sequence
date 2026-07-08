# -*- coding: utf-8 -*-
"""思想8·Trajectory: 把一个月看成 30 天的日活动轨迹, DTW 距离聚类/检测。
假设检验: DTW 对齐是否优于逐日欧氏距离(轨迹是否需要"弹性对齐")。"""
import json
import random
from datetime import datetime

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

import bp_common as C
import mem_rich as M

d = C.load_art()
users = M.load("data_cluster.jsonl", gq=M.global_quantiles("data_cluster.jsonl"))
uid2i = {u["uid"]: i for i, u in enumerate(users)}
meta = json.load(open("blueprint_meta.json"))
users = [users[uid2i[u]] for u in meta["uid"]]
print("== 思想8 Trajectory(DTW) ==", flush=True)

T0 = datetime(2024, 1, 1).timestamp()
traj = np.zeros((len(users), 30))
for i, u in enumerate(users):
    for ts in u["ts"]:
        day = min(int((ts - T0) // 86400), 29)
        traj[i, day] += 1
traj = (traj - traj.mean(1, keepdims=True)) / (traj.std(1, keepdims=True) + 1e-6)


def dtw(a, b, w=8):
    n = len(a)
    D = np.full((n + 1, n + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(n, i + w)
        for j in range(lo, hi + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1],
                                                     D[i - 1, j - 1])
    return D[n, n]


def pair_dist(idx, fn):
    n = len(idx)
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            Dm[i, j] = Dm[j, i] = fn(traj[idx[i]], traj[idx[j]])
    return Dm


def ari_of(Dm, y_true, k=4):
    lab = fcluster(linkage(squareform(Dm), "average"), k, criterion="maxclust")
    return adjusted_rand_score(y_true, lab)


rng = random.Random(42)
wi = [i for i in range(len(users)) if d["y"][i] == 0]
bi = [i for i in range(len(users)) if d["y"][i] == 1]
ws = rng.sample(wi, 400)
eu = lambda a, b: float(np.linalg.norm(a - b))

wa_d = ari_of(pair_dist(ws, dtw), d["wtype"][ws])
wa_e = ari_of(pair_dist(ws, eu), d["wtype"][ws])
ba_d = ari_of(pair_dist(bi, dtw), d["btype"][bi])
ba_e = ari_of(pair_dist(bi, eu), d["btype"][bi])
C.row("DTW 轨迹聚类", None, wa_d, ba_d)
C.row("欧氏 轨迹聚类 baseline", None, wa_e, ba_e)

# 检测: 到 200 个训练白轨迹的最近 DTW 距离
ref = rng.sample([i for i in wi if d["is_train"][i]], 200)
s = np.zeros(len(users))
for i in range(len(users)):
    s[i] = min(dtw(traj[i], traj[r]) for r in ref)
C.row("最近白轨迹 DTW 距离检测", C.det_auc(d, s))
C.save_score("e8_dtw", s)
