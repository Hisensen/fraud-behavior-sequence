# -*- coding: utf-8 -*-
"""思想6·Motif Mining: 高频行为片段(2/3-gram, 含"快速版"=相邻间隔≤5分钟)作词表,
用户=motif 计数向量。检测=未见/罕见 motif 占比; 聚类=motif 向量。
Baseline: 单事件词频向量。"""
import json
from collections import Counter

import numpy as np

import bp_common as C
import mem_rich as M

d = C.load_art()
users = M.load("data_cluster.jsonl", gq=M.global_quantiles("data_cluster.jsonl"))
uid2i = {u["uid"]: i for i, u in enumerate(users)}
meta = json.load(open("blueprint_meta.json"))
users = [users[uid2i[u]] for u in meta["uid"]]
tr = d["is_train"]
print("== 思想6 Motif Mining ==", flush=True)


def motifs(u):
    seq, gap = u["type"], u["gap"]
    out = []
    for i in range(len(seq) - 1):
        fast = "!" if gap[i + 1] <= 1 else "."      # ≤5分钟=快速连接
        out.append(f"{seq[i]}{fast}{seq[i+1]}")
    for i in range(len(seq) - 2):
        out.append(f"{seq[i]}-{seq[i+1]}-{seq[i+2]}")
    return out


cnt = Counter()
n_tr = 0
for u, t in zip(users, tr):
    if t:
        cnt.update(motifs(u))
        n_tr += 1
vocab = [m for m, c in cnt.most_common(300)]
v2i = {m: i for i, m in enumerate(vocab)}
print(f"  训练白样本 motif 词表 300 / 总模式数 {len(cnt)}", flush=True)

X = np.zeros((len(users), len(vocab)))
rare = np.zeros(len(users))
for i, u in enumerate(users):
    ms = motifs(u)
    for m in ms:
        if m in v2i:
            X[i, v2i[m]] += 1
    X[i] /= max(len(ms), 1)
    # 罕见度: motif 在训练白样本中的负对数频率均值
    rare[i] = np.mean([-np.log((cnt.get(m, 0) + 1) / (n_tr + 1)) for m in ms])

C.row("motif 罕见度检测", C.det_auc(d, rare))
C.row("motif 向量聚类", None, C.white_ari(d, X), C.black_ari(d, X))
freq = np.array([[u["type"].count(k) / len(u["type"])
                  for k in range(len(M.EVENT_TYPES))] for u in users])
C.row("单事件词频 baseline", None, C.white_ari(d, freq), C.black_ari(d, freq))
C.save_score("e6_rare", rare)
