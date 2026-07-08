# -*- coding: utf-8 -*-
"""思想5·Behavior Grammar: 违反"正常行为语法"即异常。
阶梯: 一元(纯词频,无语法) → 二元马尔可夫(局部语法) → MEM(长程条件语法)。
另做"顺序破坏"检验: 把测试序列打乱后语法分应显著上升, 证明学到的是顺序而非词频。"""
import json
import random

import numpy as np

import bp_common as C
import mem_rich as M

d = C.load_art()
users = M.load("data_cluster.jsonl", gq=M.global_quantiles("data_cluster.jsonl"))
uid2i = {u["uid"]: i for i, u in enumerate(users)}
meta = json.load(open("blueprint_meta.json"))
order = [uid2i[u] for u in meta["uid"]]
users = [users[i] for i in order]            # 对齐 npz 行序
tr = d["is_train"]
V = len(M.EVENT_TYPES)
print("== 思想5 Behavior Grammar ==", flush=True)

# 训练白样本上估计一元/二元分布(拉普拉斯平滑)
uni = np.ones(V)
big = np.ones((V, V))
for u, t in zip(users, tr):
    if not t:
        continue
    seq = u["type"]
    for a in seq:
        uni[a] += 1
    for a, b in zip(seq, seq[1:]):
        big[a][b] += 1
uni_p = uni / uni.sum()
big_p = big / big.sum(1, keepdims=True)


def nll(seq, shuffle=False, rng=None):
    s = list(seq)
    if shuffle:
        rng.shuffle(s)
    n1 = -np.mean([np.log(uni_p[a]) for a in s])
    n2 = -np.mean([np.log(big_p[a][b]) for a, b in zip(s, s[1:])])
    return n1, n2


rng = random.Random(0)
r = np.array([nll(u["type"]) for u in users])
rs = np.array([nll(u["type"], True, rng) for u in users])
C.row("一元词频 NLL(无语法)", C.det_auc(d, r[:, 0]))
C.row("二元语法 NLL", C.det_auc(d, r[:, 1]))
C.row("参照: MEM 长程语法(惊讶度)", C.det_auc(d, d["mem_topk"]))
dw = (rs[:, 1] - r[:, 1])[d["y"] == 0].mean()
db = (rs[:, 1] - r[:, 1])[d["y"] == 1].mean()
print(f"  顺序破坏检验: 打乱后二元NLL上升 白+{dw:.3f} / 黑+{db:.3f} "
      f"→ {'顺序信息真实存在' if dw > 0.05 else '顺序信息弱'}", flush=True)
C.save_score("e5_bigram", r[:, 1])
