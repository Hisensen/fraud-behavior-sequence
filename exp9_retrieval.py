# -*- coding: utf-8 -*-
"""思想9·Retrieval Memory: 不训练分类器, 只查最近邻。
(a) 无标签: 第k近白样本距离=风险分(嵌入空间/指纹空间/统计空间对比)。
(b) 有案件库: 一半黑样本(120)带手法标签入库, 另一半查 top-10 邻居 →
    邻居里黑占比=风险分, 邻居多数手法=归因; 报 Recall@10 与手法命中率。"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import bp_common as C

d = C.load_art()
tr = d["is_train"]
print("== 思想9 Retrieval Memory ==", flush=True)


def knn_score(X, k=5):
    sc = StandardScaler().fit(X[tr])
    Xs, Xtr = sc.transform(X), sc.transform(X[tr])
    nn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dist, _ = nn.kneighbors(Xs)
    return dist[:, -1]


for name, X in (("嵌入空间", d["emb"]), ("指纹空间", d["fp"]),
                ("统计空间", d["stats"])):
    s = knn_score(X)
    C.row(f"第5近白样本距离({name})", C.det_auc(d, s))
    if name == "嵌入空间":
        C.save_score("e9_knn", s)

# (b) 案件库检索
rng = np.random.RandomState(42)
bidx = np.where(d["y"] == 1)[0]
known = rng.choice(bidx, 120, replace=False)
query = np.array([i for i in bidx if i not in set(known)])
lib = np.concatenate([np.where(tr)[0], known])       # 库=训练白+已结案黑
lib_black = np.isin(lib, known)

for name, X in (("嵌入", d["emb"]), ("指纹", d["fp"])):
    sc = StandardScaler().fit(X[lib])
    nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(X[lib]))
    _, idx = nn.kneighbors(sc.transform(X[query]))
    frac_black = lib_black[idx].mean(1)               # 邻居黑占比
    hit = 0
    for qi, row in zip(query, idx):
        nb = [d["btype"][lib[j]] for j in row if lib_black[np.where(lib == lib[j])[0][0]]]
        nb = [b for b in nb if b]
        if nb and max(set(nb), key=nb.count) == d["btype"][qi]:
            hit += 1
    rec10 = float((frac_black > 0).mean())
    print(f"  案件库({name}空间): 待查黑 Recall@10={rec10:.1%}  "
          f"邻居多数手法命中率={hit/len(query):.1%}  "
          f"邻居黑占比作分数 AUC(仅黑查询集无白, 见下)", flush=True)

# 用全部测试账户查库, 邻居黑占比当风险分
test_idx = np.where(d["test"])[0]
sc = StandardScaler().fit(d["fp"][lib])
nn = NearestNeighbors(n_neighbors=10).fit(sc.transform(d["fp"][lib]))
_, idx = nn.kneighbors(sc.transform(d["fp"][test_idx]))
s_all = np.zeros(len(d["y"]))
s_all[test_idx] = lib_black[idx].mean(1)
from sklearn.metrics import roc_auc_score
print(f"  邻居黑占比风险分(指纹空间, 120案件库) 测试AUC="
      f"{roc_auc_score(d['y'][test_idx], s_all[test_idx]):.4f}", flush=True)
