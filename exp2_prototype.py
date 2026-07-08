# -*- coding: utf-8 -*-
"""思想2·Prototype Learning: K 个正常原型, 到最近原型的距离=风险分,
到各原型的距离向量=用户表示。假设检验: 多原型是否优于单一中心(K=1=DeepSVDD心)。"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import bp_common as C

d = C.load_art()
tr = d["is_train"]
print("== 思想2 Prototype Learning ==", flush=True)
sc = StandardScaler().fit(d["emb"][tr])
E, Etr = sc.transform(d["emb"]), sc.transform(d["emb"][tr])

best = None
for k in (1, 2, 4, 8, 16, 32):
    km = KMeans(k, n_init=10, random_state=42).fit(Etr)
    dist = np.linalg.norm(E[:, None] - km.cluster_centers_[None], axis=2)
    s = dist.min(1)                      # 到最近原型距离
    auc = C.det_auc(d, s)
    wa = C.white_ari(d, dist) if k >= 4 else None   # 距离向量作表示
    C.row(f"K={k:<2} 最近原型距离", auc, wa)
    if best is None or auc > best[1]:
        best = (k, auc, s)
print(f"  → 最优 K={best[0]}; K=1(单中心)与多原型的差距即'多模态收益'", flush=True)
C.save_score("e2_proto", best[2])
