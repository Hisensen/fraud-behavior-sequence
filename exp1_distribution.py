# -*- coding: utf-8 -*-
"""思想1·Distribution Learning: 正常样本成高密度分布, 密度低=异常。
假设检验: (a) 正常分布是否多模态(GMM BIC 选峰数); (b) 密度能否当风险分。
Baseline: 单高斯Mahalanobis / iForest(统计特征) / OCSVM(统计特征)。"""
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import bp_common as C

d = C.load_art()
tr = d["is_train"]
print("== 思想1 Distribution Learning ==", flush=True)

# 嵌入空间标准化(用训练白样本拟合)
sc = StandardScaler().fit(d["emb"][tr])
E, Etr = sc.transform(d["emb"]), sc.transform(d["emb"][tr])
sc2 = StandardScaler().fit(d["stats"][tr])
S, Str = sc2.transform(d["stats"]), sc2.transform(d["stats"][tr])

# (a) 多模态检验: GMM BIC 扫 K
bics = {}
for k in (1, 2, 4, 6, 8, 12):
    g = GaussianMixture(k, covariance_type="diag", random_state=42,
                        n_init=2).fit(Etr)
    bics[k] = g.bic(Etr)
kbest = min(bics, key=bics.get)
print(f"  GMM BIC 选峰数 K={kbest} "
      f"(K=1 BIC={bics[1]:.0f} vs K={kbest} BIC={bics[kbest]:.0f}) "
      f"→ 正常分布{'是多模态' if kbest > 1 else '单模态'}", flush=True)

# (b) 密度作风险分
g1 = GaussianMixture(1, covariance_type="full", random_state=42).fit(Etr)
s_g1 = -g1.score_samples(E)
gk = GaussianMixture(kbest, covariance_type="diag", random_state=42,
                     n_init=3).fit(Etr)
s_gk = -gk.score_samples(E)
maha = EmpiricalCovariance().fit(Etr)
s_mh = maha.mahalanobis(E)
ifo = IsolationForest(n_estimators=300, random_state=42).fit(Str)
s_if = -ifo.score_samples(S)
oc = OneClassSVM(nu=0.05, gamma="scale").fit(Str)
s_oc = -oc.decision_function(S)

C.row("单高斯 Mahalanobis(嵌入)", C.det_auc(d, s_mh))
C.row("单高斯 对数密度(嵌入)", C.det_auc(d, s_g1))
C.row(f"GMM K={kbest} 对数密度(嵌入)", C.det_auc(d, s_gk))
C.row("iForest(统计特征) baseline", C.det_auc(d, s_if))
C.row("OCSVM(统计特征) baseline", C.det_auc(d, s_oc))
C.row("参照: MEM 惊讶度 top5", C.det_auc(d, d["mem_topk"]))
C.save_score("e1_gmm", s_gk)
C.save_score("e1_iforest", s_if)
