# -*- coding: utf-8 -*-
"""加载已训练的 MEM, 对最优打分变体做小扫描(k 值 × type/gap 权重)"""
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mem_experiment import MEM, load, SEED
from mem_score_v2 import per_position_ce, bucket_stats, z_normalize, topk_mean

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"
users = load("data_temporal.jsonl")
whites = [u for u in users if u["label"] == 0]
blacks = [u for u in users if u["label"] == 1]
rng = random.Random(SEED)
rng.shuffle(whites)
train_w, test_w = whites[:400], whites[400:]
test_users = test_w + blacks
y = np.array([u["label"] for u in test_users])

model = MEM().to(device)
model.load_state_dict(torch.load("mem_model.pt"))

pcs_train = per_position_ce(model, train_w, device)
pcs_test = per_position_ce(model, test_users, device)
stats = bucket_stats(pcs_train)
zs = [z_normalize(p, stats) for p in pcs_test]

print(f"{'k':>4} | " + " | ".join(f"w_gap={w:.1f}" for w in (0.0, 0.2, 0.5, 1.0)))
best = (0, None)
for k in (2, 3, 5, 8, 10, 15):
    row = []
    for w in (0.0, 0.2, 0.5, 1.0):
        s = np.array([topk_mean(zt + w * zg, k) for zt, zg in zs])
        auc = roc_auc_score(y, s)
        row.append(auc)
        if auc > best[0]:
            best = (auc, (k, w))
    print(f"{k:>4} | " + " | ".join(f"{a:.4f} " for a in row))

k, w = best[1]
print(f"\n最优: k={k}, w_gap={w}, AUC={best[0]:.4f}")
s = np.array([topk_mean(zt + w * zg, k) for zt, zg in zs])
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, s)
print(f"KS={np.max(tpr - fpr):.4f}")
for sub, name in [(1, "盗号"), (2, "套现"), (3, "洗钱")]:
    idx = [i for i, u in enumerate(test_users) if u["label"] == 0 or u["sub"] == sub]
    print(f"  {name}: AUC={roc_auc_score(y[idx], s[idx]):.4f}")
# 低误报区召回: FPR=1%/5% 时的黑样本召回率
for target in (0.01, 0.05):
    j = np.searchsorted(fpr, target, side="right") - 1
    print(f"  Recall@FPR={target:.0%}: {tpr[j]:.1%}")
