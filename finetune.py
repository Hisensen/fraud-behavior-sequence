# -*- coding: utf-8 -*-
"""
实验③: MEM 预训练 + 少量黑标签微调, 比纯无监督多赚多少?
------------------------------------------------------
黑样本切分: 100 个进"标签池"(微调用), 400 个留作测试。
测试集固定: 100 留出白 + 400 黑, 三种方法同一测试集可比。

对照:
  (a) 纯无监督   : MEM 异常分(z-norm top-5 type CE), 0 黑标签
  (b) 预训练+微调 : 加载 mem_model.pt, mean-pool + 线性头, BCE 微调
  (c) 从零监督   : 同架构随机初始化, 直接监督训练
  (d) 融合       : (a) 与 (b) 的秩平均
每档 N ∈ {10, 50, 100} 个黑标签 × 3 个采样种子, 报 mean±std。
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import rankdata

from mem_experiment import MEM, load, pad_batch, SEED
from mem_score_v2 import per_position_ce, bucket_stats, z_normalize, topk_mean


class Clf(nn.Module):
    def __init__(self, mem):
        super().__init__()
        self.mem = mem
        self.head = nn.Linear(64, 1)

    def forward(self, tp, am, gp, hr, pad):
        h = self.mem.encode(tp, am, gp, hr, pad)
        m = (~pad).float().unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def train_clf(clf, data, labels, device, epochs, lr, bs=32):
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    pos_w = torch.tensor((len(labels) - sum(labels)) / max(1, sum(labels)),
                         dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    clf.train()
    idx = list(range(len(data)))
    for _ in range(epochs):
        random.shuffle(idx)
        for s in range(0, len(idx), bs):
            j = idx[s:s + bs]
            tp, am, gp, hr, pad = pad_batch([data[k] for k in j], device)
            yb = torch.tensor([labels[k] for k in j], dtype=torch.float32)
            loss = lossf(clf(tp, am, gp, hr, pad), yb)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def predict(clf, data, device, bs=64):
    clf.eval()
    out = []
    for s in range(0, len(data), bs):
        tp, am, gp, hr, pad = pad_batch(data[s:s + bs], device)
        out.append(torch.sigmoid(clf(tp, am, gp, hr, pad)).numpy())
    return np.concatenate(out)


def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return (roc_auc_score(y, s),
            float(tpr[np.searchsorted(fpr, 0.01, side="right") - 1]))


def main():
    device = "cpu"
    users = load("data_temporal.jsonl")
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_w = whites[:400], whites[400:]
    pool = random.Random(888).sample(blacks, 100)
    pool_uids = {u["uid"] for u in pool}
    test_users = test_w + [b for b in blacks if b["uid"] not in pool_uids]
    y = np.array([u["label"] for u in test_users])
    print(f"标签池: {len(pool)} 黑 | 测试集: {len(test_w)} 白 + {int(y.sum())} 黑\n")

    # (a) 纯无监督参考
    base = MEM().to(device)
    base.load_state_dict(torch.load("mem_model.pt"))
    stats = bucket_stats(per_position_ce(base, train_w, device))
    pcs = per_position_ce(base, test_users, device)
    s_unsup = np.array([topk_mean(z_normalize(p, stats)[0], 5) for p in pcs])
    auc_u, r1_u = metrics(y, s_unsup)
    print(f"(a) 纯无监督(0黑标签):            AUC={auc_u:.4f}  R@FPR1%={r1_u:.1%}\n")

    print(f"{'N黑标签':<8}{'方法':<22}{'AUC(mean±std)':<20}{'R@FPR1%':<12}")
    for N in (10, 50, 100):
        res = {"ft": [], "sc": [], "ens": []}
        for seed in range(3):
            random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
            lab_b = random.Random(seed).sample(pool, N)
            data = train_w + lab_b
            labels = [0] * len(train_w) + [1] * N

            ft = Clf(copy.deepcopy(base)).to(device)
            train_clf(ft, data, labels, device, epochs=15, lr=3e-4)
            s_ft = predict(ft, test_users, device)
            res["ft"].append(metrics(y, s_ft))

            sc = Clf(MEM()).to(device)
            train_clf(sc, data, labels, device, epochs=40, lr=1e-3)
            res["sc"].append(metrics(y, predict(sc, test_users, device)))

            s_ens = rankdata(s_ft) + rankdata(s_unsup)
            res["ens"].append(metrics(y, s_ens))

        for key, name in [("ft", "(b) 预训练+微调"), ("sc", "(c) 从零监督"),
                          ("ens", "(d) 微调+无监督融合")]:
            a = np.array([x[0] for x in res[key]])
            r = np.array([x[1] for x in res[key]])
            print(f"{N:<8}{name:<22}{a.mean():.4f}±{a.std():.4f}      "
                  f"{r.mean():>6.1%}±{r.std():.1%}")
        print()


if __name__ == "__main__":
    main()
