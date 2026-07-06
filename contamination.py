# -*- coding: utf-8 -*-
"""
实验①: 污染鲁棒性 —— 训练"白样本"里混入黑样本, MEM 还能用吗?
真实场景 unlabeled 数据含 ~0.1-1% 欺诈, 这里测 0/1%/5%/10% 四档。
所有档共用同一测试集(100 留出白 + 456 黑), 污染用的 44 个黑样本
固定从测试集中剔除, 保证各档 AUC 可比。
"""
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from mem_experiment import MEM, load, train, SEED
from mem_score_v2 import per_position_ce, bucket_stats, z_normalize, topk_mean


def main():
    device = "cpu"
    users = load("data_temporal.jsonl")
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_w = whites[:400], whites[400:]

    pool = random.Random(777).sample(blacks, 44)   # 最大污染档所需黑样本
    pool_uids = {u["uid"] for u in pool}
    test_users = test_w + [b for b in blacks if b["uid"] not in pool_uids]
    y = np.array([u["label"] for u in test_users])
    print(f"测试集: {len(test_w)} 白 + {int(y.sum())} 黑 (所有污染档共用)\n")

    print(f"{'污染档':<12}{'训练集构成':<16}{'AUC':>8}{'KS':>8}{'R@FPR1%':>10}{'R@FPR5%':>10}")
    for c, tag in [(0, "0%"), (4, "1%"), (21, "5%"), (44, "10%")]:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        train_set = train_w + pool[:c]
        model = MEM().to(device)
        train(model, train_set, device)
        stats = bucket_stats(per_position_ce(model, train_set, device))
        pcs = per_position_ce(model, test_users, device)
        s = np.array([topk_mean(z_normalize(p, stats)[0], 5) for p in pcs])
        auc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        ks = float(np.max(tpr - fpr))
        r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
        r5 = tpr[np.searchsorted(fpr, 0.05, side="right") - 1]
        print(f"{tag:<12}{f'{len(train_w)}白+{c}黑':<16}"
              f"{auc:>8.4f}{ks:>8.4f}{r1:>10.1%}{r5:>10.1%}")


if __name__ == "__main__":
    main()
