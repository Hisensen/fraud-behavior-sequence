# -*- coding: utf-8 -*-
"""
实验③补充: 公平版微调对照 —— 唯一变量是初始化(预训练 vs 随机)。
v1 的微调只训 15ep/lr3e-4, 从零训 40ep/lr1e-3, 不可比。
本脚本统一 40ep/lr1e-3, 另加线性探针(冻结编码器)变体。
"""
import copy
import random

import numpy as np
import torch

from mem_experiment import MEM, load, SEED
from finetune import Clf, train_clf, predict, metrics


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

    base = MEM().to(device)
    base.load_state_dict(torch.load("mem_model.pt"))

    print(f"{'N':<6}{'方法':<26}{'AUC(mean±std)':<20}{'R@FPR1%':<14}")
    for N in (10, 50):
        res = {"ft40": [], "probe": []}
        for seed in range(3):
            random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
            lab_b = random.Random(seed).sample(pool, N)
            data = train_w + lab_b
            labels = [0] * len(train_w) + [1] * N

            # 全参数微调, 与从零监督完全同配置(40ep, lr1e-3)
            ft = Clf(copy.deepcopy(base)).to(device)
            train_clf(ft, data, labels, device, epochs=40, lr=1e-3)
            res["ft40"].append(metrics(y, predict(ft, test_users, device)))

            # 线性探针: 冻结预训练编码器, 只训头
            pr = Clf(copy.deepcopy(base)).to(device)
            for p in pr.mem.parameters():
                p.requires_grad = False
            train_clf(pr, data, labels, device, epochs=60, lr=1e-2)
            res["probe"].append(metrics(y, predict(pr, test_users, device)))

        for key, name in [("ft40", "预训练+微调(40ep,公平)"),
                          ("probe", "线性探针(冻结编码器)")]:
            a = np.array([x[0] for x in res[key]])
            r = np.array([x[1] for x in res[key]])
            print(f"{N:<6}{name:<26}{a.mean():.4f}±{a.std():.4f}      "
                  f"{r.mean():>6.1%}±{r.std():.1%}", flush=True)
        print()


if __name__ == "__main__":
    main()
