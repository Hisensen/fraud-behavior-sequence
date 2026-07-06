# -*- coding: utf-8 -*-
"""架构探索收官: 合体与融合。
  V8 = 自回归 + 时间偏置注意力(两个赢家机制合体)
  F1 = V5(AR) 与 V7(时间偏置MEM) 分数秩融合(双通道)
  F2 = V8 与 V7 秩融合
V5/V7 同种子确定性复现, 无需缓存。"""
import random

import numpy as np
import torch
from scipy.stats import rankdata

from mem_rich import load, global_quantiles, SEED, metrics
from mem_arch import run_variant

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def main():
    device = "cpu"
    gq = global_quantiles()
    users = load(gq=gq)
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_u = whites[:1000], whites[1000:] + blacks
    y = np.array([u["label"] for u in test_u])

    a5, c5, s5, r5 = run_variant("V5 AR(复现)", dict(causal=True),
                                 train_w, test_u, y, device)
    a7, c7, s7, r7 = run_variant("V7 时间偏置+去PE(复现)",
                                 dict(time_bias=True, use_pe=False),
                                 train_w, test_u, y, device)
    a8, c8, s8, r8 = run_variant("V8 AR+时间偏置",
                                 dict(causal=True, time_bias=True),
                                 train_w, test_u, y, device)

    print("\n== 双通道融合 ==", flush=True)
    for name, pair in [("F1 = V5(AR) + V7(时间偏置)", (s5, s7)),
                       ("F2 = V8 + V7", (s8, s7)),
                       ("F3 = V5 + V8", (s5, s8))]:
        f = rankdata(pair[0]) + rankdata(pair[1])
        auc, ks, r1 = metrics(y, f)
        print(f"  {name:<28} AUC={auc:.4f}  KS={ks:.4f}  R@FPR1%={r1:.1%}",
              flush=True)

    print("\n== 收官总表 ==", flush=True)
    for n, (a, c, r) in {"V5 AR": (a5, c5, r5), "V7 偏置MEM": (a7, c7, r7),
                         "V8 AR+偏置": (a8, c8, r8)}.items():
        print(f"  {n:<12} AUC={a:.4f}  R@FPR1%={r:>6.1%}  ({c})", flush=True)


if __name__ == "__main__":
    main()
