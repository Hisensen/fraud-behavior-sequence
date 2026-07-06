# -*- coding: utf-8 -*-
"""架构探索补跑: V5 自回归(修复因果掩码) + V6 三种子集成 + V7 组合验证。
V7 = 时间偏置(低误报收益) + 去PE(省参数无损) 组合。"""
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

    R = {}
    R["V5 自回归AR"] = run_variant("V5 AR", dict(causal=True),
                                   train_w, test_u, y, device)
    R["V7 时间偏置+去PE"] = run_variant(
        "V7 组合", dict(time_bias=True, use_pe=False),
        train_w, test_u, y, device)

    print("\n== V6 三种子集成(V0 基线配置) ==", flush=True)
    ss = []
    for sd in (SEED, 7, 77):
        _, _, s, _ = run_variant(f"  V0-seed{sd}", dict(), train_w, test_u, y,
                                 device, seed=sd)
        ss.append(s)
    ens = sum(rankdata(s) for s in ss)
    auc, ks, r1 = metrics(y, ens)
    print(f"    集成 AUC={auc:.4f}  KS={ks:.4f}  R@FPR1%={r1:.1%}", flush=True)
    R["V6 三种子集成"] = (auc, "rank-avg", ens, r1)

    print("\n== 补跑总表 ==", flush=True)
    for name, (auc, cfg_s, _, r1) in R.items():
        print(f"  {name:<22} AUC={auc:.4f}  R@FPR1%={r1:>6.1%}  ({cfg_s})",
              flush=True)


if __name__ == "__main__":
    main()
