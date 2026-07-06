# -*- coding: utf-8 -*-
"""
MEM 打分方案 v2 —— 修复"似然≠异常"的熵混淆问题
------------------------------------------------
v1 的失败: 朴素平均重建误差 AUC=0.32(反向)。
根因: 黑样本爆发窗口是低熵区域(小间隔好预测), 白样本孤立敏感事件
     的间隔是高熵区域(难预测), 平均池化后方向颠倒。

v2 修复(两个正交手段):
  1. top-k 池化 —— 欺诈是局部异常(~8 个爆发位置 vs ~70 个正常位置),
     取每条序列 CE 最高的 k 个位置, 不被正常位置稀释。
  2. 难度归一化 —— 用白样本训练集统计"每个间隔桶"的 CE 均值/方差,
     对每个位置做 z-score, 消除"这个位置本来就难/易预测"的混淆,
     只留下"比同难度的正常位置异常多少"。

打分改用步长遮罩: 每轮 mask 位置 i≡r (mod S), S 轮覆盖全部位置各一次,
得到逐位置 CE, 供池化变体使用。
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from mem_experiment import (MEM, load, pad_batch, train, EVENT_TYPES,
                            SENSITIVE, N_GAP, SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@torch.no_grad()
def per_position_ce(model, users, device, stride=7, bs=64):
    """步长遮罩, 返回每个用户逐位置的 (type_ce, gap_ce, gap_bucket, type_id)"""
    model.eval()
    out = [{"t": np.zeros(len(u["types"])), "g": np.zeros(len(u["types"])),
            "gb": np.array(u["gaps"]), "tp": np.array(u["types"])}
           for u in users]
    ce = nn.functional.cross_entropy
    for r in range(stride):
        for s in range(0, len(users), bs):
            batch = users[s:s + bs]
            tp, am, gp, hr, pad = pad_batch(batch, device)
            L = tp.size(1)
            pos = torch.arange(L)
            mask = ((pos % stride) == r)[None].expand_as(pad) & ~pad
            if not mask.any():
                continue
            lt, lg = model(tp, am, gp, hr, pad, mask.to(device))
            for i in range(len(batch)):
                m = mask[i]
                if not m.any():
                    continue
                idx = m.nonzero().flatten()
                tce = ce(lt[i, idx], tp[i, idx], reduction="none").cpu().numpy()
                gce = ce(lg[i, idx], gp[i, idx], reduction="none").cpu().numpy()
                out[s + i]["t"][idx.numpy()] = tce
                out[s + i]["g"][idx.numpy()] = gce
    return out


def bucket_stats(pcs):
    """白样本训练集: 每个间隔桶的 type/gap CE 均值与标准差"""
    mu_t = np.zeros(N_GAP); sd_t = np.ones(N_GAP)
    mu_g = np.zeros(N_GAP); sd_g = np.ones(N_GAP)
    all_t = np.concatenate([p["t"] for p in pcs])
    all_g = np.concatenate([p["g"] for p in pcs])
    all_b = np.concatenate([p["gb"] for p in pcs])
    gmu_t, gsd_t = all_t.mean(), max(all_t.std(), 1e-3)
    gmu_g, gsd_g = all_g.mean(), max(all_g.std(), 1e-3)
    for b in range(N_GAP):
        m = all_b == b
        if m.sum() >= 30:
            mu_t[b], sd_t[b] = all_t[m].mean(), max(all_t[m].std(), 1e-3)
            mu_g[b], sd_g[b] = all_g[m].mean(), max(all_g[m].std(), 1e-3)
        else:
            mu_t[b], sd_t[b] = gmu_t, gsd_t
            mu_g[b], sd_g[b] = gmu_g, gsd_g
    return mu_t, sd_t, mu_g, sd_g


def z_normalize(p, stats):
    mu_t, sd_t, mu_g, sd_g = stats
    b = p["gb"]
    zt = (p["t"] - mu_t[b]) / sd_t[b]
    zg = (p["g"] - mu_g[b]) / sd_g[b]
    return zt, zg


def topk_mean(a, k):
    if len(a) == 0:
        return 0.0
    k = min(k, len(a))
    return float(np.sort(a)[-k:].mean())


def ks_stat(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def report(name, y, s):
    print(f"  {name:<36} AUC={roc_auc_score(y, s):.4f}  KS={ks_stat(y, s):.4f}")
    return roc_auc_score(y, s)


def main():
    device = "cpu"
    users = load("data_temporal.jsonl")
    whites = [u for u in users if u["label"] == 0]
    blacks = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(whites)
    train_w, test_w = whites[:400], whites[400:]
    test_users = test_w + blacks
    y = np.array([u["label"] for u in test_users])

    print("== 训练 MEM (同 v1 配置, 仅 400 白样本) ==")
    model = MEM().to(device)
    train(model, train_w, device)
    torch.save(model.state_dict(), "mem_model.pt")

    print("\n== 逐位置 CE (步长遮罩, 覆盖全部位置) ==")
    pcs_train = per_position_ce(model, train_w, device)
    pcs_test = per_position_ce(model, test_users, device)
    stats = bucket_stats(pcs_train)
    mu_t, sd_t, mu_g, sd_g = stats
    print("  白样本训练集各间隔桶的期望CE(type/gap):")
    names = ["≤1m", "1-5m", "5-30m", "0.5-1h", "1-6h", "6-24h", "1-7d", ">7d", "首事件"]
    for b in range(N_GAP):
        print(f"    {names[b]:<8} type {mu_t[b]:.3f}±{sd_t[b]:.3f}   "
              f"gap {mu_g[b]:.3f}±{sd_g[b]:.3f}")

    print("\n== 打分变体对比 (测试集: 100白 + 500黑) ==")
    raw_sum = [p["t"] + p["g"] for p in pcs_test]
    zs = [z_normalize(p, stats) for p in pcs_test]
    z_sum = [zt + zg for zt, zg in zs]

    report("V1 raw 平均(=v1复现)", y, np.array([a.mean() for a in raw_sum]))
    for k in (3, 5, 10):
        report(f"V2 raw top-{k}", y,
               np.array([topk_mean(a, k) for a in raw_sum]))
    report("V3 z-norm 平均", y, np.array([a.mean() for a in z_sum]))
    for k in (3, 5, 10):
        report(f"V4 z-norm top-{k}", y,
               np.array([topk_mean(a, k) for a in z_sum]))
    report("V5 z-norm(仅type) top-5", y,
           np.array([topk_mean(zt, 5) for zt, _ in zs]))
    report("V6 z-norm(仅gap) top-5", y,
           np.array([topk_mean(zg, 5) for _, zg in zs]))

    best = np.array([topk_mean(a, 5) for a in z_sum])
    print("\n== 最优变体(z-norm top-5) 分欺诈模式 ==")
    for sub, name in [(1, "盗号"), (2, "套现"), (3, "洗钱")]:
        idx = [i for i, u in enumerate(test_users)
               if u["label"] == 0 or u["sub"] == sub]
        report(f"  {name}", y[idx], best[idx])

    print("\n== 最优变体分数分布 ==")
    for nm, arr in [("白(测试)", best[y == 0]), ("黑", best[y == 1])]:
        print(f"  {nm}: mean={arr.mean():.3f}  p10={np.percentile(arr,10):.3f}  "
              f"p50={np.percentile(arr,50):.3f}  p90={np.percentile(arr,90):.3f}")

    # 高分位置的可解释性: 黑样本得分最高的位置是什么事件?
    print("\n== 可解释性: 每个黑样本 z 分最高位置的事件类型分布 ==")
    from collections import Counter
    hits = Counter()
    for i, u in enumerate(test_users):
        if u["label"] == 1:
            j = int(np.argmax(z_sum[i]))
            hits[EVENT_TYPES[u["types"][j]]] += 1
    for e, c in hits.most_common():
        tag = "«敏感»" if e in SENSITIVE else ""
        print(f"    {e:<6} {c:>4} {tag}")

    np.savez("mem_scores_v2.npz", score=best, label=y,
             uid=[u["uid"] for u in test_users])


if __name__ == "__main__":
    main()
