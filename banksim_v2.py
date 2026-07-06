# -*- coding: utf-8 -*-
"""
BankSim v2: 自清洗训练 —— 解决训练集 19% 欺诈污染问题。
流程: 第1轮全量训练 → 给训练窗口自打分 → 踢掉分数最高的 18% →
      第2轮用清洗后的训练集重训 → 同一测试集对比。
对照: v1(不清洗) 窗口 0.736 / 定位 0.853。
"""
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from banksim_experiment import (load_windows, encode_all, FieldMEM, train,
                                per_position_field_ce, norm_by, topk_mean,
                                N_TYPES, N_GAP, SEED)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DROP_FRAC = 0.18


def score_windows(model, pcs_ref, wins, device):
    pcs = per_position_field_ce(model, wins, device)
    za = norm_by(pcs_ref, pcs, "amt", "tp", N_TYPES)
    zt = norm_by(pcs_ref, pcs, "type", "gb", N_GAP)
    z = [a + b for a, b in zip(za, zt)]
    return np.array([topk_mean(a, 5) for a in z]), z


def evaluate(tag, model, pcs_tr_ref, test_wins, y, device):
    s, z = score_windows(model, pcs_tr_ref, test_wins, device)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(z[i].tolist()); pos_y.extend(w["fr"].tolist())
    loc = roc_auc_score(np.array(pos_y), np.array(pos_z))
    print(f"[{tag}] AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}  定位={loc:.4f}", flush=True)
    return auc


def main():
    device = "cpu"
    wins = load_windows()
    tr, tw, bl = encode_all(wins)
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    n_dirty = sum(1 for w in tr if w["label"] == 1)
    print(f"训练 {len(tr)}(真实污染 {n_dirty}={n_dirty/len(tr):.0%}), "
          f"测试 {len(tw)}白+{len(bl)}黑\n", flush=True)

    # ---- 第 1 轮: 带污染全量训练 ----
    print("== 第1轮训练(带污染, 复现 v1) ==", flush=True)
    m1 = FieldMEM().to(device)
    train(m1, tr, device)
    pcs_tr1 = per_position_field_ce(m1, tr, device)
    evaluate("第1轮(=v1基线)", m1, pcs_tr1, test_wins, y, device)

    # ---- 自清洗: 给训练集自打分, 踢掉最高的 DROP_FRAC ----
    s_tr, _ = score_windows(m1, pcs_tr1, tr, device)
    keep_idx = np.argsort(s_tr)[:int(len(tr) * (1 - DROP_FRAC))]
    tr2 = [tr[i] for i in keep_idx]
    kept_dirty = sum(1 for w in tr2 if w["label"] == 1)
    dropped = len(tr) - len(tr2)
    dropped_dirty = n_dirty - kept_dirty
    print(f"\n自清洗: 踢掉分数最高的 {dropped} 个窗口, "
          f"其中真欺诈 {dropped_dirty} 个(清洗命中率 {dropped_dirty/dropped:.0%});"
          f" 剩余污染 {kept_dirty}/{len(tr2)}={kept_dirty/len(tr2):.1%}", flush=True)

    # ---- 第 2 轮: 清洗后重训 ----
    print("\n== 第2轮训练(清洗后) ==", flush=True)
    random.seed(SEED); torch.manual_seed(SEED)
    m2 = FieldMEM().to(device)
    train(m2, tr2, device)
    pcs_tr2 = per_position_field_ce(m2, tr2, device)
    evaluate("第2轮(自清洗后)", m2, pcs_tr2, test_wins, y, device)


if __name__ == "__main__":
    main()
