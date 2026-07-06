# -*- coding: utf-8 -*-
"""
Sparkov v2: 字段级遮罩 MEM
--------------------------
v1(整事件遮罩)失败诊断: 信用卡流序列依赖弱, 欺诈签名是"字段间不一致"
(类别×金额×时段不匹配)。整事件遮罩下模型看不到类别去猜金额, 天然高熵。
v2 改为字段级遮罩: 只遮一个字段、其余可见 —
  P(金额|类别,时段,上下文) / P(类别|金额,时段,上下文) / P(间隔|其余)
欺诈交易的字段组合不符合正常条件分布 → 对应字段 CE 高。
归一化: 金额CE按可见类别分桶归一; 类别/间隔CE按间隔桶归一。
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from sparkov_experiment import (load_windows, encode_all, to_tensors, topk_mean,
                                metrics, report, N_TYPES, N_AMOUNT, N_GAP, WIN,
                                SEED)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FIELDS = ("type", "amt", "gap")


class FieldMEM(nn.Module):
    """字段级遮罩: 每个字段有独立的 [MASK] 向量, 其余字段保持可见"""
    def __init__(self, d=64):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.hour_proj = nn.Linear(2, 8)
        self.mask_t = nn.Parameter(torch.randn(32) * 0.02)
        self.mask_a = nn.Parameter(torch.randn(8) * 0.02)
        self.mask_g = nn.Parameter(torch.randn(16) * 0.02)
        self.in_proj = nn.Linear(32 + 8 + 16 + 8, d)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_t = nn.Linear(d, N_TYPES)
        self.head_a = nn.Linear(d, N_AMOUNT)
        self.head_g = nn.Linear(d, N_GAP)

    def _head(self, field):
        return {"type": self.head_t, "amt": self.head_a,
                "gap": self.head_g}[field]

    def forward(self, tp, am, gp, hr, mask, field):
        """mask: (B,L) bool; field: 被遮的字段名, 其余字段完全可见"""
        et, ea, eg = self.type_emb(tp), self.amount_emb(am), self.gap_emb(gp)
        m = mask.unsqueeze(-1)
        if field == "type":
            et = torch.where(m, self.mask_t.expand_as(et), et)
        elif field == "amt":
            ea = torch.where(m, self.mask_a.expand_as(ea), ea)
        else:
            eg = torch.where(m, self.mask_g.expand_as(eg), eg)
        x = self.in_proj(torch.cat([et, ea, eg, self.hour_proj(hr)], -1))
        x = x + self.pos_emb(torch.arange(x.size(1), device=x.device))[None]
        h = self.encoder(x)
        return self._head(field)(h)


def target(field, tp, am, gp):
    return {"type": tp, "amt": am, "gap": gp}[field]


def train(model, wins, device, epochs=20, bs=64, lr=1e-3, ratio=0.15):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    model.train()
    for ep in range(epochs):
        order = list(range(len(wins)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [wins[j] for j in order[s:s + bs]]
            tp, am, gp, hr = to_tensors(batch, device)
            field = FIELDS[nb % 3]           # 轮换字段
            mask = torch.rand(tp.shape) < ratio
            mask[:, 0] |= ~mask.any(1)
            mask = mask.to(device)
            logits = model(tp, am, gp, hr, mask, field)
            y = target(field, tp, am, gp)
            loss = ce(logits[mask], y[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_field_ce(model, wins, device, stride=8, bs=128):
    """对每个字段各做一轮步长遮罩, 返回逐位置逐字段 CE"""
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{f: np.zeros(WIN) for f in FIELDS} for _ in wins]
    for i, w in enumerate(wins):
        out[i]["gb"] = np.array(w["gaps"])
        out[i]["tp"] = np.array(w["types"])
    for field in FIELDS:
        for r in range(stride):
            for s in range(0, len(wins), bs):
                batch = wins[s:s + bs]
                tp, am, gp, hr = to_tensors(batch, device)
                pos = torch.arange(WIN)
                mask = ((pos % stride) == r)[None].expand(len(batch), WIN)
                logits = model(tp, am, gp, hr, mask.to(device), field)
                y = target(field, tp, am, gp)
                idx = mask[0].nonzero().flatten()
                for i in range(len(batch)):
                    out[s + i][field][idx] = ce(
                        logits[i, idx], y[i, idx],
                        reduction="none").cpu().numpy()
    return out


def norm_by(pcs_train, pcs, field, bucket_key, n_bucket):
    """按 bucket_key(gb=间隔桶 / tp=类别) 做难度 z-score"""
    all_v = np.concatenate([p[field] for p in pcs_train])
    all_b = np.concatenate([p[bucket_key] for p in pcs_train])
    mu = np.full(n_bucket, all_v.mean())
    sd = np.full(n_bucket, max(all_v.std(), 1e-3))
    for b in range(n_bucket):
        m = all_b == b
        if m.sum() >= 30:
            mu[b], sd[b] = all_v[m].mean(), max(all_v[m].std(), 1e-3)
    return [(p[field] - mu[np.clip(p[bucket_key], 0, n_bucket-1)]) /
            sd[np.clip(p[bucket_key], 0, n_bucket-1)] for p in pcs]


def main():
    device = "cpu"
    print("== 加载数据 (同 v1 协议) ==", flush=True)
    wins = load_windows()
    tr, tw, bl = encode_all(wins)
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    print(f"训练 {len(tr)}, 测试 {len(tw)}白+{len(bl)}黑\n", flush=True)

    print("== FieldMEM 训练 (字段级遮罩轮换) ==", flush=True)
    model = FieldMEM().to(device)
    train(model, tr, device, epochs=20)
    torch.save(model.state_dict(), "sparkov_fieldmem.pt")

    print("\n== 打分 (3 字段 × 步长遮罩) ==", flush=True)
    pcs_tr = per_position_field_ce(model, tr, device)
    pcs = per_position_field_ce(model, test_wins, device)
    # 金额CE按可见类别归一(难度由类别决定); 类别/间隔CE按间隔桶归一
    za = norm_by(pcs_tr, pcs, "amt", "tp", N_TYPES)
    zt = norm_by(pcs_tr, pcs, "type", "gb", N_GAP)
    zg = norm_by(pcs_tr, pcs, "gap", "gb", N_GAP)

    combos = {"仅amt|cat可见": za, "仅type": zt,
              "amt+type": [a+b for a,b in zip(za,zt)],
              "amt+type+gap": [a+b+c for a,b,c in zip(za,zt,zg)]}
    best_name, best_auc, best_s, best_z = None, 0, None, None
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"z-norm top-{k} ({name})", y, s)
            if auc > best_auc:
                best_name, best_auc, best_s, best_z = f"top-{k} {name}", auc, s, z

    print(f"\n== 定位能力 (变体: {best_name}) ==", flush=True)
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(best_z[i].tolist())
            pos_y.extend(w["fr"].tolist())
    pos_y = np.array(pos_y)
    print(f"  黑窗口内交易 {len(pos_y)} 笔, 欺诈 {int(pos_y.sum())} 笔")
    print(f"  位置级 AUC = {roc_auc_score(pos_y, np.array(pos_z)):.4f}")


if __name__ == "__main__":
    main()
