# -*- coding: utf-8 -*-
"""
Sparkov v4: 自回归架构移植到真实卡数据(对照 v3 字段遮罩 0.886)。
因果注意力, 位置 i 的隐状态预测第 i+1 笔的 [类别/全局金额桶/间隔桶/个人金额桶]。
打分: 逐位置四头 CE, amt/pamt 按目标类别归一, type/gap 按间隔桶归一, 网格选优。
"""
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from sparkov_experiment import (load_windows, N_TYPES, N_AMOUNT, N_GAP,
                                WIN, SEED)
from sparkov_v3 import encode_all, N_PAMT, topk_mean, norm_by

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HEAD_MAP = {"h_type": "type", "h_amt": "amt", "h_gap": "gap", "h_pamt": "pamt"}
N_CLS = {"type": N_TYPES, "amt": N_AMOUNT, "gap": N_GAP, "pamt": N_PAMT}


class ARMEM(nn.Module):
    def __init__(self, d=64, layers=2):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.pamt_emb = nn.Embedding(N_PAMT, 16)
        self.in_proj = nn.Linear(32 + 8 + 16 + 16, d)
        self.pos_emb = nn.Embedding(WIN, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=128, dropout=0.1,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.heads = nn.ModuleDict({k: nn.Linear(d, N_CLS[f])
                                    for k, f in HEAD_MAP.items()})

    def forward(self, T):
        x = self.in_proj(torch.cat([self.type_emb(T["type"]),
                                    self.amount_emb(T["amt"]),
                                    self.gap_emb(T["gap"]),
                                    self.pamt_emb(T["pamt"])], -1))
        L = x.size(1)
        x = x + self.pos_emb(torch.arange(L, device=x.device))[None]
        cm = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        h = self.encoder(x, mask=cm)
        return {k: head(h) for k, head in self.heads.items()}


def to_tensors(batch, device):
    T = {}
    for f, key in (("type", "types"), ("amt", "amounts"),
                   ("gap", "gaps"), ("pamt", "pamts")):
        T[f] = torch.tensor([w[key] for w in batch], dtype=torch.long).to(device)
    return T


def train(model, wins, device, epochs=20, bs=64, lr=1e-3):
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
            T = to_tensors(batch, device)
            out = model(T)
            loss = sum(ce(out[k][:, :-1].reshape(-1, N_CLS[f]),
                          T[f][:, 1:].reshape(-1))
                       for k, f in HEAD_MAP.items())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, wins, device, bs=128):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{f: np.zeros(WIN) for f in HEAD_MAP.values()} for _ in wins]
    for i, w in enumerate(wins):
        out[i]["gb"] = np.array(w["gaps"])
        out[i]["tp"] = np.array(w["types"])
    for s in range(0, len(wins), bs):
        batch = wins[s:s + bs]
        T = to_tensors(batch, device)
        o = model(T)
        for i in range(len(batch)):
            for k, f in HEAD_MAP.items():
                out[s+i][f][1:] = ce(o[k][i, :-1], T[f][i, 1:],
                                     reduction="none").cpu().numpy()
    return out


def report(name, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    print(f"  {name:<30} AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc


def main():
    device = "cpu"
    wins = load_windows()
    tr, tw, bl = encode_all(wins)
    tr = tr[:6000]
    test_wins = tw + bl
    y = np.array([w["label"] for w in test_wins])
    print(f"训练 {len(tr)}, 测试 {len(tw)}白+{len(bl)}黑\n", flush=True)

    print("== ARMEM 训练 (因果, 预测下一笔四字段) ==", flush=True)
    model = ARMEM().to(device)
    train(model, tr, device)

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, tr, device)
    pcs = per_position_ce(model, test_wins, device)
    za = norm_by(pcs_tr, pcs, "amt", "tp", N_TYPES)
    zp = norm_by(pcs_tr, pcs, "pamt", "tp", N_TYPES)
    zt = norm_by(pcs_tr, pcs, "type", "gb", N_GAP)
    zg = norm_by(pcs_tr, pcs, "gap", "gb", N_GAP)
    combos = {"仅pamt": zp, "仅type": zt, "type+pamt": [a+b for a, b in zip(zt, zp)],
              "SUM": [a+b+c+d for a, b, c, d in zip(zt, za, zg, zp)]}
    best = (0, None, None)
    for name, z in combos.items():
        for k in (3, 5, 10):
            s = np.array([topk_mean(a, k) for a in z])
            auc = report(f"top-{k} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"top-{k} {name}", z)
    print(f"\nv3(字段遮罩+pamt) = 0.8857 | v4(AR) 最优 = {best[0]:.4f} "
          f"({best[1]})", flush=True)
    pos_z, pos_y = [], []
    for i, w in enumerate(test_wins):
        if w["label"] == 1:
            pos_z.extend(best[2][i].tolist())
            pos_y.extend(w["fr"].tolist())
    print(f"定位 AUC = {roc_auc_score(np.array(pos_y), np.array(pos_z)):.4f} "
          f"(v3 = 0.885)", flush=True)


if __name__ == "__main__":
    main()
