# -*- coding: utf-8 -*-
"""
以太坊 v2: V8 架构(自回归+时间偏置注意力)移植到真实资金流。
对照 v1(事件遮罩 MEM3) = 0.904。任务/切分/打分网格与 v1 完全一致。
"""
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

from eth_experiment import (load, build_encoder, N_TYPES, N_AMOUNT, N_GAP,
                            GAP_EDGES, MAX_LEN, bucket_norm, topk_mean, SEED)
from mem_arch import TimeBiasLayer

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HEAD_MAP = {"h_type": "types", "h_gap": "gaps", "h_amt": "amounts"}
N_CLS = {"h_type": N_TYPES, "h_gap": N_GAP, "h_amt": N_AMOUNT}


def enc_with_ts(enc, u):
    e = enc(u)
    e["ts"] = [t for t, _, _ in u["txs"][-MAX_LEN:]]
    return e


class ARBiasMEM(nn.Module):
    def __init__(self, d=64, layers=2, nhead=4):
        super().__init__()
        self.nhead = nhead
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.amount_emb = nn.Embedding(N_AMOUNT, 8)
        self.gap_emb = nn.Embedding(N_GAP, 16)
        self.hour_proj = nn.Linear(2, 8)
        self.in_proj = nn.Linear(32 + 8 + 16 + 8, d)
        self.bias_emb = nn.Embedding(len(GAP_EDGES) + 1, nhead)
        self.layers = nn.ModuleList([TimeBiasLayer(d, nhead, 128)
                                     for _ in range(layers)])
        self.heads = nn.ModuleDict({k: nn.Linear(d, N_CLS[k])
                                    for k in HEAD_MAP})

    def forward(self, T, pad):
        h = T["hours"]
        x = self.in_proj(torch.cat(
            [self.type_emb(T["types"]), self.amount_emb(T["amounts"]),
             self.gap_emb(T["gaps"]),
             self.hour_proj(torch.stack([torch.sin(2*math.pi*h/24),
                                         torch.cos(2*math.pi*h/24)], -1))],
            -1))
        B, L = pad.shape
        dt = (T["ts"][:, :, None] - T["ts"][:, None, :]).abs()
        dtb = torch.bucketize(dt.float(),
                              torch.tensor(GAP_EDGES, dtype=torch.float32,
                                           device=dt.device))
        bias = self.bias_emb(dtb).permute(0, 3, 1, 2).reshape(B*self.nhead, L, L)
        cm = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        bias = bias + cm[None]
        for lyr in self.layers:
            x = lyr(x, bias, pad)
        return {k: head(x) for k, head in self.heads.items()}


def to_tensors(batch, device):
    n = len(batch)
    L = max(len(u["types"]) for u in batch)
    T = {}
    for f in ("types", "amounts", "gaps"):
        M = torch.zeros(n, L, dtype=torch.long)
        for i, u in enumerate(batch):
            M[i, :len(u[f])] = torch.tensor(u[f])
        T[f] = M.to(device)
    H = torch.zeros(n, L)
    TS = torch.zeros(n, L, dtype=torch.float64)
    pad = torch.ones(n, L, dtype=torch.bool)
    for i, u in enumerate(batch):
        m = len(u["hours"])
        H[i, :m] = torch.tensor(u["hours"])
        TS[i, :m] = torch.tensor(u["ts"], dtype=torch.float64)
        pad[i, :m] = False
    T["hours"] = H.to(device)
    T["ts"] = TS.to(device)
    return T, pad.to(device)


def train(model, users, device, epochs=25, bs=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.functional.cross_entropy
    model.train()
    for ep in range(epochs):
        order = list(range(len(users)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            batch = [users[j] for j in order[s:s + bs]]
            T, pad = to_tensors(batch, device)
            out = model(T, pad)
            loss = 0
            m = ~pad[:, 1:]
            for k, f in HEAD_MAP.items():
                loss = loss + ce(out[k][:, :-1][m], T[f][:, 1:][m])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss {tot/nb:.4f}", flush=True)


@torch.no_grad()
def per_position_ce(model, users, device, bs=64):
    model.eval()
    ce = nn.functional.cross_entropy
    out = [{"t": np.zeros(len(u["types"])), "g": np.zeros(len(u["types"])),
            "a": np.zeros(len(u["types"])), "gb": np.array(u["gaps"])}
           for u in users]
    key = {"h_type": "t", "h_gap": "g", "h_amt": "a"}
    for s in range(0, len(users), bs):
        batch = users[s:s + bs]
        T, pad = to_tensors(batch, device)
        o = model(T, pad)
        for i in range(len(batch)):
            m = len(batch[i]["types"])
            if m < 2:
                continue
            for k, f in HEAD_MAP.items():
                out[s+i][key[k]][1:m] = ce(o[k][i, :m-1], T[f][i, 1:m],
                                           reduction="none").cpu().numpy()
    return out


def report(name, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    r1 = tpr[np.searchsorted(fpr, 0.01, side="right") - 1]
    print(f"  {name:<28} AUC={auc:.4f}  KS={np.max(tpr-fpr):.4f}  "
          f"R@FPR1%={r1:.1%}", flush=True)
    return auc


def main():
    device = "cpu"
    users = load()
    normal = [u for u in users if u["label"] == 0]
    phish = [u for u in users if u["label"] == 1]
    rng = random.Random(SEED)
    rng.shuffle(normal)
    enc = build_encoder(normal[:4000])
    train_u = [enc_with_ts(enc, u) for u in normal[:4000]]
    test_u = ([enc_with_ts(enc, u) for u in normal[4000:5000]] +
              [enc_with_ts(enc, u) for u in phish])
    y = np.array([u["label"] for u in test_u])
    print(f"训练 {len(train_u)} | 测试 {int((y==0).sum())}白+{int(y.sum())}黑\n",
          flush=True)

    print("== ARBiasMEM 训练 (V8 架构) ==", flush=True)
    model = ARBiasMEM().to(device)
    train(model, train_u, device)

    print("\n== 打分 ==", flush=True)
    pcs_tr = per_position_ce(model, train_u, device)
    pcs = per_position_ce(model, test_u, device)
    zt = bucket_norm(pcs_tr, pcs, "t")
    zg = bucket_norm(pcs_tr, pcs, "g")
    za = bucket_norm(pcs_tr, pcs, "a")
    combos = {"仅type": zt, "仅amt": za, "仅gap": zg,
              "type+amt": [a+b for a, b in zip(zt, za)],
              "SUM": [a+b+c for a, b, c in zip(zt, za, zg)]}
    best = (0, None)
    for name, z in combos.items():
        for agg, fn in [("mean", lambda a: a.mean()),
                        ("top5", lambda a: topk_mean(a, 5)),
                        ("top10", lambda a: topk_mean(a, 10))]:
            s = np.array([fn(a) for a in z])
            auc = report(f"{agg} ({name})", y, s)
            if auc > best[0]:
                best = (auc, f"{agg} {name}")
    print(f"\nv1(事件遮罩) = 0.9035 | v2(AR+时间偏置) 最优 = {best[0]:.4f} "
          f"({best[1]})", flush=True)


if __name__ == "__main__":
    main()
